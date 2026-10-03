"""Company knowledge base endpoints: documents/PDFs -> chunks -> retrieval -> cache -> LLM."""
import asyncio, os
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from .rag import MAX_KB_DOC_CHARS, MAX_UPLOAD_BYTES, extract_pdf, make_store
from .service import answer

KB_NS = "kb"
RELATED_THRESHOLD = float(os.getenv("KB_INVALIDATE_THRESHOLD", "0.5"))

class DocReq(BaseModel):
    name: str = "document"
    text: str = Field(min_length=1)

class AskReq(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=10)
    bypass_cache: bool = False
    similarity_threshold: float | None = Field(None, ge=0, le=1)

def make_kb_router(cache, get_llm, rate_limit, store=None):
    router = APIRouter(prefix="/v1/kb")
    kb = store or make_store(lambda: cache.emb)
    router.kb = kb

    def check(text):
        if len(text) > MAX_KB_DOC_CHARS:
            raise HTTPException(413, f"document too long ({len(text)} chars, max {MAX_KB_DOC_CHARS})")

    async def save(name, text, doc_id=None):
        """Store a document; clear cached answers it makes stale. Returns the response dict."""
        check(text)
        removed = await cache.invalidate_by_source(doc_id, KB_NS) if doc_id else 0   # built from the old text
        doc_id, n, vecs = await kb.add(name, text, doc_id)
        removed += await cache.invalidate_related(vecs, KB_NS, RELATED_THRESHOLD)    # related to the new text
        return {"doc_id": doc_id, "name": name, "chunks": n, "chars": len(text), "cache_entries_removed": removed}

    @router.get("/docs")
    async def list_docs():
        return await kb.list()

    @router.get("/docs/{doc_id}")
    async def get_doc(doc_id: str):
        d = await kb.get(doc_id)
        if not d: raise HTTPException(404, "document not found")
        return d

    @router.post("/docs")
    async def add_doc(req: DocReq):
        return await save(req.name, req.text)

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        """PDF, .txt or .md file. PDFs need a text layer (scanned images are not OCR'd)."""
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
        name, pages = file.filename or "document", None
        if name.lower().endswith(".pdf"):
            try:
                text, pages = await asyncio.to_thread(extract_pdf, data)
            except Exception as e:
                raise HTTPException(422, f"could not read PDF: {type(e).__name__}: {e}")
        else:
            text = data.decode("utf-8", "ignore")
        if not text.strip():
            raise HTTPException(422, "no text found (a scanned PDF has no text layer)")
        out = await save(name, text)
        out["pages"] = pages
        return out

    @router.put("/docs/{doc_id}")
    async def update_doc(doc_id: str, req: DocReq):
        if not await kb.get(doc_id): raise HTTPException(404, "document not found")
        return await save(req.name, req.text, doc_id)

    @router.delete("/docs/{doc_id}")
    async def delete_doc(doc_id: str):
        if not await kb.remove(doc_id): raise HTTPException(404, "document not found")
        return {"cache_entries_removed": await cache.invalidate_by_source(doc_id, KB_NS)}

    @router.post("/ask")
    async def ask(req: AskReq, request: Request, response: Response):
        await rate_limit(request, response)
        if not await kb.list():
            raise HTTPException(400, "knowledge base is empty - add a document first")
        try:
            out = await answer(cache, get_llm(), req.question, ns=KB_NS, bypass=req.bypass_cache,
                               threshold=req.similarity_threshold,
                               context_fn=lambda emb: kb.search(emb, req.top_k))
        except Exception as e:
            print("LLM/KB ERROR:", type(e).__name__, e)
            raise HTTPException(502, f"request failed: {type(e).__name__}: {e}")
        response.headers["X-Cache"] = "HIT" if out["cache"]["hit"] else "MISS"
        return out

    return router