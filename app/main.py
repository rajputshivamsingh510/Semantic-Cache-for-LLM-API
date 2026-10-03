import uuid
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .config import settings
from .backends import MemoryBackend, RedisBackend
from .cache import SemanticCache
from .embeddings import get_embedder
from .llm import get_llm
from .service import answer
from .kb_routes import make_kb_router

app = FastAPI(title="Semantic Cache for LLM APIs")
backend = RedisBackend(settings.redis_url) if settings.redis_url else MemoryBackend()
cache = SemanticCache(backend, get_embedder(settings.embedder), settings.similarity_threshold,
                      settings.default_ttl, settings.max_entries_per_ns)
llm = get_llm()

class ChatReq(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    namespace: str = "default"          # e.g. tenant or policy-doc version
    bypass_cache: bool = False
    ttl: int | None = Field(None, ge=1)
    similarity_threshold: float | None = Field(None, ge=0, le=1)

DOCS: dict[str, dict] = {}          # doc_id -> {name, text}   (in-process; fine for a demo)

class DocReq(BaseModel):
    name: str = "document"
    text: str = Field(min_length=1)

class AskReq(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=2000)
    bypass_cache: bool = False
    similarity_threshold: float | None = Field(None, ge=0, le=1)

async def _rate_limit(request: Request, response: Response):
    key = request.headers.get("x-api-key") or request.client.host
    n = await backend.rate_hit(key, settings.rate_window)
    response.headers["X-RateLimit-Remaining"] = str(max(0, settings.rate_limit - n))
    if n > settings.rate_limit:
        await backend.incr_stat("rate_limited")
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(settings.rate_window)})
app.include_router(make_kb_router(cache, lambda: llm, _rate_limit))
class InvalidateReq(BaseModel):
    namespace: str | None = None
    similar_to: str | None = None       # drop entries semantically related to this text
    threshold: float = 0.7

@app.post("/v1/chat")
async def chat(req: ChatReq, request: Request, response: Response):
    await _rate_limit(request, response)
    out = await answer(cache, llm, req.query, req.namespace, req.bypass_cache, req.ttl, req.similarity_threshold)
    response.headers["X-Cache"] = "HIT" if out["cache"]["hit"] else "MISS"
    return out

@app.get("/stats")
async def stats():
    s = await backend.get_stats()
    req = s.get("requests", 0)
    return {**s, "hit_rate": round(s.get("hits", 0) / req, 4) if req else 0.0,
            "avg_latency_ms": round(s.get("latency_ms_total", 0) / req, 2) if req else 0.0}

@app.post("/cache/invalidate")
async def invalidate(req: InvalidateReq):
    if req.similar_to:
        return {"removed": await cache.invalidate_similar(req.similar_to, req.namespace or "default", req.threshold)}
    return {"removed": await cache.invalidate_ns(req.namespace)}

@app.delete("/cache/{entry_id}")
async def delete_entry(entry_id: str):
    if not await cache.invalidate_id(entry_id):
        raise HTTPException(404, "entry not found")
    return {"removed": 1}

@app.get("/health")
async def health():
    return {"status": "ok", "backend": type(backend).__name__, "embedder": settings.embedder,
            "threshold": settings.similarity_threshold}


# ---- document Q&A (the cache sits in front of "send the whole document to the LLM") ----
def _check_doc(text: str):
    if len(text) > settings.max_doc_chars:
        raise HTTPException(413, f"document too long ({len(text)} chars, max {settings.max_doc_chars})")

@app.post("/v1/docs")
async def create_doc(req: DocReq):
    _check_doc(req.text)
    doc_id = uuid.uuid4().hex[:10]
    DOCS[doc_id] = {"name": req.name, "text": req.text}
    return {"doc_id": doc_id, "chars": len(req.text), "approx_tokens": len(req.text) // 4}

@app.put("/v1/docs/{doc_id}")
async def update_doc(doc_id: str, req: DocReq):
    """New version of the document -> every cached answer for it is stale, so flush its namespace."""
    if doc_id not in DOCS: raise HTTPException(404, "document not found")
    _check_doc(req.text)
    DOCS[doc_id] = {"name": req.name, "text": req.text}
    return {"doc_id": doc_id, "cache_entries_removed": await cache.invalidate_ns(doc_id),
            "chars": len(req.text), "approx_tokens": len(req.text) // 4}

@app.post("/v1/ask")
async def ask(req: AskReq, request: Request, response: Response):
    await _rate_limit(request, response)
    doc = DOCS.get(req.doc_id)
    if not doc: raise HTTPException(404, "document not found")
    try:
        out = await answer(cache, llm, req.question, ns=req.doc_id, bypass=req.bypass_cache,
                           threshold=req.similarity_threshold, context=doc["text"])
    except Exception as e:                      # bad key, rate limit, Ollama not running...
        raise HTTPException(502, f"LLM call failed: {type(e).__name__}: {e}")
    response.headers["X-Cache"] = "HIT" if out["cache"]["hit"] else "MISS"
    return out

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
