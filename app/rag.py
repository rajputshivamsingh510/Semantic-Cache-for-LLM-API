"""Knowledge base: chunking, PDF reading, and the in-memory store.
Stores share one async interface: add / list / get / remove / search  (see also pgstore.py)."""
import asyncio, io, os, re, uuid
import numpy as np

MAX_KB_DOC_CHARS = int(os.getenv("MAX_KB_DOC_CHARS", "400000"))      # ~150+ pages of text
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024

def _split_long(s: str, size: int):
    while len(s) > size:
        cut = s.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        yield s[:cut].strip(); s = s[cut:].strip()
    if s: yield s

def chunk_text(text: str, size: int = 500, overlap: int = 1) -> list[str]:
    """Group sentences into chunks of ~size characters, repeating `overlap` sentences between chunks."""
    sents = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if s.strip():
            sents.extend(_split_long(s.strip(), size))
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            if cur and sum(len(x) + 1 for x in cur) + len(s) > size:
                cur = []                  # overlap would make the chunk too big
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def extract_pdf(data: bytes):
    """-> (text, page_count). Scanned PDFs (images only) have no text layer and give an empty string."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)        # join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # single line breaks -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip(), len(pages)

def embed_many(embedder, texts):
    f = getattr(embedder, "embed_many", None)
    if f: return f(texts)
    return np.stack([embedder.embed(t) for t in texts]) if texts else np.zeros((0, 1), np.float32)

def pick_sources(hits, k=4, min_ratio=0.5):
    """hits: [(score, doc_id, doc_name, chunk)] best first -> (prompt context, sources).
    Chunks scoring below min_ratio x the best are dropped: fewer tokens sent, and a cached answer is
    only tied to the documents it really used."""
    top = [h for h in hits[:k] if h[0] >= min_ratio * hits[0][0]] if hits else []
    context = "\n\n".join(f"[{name}] {chunk}" for _, _, name, chunk in top)
    sources = [{"doc_id": d, "doc": n, "snippet": c[:160], "score": round(s, 3)} for s, d, n, c in top]
    return context, sources

class MemoryKB:
    """Default store: numpy vectors in RAM. Lost on restart."""
    def __init__(self, get_emb):
        self.get_emb, self.docs = get_emb, {}

    async def add(self, name, text, doc_id=None):
        chunks = chunk_text(text)
        vecs = await asyncio.to_thread(embed_many, self.get_emb(), chunks)
        doc_id = doc_id or uuid.uuid4().hex[:10]
        self.docs[doc_id] = {"name": name, "text": text, "chunks": chunks, "vecs": vecs}
        return doc_id, len(chunks), vecs

    async def list(self):
        return [{"doc_id": i, "name": d["name"], "chars": len(d["text"]), "chunks": len(d["chunks"])}
                for i, d in self.docs.items()]

    async def get(self, doc_id):
        d = self.docs.get(doc_id)
        return {"doc_id": doc_id, "name": d["name"], "text": d["text"]} if d else None

    async def remove(self, doc_id) -> bool:
        return self.docs.pop(doc_id, None) is not None

    async def search(self, qvec, k=4):
        hits = []
        for doc_id, d in self.docs.items():
            if len(d["chunks"]):
                for i, s in enumerate(d["vecs"] @ qvec):
                    hits.append((float(s), doc_id, d["name"], d["chunks"][i]))
        hits.sort(key=lambda h: -h[0])
        return pick_sources(hits, k)

def make_store(get_emb):
    url = os.getenv("DATABASE_URL", "")
    if url:
        from .pgstore import PgVectorKB
        return PgVectorKB(url, get_emb)
    return MemoryKB(get_emb)