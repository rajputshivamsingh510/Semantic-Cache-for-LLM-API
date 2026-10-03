"""Pluggable embedders. All return L2-normalised float32 vectors. embed_many() is the fast batch path."""
import hashlib, re
import numpy as np

_STOP = set("a an the is are was were be do does did of to for in on at by with and or our your their "
            "what how many much can could would should please tell me us we i you it this that".split())

def _stem(w: str) -> str:
    for suf in ("ing", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v

class HashEmbedder:
    """Dependency-free lexical embedder (stemmed bag-of-words, hashed). For tests/offline
    only: it matches overlapping words, NOT meaning. Use SBert/FastEmbed for real deployments."""
    dim = 512
    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in _STOP:
                continue
            h = int(hashlib.md5(_stem(w).encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return _norm(v)
    def embed_many(self, texts):
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

class SBertEmbedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)
    def embed_many(self, texts):
        if not texts: return np.zeros((0, 384), np.float32)
        return self.model.encode(list(texts), batch_size=32, normalize_embeddings=True).astype(np.float32)

class FastEmbedEmbedder:
    """Lightweight ONNX embedder (no PyTorch). Fits small servers."""
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=name)
    def embed(self, text: str) -> np.ndarray:
        return _norm(next(iter(self.model.embed([text]))))
    def embed_many(self, texts):
        if not texts: return np.zeros((0, 384), np.float32)
        return np.stack([_norm(v) for v in self.model.embed(list(texts))])

def get_embedder(kind: str):
    if kind == "sbert":
        return SBertEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    return HashEmbedder()