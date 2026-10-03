import time, uuid
import numpy as np
from .backends import Entry

class SemanticCache:
    def __init__(self, backend, embedder, threshold=0.88, default_ttl=3600, max_entries=5000):
        self.b, self.emb = backend, embedder
        self.threshold, self.default_ttl, self.max_entries = threshold, default_ttl, max_entries

    async def lookup(self, query: str, ns="default", threshold: float | None = None, emb=None):
        """Return (entry|None, best_similarity, embedding). Cosine similarity on normalised vectors."""
        emb = self.emb.embed(query) if emb is None else emb
        entries = await self.b.all(ns)
        if not entries:
            return None, 0.0, emb
        sims = np.stack([e.embedding for e in entries]) @ emb
        i = int(np.argmax(sims))
        best = float(sims[i])
        if best >= (self.threshold if threshold is None else threshold):
            return entries[i], best, emb
        return None, best, emb

    async def store(self, query, response, emb, ns="default", ttl=None, meta=None) -> Entry:
        ttl = ttl or self.default_ttl
        now = time.time()
        e = Entry(uuid.uuid4().hex[:12], ns, query, response, emb, now, now + ttl, meta or {})
        await self.b.put(e)
        return e

    # ---- invalidation -------------------------------------------------
    async def invalidate_id(self, id): return await self.b.delete(id)
    async def invalidate_ns(self, ns=None): return await self.b.clear(ns)
    async def invalidate_similar(self, text, ns="default", threshold=0.7) -> int:
        """Drop every entry related to `text` (e.g. the policy doc just changed)."""
        emb = self.emb.embed(text)
        n = 0
        for e in await self.b.all(ns):
            if float(e.embedding @ emb) >= threshold:
                n += await self.b.delete(e.id)
        return n
