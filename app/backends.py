"""Storage backends: Redis (production) and in-memory (dev/tests). Same async interface."""
import json, time
from dataclasses import dataclass
import numpy as np

@dataclass
class Entry:
    id: str
    ns: str
    query: str
    response: str
    embedding: np.ndarray
    created_at: float
    expires_at: float
    meta: dict
    hits: int = 0

class MemoryBackend:
    def __init__(self):
        self.entries: dict[str, Entry] = {}
        self.stats: dict[str, float] = {}
        self.counters: dict[str, tuple[int, float]] = {}

    async def put(self, e: Entry):
        self.entries[e.id] = e
    async def all(self, ns: str):
        now = time.time()
        for k in [k for k, e in self.entries.items() if e.expires_at <= now]:
            del self.entries[k]
        return [e for e in self.entries.values() if e.ns == ns]
    async def delete(self, id: str) -> bool:
        return self.entries.pop(id, None) is not None
    async def clear(self, ns: str | None = None) -> int:
        ids = [k for k, e in self.entries.items() if ns is None or e.ns == ns]
        for k in ids: del self.entries[k]
        return len(ids)
    async def hit(self, id: str):
        if id in self.entries: self.entries[id].hits += 1
    async def incr_stat(self, name: str, n: float = 1):
        self.stats[name] = self.stats.get(name, 0) + n
    async def get_stats(self) -> dict:
        return dict(self.stats)
    async def rate_hit(self, key: str, window: int) -> int:
        now = time.time()
        count, start = self.counters.get(key, (0, now))
        if now - start >= window: count, start = 0, now
        self.counters[key] = (count + 1, start)
        return count + 1

class RedisBackend:
    """Layout:  sc:e:{ns}:{id}  HASH(meta json, emb bytes, hits)  with native EXPIRE for TTL
                sc:stats        HASH of counters (HINCRBYFLOAT)
                sc:rl:{key}     fixed-window rate-limit counter"""
    def __init__(self, url: str):
        import redis.asyncio as redis
        self.r = redis.from_url(url)

    def _key(self, ns, id): return f"sc:e:{ns}:{id}"

    async def put(self, e: Entry):
        meta = {"id": e.id, "ns": e.ns, "query": e.query, "response": e.response,
                "created_at": e.created_at, "expires_at": e.expires_at, "meta": e.meta}
        k = self._key(e.ns, e.id)
        pipe = self.r.pipeline()
        pipe.hset(k, mapping={"meta": json.dumps(meta), "emb": e.embedding.astype(np.float32).tobytes(), "hits": 0})
        pipe.expire(k, max(1, int(e.expires_at - time.time())))
        await pipe.execute()

    def _parse(self, d) -> Entry:
        m = json.loads(d[b"meta"])
        return Entry(m["id"], m["ns"], m["query"], m["response"], np.frombuffer(d[b"emb"], dtype=np.float32),
                     m["created_at"], m["expires_at"], m["meta"], int(d.get(b"hits", 0)))

    async def all(self, ns: str):
        keys = [k async for k in self.r.scan_iter(match=f"sc:e:{ns}:*", count=500)]
        if not keys: return []
        pipe = self.r.pipeline()
        for k in keys: pipe.hgetall(k)
        return [self._parse(d) for d in await pipe.execute() if d]

    async def delete(self, id: str) -> bool:
        keys = [k async for k in self.r.scan_iter(match=f"sc:e:*:{id}")]
        return bool(keys and await self.r.delete(*keys))

    async def clear(self, ns: str | None = None) -> int:
        keys = [k async for k in self.r.scan_iter(match=f"sc:e:{ns or '*'}:*")]
        return await self.r.delete(*keys) if keys else 0

    async def hit(self, id: str):
        async for k in self.r.scan_iter(match=f"sc:e:*:{id}"):
            await self.r.hincrby(k, "hits", 1)

    async def incr_stat(self, name: str, n: float = 1):
        await self.r.hincrbyfloat("sc:stats", name, n)
    async def get_stats(self) -> dict:
        return {k.decode(): float(v) for k, v in (await self.r.hgetall("sc:stats")).items()}

    async def rate_hit(self, key: str, window: int) -> int:
        k = f"sc:rl:{key}"
        pipe = self.r.pipeline()
        pipe.incr(k); pipe.expire(k, window, nx=True)
        return (await pipe.execute())[0]
