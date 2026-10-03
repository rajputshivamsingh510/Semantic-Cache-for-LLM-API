import asyncio, time, pytest
from app.backends import MemoryBackend, RedisBackend
from app.cache import SemanticCache
from app.embeddings import HashEmbedder
from app.llm import MockLLM
from app.service import answer
import fakeredis

def make(kind):
    if kind == "redis":
        b = RedisBackend("redis://localhost")
        b.r = fakeredis.FakeAsyncRedis()
    else:
        b = MemoryBackend()
    return SemanticCache(b, HashEmbedder(), threshold=0.8, default_ttl=60), MockLLM(0.0, 0.0)

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "redis"])
async def test_hit_miss_ttl_invalidate(kind):
    c, llm = make(kind)
    a = await answer(c, llm, "What is our leave policy?")
    assert not a["cache"]["hit"]
    b = await answer(c, llm, "what is the leave policy")             # near-duplicate -> HIT
    assert b["cache"]["hit"] and b["usage"]["cost_usd"] == 0 and b["usage"]["tokens_saved"] > 0
    d = await answer(c, llm, "How do I reset my VPN password?")      # unrelated -> MISS
    assert not d["cache"]["hit"]
    assert (await answer(c, llm, "leave policy", ns="other"))["cache"]["hit"] is False  # namespaces isolate
    assert await c.invalidate_similar("leave policy") >= 1             # semantic invalidation
    assert not (await answer(c, llm, "What is our leave policy?"))["cache"]["hit"]
    s = await c.b.get_stats()
    assert s["hits"] == 1 and s["requests"] == 5

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "redis"])
async def test_ttl_expiry(kind):
    c, llm = make(kind)
    await answer(c, llm, "What is our leave policy?", ttl=1)
    await asyncio.sleep(1.2)
    assert not (await answer(c, llm, "What is our leave policy?"))["cache"]["hit"]

@pytest.mark.asyncio
async def test_rate_limit_counter():
    b = MemoryBackend()
    assert [await b.rate_hit("k", 60) for _ in range(3)] == [1, 2, 3]
