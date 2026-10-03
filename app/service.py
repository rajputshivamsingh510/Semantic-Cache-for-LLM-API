"""Core request path, shared by the API and the benchmark."""
import asyncio, inspect, time
from .llm import cost_usd, build_prompt

async def answer(cache, llm, query, ns="default", bypass=False, ttl=None, threshold=None,
                 context=None, context_fn=None):
    """context: fixed text sent to the LLM on a miss.
    context_fn(emb) -> (context, sources), sync or async: retrieval step (RAG), only run on a miss."""
    t0 = time.perf_counter()
    b = cache.b
    emb = await asyncio.to_thread(cache.emb.embed, query)   # keep the event loop free
    entry, sim, _ = (None, 0.0, emb) if bypass else await cache.lookup(query, ns, threshold, emb)
    if entry:
        await b.hit(entry.id)
        saved = entry.meta
        out = {"response": entry.response,
               "cache": {"hit": True, "similarity": round(sim, 4), "matched_query": entry.query,
                         "entry_id": entry.id, "age_s": round(time.time() - entry.created_at, 1),
                         "expires_in_s": round(entry.expires_at - time.time(), 1)},
               "sources": saved.get("sources", []),
               "usage": {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                         "tokens_saved": saved["tokens_in"] + saved["tokens_out"],
                         "cost_saved_usd": saved["cost_usd"]}}
        await b.incr_stat("hits"); await b.incr_stat("tokens_saved", out["usage"]["tokens_saved"])
        await b.incr_stat("cost_saved_usd", saved["cost_usd"])
    else:
        sources = []
        if context_fn:
            res = context_fn(emb)
            context, sources = await res if inspect.isawaitable(res) else res   # sync or async retrieval
        r = await llm.complete(build_prompt(context, query) if context else query)
        c = cost_usd(r["tokens_in"], r["tokens_out"])
        e = await cache.store(query, r["text"], emb, ns, ttl,
                              {"tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "cost_usd": c,
                               "model": getattr(llm, "model", "mock"), "sources": sources})
        out = {"response": r["text"],
               "cache": {"hit": False, "similarity": round(sim, 4), "entry_id": e.id},
               "sources": sources,
               "usage": {"tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "cost_usd": round(c, 6),
                         "tokens_saved": 0, "cost_saved_usd": 0.0}}
        await b.incr_stat("misses"); await b.incr_stat("llm_calls")
        await b.incr_stat("tokens_spent", r["tokens_in"] + r["tokens_out"]); await b.incr_stat("cost_usd", c)
    ms = (time.perf_counter() - t0) * 1000
    out["latency_ms"] = round(ms, 2)
    await b.incr_stat("requests"); await b.incr_stat("latency_ms_total", ms)
    return out