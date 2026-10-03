"""Benchmark: no cache vs semantic cache.  python benchmark.py [--embedder hash|sbert] [--threshold 0.8]"""
import argparse, asyncio, random, statistics as st
from app.backends import MemoryBackend
from app.cache import SemanticCache
from app.embeddings import get_embedder
from app.llm import MockLLM
from app.service import answer

# topic -> paraphrases (first one is the "canonical" question)
TOPICS = {
 "leave":   ["What is our leave policy?", "How many leaves can employees take?", "What is the company's leave allowance?",
             "Tell me the leave policy for employees", "How many days of leave do I get?"],
 "wfh":     ["What is the work from home policy?", "Can employees work from home?", "How many days can I work from home?",
             "Explain our remote work policy", "Is working from home allowed?"],
 "expense": ["How do I submit an expense report?", "How to file expense claims?", "What is the process to submit expenses?",
             "Where do I submit my expense reports?", "Steps for submitting an expense claim"],
 "vpn":     ["How do I reset my VPN password?", "Reset VPN password steps", "I forgot my VPN password, how to reset it?",
             "How can I change the VPN password?", "VPN password reset process"],
 "payroll": ["When is payroll processed?", "What date do we get paid?", "When does payroll run each month?",
             "What is the payroll schedule?", "On which day is salary credited?"],
 "health":  ["What does the health insurance cover?", "Health insurance coverage details", "What is covered under company health insurance?",
             "Explain the health insurance benefits", "Does our health insurance cover dental?"],
 "travel":  ["What is the travel reimbursement policy?", "How are travel expenses reimbursed?", "Travel reimbursement rules for employees",
             "How do I get reimbursed for business travel?", "Explain the business travel policy"],
 "onboard": ["What is the onboarding process for new hires?", "How does onboarding work for new employees?", "New hire onboarding steps",
             "Onboarding checklist for new joiners", "What happens in the first week for new hires?"],
}

def workload(n, seed=7):
    rnd = random.Random(seed)
    topics = list(TOPICS)
    weights = [1 / (i + 1) for i in range(len(topics))]            # skewed popularity
    return [(t, rnd.choice(TOPICS[t])) for t in rnd.choices(topics, weights, k=n)]

async def run(use_cache, reqs, args):
    cache = SemanticCache(MemoryBackend(), get_embedder(args.embedder), args.threshold, 3600)
    llm, sem = MockLLM(args.llm_latency, 0.002), asyncio.Semaphore(args.concurrency)
    truth = {q: t for t, qs in TOPICS.items() for q in qs}
    lat, wrong = [], 0
    async def one(q):
        nonlocal wrong
        async with sem:
            r = await answer(cache, llm, q, bypass=not use_cache)
        lat.append(r["latency_ms"])
        if r["cache"]["hit"] and truth[r["cache"]["matched_query"]] != truth[q]:
            wrong += 1
    # warm-up wave sequentially for cache correctness of first-seen queries, then concurrent
    for _, q in reqs[: args.warm]: await one(q)
    await asyncio.gather(*(one(q) for _, q in reqs[args.warm:]))
    s = await cache.b.get_stats(); n = len(reqs)
    return {"avg_latency_ms": st.mean(lat), "p95_latency_ms": sorted(lat)[int(.95 * n) - 1],
            "avg_cost_usd": s.get("cost_usd", 0) / n, "total_cost_usd": s.get("cost_usd", 0),
            "hit_rate": s.get("hits", 0) / n, "llm_calls": int(s.get("llm_calls", 0)),
            "llm_calls_avoided": int(s.get("hits", 0)), "tokens_saved": int(s.get("tokens_saved", 0)),
            "wrong_hits": wrong}

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300); p.add_argument("--embedder", default="hash")
    p.add_argument("--threshold", type=float, default=0.8); p.add_argument("--llm-latency", type=float, default=0.5)
    p.add_argument("--concurrency", type=int, default=25); p.add_argument("--warm", type=int, default=10)
    a = p.parse_args(); reqs = workload(a.n)
    base, sem = await run(False, reqs, a), await run(True, reqs, a)
    rows = [("Average latency (ms)", "avg_latency_ms", "{:.1f}"), ("p95 latency (ms)", "p95_latency_ms", "{:.1f}"),
            ("Average cost / request ($)", "avg_cost_usd", "{:.6f}"), ("Total cost ($)", "total_cost_usd", "{:.4f}"),
            ("Cache hit rate", "hit_rate", "{:.1%}"), ("LLM calls made", "llm_calls", "{}"),
            ("LLM calls avoided", "llm_calls_avoided", "{}"), ("Tokens saved", "tokens_saved", "{}"),
            ("Wrong-topic hits (false positives)", "wrong_hits", "{}")]
    print(f"\n{a.n} requests | embedder={a.embedder} threshold={a.threshold} | mock LLM {a.llm_latency}s base\n")
    print(f"{'Metric':38}{'No cache':>14}{'Semantic cache':>16}"); print("-" * 68)
    for label, k, f in rows: print(f"{label:38}{f.format(base[k]):>14}{f.format(sem[k]):>16}")
    print(f"\nLatency reduction: {1 - sem['avg_latency_ms'] / base['avg_latency_ms']:.1%} | "
          f"Cost reduction: {1 - sem['total_cost_usd'] / base['total_cost_usd']:.1%}")
asyncio.run(main())
