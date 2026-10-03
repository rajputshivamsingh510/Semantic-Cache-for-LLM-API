"""Live demo against a running server:  python demo.py   (start the API first)"""
import time, httpx

URL = "http://localhost:8000"
QUERIES = [
    "What is our leave policy?",              # MISS  -> calls the LLM
    "What is the leave policy?",              # HIT   (paraphrase)
    "Tell me the leave policy for employees", # HIT   (paraphrase)
    "How do I reset my VPN password?",        # MISS  (new topic)
    "How can I reset the VPN password?",     # HIT
    "What is the work from home policy?",     # MISS
]
G, R, X = "\033[92m", "\033[93m", "\033[0m"
with httpx.Client(timeout=30) as c:
    c.post(f"{URL}/cache/invalidate", json={})            # start clean
    print(f"{'result':6} {'sim':>5} {'ms':>7} {'cost $':>9}  query")
    for q in QUERIES:
        r = c.post(f"{URL}/v1/chat", json={"query": q}).json()
        hit = r["cache"]["hit"]
        print(f"{(G+'HIT ' if hit else R+'MISS')+X:15} {r['cache']['similarity']:>5} {r['latency_ms']:>7} "
              f"{r['usage']['cost_usd']:>9}  {q}" + (f"\n        matched -> {r['cache']['matched_query']}" if hit else ""))
    s = c.get(f"{URL}/stats").json()
    print(f"\nhit_rate={s['hit_rate']:.0%}  llm_calls={int(s.get('llm_calls',0))}  "
          f"tokens_saved={int(s.get('tokens_saved',0))}  cost_saved=${s.get('cost_saved_usd',0):.5f}")
