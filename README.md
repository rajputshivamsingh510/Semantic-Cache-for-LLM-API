# Semantic Cache for LLM APIs

FastAPI service that embeds each query, looks for a semantically similar earlier query in Redis,
and returns the cached answer on a hit (otherwise calls the LLM and stores the result).

```
Client -> FastAPI -> rate limit -> embed query -> cosine search (per namespace)
                                      |-- sim >= threshold -> HIT: cached answer (0 tokens, 0 $)
                                      '-- else -> MISS: LLM -> store (TTL) -> answer
```

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload                         # in-memory backend, hash embedder (no deps)
docker compose up --build                             # Redis + sentence-transformers embedder
pytest -q                                             # tests (memory + fakeredis)
python benchmark.py --embedder sbert --threshold 0.8  # real benchmark
```
```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"query":"How many leaves can employees take?"}' | jq
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | `query`, `namespace`, `ttl`, `similarity_threshold`, `bypass_cache`. Returns answer + `cache{hit,similarity,matched_query,age_s,expires_in_s}` + `usage{tokens,cost,tokens_saved,cost_saved}` + `latency_ms`. Header `X-Cache: HIT/MISS`. |
| `GET /stats` | requests, hits, misses, hit_rate, avg latency, LLM calls, tokens/cost spent & saved, rate-limited count |
| `POST /cache/invalidate` | by `namespace`, or `similar_to: "<text>"` to drop everything related to a changed document |
| `DELETE /cache/{id}` | remove one entry |

## Design notes (interview talking points)
- **Redis role**: per-entry HASH (`sc:e:{ns}:{id}`) with native `EXPIRE` = TTL; `sc:stats` counters via `HINCRBYFLOAT`;
  `sc:rl:{key}` fixed-window rate limiter via `INCR`+`EXPIRE`. Shared by all API replicas.
- **Threshold trade-off**: higher = fewer false hits, lower hit rate. Benchmark reports wrong-topic hits so you can tune it.
  Typical sbert (MiniLM) range 0.80-0.90; start high and lower it while watching false positives.
- **Invalidation**: TTL (staleness bound), explicit delete, namespace flush (use a version in the namespace, e.g. `hr-v7`),
  and semantic invalidation (`similar_to`).
- **Namespaces** isolate tenants/doc versions so one tenant's answer is never served to another.
- **Known limits / next steps**: lookup is brute-force numpy over a namespace (fine to ~10-50k entries). To scale, move to
  RediSearch vector index (`FT.CREATE ... VECTOR HNSW`, KNN query) or an in-process matrix cache invalidated by a version key.
  No request coalescing yet (concurrent identical misses each call the LLM - add a Redis lock/single-flight).
  Don't cache personalised or time-sensitive prompts; use `bypass_cache`.

## Document Q&A demo (frontend)
```bash
python -m uvicorn app.main:app --port 8000      # set EMBEDDER=sbert for meaning-based matching
# open http://localhost:8000
```
Paste/upload a document -> ask questions. First ask of a topic = MISS (whole document sent to the LLM, slow);
a similar question afterwards = HIT (served from cache, milliseconds). Re-loading a changed document flushes
that document's namespace. Slow-path latency is simulated by the mock LLM unless `LLM_PROVIDER=anthropic`.
Endpoints: `POST /v1/docs`, `PUT /v1/docs/{id}`, `POST /v1/ask`. No RAG yet: the full document is the prompt (max 20k chars).
