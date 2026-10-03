"""Postgres + pgvector store. Chunks and vectors persist on disk and use no app RAM.
Enable with DATABASE_URL=postgresql://user:pass@host:5432/dbname  (needs `pip install "psycopg[binary]"`)."""
import asyncio, uuid
from .rag import chunk_text, embed_many, pick_sources

def _lit(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

class PgVectorKB:
    def __init__(self, url, get_emb):
        self.url, self.get_emb = url, get_emb
        self._ready, self._lock = False, None

    async def _conn(self):
        import psycopg
        # prepare_threshold=None: works behind connection poolers (Supabase/Neon pgbouncer)
        return await psycopg.AsyncConnection.connect(self.url, prepare_threshold=None)

    async def _ensure(self):
        if self._ready: return
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:
            if self._ready: return
            dim = len(await asyncio.to_thread(self.get_emb().embed, "dimension probe"))
            async with await self._conn() as c:
                await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await c.execute("""CREATE TABLE IF NOT EXISTS kb_docs (
                    id text PRIMARY KEY, name text NOT NULL, text text NOT NULL, created_at timestamptz DEFAULT now())""")
                await c.execute(f"""CREATE TABLE IF NOT EXISTS kb_chunks (
                    id bigserial PRIMARY KEY, doc_id text NOT NULL REFERENCES kb_docs(id) ON DELETE CASCADE,
                    idx int NOT NULL, content text NOT NULL, embedding vector({dim}) NOT NULL)""")
                await c.execute("CREATE INDEX IF NOT EXISTS kb_chunks_doc ON kb_chunks(doc_id)")
                cur = await c.execute("""SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = 'kb_chunks'::regclass AND attname = 'embedding'""")
                have = (await cur.fetchone())[0]
            if have != dim:
                raise RuntimeError(f"kb_chunks stores {have}-dim vectors but the embedder makes {dim}-dim ones. "
                                   "You changed EMBEDDER: run  DROP TABLE kb_chunks, kb_docs;  and re-upload.")
            try:                                   # approximate-nearest-neighbour index (pgvector >= 0.5)
                async with await self._conn() as c:
                    await c.execute("CREATE INDEX IF NOT EXISTS kb_chunks_hnsw ON kb_chunks "
                                    "USING hnsw (embedding vector_cosine_ops)")
            except Exception as e:
                print("HNSW index not created (search still works, just slower):", e)
            self._ready = True

    async def add(self, name, text, doc_id=None):
        await self._ensure()
        chunks = chunk_text(text)
        vecs = await asyncio.to_thread(embed_many, self.get_emb(), chunks)
        doc_id = doc_id or uuid.uuid4().hex[:10]
        async with await self._conn() as c:
            await c.execute("""INSERT INTO kb_docs (id, name, text) VALUES (%s, %s, %s)
                               ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, text = EXCLUDED.text""",
                            (doc_id, name, text))
            await c.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (doc_id,))
            async with c.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO kb_chunks (doc_id, idx, content, embedding) VALUES (%s, %s, %s, %s::vector)",
                    [(doc_id, i, ch, _lit(v)) for i, (ch, v) in enumerate(zip(chunks, vecs))])
        return doc_id, len(chunks), vecs

    async def list(self):
        await self._ensure()
        async with await self._conn() as c:
            cur = await c.execute("""SELECT d.id, d.name, length(d.text),
                (SELECT count(*) FROM kb_chunks k WHERE k.doc_id = d.id) FROM kb_docs d ORDER BY d.created_at""")
            return [{"doc_id": r[0], "name": r[1], "chars": r[2], "chunks": r[3]} for r in await cur.fetchall()]

    async def get(self, doc_id):
        await self._ensure()
        async with await self._conn() as c:
            row = await (await c.execute("SELECT name, text FROM kb_docs WHERE id = %s", (doc_id,))).fetchone()
        return {"doc_id": doc_id, "name": row[0], "text": row[1]} if row else None

    async def remove(self, doc_id) -> bool:
        await self._ensure()
        async with await self._conn() as c:
            cur = await c.execute("DELETE FROM kb_docs WHERE id = %s", (doc_id,))
            return cur.rowcount > 0

    async def search(self, qvec, k=4):
        await self._ensure()
        lit = _lit(qvec)
        async with await self._conn() as c:
            cur = await c.execute("""SELECT 1 - (c.embedding <=> %s::vector), c.doc_id, d.name, c.content
                FROM kb_chunks c JOIN kb_docs d ON d.id = c.doc_id
                ORDER BY c.embedding <=> %s::vector LIMIT %s""", (lit, lit, k))
            hits = [(float(s), d, n, ch) for s, d, n, ch in await cur.fetchall()]
        return pick_sources(hits, k)