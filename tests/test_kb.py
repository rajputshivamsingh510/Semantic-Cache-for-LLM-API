import os, asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.backends import MemoryBackend
from app.cache import SemanticCache
from app.embeddings import HashEmbedder
from app.kb_routes import make_kb_router
from app.llm import MockLLM
from app.rag import MemoryKB, chunk_text

PG = os.getenv("TEST_DATABASE_URL")     # e.g. postgresql://postgres:pw@localhost:5432/kbtest (needs pgvector)
HR = "Employees receive 24 days of paid leave per year. Sick leave is 12 days per year."
IT = "To reset a VPN password open the IT portal and choose Reset VPN access."

def make_pdf(pages):
    """Tiny valid PDF with one line of text per page."""
    n = len(pages); font = 3 + 2 * n
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [" + " ".join(f"{3 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>"]
    for i, t in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> >>")
        s = f"BT /F1 12 Tf 50 700 Td ({t}) Tj ET"
        objs.append(f"<< /Length {len(s)} >>\nstream\n{s}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()

@pytest.fixture(params=["memory", pytest.param("pg", marks=pytest.mark.skipif(not PG, reason="set TEST_DATABASE_URL"))])
def client(request):
    cache = SemanticCache(MemoryBackend(), HashEmbedder(), threshold=0.8, default_ttl=3600)
    get_emb = lambda: cache.emb
    if request.param == "pg":
        import psycopg
        from app.pgstore import PgVectorKB
        with psycopg.connect(PG, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS kb_chunks, kb_docs")
        store = PgVectorKB(PG, get_emb)
    else:
        store = MemoryKB(get_emb)
    async def no_limit(request, response): pass
    app = FastAPI()
    app.include_router(make_kb_router(cache, lambda: MockLLM(0, 0, 0), no_limit, store=store))
    with TestClient(app) as c:
        c.store = store
        yield c

def ask(c, q):
    return c.post("/v1/kb/ask", json={"question": q}).json()

def test_kb_flow(client):
    c = client
    hr = c.post("/v1/kb/docs", json={"name": "HR", "text": HR}).json()["doc_id"]
    it = c.post("/v1/kb/docs", json={"name": "IT", "text": IT}).json()["doc_id"]
    assert len(c.get("/v1/kb/docs").json()) == 2

    q1 = "How many days of paid leave do employees get?"
    a = ask(c, q1)
    assert not a["cache"]["hit"] and a["sources"][0]["doc"] == "HR" and "24 days" in a["response"]
    b = ask(c, "how many days of paid leave do employees get")
    assert b["cache"]["hit"] and b["sources"] == a["sources"]
    assert not ask(c, "How do I reset my VPN password?")["cache"]["hit"]

    r = c.put(f"/v1/kb/docs/{hr}", json={"name": "HR", "text": HR.replace("24", "30")}).json()
    assert r["cache_entries_removed"] >= 1                         # only answers built from HR
    assert ask(c, "How do I reset my VPN password?")["cache"]["hit"]
    n = ask(c, q1)
    assert not n["cache"]["hit"] and "30 days" in n["response"]

    assert ask(c, q1)["cache"]["hit"]
    add = c.post("/v1/kb/docs", json={"name": "HR2", "text": "Employees receive 30 days of paid leave per year."}).json()
    assert add["cache_entries_removed"] >= 1                       # new related content clears the answer
    assert not ask(c, q1)["cache"]["hit"]

    assert c.delete(f"/v1/kb/docs/{it}").json()["cache_entries_removed"] >= 1
    assert c.get(f"/v1/kb/docs/{it}").status_code == 404

def test_upload_pdf_and_txt(client):
    c = client
    pdf = make_pdf(["Employees receive 24 days of paid leave per year.", "Payroll is processed on the 28th of every month."])
    r = c.post("/v1/kb/upload", files={"file": ("handbook.pdf", pdf, "application/pdf")})
    assert r.status_code == 200 and r.json()["pages"] == 2 and r.json()["chunks"] >= 1
    a = ask(c, "How many days of paid leave do employees get?")
    assert a["sources"][0]["doc"] == "handbook.pdf" and "24 days" in a["response"]
    t = c.post("/v1/kb/upload", files={"file": ("it.txt", IT.encode(), "text/plain")})
    assert t.status_code == 200 and t.json()["pages"] is None
    assert c.post("/v1/kb/upload", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")}).status_code == 422
    assert c.post("/v1/kb/upload", files={"file": ("empty.txt", b"  ", "text/plain")}).status_code == 422

def test_empty_and_limits(client):
    c = client
    assert c.post("/v1/kb/ask", json={"question": "hi"}).status_code == 400
    assert c.post("/v1/kb/docs", json={"text": "x" * 400001}).status_code == 413

def test_chunking_long_sentence():
    chunks = chunk_text("word " * 600)            # one 3000-char "sentence" without punctuation
    assert len(chunks) >= 6 and max(map(len, chunks)) <= 520

@pytest.mark.skipif(not PG, reason="set TEST_DATABASE_URL")
def test_pg_survives_restart(client):
    from app.pgstore import PgVectorKB
    if not isinstance(client.store, PgVectorKB):
        pytest.skip("pg store only")
    client.post("/v1/kb/docs", json={"name": "HR", "text": HR})
    fresh = PgVectorKB(PG, client.store.get_emb)                 # a new process, same database
    docs = asyncio.run(fresh.list())
    assert [d["name"] for d in docs] == ["HR"]