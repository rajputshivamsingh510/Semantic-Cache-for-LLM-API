from fastapi.testclient import TestClient
from app import main
from app.llm import MockLLM

main.llm = MockLLM(0, 0, 0)
c = TestClient(main.app)
DOC = "Employees get 24 days of paid leave per year. Remote work is allowed 3 days a week. Payroll runs on the 28th."

def test_doc_flow():
    d = c.post("/v1/docs", json={"name": "hr", "text": DOC}).json()["doc_id"]
    a = c.post("/v1/ask", json={"doc_id": d, "question": "How many days of leave do employees get?"})
    assert a.status_code == 200 and not a.json()["cache"]["hit"] and "24 days" in a.json()["response"]
    b = c.post("/v1/ask", json={"doc_id": d, "question": "how many days of leave do employees get"})
    assert b.headers["x-cache"] == "HIT"
    r = c.put(f"/v1/docs/{d}", json={"name": "hr", "text": DOC.replace("24", "30")}).json()
    assert r["cache_entries_removed"] == 1                       # new version flushes stale answers
    n = c.post("/v1/ask", json={"doc_id": d, "question": "How many days of leave do employees get?"}).json()
    assert not n["cache"]["hit"] and "30 days" in n["response"]

def test_errors():
    assert c.post("/v1/ask", json={"doc_id": "nope", "question": "x"}).status_code == 404
    assert c.post("/v1/docs", json={"text": "x" * 20001}).status_code == 413
    assert c.get("/").status_code == 200
