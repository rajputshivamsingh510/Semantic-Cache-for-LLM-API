import asyncio, re
from .config import settings
from .embeddings import _STOP, _stem

DOC_HEAD, Q_MARK = "Document:\n", "\n\nQuestion: "

def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def build_prompt(context: str, query: str) -> str:
    return (f"{DOC_HEAD}{context}{Q_MARK}{query}\n"
            "Answer using only the document. If the document does not contain the answer, say so.")

class MockLLM:
    """Simulated LLM. Latency grows with prompt size. Answers by picking best-matching sentences."""
    def __init__(self, base_latency=0.8, per_token=0.002, per_token_in=0.0003):
        self.base, self.per, self.per_in = base_latency, per_token, per_token_in

    def _extract(self, prompt: str) -> str:
        doc, q = prompt[len(DOC_HEAD):].split(Q_MARK, 1)
        q = q.split("\nAnswer using")[0]
        kw = {_stem(w) for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOP}
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", doc) if s.strip()]
        score = [len(kw & {_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower())}) for s in sents]
        top = sorted(sorted(range(len(sents)), key=lambda i: -score[i])[:2])
        if not score or max(score) == 0:
            return "I couldn't find that in the document."
        return " ".join(sents[i] for i in top)

    async def complete(self, prompt: str) -> dict:
        out = self._extract(prompt) if prompt.startswith(DOC_HEAD) else \
              f"[mock answer] Based on company documents: {prompt.strip().rstrip('?')}. " * 3
        t_in, t_out = est_tokens(prompt), est_tokens(out)
        await asyncio.sleep(self.base + self.per * t_out + self.per_in * t_in)
        return {"text": out, "tokens_in": t_in, "tokens_out": t_out}

class AnthropicLLM:
    def __init__(self, model):
        import anthropic
        self.c, self.model = anthropic.AsyncAnthropic(), model
    async def complete(self, prompt: str) -> dict:
        m = await self.c.messages.create(model=self.model, max_tokens=500,
                                         messages=[{"role": "user", "content": prompt}])
        return {"text": m.content[0].text, "tokens_in": m.usage.input_tokens, "tokens_out": m.usage.output_tokens}

class OpenAICompatLLM:
    """Any /chat/completions API: Mistral, Ollama (local), Groq, OpenAI, ..."""
    def __init__(self, base_url, api_key, model):
        import httpx
        self.model = model
        self.c = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=180,
                                   headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    async def complete(self, prompt: str) -> dict:
        r = await self.c.post("/chat/completions", json={
            "model": self.model, "max_tokens": 500, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        j = r.json(); text = j["choices"][0]["message"]["content"]; u = j.get("usage") or {}
        return {"text": text, "tokens_in": u.get("prompt_tokens") or est_tokens(prompt),
                "tokens_out": u.get("completion_tokens") or est_tokens(text)}

def get_llm():
    p, m = settings.llm_provider, settings.llm_model
    if p == "anthropic":
        return AnthropicLLM(m or "claude-haiku-4-5-20251001")
    if p == "mistral":
        return OpenAICompatLLM(settings.llm_base_url or "https://api.mistral.ai/v1", settings.llm_api_key,
                               m or "mistral-small-latest")
    if p == "ollama":
        return OpenAICompatLLM(settings.llm_base_url or "http://localhost:11434/v1", "", m or "mistral")
    if p == "openai_compat":
        return OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, m)
    return MockLLM()

def cost_usd(t_in: int, t_out: int) -> float:
    return t_in / 1000 * settings.price_in_per_1k + t_out / 1000 * settings.price_out_per_1k