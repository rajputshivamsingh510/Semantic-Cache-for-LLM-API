import os
from dotenv import load_dotenv

load_dotenv() 
class Settings:
    redis_url = os.getenv("REDIS_URL", "")            # empty -> in-memory backend
    embedder = os.getenv("EMBEDDER", "hash")          # hash | sbert
    similarity_threshold = float(os.getenv("SIM_THRESHOLD", "0.8"))
    default_ttl = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    rate_limit = int(os.getenv("RATE_LIMIT", "60"))    # requests per window
    rate_window = int(os.getenv("RATE_WINDOW", "60"))  # seconds
    llm_provider = os.getenv("LLM_PROVIDER", "mock")   # mock | mistral | ollama | openai_compat | anthropic
    llm_model = os.getenv("LLM_MODEL", "")             # empty -> default model per provider
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("MISTRAL_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    price_in_per_1k = float(os.getenv("PRICE_IN_PER_1K", "0.001"))    # USD
    price_out_per_1k = float(os.getenv("PRICE_OUT_PER_1K", "0.005"))
    max_doc_chars = int(os.getenv("MAX_DOC_CHARS", "20000"))
    max_entries_per_ns = int(os.getenv("MAX_ENTRIES_PER_NS", "5000"))

settings = Settings()