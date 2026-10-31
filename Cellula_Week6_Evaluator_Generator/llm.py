from langchain_community.cache import RedisCache as LangChainRedisCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from cache import RedisCache
from config import CACHE_TTL, LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def build_llm(cache: RedisCache) -> ChatOpenAI:
    set_llm_cache(LangChainRedisCache(redis_=cache.client, ttl=CACHE_TTL))
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        timeout=120,
        max_retries=1,
    )
