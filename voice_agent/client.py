from groq import Groq

from voice_agent.config import groq_api_key

_cache = {}


def groq_client() -> Groq:
    key = groq_api_key()
    client = _cache.get(key)
    if client is None:
        client = Groq(api_key=key)
        _cache[key] = client
    return client
