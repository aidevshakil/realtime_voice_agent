from groq import Groq

from voice_agent.config import groq_api_key

_cache = {}


def groq_client() -> Groq:
    """One client per API key, built lazily so importing the package never needs credentials."""
    key = groq_api_key()
    client = _cache.get(key)
    if client is None:
        client = Groq(api_key=key)
        _cache[key] = client
    return client
