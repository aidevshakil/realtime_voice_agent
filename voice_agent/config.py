import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
ASR_MODEL = "whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
MIN_UTTERANCE_MS = 500  # ignore shorter blips — coughs, keyboard clicks, background noise
MAX_UTTERANCE_MS = 30000  # hard stop so a noisy room can't record forever

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly voice assistant. Keep replies concise and "
    "conversational since they will be spoken aloud."
)


class MissingCredentials(RuntimeError):
    """Raised when a required key isn't configured — callers can show this instead of crashing."""


def groq_api_key() -> str:
    """Read the key at call time, not import time, so the web UI can load without one
    and tell the user what's missing."""
    value = os.getenv("GROQ_API_KEY")
    if not value:
        raise MissingCredentials(
            "Missing GROQ_API_KEY. Copy .env.example to .env and fill it in, "
            "or paste a key in the sidebar."
        )
    return value


def has_groq_api_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def set_groq_api_key(value: str):
    """Let the UI supply a key for this process without touching .env."""
    os.environ["GROQ_API_KEY"] = value.strip()


def llm_model() -> str:
    return os.getenv("GROQ_LLM_MODEL") or DEFAULT_LLM_MODEL
