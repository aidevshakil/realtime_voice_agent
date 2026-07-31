import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
ASR_MODEL = "whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
MIN_UTTERANCE_MS = 700
MAX_UTTERANCE_MS = 30000

MIN_FRAME_RMS = 300
MIN_UTTERANCE_RMS = 400

BARGE_IN_MS = 400

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly voice assistant. Keep replies concise and "
    "conversational since they will be spoken aloud.\n"
    "Use a tool only when the user's request actually needs it. Greetings, thanks, "
    "goodbyes, small talk and anything you already know the answer to need no tool — "
    "just reply. Never call a tool to acknowledge something.\n"
    "If a message is garbled, empty of meaning, or looks like a mis-transcription, say "
    "you didn't catch it and ask them to repeat — do not guess or invent a topic."
)


class MissingCredentials(RuntimeError):
    """Raised when a required key isn't configured — callers can show this instead of crashing."""


def groq_api_key() -> str:
   
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
    
    os.environ["GROQ_API_KEY"] = value.strip()


def llm_model() -> str:
    return os.getenv("GROQ_LLM_MODEL") or DEFAULT_LLM_MODEL
