import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
ASR_MODEL = "whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
MIN_UTTERANCE_MS = 700  # ignore shorter blips — coughs, keyboard clicks, background noise
MAX_UTTERANCE_MS = 30000  # hard stop so a noisy room can't record forever

# VAD answers "is this speech-shaped?", not "is this loud enough to be aimed at me?" — so room
# noise and the tail of the agent's own voice leaking back through the speakers both pass it.
# Whisper turns those into confident nonsense (often in the wrong language), so gate on level
# too: a frame counts as speech only above MIN_FRAME_RMS, and an utterance is only transcribed
# if its overall level clears MIN_UTTERANCE_RMS. Int16 full scale is 32768; normal speech into a
# headset mic sits in the low thousands.
MIN_FRAME_RMS = 300
MIN_UTTERANCE_RMS = 400

# How much confirmed speech it takes to cut the agent off mid-sentence. Interrupting on the very
# first triggered frame means any echo or door slam kills the reply; waiting for a sustained
# ~400 ms means only someone actually talking does.
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
