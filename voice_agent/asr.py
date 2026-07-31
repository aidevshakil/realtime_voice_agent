import io
import re

from voice_agent.client import groq_client
from voice_agent.config import ASR_MODEL

_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "thanks for watching this video",
    "please subscribe",
    "subtitles by the amara org community",
    "subtitles by amara org",
    "amara org",
    "transcription by castingwords",
    "www mooji org",
    "music",
    "applause",
    "silence",
    "you",
    "hmm",
    "uh",
    "um",
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_LETTER = re.compile(r"\w", flags=re.UNICODE)


def is_hallucination(text: str) -> bool:
    """True if the transcript looks like Whisper filling silence rather than a real utterance."""
    if not text:
        return True
    normalized = _PUNCT.sub(" ", text).lower()
    normalized = " ".join(normalized.split())
    if not normalized or not _LETTER.search(normalized):
        return True
    return normalized in _HALLUCINATIONS


def transcribe(wav_bytes: bytes, language: str | None = None) -> str:
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "utterance.wav"
    kwargs = {"model": ASR_MODEL, "file": audio_file, "temperature": 0.0}
    if language and language != "auto":
        kwargs["language"] = language
    result = groq_client().audio.transcriptions.create(**kwargs)
    text = result.text.strip()
    return "" if is_hallucination(text) else text
