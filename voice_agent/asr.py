import io

from voice_agent.client import groq_client
from voice_agent.config import ASR_MODEL


def transcribe(wav_bytes: bytes, language: str | None = None) -> str:
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "utterance.wav"
    kwargs = {"model": ASR_MODEL, "file": audio_file}
    if language and language != "auto":
        kwargs["language"] = language
    result = groq_client().audio.transcriptions.create(**kwargs)
    return result.text.strip()
