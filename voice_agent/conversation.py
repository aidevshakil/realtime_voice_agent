import queue
import re
import sys
import threading
import time
import traceback

from voice_agent.asr import transcribe
from voice_agent.audio_io import pcm_to_wav_bytes
from voice_agent.backends import BrowserAudioBackend, LocalAudioBackend
from voice_agent.transcript import Transcript
from voice_agent.tts import discard, synthesize_to_file

__all__ = [
    "BrowserAudioBackend",
    "ConversationEngine",
    "LocalAudioBackend",
    "Transcript",
    "split_speakable",
]

IDLE = "idle"
LISTENING = "listening"
TRANSCRIBING = "transcribing"
THINKING = "thinking"
SPEAKING = "speaking"
STOPPED = "stopped"

MIN_CHUNK_CHARS = 24
MAX_CHUNK_CHARS = 220
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=\n)")


def split_speakable(buffer: str, force: bool = False) -> tuple[list[str], str]:
    """Peel complete sentences off the front of `buffer`.

    Returns (chunks ready to speak, remaining text). With `force`, whatever is left is
    flushed too — used when the model has stopped generating.
    """
    chunks = []
    rest = buffer
    while True:
        match = _SENTENCE_END.search(rest)
        if not match:
            break
        candidate, tail = rest[: match.end()], rest[match.end():]
        if len(candidate.strip()) < MIN_CHUNK_CHARS and tail:
            later = _SENTENCE_END.search(tail)
            if later:
                rest = candidate + tail
                extended, tail2 = rest[: len(candidate) + later.end()], rest[len(candidate) + later.end():]
                chunks.append(extended.strip())
                rest = tail2
                continue
            break
        chunks.append(candidate.strip())
        rest = tail

    if not chunks and len(rest) >= MAX_CHUNK_CHARS:
        cut = rest.rfind(" ", 0, MAX_CHUNK_CHARS)
        if cut > MIN_CHUNK_CHARS:
            chunks.append(rest[:cut].strip())
            rest = rest[cut:]

    if force and rest.strip():
        chunks.append(rest.strip())
        rest = ""

    return [c for c in chunks if c], rest


class ConversationEngine:
    def __init__(
        self,
        agent,
        transcript: Transcript,
        *,
        backend=None,
        language="auto",
        voice=None,
        rate=None,
        device=None,
        aggressiveness=2,
        barge_in=True,
        speak=True,
    ):
        self.agent = agent
        self.transcript = transcript
        self.backend = backend or LocalAudioBackend(
            device=device, aggressiveness=aggressiveness
        )
        self.language = language
        self.voice = voice
        self.rate = rate
        self.barge_in = barge_in
        self.speak = speak

        self._state = IDLE
        self._state_lock = threading.Lock()
        self._reply_index = None
        self._partial = ""
        self._error = None
        self._last_timings = {}
        self._interrupts = 0

        self._utterances = queue.Queue()
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._thread = None


    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._interrupt.clear()
        self._error = None
        self._set_state(LISTENING)
        self.backend.start(self._utterances.put, self._on_speech_confirmed)
        self._thread = threading.Thread(
            target=self._process_loop, name="voice-process", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._interrupt.set()
        self.backend.stop()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._set_state(STOPPED)

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "state": self._state,
                "partial": self._partial,
                "error": self._error or self.backend.error,
                "timings": dict(self._last_timings),
                "interruptions": self._interrupts,
                "running": self.running,
                "backend": self.backend.name,
            }

    def _set_state(self, state):
        with self._state_lock:
            self._state = state

    def _set_partial(self, text):
        with self._state_lock:
            self._partial = text


    def _on_speech_confirmed(self):
        """Fires once the segmenter has heard sustained speech — not on the first triggered
        frame, so a door slam or the agent's own echo can't cut the reply off. If the agent
        is mid-sentence when someone really is talking, cut it off."""
        with self._state_lock:
            speaking = self._state in (SPEAKING, THINKING)
        if speaking and self.barge_in:
            self._interrupt.set()
            self.backend.interrupt()
            with self._state_lock:
                self._interrupts += 1


    def _process_loop(self):
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.25)
            except queue.Empty:
                continue
            if self._stop.is_set():
                return
            while not self._utterances.empty():
                utterance = self._utterances.get_nowait()
            try:
                self._handle(utterance)
            except Exception as exc:
                self._report(f"{type(exc).__name__}: {exc}", traceback.format_exc())
                self._set_state(LISTENING)

    def _report(self, message: str, tb: str | None = None):
        """Surface a failed turn in the transcript — reusing the reply placeholder if the turn
        got far enough to create one, so a failure doesn't leave an empty bubble behind."""
        entry = {"role": "assistant", "text": message, "error": True, "traceback": tb}
        if self._reply_index is not None:
            self.transcript.update(self._reply_index, **entry)
        else:
            self.transcript.add(entry)

    def _handle(self, utterance: bytes):
        self._interrupt.clear()
        self._reply_index = None
        timings = {}

        self._set_state(TRANSCRIBING)
        started = time.perf_counter()
        text = transcribe(pcm_to_wav_bytes(utterance), language=self.language)
        timings["asr_ms"] = (time.perf_counter() - started) * 1000
        if not text:
            self._set_state(LISTENING)
            return

        self.transcript.add({"role": "user", "text": text})
        self._set_state(THINKING)
        turn_started = time.perf_counter()
        index = self.transcript.add({"role": "assistant", "text": "", "events": []})
        self._reply_index = index

        half_duplex = not self.barge_in and self.backend.can_mute_input
        if half_duplex:
            self.backend.mute_input(True)

        spoken = ""
        buffer = ""
        first_audio_at = None
        try:
            for chunk in self.agent.stream_respond(text, interrupt=self._interrupt):
                if self._interrupt.is_set() or self._stop.is_set():
                    break
                if chunk["type"] == "tool":
                    self.transcript.append_event(
                        index, {k: chunk[k] for k in ("tool", "args", "result")}
                    )
                    continue

                spoken += chunk["value"]
                buffer += chunk["value"]
                self._set_partial(spoken)
                self.transcript.update(index, text=spoken)

                ready, buffer = split_speakable(buffer)
                for sentence in ready:
                    if self._interrupt.is_set():
                        break
                    if self._speak_sentence(sentence) and first_audio_at is None:
                        first_audio_at = time.perf_counter()
                        timings["first_audio_ms"] = (first_audio_at - turn_started) * 1000

            if not self._interrupt.is_set() and not self._stop.is_set():
                ready, _ = split_speakable(buffer, force=True)
                for sentence in ready:
                    if self._speak_sentence(sentence) and first_audio_at is None:
                        first_audio_at = time.perf_counter()
                        timings["first_audio_ms"] = (first_audio_at - turn_started) * 1000

            timings["reply_ms"] = (time.perf_counter() - turn_started) * 1000

            if self.backend.busy:
                self._set_state(SPEAKING)
                self.backend.wait_until_idle(abort=self._interrupt)
        finally:
            self._set_partial("")
            if half_duplex:
                self.backend.mute_input(False)

        interrupted = self._interrupt.is_set()
        self.transcript.update(
            index,
            text=spoken or ("(interrupted)" if interrupted else "(no reply)"),
            interrupted=interrupted,
            timings=timings,
        )
        with self._state_lock:
            self._last_timings = timings
        self._set_state(LISTENING)

    def _speak_sentence(self, sentence: str) -> bool:
        """Synthesize one sentence and hand it to the backend. Returns True if audio is on
        its way — the caller uses that to time the first spoken word."""
        if not self.speak:
            return False
        try:
            path = synthesize_to_file(sentence, voice=self.voice, rate=self.rate)
        except Exception as exc:
            print(f"[tts] {type(exc).__name__}: {exc}", file=sys.stderr)
            return False
        try:
            self.backend.play(path)
        except Exception as exc:
            discard(path)
            print(f"[audio] {type(exc).__name__}: {exc}", file=sys.stderr)
            return False
        self._set_state(SPEAKING)
        return True

    def _fail(self, message: str):
        with self._state_lock:
            self._error = message
        self._stop.set()
        self._set_state(STOPPED)
