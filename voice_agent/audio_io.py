import collections
import contextlib
import io
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import warnings
import wave

import sounddevice as sd

with warnings.catch_warnings():  # webrtcvad imports the deprecated pkg_resources; pinned in requirements
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    import webrtcvad

from voice_agent.config import (
    FRAME_MS,
    MAX_UTTERANCE_MS,
    MIN_UTTERANCE_MS,
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
)

FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono samples


def list_input_devices() -> list[tuple[int, str]]:
    """(index, name) for every device that can record — used to populate the UI picker."""
    try:
        return [
            (i, d["name"])
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        ]
    except Exception:
        return []


class VadSegmenter:
    """Turns a stream of 30 ms PCM16 frames into utterances.

    Speech starts once most of a rolling window is voiced and ends once most of it isn't, so a
    single noisy frame neither triggers nor terminates a turn. Shared by the local microphone
    and the browser (WebRTC) path so both segment speech identically.
    """

    def __init__(
        self,
        aggressiveness=VAD_AGGRESSIVENESS,
        padding_ms=300,
        ratio=0.75,
        min_utterance_ms=MIN_UTTERANCE_MS,
        on_speech_start=None,
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._ratio = ratio
        self._min_utterance_ms = min_utterance_ms
        self.on_speech_start = on_speech_start
        self._ring = collections.deque(maxlen=max(1, int(padding_ms / FRAME_MS)))
        self.triggered = False
        self._voiced = []
        self._speech_ms = 0  # only frames VAD called speech, excluding the silence padding

    def reset(self):
        self._ring.clear()
        self.triggered = False
        self._voiced = []
        self._speech_ms = 0

    def push(self, frame: bytes):
        """Feed one frame. Returns a complete utterance (PCM16 bytes) when speech ends."""
        if len(frame) != FRAME_BYTES:
            return None
        is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

        if not self.triggered:
            self._ring.append((frame, is_speech))
            voiced = sum(1 for _, speech in self._ring if speech)
            if voiced > self._ratio * self._ring.maxlen:
                self.triggered = True
                self._voiced.extend(f for f, _ in self._ring)
                self._speech_ms = voiced * FRAME_MS
                self._ring.clear()
                if self.on_speech_start:
                    self.on_speech_start()
            return None

        self._voiced.append(frame)
        self._ring.append((frame, is_speech))
        if is_speech:
            self._speech_ms += FRAME_MS
        unvoiced = sum(1 for _, speech in self._ring if not speech)
        too_long = len(self._voiced) * FRAME_MS >= MAX_UTTERANCE_MS
        if unvoiced > self._ratio * self._ring.maxlen or too_long:
            utterance = b"".join(self._voiced) if self._speech_ms >= self._min_utterance_ms else None
            self.reset()
            return utterance
        return None


class MicListener:
    """Continuously listens to the local mic and yields one utterance (PCM16 bytes) at a time."""

    def __init__(
        self,
        padding_ms=300,
        ratio=0.75,
        aggressiveness=VAD_AGGRESSIVENESS,
        min_utterance_ms=MIN_UTTERANCE_MS,
        device=None,
    ):
        self._queue = queue.Queue()
        self._paused = False
        self._segmenter_args = {
            "aggressiveness": aggressiveness,
            "padding_ms": padding_ms,
            "ratio": ratio,
            "min_utterance_ms": min_utterance_ms,
        }
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_BYTES // 2,
            dtype="int16",
            channels=1,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        if not self._paused:
            self._queue.put(bytes(indata))

    def pause(self):
        self._paused = True
        while not self._queue.empty():
            self._queue.get_nowait()

    def resume(self):
        self._paused = False

    def listen(self, timeout_s: float | None = None, on_speech_start=None):
        """Yield utterances forever. If `timeout_s` is set, stop waiting after that many
        seconds of no speech starting — lets a web UI stay responsive between attempts.
        The clock only runs while idle; once someone starts talking they get to finish.

        `on_speech_start` fires the moment VAD triggers, before the utterance has finished.
        That's the barge-in signal: it's what lets the agent stop talking as soon as you do,
        rather than after you've stopped."""
        self._stream.start()
        segmenter = VadSegmenter(on_speech_start=on_speech_start, **self._segmenter_args)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while True:
            if not segmenter.triggered and deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    frame = self._queue.get(timeout=remaining)
                except queue.Empty:
                    return
            else:
                frame = self._queue.get()

            utterance = segmenter.push(frame)
            if utterance:
                yield utterance
                deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def close(self):
        self._stream.stop()
        self._stream.close()


def capture_utterance(timeout_s: float = 10.0, **listener_kwargs) -> bytes | None:
    """Open the mic, wait for one utterance, close. Returns None if nobody spoke in time.

    The web UI calls this instead of holding a listener open across reruns, since Streamlit
    can restart the script at any point and would otherwise leak the audio stream.
    """
    listener = MicListener(**listener_kwargs)
    try:
        for utterance in listener.listen(timeout_s=timeout_s):
            return utterance
    finally:
        listener.close()
    return None


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _player_cmd(path: str) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["afplay", path]
    if system == "Linux":
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
    raise RuntimeError(f"No audio playback configured for platform: {system}")


def play_audio_file(path: str):
    cmd = _player_cmd(path)
    returncode = subprocess.run(cmd).returncode
    if returncode < 0:
        return  # player killed by a signal (Ctrl+C reaches the whole process group) — just stop the clip
    if returncode != 0:
        raise RuntimeError(f"Playback failed: {cmd[0]} exited with {returncode}")


def read_wav(path: str):
    """(samples, sample_rate) as int16 — shaped (frames, channels) for sounddevice."""
    import numpy as np

    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit audio, got {wf.getsampwidth() * 8}-bit")
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            data = data.reshape(-1, channels)
        return data, wf.getframerate()


class Speaker:
    """Plays queued clips back-to-back on a worker thread, and can be cut off mid-clip.

    The agent speaks one sentence at a time as the LLM produces them, so playback has to start
    while more audio is still being written — and `interrupt()` has to kill both the clip that's
    playing and everything queued behind it the instant the user starts talking.

    Playback goes through sounddevice rather than `afplay`: spawning a player process costs the
    better part of a second, which lands as a silent gap between every sentence. Falls back to
    the external player if the clip can't be decoded here.
    """

    BLOCK_FRAMES = 1024  # write granularity — also how often an interrupt is noticed

    def __init__(self, delete_after=True):
        self._queue = queue.Queue()
        self._delete_after = delete_after
        self._lock = threading.Lock()
        self._pending = 0
        self._proc = None
        self._stream = None       # persistent output stream, opened on the first clip
        self._stream_spec = None  # (samplerate, channels) it was opened with
        self._skip = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, path: str):
        with self._lock:
            self._pending += 1
        self._queue.put(path)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending > 0

    def interrupt(self):
        """Drop everything queued and stop the clip currently playing."""
        self._skip.set()
        while True:
            try:
                path = self._queue.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending -= 1
            if self._delete_after:
                _discard(path)
        with self._lock:
            stream, proc = self._stream, self._proc
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()  # discard buffered audio instead of draining it
        if proc and proc.poll() is None:
            proc.terminate()

    def wait_until_idle(self, timeout_s=60.0, abort=None):
        """Block until the queue drains. `abort` is an Event that cuts the wait short."""
        deadline = time.monotonic() + timeout_s
        while self.busy and time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                return
            time.sleep(0.03)

    def close(self):
        self._closed.set()
        self.interrupt()
        self._queue.put(None)
        self._thread.join(timeout=2)
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()

    def _run(self):
        while True:
            path = self._queue.get()
            if path is None:
                return
            self._skip.clear()  # a fresh clip is never born interrupted
            try:
                self._play(path)
            except Exception as exc:  # a dead player shouldn't take the conversation with it
                print(f"[audio] playback failed: {exc}", file=sys.stderr)
            finally:
                with self._lock:
                    self._proc = None
                    self._pending -= 1
                if self._delete_after:
                    _discard(path)

    def _play(self, path: str):
        try:
            data, rate = read_wav(path)
        except Exception:  # not a plain 16-bit WAV — let the system player deal with it
            self._play_external(path)
            return

        channels = 1 if data.ndim == 1 else data.shape[1]
        try:
            stream = self._output_stream(rate, channels)
            for start in range(0, len(data), self.BLOCK_FRAMES):
                if self._skip.is_set() or self._closed.is_set():
                    return
                stream.write(data[start : start + self.BLOCK_FRAMES])
        except Exception as exc:
            if self._skip.is_set() or self._closed.is_set():
                return  # abort() during a blocked write — expected
            # The audio device can fail transiently (virtual devices, a device switching
            # underneath us). Drop the stream so the next clip reopens it, and don't lose
            # this sentence — hand it to the system player.
            print(f"[audio] output stream failed ({exc}); falling back to the system player",
                  file=sys.stderr)
            self._drop_stream()
            self._play_external(path)

    def _play_external(self, path: str):
        proc = subprocess.Popen(_player_cmd(path))
        with self._lock:
            self._proc = proc
        proc.wait()

    def _drop_stream(self):
        with self._lock:
            stream, self._stream, self._stream_spec = self._stream, None, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()

    def warm(self, rate: int = 22050, channels: int = 1):
        """Open the output stream ahead of time.

        Opening one costs ~0.5s and can fail transiently on virtual/aggregate devices. Doing it
        at startup keeps that cost off the first sentence of the first reply, which is exactly
        the moment latency is most visible.
        """
        with contextlib.suppress(Exception):
            self._output_stream(rate, channels)

    def _output_stream(self, rate: int, channels: int):
        """One stream reused across clips, so there's no gap between sentences."""
        spec = (rate, channels)
        with self._lock:
            stream, current = self._stream, self._stream_spec
        if stream is not None and current != spec:
            with contextlib.suppress(Exception):
                stream.close()
            stream = None
        if stream is None:
            try:
                stream = sd.OutputStream(samplerate=rate, channels=channels, dtype="int16")
            except Exception:
                # one retry: opening a virtual device right after another stream closed can
                # fail once and then succeed
                sd._terminate()
                sd._initialize()
                stream = sd.OutputStream(samplerate=rate, channels=channels, dtype="int16")
            with self._lock:
                self._stream, self._stream_spec = stream, spec
        if not stream.active:
            stream.start()  # abort() stops it; restart for the next clip
        return stream


def _discard(path: str):
    with contextlib.suppress(OSError, TypeError):
        os.remove(path)
