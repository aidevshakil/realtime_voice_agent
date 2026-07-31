import threading
import time

from voice_agent.audio_io import FRAME_BYTES, MicListener, Speaker, VadSegmenter, read_wav
from voice_agent.tts import discard


class LocalAudioBackend:
    def __init__(self, device=None, aggressiveness=2):
        self.device = device
        self.aggressiveness = aggressiveness
        self._listener = None
        self._speaker = None
        self._thread = None
        self._stop = threading.Event()
        self._on_utterance = None
        self._on_speech_confirmed = None
        self.error = None

    name = "local"
    can_mute_input = True

    def start(self, on_utterance, on_speech_confirmed):
        self._on_utterance = on_utterance
        self._on_speech_confirmed = on_speech_confirmed
        self._stop.clear()
        self._speaker = Speaker()
        self._speaker.warm()
        self._thread = threading.Thread(target=self._capture_loop, name="voice-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self):
        try:
            self._listener = MicListener(aggressiveness=self.aggressiveness, device=self.device)
        except Exception as exc:
            self.error = f"Microphone unavailable — {type(exc).__name__}: {exc}"
            return
        try:
            while not self._stop.is_set():
                for utterance in self._listener.listen(
                    timeout_s=0.5, on_speech_confirmed=self._on_speech_confirmed
                ):
                    if self._stop.is_set():
                        return
                    self._on_utterance(utterance)
        except Exception as exc:
            self.error = f"Capture failed — {type(exc).__name__}: {exc}"
        finally:
            if self._listener:
                self._listener.close()
                self._listener = None

    def stop(self):
        self._stop.set()
        if self._speaker:
            self._speaker.interrupt()
            self._speaker.close()
            self._speaker = None
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def play(self, path: str):
        if self._speaker:
            self._speaker.enqueue(path)

    def interrupt(self):
        if self._speaker:
            self._speaker.interrupt()

    @property
    def busy(self) -> bool:
        return bool(self._speaker and self._speaker.busy)

    def wait_until_idle(self, abort=None):
        if self._speaker:
            self._speaker.wait_until_idle(abort=abort)

    def mute_input(self, muted: bool):
        if self._listener:
            self._listener.pause() if muted else self._listener.resume()


class BrowserAudioBackend:

    name = "browser"
    can_mute_input = False

    def __init__(self, aggressiveness=2):
        self.aggressiveness = aggressiveness
        self.error = None
        self._segmenter = None
        self._on_utterance = None
        self._on_speech_confirmed = None
        self._resampler = None
        self._in_buffer = bytearray()
        self._out_buffer = bytearray()
        self._out_lock = threading.Lock()
        self._connected = threading.Event()


    def start(self, on_utterance, on_speech_confirmed):
        self._on_utterance = on_utterance
        self._on_speech_confirmed = on_speech_confirmed
        self._segmenter = VadSegmenter(
            aggressiveness=self.aggressiveness, on_speech_confirmed=self._speech_confirmed
        )

    def stop(self):
        self.interrupt()
        self._segmenter = None

    def play(self, path: str):
       
        try:
            data, rate = read_wav(path)
            pcm = _resample_int16(data, rate, 16000).tobytes()
        except Exception as exc:
            self.error = f"Could not decode reply audio — {type(exc).__name__}: {exc}"
            return
        finally:
            discard(path)
        with self._out_lock:
            self._out_buffer.extend(pcm)

    def interrupt(self):
        with self._out_lock:
            self._out_buffer.clear()

    @property
    def busy(self) -> bool:
        with self._out_lock:
            return len(self._out_buffer) > 0

    def wait_until_idle(self, abort=None, timeout_s=120.0):
        deadline = time.monotonic() + timeout_s
        while self.busy and time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                return
            time.sleep(0.03)

    def mute_input(self, muted: bool):
        pass

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _speech_confirmed(self):
        if self._on_speech_confirmed:
            self._on_speech_confirmed()


    def process_frame(self, frame):
        """Called per inbound frame on streamlit-webrtc's worker thread."""
        import av
        import numpy as np

        self._connected.set()
        try:
            if self._resampler is None:
                self._resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for resampled in self._resampler.resample(frame):
                self._in_buffer.extend(bytes(resampled.planes[0]))
        except Exception as exc:
            self.error = f"Inbound audio failed — {type(exc).__name__}: {exc}"

        while len(self._in_buffer) >= FRAME_BYTES and self._segmenter is not None:
            chunk = bytes(self._in_buffer[:FRAME_BYTES])
            del self._in_buffer[:FRAME_BYTES]
            utterance = self._segmenter.push(chunk)
            if utterance and self._on_utterance:
                self._on_utterance(utterance)

        return self._outbound_frame(frame, av, np)

    def _outbound_frame(self, frame, av, np):
        rate = frame.sample_rate
        channels = len(frame.layout.channels)
        samples = frame.samples

        needed_16k = int(round(samples * 16000 / rate))
        with self._out_lock:
            take = min(needed_16k * 2, len(self._out_buffer))
            chunk = bytes(self._out_buffer[:take])
            del self._out_buffer[:take]

        mono16k = np.frombuffer(chunk, dtype=np.int16)
        if len(mono16k) < needed_16k:
            mono16k = np.concatenate(
                [mono16k, np.zeros(needed_16k - len(mono16k), dtype=np.int16)]
            )
        mono = _resample_int16(mono16k, 16000, rate)
        mono = _fit(mono, samples, np)

        interleaved = np.repeat(mono, channels) if channels > 1 else mono
        out = av.AudioFrame.from_ndarray(
            interleaved.reshape(1, -1), format="s16", layout=frame.layout.name
        )
        out.sample_rate = rate
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out


def _fit(samples, length, np):
    if len(samples) == length:
        return samples
    if len(samples) > length:
        return samples[:length]
    return np.concatenate([samples, np.zeros(length - len(samples), dtype=np.int16)])


def _resample_int16(data, from_rate: int, to_rate: int):
    import numpy as np

    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    if from_rate == to_rate or len(data) == 0:
        return np.asarray(data, dtype=np.int16)
    count = int(round(len(data) * to_rate / from_rate))
    positions = np.linspace(0, len(data) - 1, count)
    return np.interp(positions, np.arange(len(data)), data).astype(np.int16)
