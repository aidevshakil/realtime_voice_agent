import threading


class Transcript:
    """Thread-safe conversation log shared between the engine thread and the UI.

    Lives apart from `conversation` so the UI can hold a transcript without importing the audio
    stack — a machine with no PortAudio should still get the browser recorder and text box.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._items = []

    def add(self, entry: dict) -> int:
        with self._lock:
            self._items.append(entry)
            return len(self._items) - 1

    def update(self, index: int, **fields):
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index].update(fields)

    def append_event(self, index: int, event: dict):
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index].setdefault("events", []).append(event)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items]

    def clear(self):
        with self._lock:
            self._items.clear()

    def __len__(self):
        with self._lock:
            return len(self._items)
