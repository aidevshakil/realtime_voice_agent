import contextlib
import functools
import os
import platform
import subprocess
import tempfile

_IS_MAC = platform.system() == "Darwin"


@functools.lru_cache(maxsize=1)
def list_voices() -> list[str]:
    """Installed system voice names, best-effort — an empty list just means 'use the default'."""
    if _IS_MAC:
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
            names = []
            for line in out.stdout.splitlines():
                name = line.split("  ")[0].strip()
                if name:
                    names.append(name)
            return names
        except (OSError, subprocess.SubprocessError):
            return []
    try:
        import pyttsx3

        engine = pyttsx3.init()
        return [v.name for v in engine.getProperty("voices")]
    except Exception:
        return []


def synthesize_to_file(text: str, voice: str | None = None, rate: int | None = None) -> str:
    """Render `text` to a WAV file and return its path. Caller owns the file — pass it to discard().

    macOS goes through the `say` binary rather than pyttsx3: pyttsx3's macOS driver drives
    NSSpeechSynthesizer on the calling thread, which misbehaves in a Streamlit worker thread.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    if _IS_MAC:
        cmd = ["say", "-o", path, "--data-format=LEI16@22050"]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        cmd.append(text)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            discard(path)
            raise RuntimeError(f"say failed: {result.stderr.strip() or result.returncode}")
        return path

    import pyttsx3

    engine = pyttsx3.init()
    if voice:
        for v in engine.getProperty("voices"):
            if v.name == voice:
                engine.setProperty("voice", v.id)
                break
    if rate:
        engine.setProperty("rate", rate)
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path


def synthesize_to_bytes(text: str, voice: str | None = None, rate: int | None = None) -> bytes:
    """Same as synthesize_to_file, but hands back the audio so it can be embedded in a web page."""
    path = synthesize_to_file(text, voice=voice, rate=rate)
    try:
        with open(path, "rb") as f:
            return f.read()
    finally:
        discard(path)


def discard(path: str):
    """Delete a synthesized clip once it has been played."""
    with contextlib.suppress(OSError):
        os.remove(path)
