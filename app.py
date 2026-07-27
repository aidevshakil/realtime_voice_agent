"""Streamlit web interface for the real-time voice agent.

Run with:  streamlit run app.py

Two ways to talk to it:

  Live mode — a background thread owns the mic and the speaker, so the conversation runs at
  audio speed instead of page speed: the agent starts talking before it has finished thinking,
  and you can cut it off mid-sentence. The page is a live view onto that thread.

  Manual mode — browser recorder or text box, one turn at a time. Works over the network,
  where live mode can't (its microphone is the server's).
"""

import hashlib
import json
import time
import traceback

import streamlit as st

from voice_agent import config, tools, tts
from voice_agent.asr import transcribe
from voice_agent.config import MissingCredentials
from voice_agent.llm import VoiceAgent
from voice_agent.transcript import Transcript  # audio-free, so this import can't fail

# sounddevice needs PortAudio; without it the browser recorder still works, so degrade instead of crash.
try:
    from voice_agent.audio_io import list_input_devices, pcm_to_wav_bytes
    from voice_agent.conversation import BrowserAudioBackend, ConversationEngine

    MIC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host machine
    list_input_devices = pcm_to_wav_bytes = ConversationEngine = BrowserAudioBackend = None
    MIC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    webrtc_streamer = WebRtcMode = None
    WEBRTC_ERROR = f"{type(exc).__name__}: {exc}"

BROWSER_MIC = "Browser microphone (works from any machine)"
SERVER_MIC = "This machine's microphone"

# Ask the browser for the processing that makes open speakers usable: it cancels the echo of the
# agent's own voice, which the server-mic path has no way to do.
AUDIO_CONSTRAINTS = {
    "echoCancellation": True,
    "noiseSuppression": True,
    "autoGainControl": True,
}

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
]

LANGUAGES = {
    "Auto-detect": "auto",
    "English": "en",
    "Bengali": "bn",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
}

STATE_BADGES = {
    "idle": ("⚪", "Idle"),
    "listening": ("🟢", "Listening — go ahead"),
    "transcribing": ("🟡", "Heard you — transcribing"),
    "thinking": ("🔵", "Thinking"),
    "speaking": ("🔊", "Speaking — talk over me to interrupt"),
    "stopped": ("⚫", "Stopped"),
}

st.set_page_config(page_title="Real-Time Voice Agent", page_icon="🎙️", layout="wide")


# --------------------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------------------

def init_state():
    ss = st.session_state
    ss.setdefault("transcript", Transcript())
    ss.setdefault("agent", None)
    ss.setdefault("engine", None)
    ss.setdefault("rec_nonce", 0)      # bumped to reset the browser recorder widget
    ss.setdefault("last_audio_hash", None)


def get_agent(settings) -> VoiceAgent:
    if st.session_state.agent is None:
        st.session_state.agent = VoiceAgent(
            model=settings["model"],
            system_prompt=settings["system_prompt"],
            enabled_tools=settings["enabled_tools"],
        )
    else:
        st.session_state.agent.configure(
            model=settings["model"],
            system_prompt=settings["system_prompt"],
            enabled_tools=settings["enabled_tools"],
        )
    return st.session_state.agent


def live_engine():
    engine = st.session_state.engine
    return engine if engine and engine.running else None


# --------------------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------------------

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")

        with st.expander("🔑 Credentials", expanded=not config.has_groq_api_key()):
            if config.has_groq_api_key():
                st.success("GROQ_API_KEY loaded")
            else:
                st.warning("No GROQ_API_KEY found in .env")
            key_input = st.text_input(
                "Groq API key",
                type="password",
                help="Used for both Whisper (speech-to-text) and the LLM. Set here it applies "
                "to this session only — it is not written to .env.",
            )
            if key_input:
                config.set_groq_api_key(key_input)
                st.session_state.agent = None  # rebuild against the new key

        st.subheader("🧠 Model")
        model = st.selectbox("LLM", MODELS, index=0)
        system_prompt = st.text_area(
            "System prompt", value=config.SYSTEM_INSTRUCTION, height=110
        )

        st.subheader("🛠️ Tools")
        all_tools = list(tools.TOOLS)
        enabled_tools = st.multiselect("Enabled", all_tools, default=all_tools)

        st.subheader("👂 Speech-to-text")
        language_label = st.selectbox("Language", list(LANGUAGES), index=0)

        st.subheader("🔊 Text-to-speech")
        speak = st.toggle("Speak replies", value=True)
        voices = tts.list_voices() if speak else []
        voice = None
        rate = None
        autoplay = True
        if speak:
            if voices:
                default = voices.index("Samantha") if "Samantha" in voices else 0
                voice = st.selectbox("Voice", voices, index=default)
            else:
                st.caption("No system voices detected — using the platform default.")
            rate = st.slider("Speaking rate (wpm)", 120, 300, 190, step=10)
            autoplay = st.toggle(
                "Autoplay reply (manual mode)", value=True,
                help="Live mode always plays through the server's speakers.",
            )

        st.subheader("🎤 Microphone")
        source = BROWSER_MIC
        device = None
        aggressiveness = config.VAD_AGGRESSIVENESS
        barge_in = True
        if MIC_IMPORT_ERROR:
            st.error(f"Live mode unavailable: {MIC_IMPORT_ERROR}")
        else:
            options = [BROWSER_MIC, SERVER_MIC] if not WEBRTC_ERROR else [SERVER_MIC]
            source = st.radio(
                "Audio source",
                options,
                index=0,
                help="The browser microphone is streamed to the server over WebRTC, so it works "
                "when the app is running somewhere else. The server microphone belongs to the "
                "machine running Streamlit.",
            )
            if WEBRTC_ERROR:
                st.caption(f"Browser microphone unavailable: {WEBRTC_ERROR}")

            if source == SERVER_MIC:
                devices = list_input_devices()
                labels = ["System default"] + [f"[{i}] {name}" for i, name in devices]
                picked = st.selectbox("Input device", labels, index=0)
                device = (
                    None if picked == "System default" else devices[labels.index(picked) - 1][0]
                )
            aggressiveness = st.slider(
                "VAD aggressiveness", 0, 3, config.VAD_AGGRESSIVENESS,
                help="Higher filters more non-speech, but may clip quiet talkers.",
            )
            barge_in = st.toggle(
                "Allow interrupting the agent", value=True,
                help="Stops playback the moment you start speaking. With the server "
                "microphone on open speakers it can hear the agent and interrupt itself — use "
                "headphones, or turn this off to mute the mic while it talks. The browser "
                "microphone cancels its own echo.",
            )
            if live_engine():
                st.caption("Source, device and VAD changes apply after you restart live mode.")

        st.divider()
        if st.button("🧹 Clear conversation", width="stretch"):
            st.session_state.transcript.clear()
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.rerun()

    return {
        "model": model,
        "system_prompt": system_prompt,
        "enabled_tools": enabled_tools,
        "language": LANGUAGES[language_label],
        "speak": speak,
        "voice": voice,
        "rate": rate,
        "autoplay": autoplay,
        "source": source,
        "device": device,
        "aggressiveness": aggressiveness,
        "barge_in": barge_in,
    }


# --------------------------------------------------------------------------------------
# live mode
# --------------------------------------------------------------------------------------

def start_live(settings, backend=None):
    agent = get_agent(settings)
    engine = ConversationEngine(
        agent,
        st.session_state.transcript,
        backend=backend,
        language=settings["language"],
        voice=settings["voice"],
        rate=settings["rate"],
        device=settings["device"],
        aggressiveness=settings["aggressiveness"],
        barge_in=settings["barge_in"],
        speak=settings["speak"],
    )
    engine.start()
    st.session_state.engine = engine


def browser_backend(settings):
    """One backend per session, kept across reruns — the WebRTC worker calls into it from its
    own thread, so it must outlive the script run that created it."""
    backend = st.session_state.get("browser_backend")
    if backend is None:
        backend = BrowserAudioBackend(aggressiveness=settings["aggressiveness"])
        st.session_state.browser_backend = backend
    return backend


def sync_live_settings(settings):
    """Push the settings that can change without reopening the audio devices."""
    engine = live_engine()
    if not engine:
        return
    engine.language = settings["language"]
    engine.voice = settings["voice"]
    engine.rate = settings["rate"]
    engine.barge_in = settings["barge_in"]
    engine.speak = settings["speak"]
    get_agent(settings)  # model / prompt / tool changes apply to the shared agent


def render_live_status():
    engine = st.session_state.engine
    if engine is None:
        st.caption("Live mode is off — the microphone is not being read.")
        return

    snap = engine.snapshot()
    if snap["error"]:
        st.error(snap["error"])
        return

    icon, label = STATE_BADGES.get(snap["state"], ("⚪", snap["state"]))
    cols = st.columns([3, 1, 1, 1])
    cols[0].markdown(f"### {icon} {label}")
    timings = snap["timings"]
    cols[1].metric("ASR", f"{timings.get('asr_ms', 0):.0f} ms" if timings else "—")
    cols[2].metric(
        "To first word",
        f"{timings['first_audio_ms']:.0f} ms" if timings.get("first_audio_ms") else "—",
        help="From the end of your sentence to the agent starting to speak.",
    )
    cols[3].metric("Interruptions", snap["interruptions"])

    if snap["partial"]:
        st.caption(f"💭 {snap['partial']}")


# --------------------------------------------------------------------------------------
# manual turns
# --------------------------------------------------------------------------------------

def run_turn(text: str, settings: dict, user_audio: bytes | None = None, asr_ms: float = 0.0):
    """One non-live turn: agent replies, and the reply is returned as an inline audio player."""
    transcript = st.session_state.transcript
    transcript.add({"role": "user", "text": text, "audio": user_audio})

    events = []
    agent = get_agent(settings)
    agent.on_event = events.append

    entry = {"role": "assistant", "events": events, "timings": {"asr_ms": asr_ms}}
    try:
        started = time.perf_counter()
        reply = agent.respond(text)
        entry["timings"]["llm_ms"] = (time.perf_counter() - started) * 1000
        entry["text"] = reply or "(empty reply)"

        if settings["speak"] and reply:
            started = time.perf_counter()
            try:
                entry["audio"] = tts.synthesize_to_bytes(
                    reply, voice=settings["voice"], rate=settings["rate"]
                )
                entry["timings"]["tts_ms"] = (time.perf_counter() - started) * 1000
            except Exception as exc:  # a broken voice shouldn't lose the text reply
                entry["tts_error"] = f"{type(exc).__name__}: {exc}"
    except MissingCredentials as exc:
        entry["text"] = str(exc)
        entry["error"] = True
    except Exception as exc:
        entry["text"] = f"{type(exc).__name__}: {exc}"
        entry["error"] = True
        entry["traceback"] = traceback.format_exc()
    finally:
        agent.on_event = None

    transcript.add(entry)


def transcribe_audio(wav_bytes: bytes, settings: dict) -> tuple[str, float]:
    started = time.perf_counter()
    text = transcribe(wav_bytes, language=settings["language"])
    return text, (time.perf_counter() - started) * 1000


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def render_message(msg: dict, autoplay: bool):
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg.get("text", ""))
            if msg.get("traceback"):
                with st.expander("Traceback"):
                    st.code(msg["traceback"])
        else:
            st.markdown(msg.get("text") or "…")

        if msg.get("interrupted"):
            st.caption("⏹️ interrupted")
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/wav", autoplay=autoplay)
        if msg.get("tts_error"):
            st.caption(f"⚠️ Speech synthesis failed — {msg['tts_error']}")

        for event in msg.get("events") or []:
            with st.expander(f"🛠️ `{event['tool']}` called"):
                st.write("**Arguments**")
                st.json(event["args"] or {})
                st.write("**Result**")
                st.code(event["result"])

        timings = msg.get("timings")
        if timings:
            parts = [
                f"{name.replace('_ms', '').replace('_', ' ')} {value:.0f} ms"
                for name, value in timings.items()
                if value
            ]
            if parts:
                st.caption(" · ".join(parts))


def render_conversation(settings: dict):
    """Status + transcript. Called directly, or wrapped in an auto-refreshing fragment."""
    render_live_status()
    st.divider()

    messages = st.session_state.transcript.snapshot()
    if not messages:
        st.info(
            "Start **live conversation** below to just talk, or use the recorder / text box "
            "for one-off turns."
        )
        return

    last_assistant = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
    )
    for i, msg in enumerate(messages):
        render_message(msg, autoplay=settings["autoplay"] and i == last_assistant)


@st.fragment(run_every=0.6)
def render_conversation_live(settings: dict):
    """Same view, refreshed twice a second while the engine is running — that's how the page
    tracks a conversation happening on another thread without reloading the whole script."""
    render_conversation(settings)


def render_browser_controls(settings: dict):
    """Live mode over WebRTC. The component's own START/STOP drives the engine: once audio is
    flowing there is nothing to press per turn — you just talk."""
    backend = browser_backend(settings)
    try:
        ctx = webrtc_streamer(
            key="voice-agent",
            mode=WebRtcMode.SENDRECV,  # send the mic up, get the agent's voice back down
            audio_frame_callback=backend.process_frame,
            media_stream_constraints={"video": False, "audio": AUDIO_CONSTRAINTS},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"Could not start the browser audio connection: {type(exc).__name__}: {exc}")
        st.caption("Switch to the server microphone in the sidebar, or use manual input below.")
        return

    engine = live_engine()
    playing = bool(ctx.state.playing)
    if playing and engine is None:
        start_live(settings, backend=backend)
        st.rerun()
    elif not playing and engine is not None:
        engine.stop()
        st.rerun()

    if playing:
        st.caption(
            "Just talk — the agent answers when you stop, and replies through your browser."
            + (" Talk over it to cut it off." if settings["barge_in"] else "")
        )
    else:
        st.caption(
            "Press **START** above and allow microphone access. Your browser handles echo "
            "cancellation, so speakers are fine. Opening the app from another machine needs "
            "**HTTPS** — browsers only grant microphone access on a secure origin (`localhost` "
            "counts as secure)."
        )
    if backend.error:
        st.error(backend.error)


def render_chat_tab(settings: dict):
    engine = live_engine()
    browser_mode = settings["source"] == BROWSER_MIC and not WEBRTC_ERROR and not MIC_IMPORT_ERROR

    if browser_mode:
        render_browser_controls(settings)
    else:
        left, right = st.columns([1, 3])
        with left:
            if MIC_IMPORT_ERROR:
                st.button("🔴 Start live conversation", disabled=True, width="stretch")
            elif engine:
                if st.button("⏹️ Stop live conversation", type="primary", width="stretch"):
                    engine.stop()
                    st.rerun()
            else:
                if st.button("🔴 Start live conversation", type="primary", width="stretch"):
                    start_live(settings)
                    st.rerun()
        with right:
            if MIC_IMPORT_ERROR:
                st.caption("Live mode needs PortAudio — `brew install portaudio`.")
            elif engine:
                st.caption(
                    "Just talk. The agent replies out loud through this machine's speakers, "
                    "starting before it has finished thinking."
                    + (" Talk over it to cut it off." if settings["barge_in"] else "")
                )
            else:
                st.caption(
                    "Opens the microphone on the machine running Streamlit and keeps it open."
                )

    engine = live_engine()
    sync_live_settings(settings)

    if engine:
        render_conversation_live(settings)
    else:
        render_conversation(settings)

    st.divider()

    if engine:
        st.caption("🎙️ Live mode is running — stop it to use manual input.")
        return

    recording = st.audio_input(
        "🎙️ Record one message", key=f"recorder_{st.session_state.rec_nonce}"
    )
    typed = st.chat_input("Type a message instead…")

    if recording is not None:
        wav_bytes = recording.getvalue()
        digest = hashlib.sha1(wav_bytes).hexdigest()
        if digest != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = digest
            with st.spinner("Transcribing…"):
                try:
                    text, asr_ms = transcribe_audio(wav_bytes, settings)
                except Exception as exc:
                    st.error(f"Transcription failed: {type(exc).__name__}: {exc}")
                    text, asr_ms = "", 0.0
            if text:
                with st.spinner("Thinking…"):
                    run_turn(text, settings, user_audio=wav_bytes, asr_ms=asr_ms)
            else:
                st.warning("Nothing recognizable in that recording — try again.")
            st.session_state.rec_nonce += 1  # clears the recorder for the next take
            st.rerun()

    if typed:
        with st.spinner("Thinking…"):
            run_turn(typed, settings)
        st.rerun()


def render_tools_tab(settings: dict):
    st.subheader("Registered tools")
    st.caption(
        "Defined in [voice_agent/tools.py](voice_agent/tools.py). Add a function, register it in "
        "`TOOLS`, and describe it in `TOOL_DECLARATIONS` — it shows up here automatically."
    )
    for decl in tools.TOOL_DECLARATIONS:
        enabled = decl["name"] in settings["enabled_tools"]
        with st.expander(f"{'✅' if enabled else '⛔'} `{decl['name']}` — {decl['description']}"):
            st.json(decl["parameters"])
            params = decl["parameters"].get("properties", {})
            with st.form(f"try_{decl['name']}"):
                st.write("**Try it**")
                args = {
                    name: st.text_input(name, key=f"arg_{decl['name']}_{name}")
                    for name in params
                }
                if st.form_submit_button("Run"):
                    fn = tools.TOOLS[decl["name"]]
                    try:
                        st.success(fn(**{k: v for k, v in args.items() if v}))
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("Tool calls this session")
    calls = [
        e for m in st.session_state.transcript.snapshot() for e in (m.get("events") or [])
    ]
    if calls:
        st.dataframe(
            [{"tool": c["tool"], "args": json.dumps(c["args"]), "result": c["result"]} for c in calls],
            width="stretch",
        )
    else:
        st.caption("No tool calls yet.")


def render_session_tab(settings: dict):
    messages = st.session_state.transcript.snapshot()
    replies = [m for m in messages if m["role"] == "assistant"]
    first_audio = [
        m["timings"]["first_audio_ms"]
        for m in replies
        if m.get("timings", {}).get("first_audio_ms")
    ]
    errors = sum(1 for m in replies if m.get("error"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turns", len(replies))
    c2.metric("Errors", errors)
    c3.metric(
        "Avg time to first word",
        f"{sum(first_audio) / len(first_audio):.0f} ms" if first_audio else "—",
    )
    c4.metric("Interrupted replies", sum(1 for m in replies if m.get("interrupted")))

    if first_audio:
        st.line_chart({"time to first spoken word (ms)": first_audio})

    st.subheader("Transcript")
    lines = [
        f"**{'You' if m['role'] == 'user' else 'Agent'}:** {m.get('text', '')}"
        for m in messages
    ]
    st.markdown("\n\n".join(lines) if lines else "_Nothing yet._")
    if lines:
        st.download_button(
            "⬇️ Download transcript (JSON)",
            data=json.dumps(
                [{k: v for k, v in m.items() if k != "audio"} for m in messages], indent=2
            ),
            file_name="voice_agent_transcript.json",
            mime="application/json",
        )

    with st.expander("Raw message history sent to the model"):
        agent = st.session_state.agent
        st.json(agent.messages if agent else [])

    with st.expander("Active configuration"):
        st.json({k: v for k, v in settings.items() if k != "system_prompt"})


def render_about_tab():
    st.markdown(
        """
### Live mode — how the real-time part works

There is no press-to-send: voice activity detection decides when your turn ended. A background
thread owns the audio, so nothing waits on the page:

| | |
|---|---|
| **Segmenter** | Watches 30 ms frames and marks where speech starts and ends. The instant it triggers it fires the barge-in signal — it doesn't wait for you to finish. |
| **Process thread** | Transcribes the utterance (Groq Whisper), then **streams** the reply from the LLM. As each sentence completes it is synthesized and queued, so the agent starts speaking while the rest of the answer is still being generated. |
| **Playback** | Gapless between sentences, and abortable mid-word. |

Interrupting the agent stops playback, abandons the rest of the stream, and commits what was
actually said to the conversation history, so the follow-up question still makes sense.

### Two microphones

| | Browser microphone | Server microphone |
|---|---|---|
| Where it runs | Your browser, streamed over WebRTC | The machine running Streamlit |
| Works from another machine | **Yes** | No |
| Echo handling | The browser cancels its own echo — speakers are fine | None: use headphones, or turn off *Allow interrupting the agent* |
| Requirement | HTTPS unless you're on `localhost` — browsers only grant microphone access on a secure origin | PortAudio |

The reply travels back down the same WebRTC connection in browser mode, so the audio comes out
of whichever machine you're sitting at.

### Where the latency goes

Transcription and the first LLM token are one API round-trip each; the first sentence of speech
has to be synthesized before anything can play. Expect roughly a second between you stopping and
the agent starting. The Session tab charts it per turn.

### Setup

```bash
brew install portaudio
pip install -r requirements.txt
cp .env.example .env            # then fill in GROQ_API_KEY
streamlit run app.py
```

The terminal version is still there: `python main.py`.
"""
    )


# --------------------------------------------------------------------------------------

def main():
    init_state()
    st.title("🎙️ Real-Time Voice Agent")
    st.caption("Groq Whisper → streaming Llama with tool calling → sentence-by-sentence speech")

    settings = render_sidebar()

    if not config.has_groq_api_key():
        st.warning("Add your Groq API key in the sidebar to start talking.", icon="🔑")

    chat, tools_tab, session, about = st.tabs(["💬 Chat", "🛠️ Tools", "📊 Session", "ℹ️ About"])
    with chat:
        render_chat_tab(settings)
    with tools_tab:
        render_tools_tab(settings)
    with session:
        render_session_tab(settings)
    with about:
        render_about_tab()


main()
