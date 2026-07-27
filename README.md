# Real-Time Voice AI Agent

Hold a spoken conversation with an LLM. The mic stays open, speech boundaries are found with
voice activity detection, and the agent **starts talking before it has finished thinking** —
each sentence is spoken as the model produces it. Talk over it and it stops mid-word.

Two front-ends over the same `voice_agent` package:

- **Web UI** — `streamlit run app.py`
- **Terminal** — `python main.py`

## Stack

- **ASR:** Groq-hosted Whisper (`whisper-large-v3-turbo`, free tier)
- **LLM:** Groq (Llama 3.3 70B by default, streamed, function calling enabled)
- **TTS:** offline system voices — macOS `say`, `pyttsx3` elsewhere (no API key, no quota)
- **Audio I/O:** sounddevice (mic capture and gapless playback) + webrtcvad (speech detection)
- **Web UI:** Streamlit

## Project structure

```
realtime_voice_agent/
├── app.py               # Streamlit web interface
├── main.py              # terminal entry point
├── requirements.txt
├── .env.example
└── voice_agent/         # package with all the logic
    ├── config.py         # env vars + constants (read lazily, so the UI can load without a key)
    ├── client.py         # lazily-built, cached Groq client
    ├── audio_io.py       # mic capture, VAD segmentation, interruptible playback
    ├── backends.py       # where audio comes from / goes to: local sound card or browser (WebRTC)
    ├── conversation.py   # the real-time loop: turn-taking, streaming, barge-in
    ├── transcript.py     # thread-safe conversation log shared with the UI
    ├── asr.py            # Groq Whisper transcription
    ├── llm.py            # Groq's Llama 3.3 agent + tool-calling loop (streaming and not)
    ├── tools.py          # tool implementations + schemas
    └── tts.py            # offline speech synthesis
```

## Setup

1. Install system audio dependency (macOS):
   ```
   brew install portaudio
   ```

2. Install Python dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure API keys:
   ```
   cp .env.example .env
   ```
   Then fill in `GROQ_API_KEY` in `.env` (no key needed for TTS — it's fully offline).

4. Run it:
   ```
   streamlit run app.py     # web interface at http://localhost:8501
   python main.py           # or the terminal version (Ctrl+C to quit)
   ```

   In the web UI, press **START** on the audio widget (browser microphone) or
   **Start live conversation** (server microphone), then just talk. In the terminal, just talk.
   There is nothing to press per turn — VAD decides when your turn ended.

## How the real-time loop works

`voice_agent/conversation.py` runs the conversation off the UI thread, so nothing waits on the
page:

| | |
|---|---|
| **Segmenter** | `webrtcvad` over 30 ms frames, with a padding-based trigger so brief noise neither starts nor ends a turn. The moment it triggers it fires the barge-in signal — it doesn't wait for you to finish the sentence. |
| **Process thread** | Transcribes the utterance with Groq Whisper, then **streams** the reply from the LLM. Each time a sentence completes it is synthesized and queued, so speech starts while the rest of the answer is still being generated. Tool calls are assembled from the stream and executed in-line. |
| **Playback** | Gapless between sentences, and abortable mid-word. |

Interrupting stops playback, abandons the rest of the LLM stream, and commits what was actually
spoken to the conversation history, so your follow-up still makes sense.

Roughly a second passes between you stopping and the agent starting: one round-trip for
transcription, one for the first token, then synthesis of the first sentence. The web UI charts
that per turn under **Session**.

## Two microphones

Chosen in the sidebar under **Audio source** (`voice_agent/backends.py`):

| | Browser microphone | Server microphone |
|---|---|---|
| Where it runs | Your browser, streamed over WebRTC (`streamlit-webrtc`) | The machine running Streamlit (`sounddevice`) |
| Works from another machine | **Yes** — this is the one to use if the app isn't on your machine | No |
| Where the reply comes out | Back down the same WebRTC connection, i.e. your speakers | The server's speakers |
| Echo | The browser cancels its own echo, so open speakers are fine | None — use headphones, or turn off *Allow interrupting the agent* to mute the mic while it speaks |
| Needs | HTTPS unless you're on `localhost` (see below) | PortAudio |

### HTTPS for remote access

Browsers only grant microphone access on a secure origin. `http://localhost:8501` counts as
secure, but `http://192.168.x.x:8501` does not — the browser will refuse the mic with no
useful error. To use the browser microphone from another machine, put the app behind HTTPS:

```bash
# a quick tunnel (gives you an https URL)
cloudflared tunnel --url http://localhost:8501

# or Streamlit's own TLS, with a certificate you supply
streamlit run app.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem
```

## Web interface

- **💬 Chat** — live conversation controls and status (listening / thinking / speaking, time to
  first spoken word, interruption count), the running transcript, and — when live mode is off —
  a press-to-record widget and text box for one-off turns.
- **🛠️ Tools** — every registered tool with its JSON schema, a form to invoke it directly, and a
  log of every tool call the model made this session.
- **📊 Session** — turn/error counts, latency chart, raw message history, transcript download.
- **ℹ️ About** — the same notes, in the app.

The sidebar controls the API key (session-only, if you'd rather not use `.env`), LLM model,
system prompt, exposed tools, transcription language, voice + speaking rate, audio source, mic
device, VAD aggressiveness, and barge-in. Most apply immediately; audio source, device and VAD
changes need live mode restarted.

If PortAudio isn't installed the app still runs — only live mode is disabled. If
`streamlit-webrtc` isn't installed, only the browser microphone is.

## Adding a tool

Add a function to `voice_agent/tools.py`, register it in `TOOLS`, and describe its schema in
`TOOL_DECLARATIONS`. Groq's Llama 3.3 will call it automatically when relevant, and it appears
in the web UI's Tools tab with no extra wiring.

## Notes

- Speech synthesis shells out to `say` on macOS rather than using `pyttsx3` directly: pyttsx3's
  macOS driver runs NSSpeechSynthesizer on the calling thread, which misbehaves in Streamlit's
  worker thread. `pyttsx3` is still the path on other platforms.
- Playback writes to a `sounddevice` output stream. Spawning `afplay` per clip costs ~0.8 s of
  process startup, which lands as a silent gap between every sentence; the external player is
  kept only as a fallback if the device fails or the clip isn't decodable.
- Whisper's API transcribes complete utterances rather than streaming partial text — this
  keeps things simple but adds the round-trip latency of one API call per utterance. It is the
  largest remaining chunk of the delay before the agent starts speaking.
- Models occasionally emit a malformed tool call, which Groq rejects server-side mid-stream.
  `voice_agent/llm.py` retries, then falls back to answering without tools, and never lets a
  bad tool name into the message history (the API would reject every later turn if it did).
- Groq's free tier has rate limits; if you hit them frequently, consider switching to local
  Whisper via `faster-whisper` for fully offline, unlimited transcription.
