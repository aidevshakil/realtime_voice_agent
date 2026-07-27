import sys
import time

from voice_agent.config import MissingCredentials, groq_api_key
from voice_agent.conversation import ConversationEngine, Transcript
from voice_agent.llm import VoiceAgent


def main():
    try:
        groq_api_key()  # fail fast with a readable message instead of mid-conversation
    except MissingCredentials as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 1

    transcript = Transcript()
    engine = ConversationEngine(VoiceAgent(), transcript, barge_in=True)
    engine.start()

    print("Listening... just talk (Ctrl+C to quit). Talk over the agent to interrupt it.")
    printed = 0
    try:
        while True:
            snapshot = engine.snapshot()
            if snapshot["error"]:
                print(f"[error] {snapshot['error']}", file=sys.stderr)
                return 1

            entries = transcript.snapshot()
            # a reply is filled in as it streams, so only print entries that are settled
            settled = len(entries) if snapshot["state"] == "listening" else len(entries) - 1
            for entry in entries[printed:settled]:
                who = "You" if entry["role"] == "user" else "Agent"
                suffix = " [interrupted]" if entry.get("interrupted") else ""
                print(f"{who}: {entry.get('text', '')}{suffix}")
            printed = max(printed, settled)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        engine.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)
