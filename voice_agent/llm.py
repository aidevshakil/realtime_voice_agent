import contextlib
import json

from voice_agent.client import groq_client
from voice_agent.config import SYSTEM_INSTRUCTION, llm_model
from voice_agent.tools import TOOL_DECLARATIONS, TOOLS


MALFORMED_REPLY = "Sorry, I garbled that one. Could you say it again?"

# Groq validates the model's generated tool call server-side and rejects a mangled one mid-stream.
# It's a sampling fluke, so the same request usually succeeds on a retry.
_MALFORMED_MARKERS = ("failed to call a function", "tool call validation failed")
_MAX_TOOL_RETRIES = 3  # last attempt drops tools entirely so the user still gets an answer


def _is_malformed_tool_call(exc) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _MALFORMED_MARKERS)


def _valid_tool_calls(tool_calls):
    """Drop calls naming a tool that doesn't exist.

    Models occasionally emit a mangled function name (the arguments spliced into the name, say).
    Echoing that back in the next request makes the API reject the *whole conversation* from then
    on — 'attempted to call tool X which was not in tools' — so it never reaches history.
    """
    return [c for c in tool_calls if c["function"]["name"] in TOOLS]


def _tools_schema(names=None):
    decls = TOOL_DECLARATIONS
    if names is not None:
        decls = [d for d in decls if d["name"] in names]
    return [{"type": "function", "function": decl} for decl in decls]


class VoiceAgent:
    """Holds the conversation and runs the tool-calling loop.

    `on_event`, if given, is called with a dict per tool call — the web UI uses it to show
    what the model actually did during a turn.
    """

    def __init__(self, model=None, system_prompt=None, enabled_tools=None, on_event=None):
        self.model = model or llm_model()
        self.system_prompt = system_prompt or SYSTEM_INSTRUCTION
        self.enabled_tools = enabled_tools  # None means "all tools"
        self.on_event = on_event
        self._messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def messages(self):
        return self._messages

    def reset(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]

    def configure(self, model=None, system_prompt=None, enabled_tools=None):
        """Apply setting changes mid-conversation. A new system prompt rewrites the existing
        system message so it takes effect on the very next turn."""
        if model:
            self.model = model
        if enabled_tools is not None:
            self.enabled_tools = enabled_tools
        if system_prompt and system_prompt != self.system_prompt:
            self.system_prompt = system_prompt
            self._messages[0] = {"role": "system", "content": system_prompt}

    def respond(self, user_text: str) -> str:
        checkpoint = len(self._messages)
        self._messages.append({"role": "user", "content": user_text})
        try:
            return self._resolve()
        except Exception:
            # roll back so a failed turn can't leave tool_calls without their results,
            # which the API rejects on every subsequent turn
            del self._messages[checkpoint:]
            raise

    def stream_respond(self, user_text: str, interrupt=None):
        """Same as respond(), but yields the reply as it is generated.

        Yields dicts: {"type": "text", "value": <token chunk>} and
        {"type": "tool", "tool": name, "args": {...}, "result": str}.

        `interrupt` is an optional threading.Event — when it gets set the stream is abandoned
        and whatever was said so far is committed to history, so the conversation stays coherent
        after the user talks over the agent.
        """
        checkpoint = len(self._messages)
        self._messages.append({"role": "user", "content": user_text})
        try:
            yield from self._resolve_stream(interrupt)
        except Exception:
            del self._messages[checkpoint:]
            raise

    def _emit(self, event):
        if self.on_event:
            self.on_event(event)

    def _run_tools(self, tool_calls):
        """Execute assembled tool calls, appending each result to history. Yields UI events."""
        for call in tool_calls:
            name = call["function"]["name"]
            fn = TOOLS.get(name)
            args = {}
            if not fn:
                result = f"Unknown tool: {name}"
            else:
                try:  # hand tool failures back to the model instead of killing the turn
                    args = json.loads(call["function"]["arguments"] or "{}") or {}
                    result = fn(**args)
                except Exception as exc:
                    result = f"Tool {name} failed: {type(exc).__name__}: {exc}"
            event = {"tool": name, "args": args, "result": str(result)}
            self._emit(event)
            self._messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": str(result)}
            )
            yield {"type": "tool", **event}

    def _resolve_stream(self, interrupt=None):
        schema = _tools_schema(self.enabled_tools)
        for _ in range(5):  # cap tool-call round-trips
            content = ""
            partials = {}  # index -> tool call being assembled across deltas

            for attempt in range(_MAX_TOOL_RETRIES):
                content = ""
                partials = {}
                emitted = False  # once text has been spoken, retrying would repeat it
                stream = None
                kwargs = {"model": self.model, "messages": self._messages, "stream": True}
                if schema:
                    kwargs["tools"] = schema
                    kwargs["tool_choice"] = "auto"
                try:
                    stream = groq_client().chat.completions.create(**kwargs)
                    for event in stream:
                        if interrupt is not None and interrupt.is_set():
                            self._messages.append({"role": "assistant", "content": content})
                            return
                        if not event.choices:
                            continue
                        delta = event.choices[0].delta
                        if delta.content:
                            content += delta.content
                            emitted = True
                            yield {"type": "text", "value": delta.content}
                        for tc in delta.tool_calls or []:
                            slot = partials.setdefault(
                                tc.index, {"id": "", "name": "", "args": ""}
                            )
                            if tc.id:
                                slot["id"] = tc.id
                            if tc.function and tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                slot["args"] += tc.function.arguments
                    break
                except Exception as exc:
                    if emitted or not _is_malformed_tool_call(exc):
                        raise
                    if attempt == _MAX_TOOL_RETRIES - 1:
                        raise
                    if attempt == _MAX_TOOL_RETRIES - 2:
                        schema = []  # it keeps mangling the call — answer without tools
                finally:
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            stream.close()

            if not partials:
                self._messages.append({"role": "assistant", "content": content})
                return

            tool_calls = _valid_tool_calls(
                [
                    {
                        "id": slot["id"],
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot["args"]},
                    }
                    for _, slot in sorted(partials.items())
                ]
            )
            if not tool_calls:  # every call was malformed — salvage the turn as plain text
                text = content or MALFORMED_REPLY
                self._messages.append({"role": "assistant", "content": text})
                if not content:
                    yield {"type": "text", "value": text}
                return

            self._messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
            yield from self._run_tools(tool_calls)

        final = "Sorry, I had trouble completing that request."
        self._messages.append({"role": "assistant", "content": final})
        yield {"type": "text", "value": final}

    def _complete(self, schema):
        """One non-streaming request, retrying a server-rejected malformed tool call.
        Returns (completion, schema) — the schema may come back empty if tools had to be
        dropped to get an answer at all."""
        for attempt in range(_MAX_TOOL_RETRIES):
            kwargs = {"model": self.model, "messages": self._messages}
            if schema:  # sending an empty tools list is rejected; omit it instead
                kwargs["tools"] = schema
                kwargs["tool_choice"] = "auto"
            try:
                return groq_client().chat.completions.create(**kwargs), schema
            except Exception as exc:
                if not _is_malformed_tool_call(exc) or attempt == _MAX_TOOL_RETRIES - 1:
                    raise
                if attempt == _MAX_TOOL_RETRIES - 2:
                    schema = []

    def _resolve(self) -> str:
        schema = _tools_schema(self.enabled_tools)
        for _ in range(5):  # cap tool-call round-trips
            completion, schema = self._complete(schema)
            message = completion.choices[0].message
            payload = message.model_dump(exclude_none=True)

            if not message.tool_calls:
                self._messages.append(payload)
                return (message.content or "").strip()

            tool_calls = _valid_tool_calls(payload.get("tool_calls", []))
            if not tool_calls:  # every call was malformed — salvage the turn as plain text
                text = (message.content or "").strip() or MALFORMED_REPLY
                self._messages.append({"role": "assistant", "content": text})
                return text

            payload["tool_calls"] = tool_calls
            self._messages.append(payload)
            for _ in self._run_tools(tool_calls):
                pass

        return "Sorry, I had trouble completing that request."
