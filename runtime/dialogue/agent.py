"""
Dialogue agent (plan 4.4): an LLM call that returns *structured* output
-- phrase-length segments, each with an affect distribution over the
11-label vocabulary and an intensity. The shape is enforced three times:
by the API's structured-output mode, by the JSON schema below, and again
by `affect.validate_reply` (never trust, always validate).

GeminiAgent is the dialogue backend (`await agent.reply(text) ->
[Segment]`; needs GEMINI_API_KEY). Small model, minimal thinking --
the plan's latency budget. STALL_REPLY is the failure-path utterance
served by the Conversation when the API times out or errors; it is
part of the never-freeze ladder, not a dialogue backend.
"""

import asyncio
import json

import runtime.config as C
from runtime.dialogue.affect import validate_reply

SYSTEM_PROMPT = f"""You are the voice of a small desk lamp robot. You \
cannot see; you hear and speak, and your body expresses emotion through \
motion and light. You are curious, warm, and a little playful. Keep \
replies short: usually 1-3 segments, each a phrase of 1-12 words.

Reply ONLY with JSON: {{"segments": [{{"text", "affect", "intensity"}}]}}.
- "text": one spoken phrase (1-12 words).
- "affect": how you feel saying it, 1-3 keys from exactly this list: \
{", ".join(C.EMOTIONS)}. Weights are positive numbers.
- "intensity": 0 to 1, how strongly to perform it.
At most {C.MAX_SEGMENTS} segments."""

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "affect": {
                        "type": "object",
                        "properties": {e: {"type": "number"}
                                       for e in C.EMOTIONS},
                        "additionalProperties": False,
                    },
                    "intensity": {"type": "number"},
                },
                "required": ["text", "affect", "intensity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["segments"],
    "additionalProperties": False,
}

# The "give me a second" ladder (plan section 5): served on LLM timeout
# or error so the lamp never freezes mid-conversation.
STALL_REPLY = {"segments": [
    {"text": "Hmm, give me a second.",
     "affect": {"confusion": 0.6, "interest": 0.4}, "intensity": 0.4},
]}


class AgentError(Exception):
    pass


class SegmentScanner:
    """Pulls complete segments out of a JSON reply as it streams in.

    The reply is `{"segments": [{...}, {...}]}` and the lamp can start
    speaking the first phrase the moment that first object closes --
    typically well before the model has finished the last one. Waiting
    for the closing brace cost most of the LLM's measured 0.82 s on
    every turn.

    Deliberately not a full JSON parser: it tracks brace depth from the
    start of the array, respecting strings and escapes so a `{` inside
    the lamp's own words cannot desynchronize it, and hands each balanced
    object to json.loads. Anything it cannot parse it skips -- the caller
    still has the accumulated text and the blocking path to fall back on."""

    def __init__(self):
        self.buf = ""
        self._i = None      # index just after the segments array's '['
        self._depth = 0
        self._start = None
        self._in_str = False
        self._esc = False

    def feed(self, chunk):
        """Add streamed text; yield every segment that just completed."""
        self.buf += chunk or ""
        if self._i is None:
            k = self.buf.find('"segments"')
            if k < 0:
                return []
            k = self.buf.find("[", k)
            if k < 0:
                return []
            self._i = k + 1
        out = []
        while self._i < len(self.buf):
            ch = self.buf[self._i]
            self._i += 1
            if self._in_str:
                if self._esc:
                    self._esc = False
                elif ch == "\\":
                    self._esc = True
                elif ch == '"':
                    self._in_str = False
                continue
            if ch == '"':
                self._in_str = True
            elif ch == "{":
                if self._depth == 0:
                    self._start = self._i - 1
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0 and self._start is not None:
                    raw = self.buf[self._start:self._i]
                    self._start = None
                    try:
                        out.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass            # skip the malformed one, keep going
        return out


def _gemini_schema(node):
    """REPLY_SCHEMA -> the OpenAPI subset Gemini accepts (drop
    `additionalProperties`, keep structure otherwise)."""
    if isinstance(node, dict):
        return {k: _gemini_schema(v) for k, v in node.items()
                if k != "additionalProperties"}
    if isinstance(node, list):
        return [_gemini_schema(v) for v in node]
    return node


GEMINI_REPLY_SCHEMA = _gemini_schema(REPLY_SCHEMA)


class GeminiAgent:
    """Structured reply via the Gemini API (google-genai SDK). History
    is kept here, one instance per conversation session."""

    def __init__(self, model=None, client=None):
        if client is None:
            import os

            from google import genai
            key = os.environ.get("GEMINI_API_KEY") \
                or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise AgentError("GEMINI_API_KEY not set (put it in the "
                                 "repo-root .env; see .env.example)")
            # the Gemini API endpoint, not Vertex: the project's key is
            # service-restricted to generativelanguage.googleapis.com
            # (Vertex calls return API_KEY_SERVICE_BLOCKED)
            client = genai.Client(api_key=key, vertexai=False)
        self.client = client
        self.model = model or C.GEMINI_MODEL
        self.history = []          # genai contents: role user|model

    def _config(self, mood):
        from google.genai import types
        system = SYSTEM_PROMPT if mood is None else \
            f"{SYSTEM_PROMPT}\n\nRight now you feel: {mood.describe()}. " \
            f"Let that colour the reply; do not announce it."
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=C.LLM_MAX_TOKENS,
            response_mime_type="application/json",
            response_schema=GEMINI_REPLY_SCHEMA,
            # keep thinking minimal: the section-7 latency budget
            # (thinking_budget=0 is the pre-Gemini-3 knob and 400s)
            thinking_config=types.ThinkingConfig(thinking_level="low"))

    def _remember(self, raw):
        self.history.append({"role": "model", "parts": [{"text": raw}]})
        if len(self.history) > 2 * C.MAX_HISTORY_TURNS:
            self.history = self.history[-2 * C.MAX_HISTORY_TURNS:]

    async def reply_stream(self, text, mood=None):
        """Yield Segments as the model finishes them, not after the last.

        A reply is one to three phrases and the lamp can start speaking
        the first as soon as it closes; waiting for the whole object cost
        most of the measured 0.82 s per turn. Falls back to nothing --
        the caller's stall reply -- if the stream produces no usable
        segment, so a malformed response degrades exactly as before.

        `mood` goes into the system instruction rather than the history:
        it costs nothing per turn and always reflects *now*, where a
        history entry would keep restating a mood it no longer has."""
        self.history.append({"role": "user", "parts": [{"text": text}]})
        scanner = SegmentScanner()
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=self.history,
                config=self._config(mood))
            async for chunk in stream:
                for raw in scanner.feed(getattr(chunk, "text", "") or ""):
                    for seg in validate_reply({"segments": [raw]}):
                        yield seg
        except asyncio.CancelledError:
            self.history.pop()
            raise
        except Exception as e:
            self.history.pop()
            raise AgentError(str(e)) from e
        self._remember(scanner.buf)

    async def reply(self, text, mood=None):
        """The whole reply at once. `reply_stream` is the path the
        runtime takes; this stays for callers that want it in one piece
        and as the reference for what the streamed version must match."""
        self.history.append({"role": "user", "parts": [{"text": text}]})
        try:
            resp = await self.client.aio.models.generate_content(
                model=self.model, contents=self.history,
                config=self._config(mood))
        except asyncio.CancelledError:
            self.history.pop()
            raise
        except Exception as e:
            self.history.pop()          # keep history consistent for retry
            raise AgentError(str(e)) from e

        raw = resp.text or ""
        self._remember(raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise AgentError(f"unparseable reply: {e}") from e
        return validate_reply(payload)
