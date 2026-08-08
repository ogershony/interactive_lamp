"""
The behavior layer (plan sections 3 and 9): owns the turn state machine
and drives one conversational turn end to end --

    endpoint -> REACT (cached clip, instantly)
             -> ASR final -> LLM structured reply -> per segment:
                TTS (measured duration) -> affect director -> motion
                (cache or engine) -> fit to duration -> schedule aligned

Every slow stage has a timeout and a fallback (plan section 5): the
ladder is cached clip, then idle breathing, and for dialogue a canned
utterance. The lamp never freezes; worst case it breathes and says
"give me a second".

The audio front-end (`feed_block`) is synchronous and cheap -- it runs
per 10 ms block on the consumer side of the mic ring. The slow half
(`handle_turn`) is async and lives on the event loop.
"""

import asyncio

import numpy as np

import runtime.config as C
from runtime.audio.envelope import envelope_30hz, resample
from runtime.audio.tts import SilentTts
from runtime.audio.vad import ENDPOINT, ONSET, EnergyVad, Endpointer
from runtime.dialogue.affect import (direct, estimate_react_affect,
                                     validate_reply)
from runtime.dialogue.agent import STALL_REPLY
from runtime.motion.align import fit_to_duration
from runtime.types import ScheduledClip, TurnState

SORRY_REPLY = {"segments": [
    {"text": "Sorry, I missed that.",
     "affect": {"understanding": 0.7, "confusion": 0.3}, "intensity": 0.4},
]}

REACT_PRIORITY, SPEAK_PRIORITY, AMBIENT_PRIORITY = 2, 1, 0
CHAIN_MAX = 8      # clips chained to cover one long speech segment


class Conversation:
    def __init__(self, scheduler, asr, agent, tts, cache=None, engine=None,
                 audio=None, recorder=None, vad=None, endpointer=None,
                 sr=None, rng=None):
        self.sched = scheduler
        self.asr = asr
        self.agent = agent
        self.tts = tts
        self.cache = cache
        self.engine = engine
        self.audio = audio
        self.rec = recorder
        self.vad = vad or EnergyVad()
        self.endpointer = endpointer or Endpointer()
        self.sr = sr or C.AUDIO_SR
        self.rng = rng or np.random.default_rng()
        self.state = TurnState.IDLE
        self._utterance = []           # mic blocks of the current turn
        self._silent_tts = SilentTts()
        # between clips the lamp performs ambient samples from the
        # prior rather than sitting in procedural idle
        if self.cache is not None and scheduler.filler_fn is None:
            scheduler.filler_fn = self.ambient_filler

    def _event(self, kind, **kw):
        if self.rec:
            self.rec.event(kind, **kw)

    # ---- audio front-end (sync, per 10 ms block) ---------------------------
    def feed_block(self, block):
        """Feed one mic block. Returns the utterance PCM when a turn
        endpoint fires, else None."""
        ev = self.endpointer.update(self.vad(block))
        if ev == ONSET:
            self.state = TurnState.LISTEN
            self._utterance = []
            self._event("speech_onset")
        if self.endpointer.in_utterance or ev == ENDPOINT:
            self._utterance.append(np.asarray(block, np.int16))
        if ev == ENDPOINT:
            pcm = np.concatenate(self._utterance) if self._utterance \
                else np.zeros(0, np.int16)
            self._utterance = []
            return pcm
        return None

    # ---- one full turn (async) ---------------------------------------------
    async def handle_turn(self, pcm=None, text=None):
        """Run a turn from utterance PCM (normal path) or from typed
        text (--turn demo path, skips ASR). Returns a summary dict."""
        self._event("endpoint")
        self.state = TurnState.REACT
        self._react(pcm, text)

        if text is None:
            text = await self._transcribe(pcm)
        # subtitle/replay track: what the user said, on the frame clock
        self._event("user_turn", text=text, frame=self.sched.frame)
        segments = await self._reply(text)

        self.state = TurnState.SPEAK
        planned = await self._speak(segments)
        self.state = TurnState.LISTEN
        self._event("turn_done", n_segments=len(planned))
        return dict(text=text, segments=planned)

    # ---- REACT (plan 6.4): motion within 200 ms of the endpoint ------------
    def _react(self, pcm, text):
        if self.cache is None or len(self.cache) == 0:
            return
        v = estimate_react_affect(text=text, pcm=pcm, sr=self.sr)
        hit = self.cache.react_clip(v, rng=self.rng)
        if hit is None:
            return
        self.sched.preempt(ScheduledClip(
            x=hit["x"].copy(), start_frame=self.sched.frame,
            priority=REACT_PRIORITY, tag="react"))
        self._event("react_motion", tag=hit["tag"])

    # ---- ASR ---------------------------------------------------------------
    async def _transcribe(self, pcm):
        if pcm is None or len(pcm) == 0:
            return ""
        try:
            tr = await asyncio.wait_for(self.asr.transcribe(pcm, self.sr),
                                        C.TIMEOUT_ASR)
            self._event("asr_final", chars=len(tr.text))
            return tr.text.strip()
        except Exception as e:  # noqa: BLE001 -- timeout or backend failure
            self._event("asr_failed", error=type(e).__name__)
            return ""

    # ---- dialogue ----------------------------------------------------------
    async def _reply(self, text):
        if not text or len(text) > 500:          # ASR->dialogue validator
            self._event("reply_fallback", reason="empty_or_long")
            return validate_reply(SORRY_REPLY)
        try:
            segments = await asyncio.wait_for(self.agent.reply(text),
                                              C.TIMEOUT_LLM + 0.5)
            if segments:
                self._event("llm_reply", n=len(segments))
                return segments
            reason = "no_segments"
        except asyncio.TimeoutError:
            reason = "timeout"
        except Exception as e:  # noqa: BLE001 -- any vendor failure stalls
            reason = type(e).__name__
        self._event("reply_fallback", reason=reason)
        return validate_reply(STALL_REPLY)

    # ---- SPEAK: synth, align, schedule -------------------------------------
    async def _speak(self, segments):
        # reply clips start when the react clip ends (a cut-off reaction
        # is worse than a slightly late reply -- plan 6.4); the react
        # clip may be active or still queued for the next tick
        base = self.sched.frame + 2
        if self.sched.active is not None \
                and self.sched.active.tag == "react":
            base = max(base, self.sched.active_end())
        for cl in self.sched.queue:
            if cl.tag == "react":
                base = max(base, cl.start_frame + len(cl.x))
        planned, t_cursor = [], 0.0
        for i, seg in enumerate(segments):
            tts_res = await self._synth(seg.text)
            dur = tts_res.duration_s
            start = base + int(round(t_cursor / C.DT))
            parts = await self._motion_chain(seg, dur, i)
            env = envelope_30hz(tts_res.pcm, tts_res.sample_rate)
            off = 0
            for j, x in enumerate(parts):
                self.sched.submit(ScheduledClip(
                    x=x, start_frame=start + off, priority=SPEAK_PRIORITY,
                    tag=f"speak:{i}" if j == 0 else f"speak:{i}+",
                    envelope=env[off:off + len(x)]))
                off += len(x)
            self._event("segment_planned", seg=i, start_frame=start,
                        seconds=round(dur, 3), text=seg.text,
                        n_clips=len(parts))
            if self.audio is not None:
                self.audio.play(resample(tts_res.pcm, tts_res.sample_rate,
                                         self.audio.sr))
            planned.append(dict(seg=i, text=seg.text, seconds=dur,
                                start_frame=start, has_motion=bool(parts)))
            t_cursor += dur
        return planned

    async def _motion_chain(self, seg, dur, i):
        """Clips covering the whole segment. When one sample runs out
        before the speech does, chain another with the same affect
        (the seam is absorbed by the scheduler's continuity blending)
        instead of freezing in a hold."""
        parts, remaining = [], dur
        for _ in range(CHAIN_MAX):
            if remaining <= 2 * C.DT:
                break
            x = await self._motion_for(direct(seg, seconds=remaining,
                                              tag=f"speak:{i}"))
            if x is None:
                break
            if len(x) * C.DT >= remaining * (1.0 - C.STRETCH_MAX):
                parts.append(fit_to_duration(x, remaining))
                remaining = 0.0
                break
            parts.append(np.asarray(x, np.float64))
            remaining -= len(x) * C.DT
        if parts and remaining > 2 * C.DT:
            # every source exhausted mid-segment: stretch/hold the tail
            last = parts.pop()
            parts.append(fit_to_duration(
                last, len(last) * C.DT + remaining))
        return parts

    def ambient_filler(self, frame, last_cmd):
        """Scheduler callback for empty moments: a calm sample from the
        prior (cache lookup, microseconds) so the lamp keeps performing
        between phrases and turns. Lowest priority -- anything real
        preempts it."""
        if self.cache is None or len(self.cache) == 0:
            return None
        v = np.zeros(len(C.EMOTIONS), np.float32)
        v[C.EMOTIONS.index(self.rng.choice(C.AMBIENT_AFFECTS))] = 1.0
        hit = self.cache.lookup(v, seconds=C.AMBIENT_SECONDS, rng=self.rng)
        if hit is None:
            return None
        return ScheduledClip(x=hit["x"].copy(), start_frame=frame,
                             priority=AMBIENT_PRIORITY, tag="ambient")

    async def _synth(self, text):
        try:
            return await asyncio.wait_for(self.tts.synth(text),
                                          C.TIMEOUT_TTS + 0.5)
        except Exception as e:  # noqa: BLE001
            self._event("tts_fallback", error=type(e).__name__)
            return await self._silent_tts.synth(text)

    async def _motion_for(self, req):
        """Cache if it is a good hit, else live generation with a
        timeout, else cache regardless of threshold, else None (the
        scheduler holds pose / breathes). Every request logs a
        `motion_source` event so sessions can report their live-vs-
        cached split (eval/metrics.motion_sources)."""
        hit = self.cache.lookup(req.affect, seconds=req.seconds,
                                cfg=req.cfg, rng=self.rng) \
            if self.cache is not None else None
        if hit is not None:
            self._event("motion_source", tag=req.tag, source="cache")
            return hit["x"].copy()
        if self.engine is not None:
            try:
                x = await asyncio.wait_for(
                    asyncio.to_thread(self.engine.clip, req),
                    C.TIMEOUT_MOTION)
                self._event("motion_source", tag=req.tag, source="engine")
                return x
            except Exception as e:  # noqa: BLE001 -- timeout or torch error
                self._event("motion_fallback", error=type(e).__name__)
        if self.cache is not None and len(self.cache):
            best = self.cache.lookup(req.affect, seconds=req.seconds,
                                     rng=self.rng)
            self._event("motion_source", tag=req.tag, source="cache_forced")
            if best is None:        # below threshold: take nearest anyway
                cos = self.cache._A @ (req.affect
                                       / np.linalg.norm(req.affect))
                return self.cache.entries[int(np.argmax(cos))]["x"].copy()
            return best["x"].copy()
        self._event("motion_source", tag=req.tag, source="none")
        return None

    # ---- hardware loop (Pi): mic ring -> turns -----------------------------
    async def run(self, ring_reader, stop=None):
        """Consume mic blocks from an audio ring reader, dispatching
        handle_turn() per endpoint. Half-duplex gating happens upstream
        in AudioIO, so blocks arriving here are silence while the lamp
        speaks."""
        while stop is None or not stop.is_set():
            for block in ring_reader.read_blocks():
                pcm = self.feed_block(block)
                if pcm is not None:
                    await self.handle_turn(pcm=pcm)
            await asyncio.sleep(C.AUDIO_BLOCK_MS / 1000.0)
