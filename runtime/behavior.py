"""
The behavior layer (plan sections 3 and 9): owns the turn state machine
and drives one conversational turn end to end --

    endpoint -> REACT (prefetched bank clip, instantly)
             -> ASR final -> LLM structured reply -> per segment:
                TTS (measured duration) -> affect director -> motion
                (generation service, MotionPool fallback) -> fit to
                duration -> schedule aligned

Every slow stage has a timeout and a fallback (plan section 5): the
ladder is live generation, then the nearest prefetched pool clip, then
idle breathing; for dialogue a one-line stall utterance. The lamp
never freezes; worst case it breathes and says "give me a second".

The audio front-end (`feed_block`) is synchronous and cheap -- it runs
per 10 ms block on the consumer side of the mic ring. The slow half
(`handle_turn`) is async and lives on the event loop.
"""

import asyncio
import difflib
import inspect
import time
from collections import deque

import numpy as np

import runtime.config as C
from runtime.audio.envelope import envelope_30hz, resample
from runtime.audio.tts import SilentTts
from runtime.audio.vad import (ENDPOINT, ONSET, SPECULATE, EnergyVad,
                               Endpointer)
from runtime.dialogue.affect import (blend_affect, direct, estimate_arousal,
                                     estimate_react_affect, intensity_to_cfg,
                                     mean_affect, named, valence_arousal,
                                     validate_reply)
from runtime.dialogue.agent import STALL_REPLY
from runtime.dialogue.mood import Mood
from runtime.motion.align import fit_to_duration
from runtime.types import ScheduledClip, TurnState

SELF_ECHO_RATIO = 0.75     # transcript vs the tail of a recent lamp phrase
SELF_ECHO_WINDOW_S = 6.0   # how long a phrase stays a plausible echo


async def _aiter(x):
    """Iterate either an async stream of segments or a plain list, so
    `_speak` does not care whether the backend streams."""
    if hasattr(x, "__aiter__"):
        async for item in x:
            yield item
    else:
        for item in x:
            yield item


def _normalize(s):
    return " ".join(str(s).lower().translate(
        str.maketrans("", "", ".,!?;:'\"")).split())


SORRY_REPLY = {"segments": [
    {"text": "Sorry, I missed that.",
     "affect": {"understanding": 0.7, "confusion": 0.3}, "intensity": 0.4},
]}

# SPEAK outranks REACT. The reaction exists to fill the hole while ASR and
# the LLM run; once the reply is ready the reaction has done its job, and
# letting it finish delays the answer *and* desynchronizes it -- the audio
# starts immediately while the motion waits behind the reaction, which
# measured at 1.1 s p95 against a 100 ms sync target. Ranking SPEAK higher
# lets the due reply take the reaction over, and the scheduler's tail blend
# (submit) plus offset decay (activation) make the cut continuous.
REACT_PRIORITY, SPEAK_PRIORITY, AMBIENT_PRIORITY = 1, 2, 0
CHAIN_MAX = 8      # clips chained to cover one long speech segment


class Conversation:
    def __init__(self, scheduler, asr, agent, tts, engine=None, pool=None,
                 audio=None, recorder=None, vad=None, endpointer=None,
                 sr=None, rng=None, mood=None):
        self.sched = scheduler
        self.asr = asr
        self.agent = agent
        self.tts = tts
        self.engine = engine       # RemoteEngine (or local MotionEngine)
        self.pool = pool           # MotionPool: react bank + ambient FIFO
        self.audio = audio
        self.rec = recorder
        self.vad = vad or EnergyVad()
        self.endpointer = endpointer or Endpointer(
            on_discard=lambda ms: self._event("speech_discarded",
                                              speech_ms=ms))
        self.sr = sr or C.AUDIO_SR
        self.rng = rng or np.random.default_rng()
        self.state = TurnState.IDLE
        # the conversation's affect memory: what the lamp still feels
        # between turns, and the thing that stops each reply from looking
        # like the lamp's first moment alive (dialogue/mood.py)
        self.mood = mood if mood is not None else Mood()
        self._pushed_level = -1.0      # last level given to the prefetcher
        self._utterance = []           # mic blocks of the current turn
        self._silent_tts = SilentTts()
        self._agent_mood = None        # does agent.reply take mood=? (lazy)
        self._seg_seq = 0              # session-unique speech segment ids
        self._in_turn = False          # a turn is between endpoint and reply
        self._was_gated = False        # mic muted on the previous block
        self._guess = None             # (n_samples, t0, task) speculative ASR
        self._said = deque(maxlen=8)   # (t, text) recently spoken, for the
        #                                echo guard in _self_echo
        #                                (TurnState.LISTEN is also the resting
        #                                 state, so it cannot answer this)
        # between clips the lamp performs ambient samples from the
        # prior rather than sitting in procedural idle
        if self.pool is not None and scheduler.filler_fn is None:
            scheduler.filler_fn = self.ambient_filler
        self._push_mood()

    def _event(self, kind, **kw):
        if self.rec:
            self.rec.event(kind, **kw)

    def _self_echo(self, text):
        """Is this transcript the lamp's own voice coming back?

        The half-duplex gate is supposed to prevent this, but it depends
        on OUTPUT_LAG_MS matching hardware nobody can interrogate --
        PortAudio reports the ALSA latency and knows nothing about the
        sound server, and the true figure moves with CPU load. A live
        session had the lamp ask "Did something happen?", transcribe
        itself, and answer its own question.

        So this is checked on content instead of on timing: compare the
        transcript against what the lamp recently said. Costs
        microseconds, needs no calibration, and holds on any box. A user
        who genuinely parrots the lamp loses one turn -- a fair price for
        never watching it talk to itself.

        The comparison is against the *tail* of each phrase, matched word
        for word. What leaks past the gate is the end of a sentence, and
        ASR mangles it, so a whole-string ratio is far too blunt: on the
        session that motivated this, whole-string scored the three real
        echoes at 0.51-1.00 against 0.28-0.38 for genuine turns -- the
        ranges touch. Tail-matched they score 0.79-1.00 against
        0.29-0.38 for the turns in that session.

        The ratio alone is still not enough -- plausible user speech
        reaches 0.72-0.73 against a short lamp phrase ("that is wonderful
        news isn't it" after the lamp said "That is wonderful!"), which
        is uncomfortably close to the weakest real echo at 0.79. So a
        candidate must also be no *longer* than what the lamp said: what
        bleeds through is the end of a sentence, and a suffix cannot
        outrun the whole. That test carries most of the weight and the
        ratio only has to separate what survives it.

        This is a backstop, not the defence. The half-duplex gate is;
        when this fires it means OUTPUT_LAG_MS is too small for the box,
        which is why it logs rather than passing quietly."""
        norm = _normalize(text)
        words = norm.split()
        if len(norm) < 8 or not words:     # too short to match on
            return None
        now = self.sched.frame * C.DT      # same clock live and simulated
        for said_at, said in self._said:
            if now - said_at > SELF_ECHO_WINDOW_S:
                continue
            other = _normalize(said)
            other_words = other.split()
            if not other_words or len(words) > len(other_words):
                continue                   # longer than the source: not an echo
            tail = " ".join(other_words[-len(words):])
            if difflib.SequenceMatcher(None, norm, tail).ratio() \
                    >= SELF_ECHO_RATIO or norm in other:
                return said
        return None

    def _push_mood(self):
        """Publish the mood to the ambient prefetcher and the session log.
        Called after every observation, so `eval/metrics.mood_track` can
        answer "did it stay sad?" without anyone watching the video."""
        vec, level = self.mood.vector(), self.mood.intensity()
        self._pushed_level = level
        if self.pool is not None:
            self.pool.set_mood(vec, level)
        self._event("mood", emotion=self.mood.dominant(),
                    level=round(level, 4))

    def _settle_mood(self):
        """Let the mood relax with the clock and republish when it has
        moved enough to matter.

        Decay is lazy -- it happens when someone reads the mood -- and
        between turns nobody does. Without this the ambient prefetcher
        keeps generating at whatever exaggeration the last turn left,
        and a mood that is supposed to subside over a minute of silence
        simply doesn't: the one thing the decay exists for."""
        level = self.mood.intensity()
        if abs(level - self._pushed_level) >= C.MOOD_PUBLISH_DELTA:
            self._push_mood()

    # ---- audio front-end (sync, per 10 ms block) ---------------------------
    def feed_block(self, block):
        """Feed one mic block. Returns the utterance PCM when a turn
        endpoint fires, else None."""
        if self.audio is not None:
            gated = self.audio.is_gated
            if self._was_gated and not gated:
                # The mic just re-opened. Anything the endpointer had
                # started accumulating is the tail of the lamp's own
                # sentence bleeding past the gate, not the user -- a live
                # session had it transcribe "Did something happen?" from
                # its own voice and then answer itself.
                self.endpointer.reset()
                self._utterance = []
            self._was_gated = gated
        ev = self.endpointer.update(self.vad(block))
        if ev == ONSET:
            self.state = TurnState.LISTEN
            self._utterance = []
            self._event("speech_onset")
            self._drop_guess()          # a new utterance invalidates any
        if self.endpointer.in_utterance or ev == ENDPOINT:
            self._utterance.append(np.asarray(block, np.int16))
        if ev == SPECULATE:
            self._guess_transcript()
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
        self._in_turn = True
        # the utterance *ended* at this endpoint, so it began that many
        # frames back. Everything about this turn that belongs on the
        # timeline -- the audio, the subtitle -- is anchored here, not at
        # whatever frame the code happens to reach later.
        speech_frame = self.sched.frame
        if pcm is not None and len(pcm):
            speech_frame -= len(pcm) / self.sr / C.DT
            if self.rec is not None:
                self.rec.audio(pcm, self.sr, speech_frame, who="user")
        try:
            self._react(pcm, text)

            if text is None:
                text = await self._transcribe(pcm)
            echo = self._self_echo(text) if text else None
            if echo is not None:
                self._event("self_heard", text=text, matched=echo)
                self.state = TurnState.LISTEN
                return dict(text=text, segments=[], self_heard=True)
            # Subtitle/replay track. `frame` is when the user *spoke*, not
            # when ASR finished saying so: this event is emitted after
            # transcription, so using the current frame put every user
            # subtitle 2.5-4.3 s late against its own audio -- the whole
            # utterance plus the recognition that followed it.
            self._event("user_turn", text=text, frame=int(round(speech_frame)))
            if text:
                # now that the words exist, revise the mood. _react had to
                # guess from prosody alone -- the transcript is the first
                # point at which "I've had a sad day" is distinguishable
                # from an animated hello.
                self.mood.observe(estimate_react_affect(text=text),
                                  estimate_arousal(pcm, self.sr),
                                  C.MOOD_W_USER)
                self._push_mood()
            # an async stream: _speak starts on segment 0 while
            # the model is still writing the rest
            segments = self._reply(text)

            self.state = TurnState.SPEAK
            planned = await self._speak(segments)
        except asyncio.CancelledError:
            # barge-in: the caller has already dropped this turn's audio
            # and clips, so just leave the state machine listening
            self._in_turn, self.state = False, TurnState.LISTEN
            self._event("turn_cancelled")
            raise
        finally:
            self._in_turn = False
        self.state = TurnState.LISTEN
        self._event("turn_done", n_segments=len(planned))
        return dict(text=text, segments=planned)

    # ---- REACT (plan 6.4): motion within 200 ms of the endpoint ------------
    def _react(self, pcm, text):
        """Instant reaction from the pool's prefetched bank -- never a
        network call, so the 200 ms promise survives service outages.

        The observation feeds the mood; the clip is picked from the
        observation blended *toward* the mood, so a reaction while the
        lamp is already sad is a sad reaction rather than a snap back to
        the neutral fallback affect."""
        obs = estimate_react_affect(text=text, pcm=pcm, sr=self.sr)
        # a light touch: on the mic path this is prosody only, and the
        # transcript arrives a beat later to revise it (handle_turn)
        self.mood.observe(obs, estimate_arousal(pcm, self.sr),
                          C.MOOD_W_PROSODY)
        self._push_mood()
        if self.pool is None:
            return
        v = blend_affect(obs, self.mood.vector(), C.MOOD_REACT_PRIOR)
        hit = self.pool.react_clip(v)
        if hit is None:
            return
        x, tag = hit
        self.sched.preempt(ScheduledClip(
            x=x, start_frame=self.sched.frame,
            priority=REACT_PRIORITY, tag="react"))
        self._event("react_motion", tag=tag)

    # ---- ASR ---------------------------------------------------------------
    def _guess_transcript(self):
        """Start transcribing before the turn is officially over.

        The endpointer needs T_END_MS of silence to be sure, and that
        window used to be dead time -- a measured live session spent
        700 ms proving the turn had ended and only then began the 870 ms
        of recognition. Both can happen at once. The snapshot is the
        utterance so far; the blocks that arrive after it are silence,
        which changes no transcript."""
        if not self._utterance:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return                  # driven synchronously (tests): skip
        pcm = np.concatenate(self._utterance)
        self._guess = (len(pcm), time.monotonic(),
                       asyncio.create_task(self.asr.transcribe(pcm, self.sr)))

    def _drop_guess(self):
        """Throw away a speculative transcription -- the user carried on
        talking, so it was of half a sentence."""
        if self._guess is not None:
            task = self._guess[2]
            if not task.done():
                task.cancel()
            elif not task.cancelled():
                task.exception()    # retrieved, so asyncio stays quiet
            self._guess = None

    async def _transcribe(self, pcm):
        if pcm is None or len(pcm) == 0:
            self._drop_guess()
            return ""
        t0 = time.monotonic()
        guess, self._guess = self._guess, None
        try:
            if guess is not None and guess[0] <= len(pcm):
                # started early; only trailing silence has been added since
                tr = await asyncio.wait_for(guess[2], C.TIMEOUT_ASR)
                self._event("asr_final", chars=len(tr.text),
                            ms=round(1000 * (time.monotonic() - t0)),
                            speculative=True,
                            saved_ms=round(1000 * (t0 - guess[1])))
                return tr.text.strip()
            if guess is not None:
                guess[2].cancel()
            tr = await asyncio.wait_for(self.asr.transcribe(pcm, self.sr),
                                        C.TIMEOUT_ASR)
            self._event("asr_final", chars=len(tr.text),
                        ms=round(1000 * (time.monotonic() - t0)))
            return tr.text.strip()
        except Exception as e:  # noqa: BLE001 -- timeout or backend failure
            self._event("asr_failed", error=type(e).__name__,
                        ms=round(1000 * (time.monotonic() - t0)))
            return ""

    # ---- dialogue ----------------------------------------------------------
    async def _reply(self, text):
        """Yield the reply's segments as they become available.

        An async generator rather than a list: with a streaming backend
        the first phrase is usable long before the model has finished the
        last, and `_speak` can start synthesizing it immediately. Waiting
        for the whole object cost most of a measured 0.82 s per turn.

        The failure ladder is unchanged -- an empty or overlong
        transcript, a timeout waiting for the *first* segment, or a
        backend error all yield the stall reply instead."""
        if not text or len(text) > 500:          # ASR->dialogue validator
            self._event("reply_fallback", reason="empty_or_long")
            for seg in validate_reply(SORRY_REPLY):
                yield seg
            return
        t0 = time.monotonic()
        n = 0
        try:
            async for seg in self._ask(text):
                if n == 0:
                    self._event("llm_first", ms=round(
                        1000 * (time.monotonic() - t0)))
                n += 1
                yield seg
        except asyncio.TimeoutError:
            reason = "timeout"
        except Exception as e:  # noqa: BLE001 -- any vendor failure stalls
            reason = type(e).__name__
        else:
            if n:
                self._event("llm_reply", n=n,
                            ms=round(1000 * (time.monotonic() - t0)))
                return
            reason = "no_segments"
        if n:
            # it already said something, so stalling on top would be worse
            # than stopping -- but a stream that dies halfway is a real
            # failure and used to leave no trace at all
            self._event("reply_truncated", reason=reason, spoken=n)
            return
        self._event("reply_fallback", reason=reason)
        for seg in validate_reply(STALL_REPLY):
            yield seg

    async def _ask(self, text):
        """Segments from the dialogue backend, told how the body feels.

        Keeping the mood in front of the LLM is what stops the words and
        the motion drifting apart over a long conversation -- it will not
        answer a sad lamp's question brightly.

        Streams when the backend offers `reply_stream`, otherwise calls
        `reply` and yields its list, so a backend that only knows how to
        return everything at once still works. `mood` support is decided
        by signature inspection rather than a caught TypeError, which
        would also swallow a TypeError raised inside the backend."""
        stream = getattr(self.agent, "reply_stream", None)
        fn = stream or self.agent.reply
        if self._agent_mood is None:
            try:
                self._agent_mood = "mood" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                self._agent_mood = False
        kw = {"mood": self.mood} if self._agent_mood else {}
        if stream is not None:
            # the timeout guards the *first* segment; once the lamp is
            # talking a slow tail is not a failure
            it = fn(text, **kw).__aiter__()
            first = True
            while True:
                nxt = it.__anext__()
                try:
                    seg = await (asyncio.wait_for(nxt, C.TIMEOUT_LLM + 0.5)
                                 if first else nxt)
                except StopAsyncIteration:
                    return
                first = False
                yield seg
        for seg in await asyncio.wait_for(fn(text, **kw),
                                          C.TIMEOUT_LLM + 0.5):
            yield seg

    # ---- SPEAK: synth, align, schedule -------------------------------------
    async def _speak(self, segments):
        # the reply starts as soon as it is ready; whatever is left of the
        # reaction is taken over by priority (see the constants above)
        base = self.sched.frame + 2
        planned, t_cursor, spoken = [], 0.0, []
        async for seg in _aiter(segments):
            spoken.append(seg)
            # segment ids are session-unique, not per-turn: the audio and
            # motion start events are paired by id downstream, and a
            # per-turn index silently matched turn 3's first segment
            # against turn 1's
            i = self._seg_seq
            self._seg_seq += 1
            prosody = valence_arousal(seg.affect, seg.intensity)
            tts_res = await self._synth(seg.text, prosody)
            dur = tts_res.duration_s
            parts = await self._motion_chain(seg, dur, i)
            # anchor *after* the awaits, not before: TTS and generation take
            # 200-400 ms on CPU, so a start frame chosen up front is already
            # in the past by the time the clip is submitted, and the clip
            # activates late against a plan that promised otherwise. Slipping
            # the whole reply keeps audio, motion and subtitle together.
            start = base + int(round(t_cursor / C.DT))
            if start < self.sched.frame + 1:
                base += self.sched.frame + 1 - start
                start = self.sched.frame + 1
            env = envelope_30hz(tts_res.pcm, tts_res.sample_rate)
            off = 0
            for j, x in enumerate(parts):
                self.sched.submit(ScheduledClip(
                    x=x, start_frame=start + off, priority=SPEAK_PRIORITY,
                    tag=f"speak:{i}" if j == 0 else f"speak:{i}+",
                    envelope=env[off:off + len(x)]))
                off += len(x)
            # affect/intensity/cfg are what actually conditioned the prior
            # for this segment; without them the log says what the lamp
            # said but never why it moved that way
            self._event("segment_planned", seg=i, start_frame=start,
                        seconds=round(dur, 3), text=seg.text,
                        n_clips=len(parts), affect=named(seg.affect),
                        intensity=round(seg.intensity, 3),
                        cfg=round(intensity_to_cfg(seg.intensity), 3),
                        prosody=prosody.as_dict())
            self._said.append((self.sched.frame * C.DT, seg.text))
            pcm = tts_res.pcm
            if self.audio is not None:
                pcm = resample(pcm, tts_res.sample_rate, self.audio.sr)
                self.audio.play(pcm)
                # paired with the scheduler's `motion_seg_start` by
                # eval/metrics.sync_errors (target p95 < 100 ms)
                self._event("audio_seg_start", seg=i)
            if self.rec is not None:
                # on the motion's frame, not the queueing moment: audio,
                # clips and subtitle then share one timeline in the replay
                self.rec.audio(pcm, self.audio.sr if self.audio is not None
                               else tts_res.sample_rate, start, who="lamp")
            planned.append(dict(seg=i, text=seg.text, seconds=dur,
                                start_frame=start, has_motion=bool(parts)))
            t_cursor += dur
        if spoken:
            # what the lamp just said, as a whole, becomes what it still
            # feels -- and through the mood, what it performs while the
            # user thinks about their answer
            vec, intensity = mean_affect(
                spoken, [p["seconds"] for p in planned])
            self.mood.observe(vec, intensity, C.MOOD_W_LAMP)
            self._push_mood()
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
        """Scheduler callback for empty moments: pop a prefetched clip
        from the pool (lock-and-go -- this runs inside the 30 Hz tick)
        so the lamp keeps performing between phrases and turns. Empty
        pool -> None -> procedural breathing. Lowest priority --
        anything real preempts it.

        Clips played while a turn is in flight are tagged `listen` rather
        than `ambient`: same mood-driven motion, but the report can then
        separate "thinking with the user waiting" (the gap that matters,
        between the reaction and the reply) from "idling between turns"
        (nobody is waiting)."""
        if self.pool is None:
            return None
        self._settle_mood()      # the moment the next ambient clip is chosen
        hit = self.pool.pop_ambient()
        if hit is None:
            return None
        x, tag = hit
        if self._in_turn:
            tag = "listen" + tag[len("ambient"):]
        return ScheduledClip(x=x, start_frame=frame,
                             priority=AMBIENT_PRIORITY, tag=tag)

    async def _synth(self, text, prosody=None):
        try:
            return await asyncio.wait_for(self.tts.synth(text, prosody),
                                          C.TIMEOUT_TTS + 0.5)
        except Exception as e:  # noqa: BLE001
            self._event("tts_fallback", error=type(e).__name__)
            return await self._silent_tts.synth(text, prosody)

    async def _motion_for(self, req):
        """Live generation (remote service or local engine) with a
        timeout; on failure the nearest in-memory pool clip
        (`pool_forced`); else None (the scheduler breathes). Every
        request logs a `motion_source` event so sessions can report
        their provenance split (eval/metrics.motion_sources)."""
        t0 = time.monotonic()
        # the exact conditioning handed to the flow-matching prior
        cond = dict(affect=named(req.affect), cfg=round(req.cfg, 3),
                    seconds=None if req.seconds is None
                    else round(req.seconds, 3), seed=req.seed)
        if self.engine is not None:
            try:
                x = await asyncio.wait_for(
                    asyncio.to_thread(self.engine.clip, req),
                    C.TIMEOUT_MOTION)
                self._event("motion_source", tag=req.tag, source="engine",
                            ms=round(1000 * (time.monotonic() - t0)), **cond)
                return x
            except Exception as e:  # noqa: BLE001 -- timeout, breaker,
                #                        transport, or torch error
                self._event("motion_fallback", error=type(e).__name__,
                            ms=round(1000 * (time.monotonic() - t0)))
        if self.pool is not None:
            x = self.pool.nearest(req.affect)
            if x is not None:
                self._event("motion_source", tag=req.tag,
                            source="pool_forced",
                            ms=round(1000 * (time.monotonic() - t0)), **cond)
                return x
        self._event("motion_source", tag=req.tag, source="none",
                    ms=round(1000 * (time.monotonic() - t0)), **cond)
        return None

    # ---- barge-in -----------------------------------------------------------
    def abort(self):
        """Drop everything the current turn has produced: queued speech
        audio, and its motion clips. The reaction and ambient stay -- the
        lamp should keep moving while it changes its mind."""
        if self.audio is not None:
            self.audio.stop_playback()
        self.sched.queue = [c for c in self.sched.queue
                            if not c.tag.startswith("speak")]
        if self.sched.active is not None \
                and self.sched.active.tag.startswith("speak"):
            self.sched.active = None

    def dispatch(self, pcm, turn=None):
        """Start a turn for `pcm`, barging in on `turn` if one is still
        running. The single place that policy lives, so the live loop and
        the simulation driver cannot drift apart on it."""
        if turn is not None:
            if not turn.done():
                self._event("barge_in")
                turn.cancel()
                self.abort()
            elif not turn.cancelled():
                turn.exception()     # retrieved, so asyncio stays quiet
        return asyncio.create_task(self.handle_turn(pcm=pcm))

    # ---- hardware loop (Pi): mic ring -> turns -----------------------------
    async def run(self, ring_reader, stop=None):
        """Consume mic blocks from the ring, dispatching a turn per
        endpoint.

        The turn runs as a *task*, never awaited inline. Awaiting it here
        froze this reader for the whole 3 s of a turn; the ring kept
        filling, and when the turn returned the backlog drained in
        milliseconds -- so an utterance spoken while the lamp was
        thinking was endpointed several seconds late, its reply queued
        behind the one already playing, and the lamp monologued for nine
        seconds straight. Draining every block keeps event timestamps
        honest and replies from stacking.

        A new endpoint during a turn is a barge-in: the in-flight turn is
        cancelled and the new utterance takes over, which is what a
        person does when interrupted. Safe because the mic is muted while
        the lamp is audible, so the only thing that can interrupt it is
        the user."""
        turn = None
        while stop is None or not stop.is_set():
            for block in ring_reader.read_blocks():
                pcm = self.feed_block(block)
                if pcm is not None:
                    turn = self.dispatch(pcm, turn)
            if turn is not None and turn.done():
                try:
                    await turn                  # surface a real failure
                except asyncio.CancelledError:
                    pass
                turn = None
            await asyncio.sleep(C.AUDIO_BLOCK_MS / 1000.0)
        if turn is not None:
            turn.cancel()
