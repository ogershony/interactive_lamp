"""
The simulated session driver: a virtual clock, a fake speaker, and the
loop that drives the real Conversation from a SimMic.

Time. The live runtime has two clocks -- a 30 Hz scheduler thread on the
wall clock, and an asyncio loop reading the mic ring. The simulator has
one: the mic. Every 10 ms block consumed advances simulated time by
10 ms, and the scheduler is ticked until its frame clock catches up. So
the audio and the motion cannot drift apart by construction, and the
whole thing is single-threaded and reproducible.

Speed. Scripted silence fast-forwards -- there is nothing to wait for,
and a twelve-second pause should not cost twelve seconds to test. But
while a turn is in flight, simulated time is paced to *wall* time, so
the real cost of Whisper, the LLM and the motion engine lands in the
timeline at full size. A turn that takes three seconds of compute
consumes three seconds of script, exactly as it would live. This is the
one property that makes the reported latencies and motion gaps mean
something; `--realtime` paces everything that way at the cost of running
as slowly as a real conversation.

Fast-forwarding does break one thing honestly: the MotionPool's refill
thread runs on the wall clock, so during a fast-forwarded silence the
scheduler can consume ambient clips faster than any real box would.
`pump()` tops the pool up synchronously in that window so the gap
analysis measures the *policy* rather than this machine's clip
throughput -- which is reported separately, from the engine's own
timings.
"""

import asyncio
import time

import numpy as np

import runtime.config as C


class SimClock:
    """The session's one source of time, created before anything that
    reads it. Everything that would otherwise call time.monotonic() --
    the recorder's event stamps, the half-duplex gate -- takes this
    instead, so events, frames and audio share a single timeline even
    while silence is being fast-forwarded."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


class SimAudio:
    """Stands in for AudioIO: swallows playback, remembers how long it
    would have taken, and reproduces the half-duplex gate.

    It also models `speaker_lag_s` -- how long after the playback queue
    empties the sound is still coming out of a real speaker. That is not
    a detail: a live session with the gate opening 150 ms after the queue
    emptied had the lamp transcribe the tail of its own sentences as the
    user's next turn. `bleed()` says whether the lamp is *audible* right
    now, which the mic must be told about, and setting `speaker_lag_s`
    above the gate reproduces the bug on demand."""

    def __init__(self, clock, sr=None, speaker_lag_s=None):
        self.sr = int(sr or C.AUDIO_SR)
        self.clock = clock             # () -> simulated seconds
        self.speaker_lag_s = C.OUTPUT_LAG_MS / 1000.0 \
            if speaker_lag_s is None else float(speaker_lag_s)
        self.busy_until = 0.0
        self.spoken_s = 0.0
        self.segments = []             # (start_s, seconds)

    def play(self, pcm):
        seconds = len(pcm) / self.sr
        start = max(self.clock(), self.busy_until)
        self.busy_until = start + seconds
        self.spoken_s += seconds
        self.segments.append((start, seconds))

    def stop_playback(self):
        self.busy_until = self.clock()

    @property
    def is_playing(self):
        """Samples still queued -- not the same as audible."""
        return self.clock() < self.busy_until

    @property
    def is_gated(self):
        """Is the mic muted? What the runtime believes."""
        return self.clock() < self.busy_until \
            + (C.OUTPUT_LAG_MS + C.HALF_DUPLEX_TAIL_MS) / 1000.0

    def bleed(self):
        """Is the lamp still physically audible? What the room knows.
        When this outlasts `is_gated`, the mic records the lamp."""
        return self.clock() < self.busy_until + self.speaker_lag_s


class SimDriver:
    def __init__(self, convo, sched, mic, audio, clock, pool=None,
                 realtime=False, max_seconds=None, on_tick=None):
        self.convo = convo
        self.sched = sched
        self.mic = mic
        self.audio = audio
        self.clock = clock
        self.pool = pool
        self.realtime = realtime
        self.max_seconds = max_seconds
        self.on_tick = on_tick
        self.turns = []               # summaries returned by handle_turn
        self.muted_blocks = 0         # blocks the gate zeroed
        self.bleed_blocks = 0         # blocks carrying the lamp's own voice
        self._voice_phase = 0.0

    def _lamp_voice(self):
        """A speech-like block standing in for the lamp's own audio
        arriving back at its microphone. Only has to be voiced enough for
        the VAD -- what matters is whether it forms a turn, not what a
        transcript of it would say."""
        n = self.mic.block
        t = (self._voice_phase + np.arange(n)) / C.AUDIO_SR
        self._voice_phase += n
        wave = sum(np.sin(2 * np.pi * f * t) for f in (150, 300, 450, 600))
        wave *= 0.2 * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))
        return (np.clip(wave, -1, 1) * 32767).astype(np.int16)

    @property
    def t(self):
        return self.clock.t

    def now(self):
        return self.clock.t

    def _tick_to(self, t):
        """Advance the 30 Hz control loop up to simulated time `t`."""
        while (self.sched.frame + 1) * C.DT <= t:
            self.sched.tick()
            if self.on_tick is not None:
                self.on_tick(self.sched)

    def pump(self, budget=2):
        """Top the motion pool up synchronously (see the module note)."""
        if self.pool is None:
            return
        for _ in range(budget):
            if not self.pool._refill_once():
                return

    def _talking(self):
        """Is the lamp still mid-reply? Ambient motion never stops, so the
        session cannot end on 'nothing is playing' -- it ends when the
        last *speech* clip and its audio are done."""
        if self.audio.is_playing:
            return True
        active = self.sched.active
        if active is not None and active.tag.startswith("speak"):
            return True
        return any(c.tag.startswith("speak") for c in self.sched.queue)

    async def run(self, tail_s=2.0):
        """Drive the script to its end, then keep going until the lamp
        has finished replying, plus `tail_s` of ambient so the recording
        shows what it does when the conversation stops."""
        blocks = self.mic.blocks()
        block_s = C.AUDIO_BLOCK_MS / 1000.0
        scripted = self.mic.scripted_blocks
        limit = self.max_seconds or (self.mic.script.seconds + 120.0)
        zeros = np.zeros(self.mic.block, np.int16)
        turn = None
        anchor = None                 # (sim_t, wall_t) while a turn runs
        tail_until = None
        i = 0
        while self.t <= limit:
            if turn is not None and turn.done():
                try:
                    self.turns.append(await turn)
                except asyncio.CancelledError:
                    pass                       # barged in on
                turn, anchor = None, None
            if i >= scripted and turn is None and not self._talking():
                if tail_until is None:
                    tail_until = self.t + tail_s
                elif self.t >= tail_until:
                    break
            else:
                tail_until = None

            block = next(blocks)
            i += 1
            # The room first: while the lamp is audible, that is what the
            # microphone picks up, whatever the user is doing.
            if self.audio.bleed():
                block = self._lamp_voice()
                self.bleed_blocks += 1
            # ...then the gate the runtime believes in. When the gate is
            # shorter than the bleed, the difference is recorded as the
            # user -- the self-hearing bug, reproduced.
            if self.audio.is_gated:
                block = zeros
                self.muted_blocks += 1
            self.clock.advance(block_s)

            pcm = self.convo.feed_block(block)
            self._tick_to(self.t)
            if pcm is not None:
                # same dispatch policy as the live loop, barge-in included
                turn = self.convo.dispatch(pcm, turn)
                anchor = (self.t, time.monotonic())

            if turn is not None or self.realtime:
                # pace simulated time to wall time so real compute lands
                # in the timeline at its real size
                if anchor is None:
                    anchor = (self.t, time.monotonic())
                lag = (self.t - anchor[0]) - (time.monotonic() - anchor[1])
                await asyncio.sleep(max(0.0, lag))
            else:
                self.pump()
                if i % 16 == 0:
                    await asyncio.sleep(0)   # let the loop breathe
        if turn is not None:
            try:
                self.turns.append(await turn)   # hit the time limit mid-turn
            except asyncio.CancelledError:
                pass
        return self.turns
