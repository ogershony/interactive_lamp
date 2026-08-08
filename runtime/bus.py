"""
Typed in-process pub/sub. Subscribers register per event *type* and get
an asyncio.Queue; publishers can be any thread (`publish` marshals onto
the loop with call_soon_threadsafe when a loop is attached, or delivers
synchronously in loop-less/test mode).

Deliberately tiny: no topics-as-strings, no wildcards. The event's class
is the topic, which keeps producers and consumers honest about payload
shapes.
"""

import asyncio


class Bus:
    def __init__(self, loop=None):
        self.loop = loop
        self._subs = {}          # type -> list[asyncio.Queue]

    def subscribe(self, etype, maxsize=64):
        q = asyncio.Queue(maxsize=maxsize)
        self._subs.setdefault(etype, []).append(q)
        return q

    def publish(self, event):
        for q in self._subs.get(type(event), []):
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self._put, q, event)
            else:
                self._put(q, event)

    @staticmethod
    def _put(q, event):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # drop the oldest: a slow consumer should see fresh events,
            # not a backlog of stale ones
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(event)
