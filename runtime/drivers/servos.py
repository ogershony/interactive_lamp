"""
Feetech servo bus driver: rad -> ticks with per-joint sign/offset
calibration, joint-limit clamp as the last line of defense, and a mock
bus so everything above P0 runs and tests without hardware.

The STS3215 is a 4096-ticks-per-revolution bus servo; sign and offset
calibration (SERVO_SIGNS / SERVO_OFFSETS_TICKS in runtime/config.py)
happens against the physical lamp at P0 and defaults to identity here.
"""

import numpy as np

import runtime.config as C

_TICKS_PER_RAD = C.SERVO_TICKS_PER_REV / (2.0 * np.pi)


def rad_to_ticks(q):
    """(5,) rad -> (5,) int ticks, clamped to joint limits (with margin)
    first: the driver never trusts its caller."""
    q = np.clip(np.asarray(q, np.float64),
                C.JOINT_LO + C.LIMIT_MARGIN, C.JOINT_HI - C.LIMIT_MARGIN)
    ticks = (C.SERVO_CENTER_TICKS + C.SERVO_OFFSETS_TICKS
             + C.SERVO_SIGNS * q * _TICKS_PER_RAD)
    return np.round(ticks).astype(int)


def ticks_to_rad(ticks):
    return ((np.asarray(ticks, np.float64) - C.SERVO_CENTER_TICKS
             - C.SERVO_OFFSETS_TICKS) / (C.SERVO_SIGNS * _TICKS_PER_RAD))


class MockServoBus:
    """Records every write; stands in for hardware in tests, eval
    replays, and desk development."""

    def __init__(self):
        self.writes = []         # list of (5,) tick arrays

    def write(self, q):
        self.writes.append(rad_to_ticks(q))

    @property
    def last(self):
        return self.writes[-1] if self.writes else None

    def close(self):
        pass


class FeetechBus:
    """Real bus via the Feetech SDK (scservo_sdk), sync-write so all
    five servos latch the same tick. Import is deferred so this module
    loads on boxes without the SDK."""

    def __init__(self, port="/dev/ttyACM0", baud=1_000_000, ids=None):
        import scservo_sdk as scs         # hardware boxes only
        self.scs = scs
        self.ids = ids or C.SERVO_IDS
        self.port = scs.PortHandler(port)
        if not self.port.openPort() or not self.port.setBaudRate(baud):
            raise RuntimeError(f"cannot open Feetech bus on {port}")
        self.packet = scs.PacketHandler(0)   # STS protocol
        # goal-position register for STS-series is 42, 2 bytes
        self.sync = scs.GroupSyncWrite(self.port, self.packet, 42, 2)

    def write(self, q):
        ticks = rad_to_ticks(q)
        self.sync.clearParam()
        for sid, t in zip(self.ids, ticks):
            t = int(np.clip(t, 0, C.SERVO_TICKS_PER_REV - 1))
            self.sync.addParam(sid, [t & 0xFF, (t >> 8) & 0xFF])
        self.sync.txPacket()

    def close(self):
        self.port.closePort()
