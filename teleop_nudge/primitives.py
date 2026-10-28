"""Signal conditioning every nudge can reuse. All stateful, all reset at episode boundaries."""

from __future__ import annotations


class Hysteresis:
    """Boolean with separate on and off thresholds."""

    def __init__(self, on: float, off: float):
        self.on, self.off = on, off
        self.state = False

    def reset(self) -> None:
        self.state = False

    def update(self, x: float) -> bool:
        """Returns the new state for input x."""
        if self.state and x < self.off:
            self.state = False
        elif not self.state and x > self.on:
            self.state = True
        return self.state


class Refractory:
    """Allows an event at most once per period."""

    def __init__(self, period_s: float):
        self.period_s = period_s
        self._last_ns = None

    def reset(self) -> None:
        self._last_ns = None

    def allow(self, t_ns: int) -> bool:
        """True and arms the timer if the period has passed since the last allowed event."""
        if self._last_ns is not None and (t_ns - self._last_ns) / 1e9 < self.period_s:
            return False
        self._last_ns = t_ns
        return True


class RisingEdge:
    """True only on the tick a boolean turns on."""

    def __init__(self):
        self._prev = False

    def reset(self) -> None:
        self._prev = False

    def update(self, x: bool) -> bool:
        """Returns True on a False to True transition."""
        edge = x and not self._prev
        self._prev = x
        return edge


class Ramp:
    """Maps x linearly from [lo, hi] onto [0, 1], clamped."""

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def __call__(self, x: float) -> float:
        if self.hi <= self.lo:
            return 1.0 if x >= self.hi else 0.0
        return min(1.0, max(0.0, (x - self.lo) / (self.hi - self.lo)))


class Debounce:
    """Boolean that must hold for min_on_s before it is reported."""

    def __init__(self, min_on_s: float):
        self.min_on_s = min_on_s
        self._since_ns = None

    def reset(self) -> None:
        self._since_ns = None

    def update(self, x: bool, t_ns: int) -> bool:
        """Returns True once x has been True for min_on_s."""
        if not x:
            self._since_ns = None
            return False
        if self._since_ns is None:
            self._since_ns = t_ns
        return (t_ns - self._since_ns) / 1e9 >= self.min_on_s


class Baseline:
    """Slow tracker of a signal's resting level, frozen while `hold` is set; returns the deviation."""

    def __init__(self, tau_s: float):
        self.tau_s = tau_s
        self._base = None
        self._t_ns = None

    def reset(self) -> None:
        self._base = None
        self._t_ns = None

    def update(self, x: float, t_ns: int, hold: bool) -> float:
        """Returns x minus the tracked baseline; the baseline follows x only while hold is False."""
        if self._base is None:
            self._base, self._t_ns = x, t_ns
            return 0.0
        if not hold:
            dt = max(0.0, (t_ns - self._t_ns) / 1e9)
            a = min(1.0, dt / self.tau_s) if self.tau_s > 0 else 1.0
            self._base += a * (x - self._base)
        self._t_ns = t_ns
        return x - self._base
