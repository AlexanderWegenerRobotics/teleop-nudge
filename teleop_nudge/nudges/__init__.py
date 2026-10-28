"""Nudge plugins. A nudge is one class in one file, registered by name; the config switches it on."""

from .base import Nudge, REGISTRY, register
from . import contact

__all__ = ["Nudge", "REGISTRY", "register"]
