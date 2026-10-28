"""Structured latent BO with explicit partial observations and EP inference."""

from .records import DesignRecord
from .selector import PartialLatentBO, PartialBOSettings

__all__ = ["DesignRecord", "PartialLatentBO", "PartialBOSettings"]
