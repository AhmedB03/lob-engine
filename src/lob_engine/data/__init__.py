"""Data layer: synthetic feed generation, canonical schema, memory optimization."""

from lob_engine.data.generator import LOBConfig, SyntheticLOBGenerator, ar1_process
from lob_engine.data.schema import frame_memory_mb, optimize_memory

__all__ = [
    "LOBConfig",
    "SyntheticLOBGenerator",
    "ar1_process",
    "optimize_memory",
    "frame_memory_mb",
]
