from dataclasses import dataclass
from typing import Tuple

from .surfaces import SurfaceType


@dataclass(frozen=True)
class FaceDescriptor:
    index: int
    surface_type: SurfaceType
    area: float









