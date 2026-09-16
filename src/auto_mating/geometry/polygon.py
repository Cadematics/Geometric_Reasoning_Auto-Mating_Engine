from dataclasses import dataclass
from typing import Tuple

from .planar import Point2D


@dataclass(frozen=True)
class Polygon2D:
    """
    A simple 2D polygon represented by an ordered sequence
    of boundary points.
    """

    points: Tuple[Point2D, ...]