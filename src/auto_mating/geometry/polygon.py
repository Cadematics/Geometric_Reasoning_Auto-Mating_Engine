from dataclasses import dataclass
from typing import Tuple

from .planar import Point2D


@dataclass(frozen=True)
class Polygon2D:
    points: Tuple[Point2D, ...]

    def area(self) -> float:
        """Return the absolute area of the polygon."""
        if len(self.points) < 3:
            return 0.0

        total = 0.0

        for i, current in enumerate(self.points):
            next_point = self.points[(i + 1) % len(self.points)]

            total += (
                current.x * next_point.y
                - next_point.x * current.y
            )

        return abs(total) * 0.5


def cross_2d(
    a: Point2D,
    b: Point2D,
    c: Point2D,
) -> float:
    """
    2D cross product of AB and AC.

    Positive:
        C is to the left of AB

    Negative:
        C is to the right of AB

    Zero:
        A, B, C are collinear
    """
    return (
        (b.x - a.x) * (c.y - a.y)
        - (b.y - a.y) * (c.x - a.x)
    )


def polygon_orientation(polygon: Polygon2D) -> float:
    """
    Return signed polygon area.

    Positive -> counter-clockwise
    Negative -> clockwise
    """
    if len(polygon.points) < 3:
        return 0.0

    total = 0.0

    for i, current in enumerate(polygon.points):
        next_point = polygon.points[(i + 1) % len(polygon.points)]

        total += (
            current.x * next_point.y
            - next_point.x * current.y
        )

    return 0.5 * total


def _inside(
    point: Point2D,
    edge_start: Point2D,
    edge_end: Point2D,
    orientation: float,
    tolerance: float,
) -> bool:
    value = cross_2d(edge_start, edge_end, point)

    if orientation > 0.0:
        return value >= -tolerance

    return value <= tolerance


def _line_intersection(
    p1: Point2D,
    p2: Point2D,
    q1: Point2D,
    q2: Point2D,
    tolerance: float = 1.0e-12,
) -> Point2D:
    """
    Intersection of the infinite lines P1-P2 and Q1-Q2.
    """

    dx1 = p2.x - p1.x
    dy1 = p2.y - p1.y

    dx2 = q2.x - q1.x
    dy2 = q2.y - q1.y

    denominator = dx1 * dy2 - dy1 * dx2

    if abs(denominator) <= tolerance:
        # Parallel lines. In normal polygon clipping this should
        # only occur at a boundary transition where the exact
        # intersection is not needed.
        return p2

    t = (
        (q1.x - p1.x) * dy2
        - (q1.y - p1.y) * dx2
    ) / denominator

    return Point2D(
        x=p1.x + t * dx1,
        y=p1.y + t * dy1,
    )


def intersect_convex_polygons(
    subject: Polygon2D,
    clip: Polygon2D,
    tolerance: float = 1.0e-9,
) -> Polygon2D:
    """
    Calculate the intersection of two convex 2D polygons.

    Uses the Sutherland-Hodgman polygon clipping algorithm.

    Both polygons must be convex.
    """

    if len(subject.points) < 3 or len(clip.points) < 3:
        return Polygon2D(points=())

    output = list(subject.points)

    orientation = polygon_orientation(clip)

    if abs(orientation) <= tolerance:
        return Polygon2D(points=())

    for i, clip_start in enumerate(clip.points):
        clip_end = clip.points[(i + 1) % len(clip.points)]

        if not output:
            break

        input_points = output
        output = []

        previous = input_points[-1]

        previous_inside = _inside(
            previous,
            clip_start,
            clip_end,
            orientation,
            tolerance,
        )

        for current in input_points:

            current_inside = _inside(
                current,
                clip_start,
                clip_end,
                orientation,
                tolerance,
            )

            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(
                            previous,
                            current,
                            clip_start,
                            clip_end,
                        )
                    )

                output.append(current)

            elif previous_inside:
                output.append(
                    _line_intersection(
                        previous,
                        current,
                        clip_start,
                        clip_end,
                    )
                )

            previous = current
            previous_inside = current_inside

    return Polygon2D(
        points=tuple(output)
    )