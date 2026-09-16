from auto_mating.geometry.planar import Point2D
from auto_mating.geometry.polygon import (
    Polygon2D,
    intersect_convex_polygons,
)


def rectangle(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Polygon2D:
    return Polygon2D(
        points=(
            Point2D(xmin, ymin),
            Point2D(xmax, ymin),
            Point2D(xmax, ymax),
            Point2D(xmin, ymax),
        )
    )


def test_polygon_area():
    polygon = rectangle(0.0, 0.0, 10.0, 5.0)

    assert polygon.area() == 50.0


def test_overlapping_rectangles():
    a = rectangle(0.0, 0.0, 10.0, 10.0)
    b = rectangle(5.0, 2.0, 15.0, 8.0)

    intersection = intersect_convex_polygons(a, b)

    assert intersection.area() == 30.0


def test_no_overlap():
    a = rectangle(0.0, 0.0, 10.0, 10.0)
    b = rectangle(20.0, 20.0, 30.0, 30.0)

    intersection = intersect_convex_polygons(a, b)

    assert intersection.area() == 0.0


def test_contained_rectangle():
    a = rectangle(0.0, 0.0, 100.0, 50.0)
    b = rectangle(20.0, 5.0, 80.0, 45.0)

    intersection = intersect_convex_polygons(a, b)

    assert abs(intersection.area() - 2400.0) < 1.0e-9