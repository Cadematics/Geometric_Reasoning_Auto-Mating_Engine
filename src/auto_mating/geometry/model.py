from dataclasses import dataclass

from .surfaces import SurfaceType


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PlaneGeometry:
    origin: Point3D
    axis_direction: Vector3D


@dataclass(frozen=True)
class CylinderGeometry:
    axis_origin: Point3D
    axis_direction: Vector3D
    radius: float

    

@dataclass(frozen=True)
class FaceDescriptor:
    index: int
    surface_type: SurfaceType
    area: float
    centroid: Point3D
    normal: Vector3D