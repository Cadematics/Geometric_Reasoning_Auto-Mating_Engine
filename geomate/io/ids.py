from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface








def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> str:
    """
    Stable-enough fingerprint for this portfolio project.
    If you need absolute stability across copies, hash file bytes instead.
    """
    st = path.stat()
    return _sha1(f"{path.name}|{st.st_size}|{int(st.st_mtime)}")


def _round3(x: float) -> float:
    return float(f"{x:.3f}")


def stable_face_key(face) -> str:
    """
    Create a geometric fingerprint stable enough for the same STEP:
      - surface type
      - key geom params (plane location+normal, cylinder axis+radius)
      - area + centroid (rounded)
    """
    surf = BRepAdaptor_Surface(face, True)
    stype = surf.GetType()

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area_r = _round3(props.Mass())
    c = props.CentreOfMass()
    centroid = (_round3(c.X()), _round3(c.Y()), _round3(c.Z()))

    if stype == GeomAbs_Plane:
        pln = surf.Plane()
        p = pln.Location()
        n = pln.Axis().Direction()
        key = (
            "plane",
            (_round3(p.X()), _round3(p.Y()), _round3(p.Z())),
            (_round3(n.X()), _round3(n.Y()), _round3(n.Z())),
            area_r,
            centroid,
        )
    elif stype == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax = cyl.Axis()
        p = ax.Location()
        d = ax.Direction()
        r = _round3(cyl.Radius())
        key = (
            "cyl",
            (_round3(p.X()), _round3(p.Y()), _round3(p.Z())),
            (_round3(d.X()), _round3(d.Y()), _round3(d.Z())),
            r,
            area_r,
            centroid,
        )
    else:
        # fallback: surface type id + area + centroid
        key = ("other", int(stype), area_r, centroid)

    return _sha1(str(key))


@dataclass(frozen=True)
class PartId:
    value: str

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_step(step_path: Path, root_index: int, solid_index: int) -> "PartId":
        fp = _file_fingerprint(step_path)
        return PartId(f"part_{fp[:10]}_r{root_index}_s{solid_index}")


@dataclass(frozen=True)
class FaceId:
    value: str

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_face(part_id: PartId, face) -> "FaceId":
        key = stable_face_key(face)
        return FaceId(f"{part_id}_face_{key[:12]}")
