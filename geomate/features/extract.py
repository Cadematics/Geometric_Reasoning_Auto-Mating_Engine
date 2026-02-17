from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Dir

from geomate.io.step_io import load_step_shapes
from geomate.io.topo import iter_solids, iter_faces
from geomate.io.ids import PartId, FaceId


def _pnt_to_list(p: gp_Pnt) -> List[float]:
    return [float(p.X()), float(p.Y()), float(p.Z())]


def _dir_to_list(d: gp_Dir) -> List[float]:
    return [float(d.X()), float(d.Y()), float(d.Z())]


def _props_area_centroid(face) -> Tuple[float, List[float]]:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = float(props.Mass())
    c = props.CentreOfMass()
    return area, _pnt_to_list(c)


def extract_features_for_solid(part_id: PartId, solid) -> Dict[str, Any]:
    """
    Extract plane and cylinder surface features from all faces of the solid.
    """
    features: List[Dict[str, Any]] = []

    for face in iter_faces(solid):
        face_id = FaceId.from_face(part_id, face)
        surf = BRepAdaptor_Surface(face, True)
        stype = surf.GetType()

        area, centroid = _props_area_centroid(face)

        if stype == GeomAbs_Plane:
            pln = surf.Plane()
            origin = pln.Location()
            normal = pln.Axis().Direction()

            features.append({
                "feature_id": f"{face_id}_plane",
                "type": "plane",
                "face_id": str(face_id),
                "tags": [],
                "geom": {
                    "origin": _pnt_to_list(origin),
                    "normal": _dir_to_list(normal)
                },
                "meta": {
                    "area": area,
                    "centroid": centroid
                }
            })

        elif stype == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            ax = cyl.Axis()
            axis_point = ax.Location()
            axis_dir = ax.Direction()
            radius = float(cyl.Radius())

            features.append({
                "feature_id": f"{face_id}_cyl",
                "type": "cylinder",
                "face_id": str(face_id),
                "tags": [],
                "geom": {
                    "axis_point": _pnt_to_list(axis_point),
                    "axis_dir": _dir_to_list(axis_dir),
                    "radius": radius
                },
                "meta": {
                    "area": area,
                    "centroid": centroid
                }
            })

        else:
            # ignore other surface types for now (cones, bspline, etc.)
            continue

    return {
        "schema_version": "0.1",
        "part_id": str(part_id),
        "source": {},
        "units": "mm",
        "features": features
    }


def extract_all(step_path: Path, out_dir: Path) -> List[Path]:
    step_path = step_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = load_step_shapes(step_path)

    written: List[Path] = []
    for root_i, root_shape in enumerate(shapes):
        for solid_i, solid in enumerate(iter_solids(root_shape)):
            part_id = PartId.from_step(step_path, root_i, solid_i)
            payload = extract_features_for_solid(part_id, solid)
            payload["source"] = {"step_path": str(step_path), "solid_index": solid_i}

            part_folder = out_dir / "parts" / str(part_id)
            part_folder.mkdir(parents=True, exist_ok=True)

            out_file = part_folder / "PartFeatures.json"
            out_file.write_text(json.dumps(payload, indent=2))
            written.append(out_file)

    return written
