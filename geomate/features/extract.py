# geomate/features/extract.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps

from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepTools import breptools

from OCC.Core.gp import gp_Pnt, gp_Dir

from geomate.io.step_io import load_step_shapes
from geomate.io.topo import iter_solids, iter_faces
from geomate.io.ids import PartId, FaceId


def _pnt_to_list(p) -> List[float]:
    return [float(p.X()), float(p.Y()), float(p.Z())]


def _dir_to_list(d) -> List[float]:
    return [float(d.X()), float(d.Y()), float(d.Z())]


def _props_area_centroid(face) -> Tuple[float, List[float]]:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = float(props.Mass())
    c = props.CentreOfMass()
    return area, _pnt_to_list(c)


def _canonical_dir(d: gp_Dir) -> gp_Dir:
    """
    Deterministic sign choice: make the dominant component positive.
    This helps matching so axes/normals don't randomly flip between parts.
    """
    x, y, z = d.X(), d.Y(), d.Z()
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        if x < 0:
            return gp_Dir(-x, -y, -z)
    elif ay >= ax and ay >= az:
        if y < 0:
            return gp_Dir(-x, -y, -z)
    else:
        if z < 0:
            return gp_Dir(-x, -y, -z)
    return d


def _solid_state(solid, p: gp_Pnt, tol: float = 1e-6):
    clf = BRepClass3d_SolidClassifier()
    clf.Load(solid)
    clf.Perform(p, tol)
    return clf.State()


def _is_in_or_on(solid, p: gp_Pnt, tol: float = 1e-6) -> bool:
    st = _solid_state(solid, p, tol)
    return st in (TopAbs_IN, TopAbs_ON)


def classify_cylinder_hole_vs_shaft(solid, face, tol: float = 1e-6) -> str:
    """
    Robust classification using face UV bounds + surface normal:
      - sample a few interior (u,v) points
      - compute normal at each
      - test +normal and -normal points with solid classifier
      - if +n is inside and -n is outside => HOLE
      - if -n is inside and +n is outside => SHAFT
    Mirrors the working Strategy A from your test output.
    """
    surf_adapt = BRepAdaptor_Surface(face, True)
    if surf_adapt.GetType() != GeomAbs_Cylinder:
        return "unknown"

    u1, u2, v1, v2 = breptools.UVBounds(face)

    uv_samples = [
        (0.5 * (u1 + u2), 0.5 * (v1 + v2)),
        (u1 + 0.33 * (u2 - u1), v1 + 0.33 * (v2 - v1)),
        (u1 + 0.67 * (u2 - u1), v1 + 0.67 * (v2 - v1)),
    ]

    r = float(surf_adapt.Cylinder().Radius())
    eps_list = [
        max(0.05, r * 0.02),
        max(0.2, r * 0.05),
        max(0.5, r * 0.10),
        1.0,
        2.0,
    ]

    for (u, v) in uv_samples:
        props = BRepLProp_SLProps(surf_adapt, u, v, 1, tol)
        if not props.IsNormalDefined():
            continue

        p = props.Value()
        n = props.Normal()

        for eps in eps_list:
            p_plus = gp_Pnt(p.X() + n.X() * eps, p.Y() + n.Y() * eps, p.Z() + n.Z() * eps)
            p_minus = gp_Pnt(p.X() - n.X() * eps, p.Y() - n.Y() * eps, p.Z() - n.Z() * eps)

            plus_in = _is_in_or_on(solid, p_plus, tol)
            minus_in = _is_in_or_on(solid, p_minus, tol)

            if plus_in and not minus_in:
                return "hole"
            if minus_in and not plus_in:
                return "shaft"

    return "unknown"


def extract_features_for_solid(part_id: PartId, solid, step_path: Path, solid_index: int) -> Dict[str, Any]:
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
            normal = _canonical_dir(pln.Axis().Direction())

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
            axis_dir = _canonical_dir(ax.Direction())
            radius = float(cyl.Radius())

            kind = classify_cylinder_hole_vs_shaft(solid, face)
            tags = [kind] if kind != "unknown" else []

            features.append({
                "feature_id": f"{face_id}_cyl",
                "type": "cylinder",
                "face_id": str(face_id),
                "tags": tags,
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
        "source": {"step_path": str(step_path), "solid_index": int(solid_index)},
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

            payload = extract_features_for_solid(part_id, solid, step_path, solid_i)

            # console summary
            n_plane = sum(1 for f in payload["features"] if f["type"] == "plane")
            n_cyl = sum(1 for f in payload["features"] if f["type"] == "cylinder")
            n_hole = sum(1 for f in payload["features"] if f["type"] == "cylinder" and "hole" in f.get("tags", []))
            n_shaft = sum(1 for f in payload["features"] if f["type"] == "cylinder" and "shaft" in f.get("tags", []))
            print(f"{part_id}: planes={n_plane}, cylinders={n_cyl} (hole={n_hole}, shaft={n_shaft})")

            part_folder = out_dir / "parts" / str(part_id)
            part_folder.mkdir(parents=True, exist_ok=True)

            out_file = part_folder / "PartFeatures.json"
            out_file.write_text(json.dumps(payload, indent=2))
            written.append(out_file)

    return written



# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Any, List, Tuple
# import json

# from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
# from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
# from OCC.Core.BRepGProp import brepgprop
# from OCC.Core.GProp import GProp_GProps
# from OCC.Core.gp import gp_Pnt, gp_Dir

# from OCC.Core.BRepLProp import BRepLProp_SLProps
# from OCC.Core.BRepTools import breptools_UVBounds

# from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON


# from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
# from OCC.Core.TopAbs import TopAbs_IN
# from OCC.Core.gp import gp_Vec
# from OCC.Core.GeomLProp import GeomLProp_SLProps



# from geomate.io.step_io import load_step_shapes
# from geomate.io.topo import iter_solids, iter_faces
# from geomate.io.ids import PartId, FaceId


# def _pnt_to_list(p: gp_Pnt) -> List[float]:
#     return [float(p.X()), float(p.Y()), float(p.Z())]


# def _dir_to_list(d: gp_Dir) -> List[float]:
#     return [float(d.X()), float(d.Y()), float(d.Z())]


# def _props_area_centroid(face) -> Tuple[float, List[float]]:
#     props = GProp_GProps()
#     brepgprop.SurfaceProperties(face, props)
#     area = float(props.Mass())
#     c = props.CentreOfMass()
#     return area, _pnt_to_list(c)

# def _canonical_dir(d: gp_Dir) -> gp_Dir:
#     # deterministic sign rule: make the dominant component positive
#     x, y, z = d.X(), d.Y(), d.Z()
#     ax, ay, az = abs(x), abs(y), abs(z)
#     if ax >= ay and ax >= az:
#         if x < 0: return gp_Dir(-x, -y, -z)
#     elif ay >= ax and ay >= az:
#         if y < 0: return gp_Dir(-x, -y, -z)
#     else:
#         if z < 0: return gp_Dir(-x, -y, -z)
#     return d


# def _point_state_in_solid(solid, p, tol: float = 1e-6):
#     clf = BRepClass3d_SolidClassifier()
#     clf.Load(solid)
#     clf.Perform(p, tol)
#     return clf.State()

# def _is_inside_or_on(solid, p, tol: float = 1e-6) -> bool:
#     st = _point_state_in_solid(solid, p, tol)
#     return st in (TopAbs_IN, TopAbs_ON)


# def classify_cylinder_hole_vs_shaft(solid, face, tol: float = 1e-6) -> str:
#     """
#     Robust test:
#       - get trimmed UV bounds from the FACE (not the adaptor)
#       - evaluate point+normal using BRepLProp_SLProps
#       - test both sides (+normal and -normal) with solid classifier
#       - whichever side is inside determines outward/inward normal direction
#     """
#     surf_adapt = BRepAdaptor_Surface(face, True)
#     if surf_adapt.GetType() != GeomAbs_Cylinder:
#         return "unknown"

#     # trimmed UV bounds (more reliable than adaptor First/Last)
#     u1, u2, v1, v2 = breptools_UVBounds(face)
#     u = 0.5 * (u1 + u2)
#     v = 0.5 * (v1 + v2)

#     props = BRepLProp_SLProps(surf_adapt, u, v, 1, tol)
#     if not props.IsNormalDefined():
#         # try a slightly different sample (avoid seams)
#         u = u1 + 0.33 * (u2 - u1)
#         v = v1 + 0.33 * (v2 - v1)
#         props = BRepLProp_SLProps(surf_adapt, u, v, 1, tol)
#         if not props.IsNormalDefined():
#             return "unknown"

#     p = props.Value()     # gp_Pnt
#     n = props.Normal()    # gp_Dir

#     r = float(surf_adapt.Cylinder().Radius())
#     # try a couple eps values in case we land ON boundary
#     for eps in (max(0.05, r * 0.02), max(0.2, r * 0.05), max(0.5, r * 0.10)):
#         p_plus = gp_Pnt(p.X() + n.X() * eps, p.Y() + n.Y() * eps, p.Z() + n.Z() * eps)
#         p_minus = gp_Pnt(p.X() - n.X() * eps, p.Y() - n.Y() * eps, p.Z() - n.Z() * eps)

#         plus_in = _is_inside_or_on(solid, p_plus, tol)
#         minus_in = _is_inside_or_on(solid, p_minus, tol)

#         # Decision:
#         # If +normal goes inside, then normal points into material => HOLE surface
#         # If -normal goes inside, then normal points out of material => SHAFT surface
#         if plus_in and not minus_in:
#             return "hole"
#         if minus_in and not plus_in:
#             return "shaft"

#     return "unknown"

# def extract_features_for_solid(part_id: PartId, solid) -> Dict[str, Any]:
#     """
#     Extract plane and cylinder surface features from all faces of the solid.
#     """
#     features: List[Dict[str, Any]] = []

#     for face in iter_faces(solid):
#         face_id = FaceId.from_face(part_id, face)
#         surf = BRepAdaptor_Surface(face, True)
#         stype = surf.GetType()

#         area, centroid = _props_area_centroid(face)

#         if stype == GeomAbs_Plane:
#             pln = surf.Plane()
#             origin = pln.Location()
#             # normal = pln.Axis().Direction()
#             normal = _canonical_dir(pln.Axis().Direction())


#             features.append({
#                 "feature_id": f"{face_id}_plane",
#                 "type": "plane",
#                 "face_id": str(face_id),
#                 "tags": [],
#                 "geom": {
#                     "origin": _pnt_to_list(origin),
#                     "normal": _dir_to_list(normal)
#                 },
#                 "meta": {
#                     "area": area,
#                     "centroid": centroid
#                 }
#             })

#         elif stype == GeomAbs_Cylinder:
#             cyl = surf.Cylinder()
#             ax = cyl.Axis()
#             axis_point = ax.Location()
#             # axis_dir = ax.Direction()
#             axis_dir = _canonical_dir(ax.Direction())

#             radius = float(cyl.Radius())
#             kind = classify_cylinder_hole_vs_shaft(solid, face)
#             tags = [kind] if kind != "unknown" else []
#             if kind == "unknown":
#                 print("UNKNOWN cylinder:", part_id, face_id, "r=", radius)



#             features.append({
#                 "feature_id": f"{face_id}_cyl",
#                 "type": "cylinder",
#                 "face_id": str(face_id),
#                 "tags": [tags],
#                 "geom": {
#                     "axis_point": _pnt_to_list(axis_point),
#                     "axis_dir": _dir_to_list(axis_dir),
#                     "radius": radius
#                 },
#                 "meta": {
#                     "area": area,
#                     "centroid": centroid
#                 }
#             })

#         else:
#             # ignore other surface types for now (cones, bspline, etc.)
#             continue

#     return {
#         "schema_version": "0.1",
#         "part_id": str(part_id),
#         "source": {},
#         "units": "mm",
#         "features": features
#     }


# def extract_all(step_path: Path, out_dir: Path) -> List[Path]:
#     step_path = step_path.expanduser().resolve()
#     out_dir = out_dir.expanduser().resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     shapes = load_step_shapes(step_path)

#     written: List[Path] = []
#     for root_i, root_shape in enumerate(shapes):
#         for solid_i, solid in enumerate(iter_solids(root_shape)):
#             part_id = PartId.from_step(step_path, root_i, solid_i)
#             payload = extract_features_for_solid(part_id, solid)
#             payload["source"] = {"step_path": str(step_path), "solid_index": solid_i}

#             part_folder = out_dir / "parts" / str(part_id)
#             part_folder.mkdir(parents=True, exist_ok=True)

#             out_file = part_folder / "PartFeatures.json"
#             out_file.write_text(json.dumps(payload, indent=2))
#             written.append(out_file)
#     n_plane = sum(1 for f in payload["features"] if f["type"] == "plane")
#     n_cyl = sum(1 for f in payload["features"] if f["type"] == "cylinder")
#     n_hole = sum(1 for f in payload["features"] if f["type"] == "cylinder" and "hole" in f.get("tags", []))
#     n_shaft = sum(1 for f in payload["features"] if f["type"] == "cylinder" and "shaft" in f.get("tags", []))
#     print(f"{part_id}: planes={n_plane}, cylinders={n_cyl} (hole={n_hole}, shaft={n_shaft})")

#     return written
