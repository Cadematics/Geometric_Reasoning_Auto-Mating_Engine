#!/usr/bin/env python3
"""
test_solid_classification.py

Goal:
  Diagnose why cylinder faces are not being classified as HOLE vs SHAFT.

What it does:
  - Loads a STEP
  - Iterates solids and finds cylindrical faces
  - For each cylinder face, tests multiple classification strategies:
      A) Surface normal test (BRepLProp_SLProps) with +n / -n stepping
      B) Axis/radial heuristic (build radial vector) with +rad / -rad stepping
      C) Multiple (u,v) sample points (mid, 1/3, 2/3) + multiple eps values
  - Prints the TopAbs state returned by BRepClass3d_SolidClassifier for each test point
  - Validates solid with BRepCheck_Analyzer
  - Prints solid volume (helps detect "not a real solid" / orientation issues)

Usage:
  python test_solid_classification.py ./PLATE.STEP
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopoDS import topods

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder

from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON, TopAbs_UNKNOWN

from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps

# UV bounds: prefer new API if available
try:
    from OCC.Core.BRepTools import breptools  # new style
    def uv_bounds(face):
        return breptools.UVBounds(face)
except Exception:
    from OCC.Core.BRepTools import breptools_UVBounds  # deprecated but works
    def uv_bounds(face):
        return breptools_UVBounds(face)


def state_name(st) -> str:
    if st == TopAbs_IN: return "IN"
    if st == TopAbs_OUT: return "OUT"
    if st == TopAbs_ON: return "ON"
    if st == TopAbs_UNKNOWN: return "UNKNOWN"
    return str(int(st))


def load_step_shapes(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path} (status={status})")
    n = reader.TransferRoots()
    if n == 0:
        raise RuntimeError(f"STEP transfer failed (no roots): {step_path}")

    shapes = []
    for i in range(1, reader.NbShapes() + 1):
        shapes.append(reader.Shape(i))
    return shapes


def iter_solids(shape):
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield topods.Solid(exp.Current())
        exp.Next()


def iter_faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()


def solid_state(solid, p: gp_Pnt, tol=1e-6):
    clf = BRepClass3d_SolidClassifier()
    clf.Load(solid)
    clf.Perform(p, tol)
    return clf.State()


def solid_volume(solid) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    return float(props.Mass())


def safe_normal_at(face, surf_adapt, u, v, tol=1e-6):
    props = BRepLProp_SLProps(surf_adapt, u, v, 1, tol)
    if not props.IsNormalDefined():
        return None
    return props.Value(), props.Normal()


def pick_uv_samples(face) -> List[Tuple[float, float, str]]:
    u1, u2, v1, v2 = uv_bounds(face)
    # avoid exact seams: sample a few interior points
    return [
        (0.5*(u1+u2), 0.5*(v1+v2), "mid"),
        (u1 + 0.33*(u2-u1), v1 + 0.33*(v2-v1), "one_third"),
        (u1 + 0.67*(u2-u1), v1 + 0.67*(v2-v1), "two_third"),
    ]


def radial_dir_from_axis(axis_dir: gp_Dir) -> gp_Vec:
    # Build a radial direction perpendicular to axis_dir using projection method
    ref = gp_Dir(1, 0, 0)
    if abs(axis_dir.Dot(ref)) > 0.95:
        ref = gp_Dir(0, 1, 0)

    ref_vec = gp_Vec(ref.X(), ref.Y(), ref.Z())
    axis_vec = gp_Vec(axis_dir.X(), axis_dir.Y(), axis_dir.Z())
    proj = axis_vec.Multiplied(ref_vec.Dot(axis_vec))
    radial = ref_vec.Subtracted(proj)
    if radial.Magnitude() < 1e-9:
        # fallback
        ref = gp_Dir(0, 0, 1)
        ref_vec = gp_Vec(ref.X(), ref.Y(), ref.Z())
        proj = axis_vec.Multiplied(ref_vec.Dot(axis_vec))
        radial = ref_vec.Subtracted(proj)
    if radial.Magnitude() < 1e-9:
        return gp_Vec(1, 0, 0)
    radial.Normalize()
    return radial


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_solid_classification.py /path/to/file.step")
        sys.exit(2)

    step_path = Path(sys.argv[1]).expanduser().resolve()
    shapes = load_step_shapes(step_path)

    print(f"STEP: {step_path}")
    print(f"root shapes: {len(shapes)}")
    print("-"*80)

    for root_i, root in enumerate(shapes):
        for solid_i, solid in enumerate(iter_solids(root)):
            print(f"[root {root_i}] solid {solid_i}")

            an = BRepCheck_Analyzer(solid)
            print(f"  solid valid: {an.IsValid()}")
            try:
                vol = solid_volume(solid)
                print(f"  volume: {vol}")
            except Exception as e:
                print(f"  volume: ERROR ({e})")

            cyl_faces = []
            for face in iter_faces(solid):
                surf = BRepAdaptor_Surface(face, True)
                if surf.GetType() == GeomAbs_Cylinder:
                    cyl_faces.append((face, surf))

            print(f"  cylinder faces: {len(cyl_faces)}")
            print()

            for idx, (face, surf) in enumerate(cyl_faces):
                cyl = surf.Cylinder()
                ax = cyl.Axis()
                axis_p = ax.Location()
                axis_d = ax.Direction()
                r = float(cyl.Radius())

                print(f"  --- CYL FACE #{idx} ---")
                print(f"    radius: {r}")
                print(f"    axis_p: ({axis_p.X():.6f}, {axis_p.Y():.6f}, {axis_p.Z():.6f})")
                print(f"    axis_d: ({axis_d.X():.6f}, {axis_d.Y():.6f}, {axis_d.Z():.6f})")

                eps_list = [max(0.05, r*0.02), max(0.2, r*0.05), max(0.5, r*0.1), 1.0, 2.0]

                # Strategy A: normal test at several UV samples
                print("    Strategy A: surface normal (+n / -n)")
                uv_samples = pick_uv_samples(face)
                for (u, v, tag) in uv_samples:
                    nv = safe_normal_at(face, surf, u, v, tol=1e-6)
                    if nv is None:
                        print(f"      [{tag}] normal: UNDEFINED at (u={u:.6f}, v={v:.6f})")
                        continue
                    p, n = nv
                    print(f"      [{tag}] p=({p.X():.6f},{p.Y():.6f},{p.Z():.6f}) n=({n.X():.6f},{n.Y():.6f},{n.Z():.6f})")
                    for eps in eps_list:
                        p_plus = gp_Pnt(p.X()+n.X()*eps, p.Y()+n.Y()*eps, p.Z()+n.Z()*eps)
                        p_minus = gp_Pnt(p.X()-n.X()*eps, p.Y()-n.Y()*eps, p.Z()-n.Z()*eps)
                        st_plus = solid_state(solid, p_plus)
                        st_minus = solid_state(solid, p_minus)
                        print(f"        eps={eps:.3f}  +n:{state_name(st_plus):>7}  -n:{state_name(st_minus):>7}")
                print()

                # Strategy B: radial test (axis + constructed radial)
                print("    Strategy B: radial (+rad / -rad) from axis")
                radial = radial_dir_from_axis(axis_d)
                # point on cylinder surface using axis_p + radial*r (this is not trimmed-aware but useful)
                p_surf = gp_Pnt(
                    axis_p.X() + radial.X() * r,
                    axis_p.Y() + radial.Y() * r,
                    axis_p.Z() + radial.Z() * r,
                )
                print(f"      p_surf=({p_surf.X():.6f},{p_surf.Y():.6f},{p_surf.Z():.6f}) rad=({radial.X():.6f},{radial.Y():.6f},{radial.Z():.6f})")
                for eps in eps_list:
                    p_plus = gp_Pnt(p_surf.X()+radial.X()*eps, p_surf.Y()+radial.Y()*eps, p_surf.Z()+radial.Z()*eps)
                    p_minus = gp_Pnt(p_surf.X()-radial.X()*eps, p_surf.Y()-radial.Y()*eps, p_surf.Z()-radial.Z()*eps)
                    st_plus = solid_state(solid, p_plus)
                    st_minus = solid_state(solid, p_minus)
                    print(f"        eps={eps:.3f}  +rad:{state_name(st_plus):>7}  -rad:{state_name(st_minus):>7}")

                print()
                print("    Interpretation hints:")
                print("      If +n is IN/ON and -n is OUT => normal points into material => HOLE-like cylinder face")
                print("      If -n is IN/ON and +n is OUT => normal points outward => SHAFT-like cylinder face")
                print("      If both sides are ON or both IN/ON => may be thin solid / tolerance / classifier ambiguity")
                print("      If normal UNDEFINED for all UV samples => trimmed seam; need different sampling or shape healing")
                print()

            print("="*80)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
