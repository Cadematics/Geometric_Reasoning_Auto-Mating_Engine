import argparse
from pathlib import Path
import json

from geomate.io.step_io import load_step_shapes
from geomate.io.topo import iter_solids, iter_faces
from geomate.io.ids import PartId, FaceId, stable_face_key


def cmd_import(args: argparse.Namespace) -> int:
    step_path = Path(args.step).expanduser().resolve()
    shapes = load_step_shapes(step_path)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": "0.1",
        "step_path": str(step_path),
        "num_root_shapes": len(shapes),
        "parts": []
    }

    for root_i, root_shape in enumerate(shapes):
        for solid_i, solid in enumerate(iter_solids(root_shape)):
            part_id = PartId.from_step(step_path, root_i, solid_i)

            faces_payload = []
            for face in iter_faces(solid):
                fid = FaceId.from_face(part_id, face)
                faces_payload.append({
                    "face_id": str(fid),
                    "key": stable_face_key(face),
                })

            index["parts"].append({
                "part_id": str(part_id),
                "root_index": root_i,
                "solid_index": solid_i,
                "num_faces": len(faces_payload),
                "faces": faces_payload
            })

    out_file = out_dir / "part_index.json"
    out_file.write_text(json.dumps(index, indent=2))
    print(f"Wrote: {out_file} (parts={len(index['parts'])})")
    return 0


def cmd_extract_features(_: argparse.Namespace) -> int:
    raise SystemExit("extract_features not implemented yet (Phase 2).")


def cmd_propose_mates(_: argparse.Namespace) -> int:
    raise SystemExit("propose_mates not implemented yet (Phase 3/4).")


def cmd_solve_assembly(_: argparse.Namespace) -> int:
    raise SystemExit("solve_assembly not implemented yet (Phase 5+).")


def cmd_export_assembly(_: argparse.Namespace) -> int:
    raise SystemExit("export_assembly not implemented yet (Phase 9).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geomate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import_step", help="Load STEP and create part+face index with stable IDs")
    p_imp.add_argument("--step", required=True, help="Path to STEP file")
    p_imp.add_argument("--out", default="outputs", help="Output directory")
    p_imp.set_defaults(func=cmd_import)

    p_ext = sub.add_parser("extract_features", help="Extract plane/cylinder features per part (Phase 2)")
    p_ext.set_defaults(func=cmd_extract_features)

    p_mat = sub.add_parser("propose_mates", help="Generate candidate mates (Phase 3/4)")
    p_mat.set_defaults(func=cmd_propose_mates)

    p_sol = sub.add_parser("solve_assembly", help="Search for best assembly (Phase 5+)")
    p_sol.set_defaults(func=cmd_solve_assembly)

    p_exp = sub.add_parser("export_assembly", help="Export final assembly (Phase 9)")
    p_exp.set_defaults(func=cmd_export_assembly)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
