import argparse
from pathlib import Path

from auto_mating.io import read_step
from auto_mating.topology import extract_topology


def inspect_step(path: str) -> None:
    path = Path(path)

    shape = read_step(path)
    topology = extract_topology(shape)

    print(f"STEP Model: {path.name}")
    print()
    print("Topology")
    print("────────")
    print(f"Solids:   {topology.solids}")
    print(f"Shells:   {topology.shells}")
    print(f"Faces:    {topology.faces}")
    print(f"Edges:    {topology.edges}")
    print(f"Vertices: {topology.vertices}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="auto-mating",
        description="Geometric reasoning tools for CAD models.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect STEP model topology.",
    )

    inspect_parser.add_argument(
        "step_file",
        help="Path to STEP file.",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_step(args.step_file)


if __name__ == "__main__":
    main()