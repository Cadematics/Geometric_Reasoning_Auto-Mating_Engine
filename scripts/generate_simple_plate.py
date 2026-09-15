from pathlib import Path

from OCCT.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCCT.STEPControl import STEPControl_AsIs, STEPControl_Writer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "simple_plate" / "plate.step"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 100 x 50 x 10 mm rectangular solid.
    box = BRepPrimAPI_MakeBox(100.0, 50.0, 10.0).Shape()

    writer = STEPControl_Writer()

    writer.Transfer(box, STEPControl_AsIs)

    status = writer.Write(str(OUTPUT))

    if status != 1:
        raise RuntimeError(
            f"Failed to write STEP file: {OUTPUT}"
        )

    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()  