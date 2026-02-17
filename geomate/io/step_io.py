from pathlib import Path
from typing import List

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone


def load_step_shapes(step_path: Path) -> List["TopoDS_Shape"]:
    """
    Read a STEP file and return the transferred root shapes.
    Each returned item is a TopoDS_Shape.
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path} (status={status})")

    n = reader.TransferRoots()
    if n == 0:
        raise RuntimeError(f"STEP transfer failed (no roots): {step_path}")

    shapes: List["TopoDS_Shape"] = []
    for i in range(1, reader.NbShapes() + 1):
        shapes.append(reader.Shape(i))
    return shapes
