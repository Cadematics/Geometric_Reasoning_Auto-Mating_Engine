from pathlib import Path
from typing import Union

from OCCT.IFSelect import IFSelect_RetDone
from OCCT.STEPControl import STEPControl_Reader
from OCCT.TopoDS import TopoDS_Shape


def read_step(path: Union[str, Path]) -> TopoDS_Shape:
    """
    Read a STEP file and return the resulting OpenCASCADE shape.

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    TopoDS_Shape
        Shape produced by OpenCASCADE's STEP translator.

    Raises
    ------
    FileNotFoundError
        If the STEP file does not exist.
    ValueError
        If the supplied path is not a regular file.
    RuntimeError
        If OpenCASCADE fails to read or transfer the STEP model.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    if not path.is_file():
        raise ValueError(f"STEP path is not a file: {path}")

    reader = STEPControl_Reader()

    status = reader.ReadFile(str(path))

    if status != IFSelect_RetDone:
        raise RuntimeError(
            f"OpenCASCADE failed to read STEP file: {path}"
        )

    transferred = reader.TransferRoots()

    if transferred == 0:
        raise RuntimeError(
            f"OpenCASCADE transferred no STEP roots: {path}"
        )

    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError(
            f"STEP file produced a null OpenCASCADE shape: {path}"
        )

    return shape