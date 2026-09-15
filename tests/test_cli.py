from pathlib import Path

from auto_mating.cli import inspect_step


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_inspect_step(capsys):
    inspect_step(str(EXAMPLE_STEP))

    captured = capsys.readouterr()

    assert "STEP Model: plate.step" in captured.out
    assert "Solids:   1" in captured.out
    assert "Shells:   1" in captured.out
    assert "Faces:    6" in captured.out
    assert "Edges:    12" in captured.out
    assert "Vertices: 8" in captured.out