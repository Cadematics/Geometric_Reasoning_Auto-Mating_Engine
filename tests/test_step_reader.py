from pathlib import Path

import pytest

from auto_mating.io import read_step


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_read_step_returns_shape():
    shape = read_step(EXAMPLE_STEP)

    assert not shape.IsNull()


def test_missing_step_file():
    with pytest.raises(FileNotFoundError):
        read_step("does_not_exist.step")


def test_step_path_must_be_file(tmp_path):
    directory = tmp_path / "not_a_file"
    directory.mkdir()

    with pytest.raises(ValueError):
        read_step(directory)