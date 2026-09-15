from pathlib import Path

from auto_mating.io import read_step
from auto_mating.topology import extract_topology


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_simple_plate_topology():
    shape = read_step(EXAMPLE_STEP)

    topology = extract_topology(shape)

    assert topology.solids == 1
    assert topology.shells == 1
    assert topology.faces == 6
    assert topology.edges == 12
    assert topology.vertices == 8