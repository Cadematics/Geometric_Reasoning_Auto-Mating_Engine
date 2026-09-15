from pathlib import Path

from auto_mating.io import read_step
from auto_mating.topology import extract_topology




from auto_mating.io import read_step
from auto_mating.topology import (
    TopologyIndex,
    extract_topology,
)

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




def test_face_edge_relationships():
    shape = read_step(EXAMPLE_STEP)
    index = TopologyIndex.build(shape)

    for face_index in range(1, index.face_count + 1):
        edges = index.face_edges(face_index)

        assert len(edges) == 4
        assert len(set(edges)) == 4


def test_edge_face_relationships():
    shape = read_step(EXAMPLE_STEP)
    index = TopologyIndex.build(shape)

    for edge_index in range(1, index.edge_count + 1):
        faces = index.edge_faces(edge_index)

        assert len(faces) == 2
        assert faces[0] != faces[1]


def test_face_adjacency():
    shape = read_step(EXAMPLE_STEP)
    index = TopologyIndex.build(shape)

    adjacency = index.face_adjacency()

    assert len(adjacency) == 6

    for face_index, neighbors in adjacency.items():
        assert len(neighbors) == 4
        assert face_index not in neighbors

        for neighbor in neighbors:
            assert face_index in adjacency[neighbor]


def test_face_adjacency_edge_count():
    shape = read_step(EXAMPLE_STEP)
    index = TopologyIndex.build(shape)

    adjacency = index.face_adjacency()

    total_adjacencies = sum(
        len(neighbors)
        for neighbors in adjacency.values()
    )

    assert total_adjacencies == 24