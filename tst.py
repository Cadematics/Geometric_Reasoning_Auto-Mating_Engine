from auto_mating.io import read_step
from auto_mating.topology import iter_faces, iter_edges, iter_vertices

shape = read_step("examples/simple_plate/plate.step")

faces = list(iter_faces(shape))
edges = list(iter_edges(shape))
vertices = list(iter_vertices(shape))

print("Faces:", len(faces))
print("Edges:", len(edges))
print("Vertices:", len(vertices))

print(type(faces[0]))
print(type(edges[0]))
print(type(vertices[0]))
