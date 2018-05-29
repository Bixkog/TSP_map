import pickle
import igraph
import sys
import numpy as np


def tsp_objective_function(p):
    s = 0.0
    for i in range(n):
        s += A[p[i-1], p[i]]
    return s

def parse_tsp(filename):
    data = ""
    with open(filename) as fh:
        data = fh.read().split("\n")
    n = int(data[3].split()[1])
    e_type = data[4].split()[1]
    e_format = ""
    if e_type == "EXPLICIT":
        e_format = data[5].split()[1]
    
    A = np.empty((n, n), dtype=int)
    if e_type == "EXPLICIT" and e_format == "UPPER_ROW":
        for i in range(n-1):
            edges = map(int, data[i+8].split())
            for j, v in enumerate(edges):
                A[i, i + j + 1] = v
                A[i + j + 1, i] = v
        for i in range(n):
            A[i, i] = 0

    if e_type == "EXPLICIT" and e_format == "FULL_MATRIX":
        for i in range(n):
            edges = map(int, data[i+8].split())
            for j, v in enumerate(edges):
                A[i, j] = v
    if e_type == "EUC_2D":
        coords = np.empty((n,2))
        for i in range(n):
            coord = list(map(int, data[i+6].split()))
            coords[coord[0]-1, :] = np.array([coord[1:]])
        for i in range(n):
            for j in range(n):
                A[i, j] = np.sqrt(((coords[i, :] - coords[j, :])**2).sum())
    return A, n

def print_graph(V, E, g_min):
	results = sorted([tsp_objective_function(s) for s in V])
	threshold = results[len(results) // 20]
	V = { s for s in V if tsp_objective_function(s) <= threshold}
	V_ = { s : (i, tsp_objective_function(s)) for i, s in enumerate(V)}
	V_c = [0 for i in range(len(V_))]
	for s, (i, r) in V_.items():
		V_c[i] = (s, r)
	E_ = [(V_[s1][0], V_[s2][0]) for s1, s2 in E if s1 in V_ and s2 in V_]

	g = igraph.Graph()
	g.add_vertices(len(V_))
	g.add_edges(E_)

	visual_style = {}
	visual_style["layout"] = g.layout_fruchterman_reingold()
	visual_style["vertex_color"] = ['red' if t[1] == g_min else 'blue'
				for t in V_c]

	igraph.summary(g)
	igraph.plot(g, **visual_style)._repr_svg_()
 
def load_graph(file_name):
    with open(file_name, "rb") as fh:
        (V, E) = pickle.load(fh)
        return V, E

if __name__ == "__main__":
	file_name = sys.argv[1]
	V, E = load_graph(file_name + ".g")
	A, n = parse_tsp(file_name + ".tsp")
	print_graph(V, E, int(sys.argv[2]))