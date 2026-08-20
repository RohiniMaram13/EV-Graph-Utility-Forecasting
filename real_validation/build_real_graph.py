import json
from pathlib import Path

import numpy as np
import pandas as pd
from acnportal import acnsim


DATA = Path("data/real_acn")

MAPPING_FILE = DATA / "caltech_station_mapping.csv"

ADJ_FILE = DATA / "caltech_pod_adjacency.npy"
META_FILE = DATA / "caltech_pod_graph_metadata.json"
EDGE_FILE = DATA / "caltech_pod_edges.csv"


# ============================================================
# LOAD FROZEN 37-NODE MAPPING
# ============================================================

mapping = pd.read_csv(MAPPING_FILE)
mapping = mapping.sort_values("node_id").reset_index(drop=True)

N = len(mapping)

assert N == 37, f"Expected 37 nodes, got {N}"

space_to_node = {
    str(row.spaceID): int(row.node_id)
    for row in mapping.itertuples()
}


# ============================================================
# GET OFFICIAL ACN-SIM NETWORK
# ============================================================

net = acnsim.sites.caltech_acn()

M = np.asarray(net.constraint_matrix)

constraint_names = list(net.constraint_index)
station_ids = list(net.station_ids)


# ============================================================
# FREEZE GRAPH RULE
#
# ONLY explicit local pod constraints:
#   CC Pod
#   AV Pod
#
# Secondary and Primary electrical equations are deliberately
# excluded from ordinary graph adjacency.
# ============================================================

POD_CONSTRAINTS = [
    "CC Pod",
    "AV Pod",
]

A = np.zeros((N, N), dtype=np.float32)

pod_members = {}
edges = []


for pod_name in POD_CONSTRAINTS:

    row_idx = constraint_names.index(pod_name)

    active_cols = np.where(
        np.abs(M[row_idx]) > 1e-12
    )[0]

    all_members = [
        station_ids[j]
        for j in active_cols
    ]

    # Keep only EVSEs represented in our frozen 37-node data.
    selected_members = [
        sid
        for sid in all_members
        if sid in space_to_node
    ]

    pod_members[pod_name] = selected_members

    print()
    print(pod_name)
    print("ACN-Sim members:", len(all_members))
    print("Selected members:", len(selected_members))
    print(selected_members)

    # Clique within each explicit pod.
    for a in range(len(selected_members)):
        for b in range(a + 1, len(selected_members)):

            sid_i = selected_members[a]
            sid_j = selected_members[b]

            i = space_to_node[sid_i]
            j = space_to_node[sid_j]

            A[i, j] = 1.0
            A[j, i] = 1.0

            edges.append({
                "node_i": i,
                "node_j": j,
                "space_i": sid_i,
                "space_j": sid_j,
                "station_i": mapping.loc[
                    mapping.node_id == i,
                    "stationID"
                ].iloc[0],
                "station_j": mapping.loc[
                    mapping.node_id == j,
                    "stationID"
                ].iloc[0],
                "constraint": pod_name,
            })


# ============================================================
# QC
# ============================================================

assert np.allclose(A, A.T), "Adjacency is not symmetric."
assert np.all(np.diag(A) == 0), "Unexpected self edges."

undirected_edges = int(A.sum() / 2)

degrees = A.sum(axis=1).astype(int)

nodes_with_neighbors = int(np.sum(degrees > 0))
isolated_nodes = int(np.sum(degrees == 0))


print()
print("=" * 72)
print("FROZEN REAL ACN GRAPH")
print("=" * 72)

print("Nodes:", N)
print("Undirected edges:", undirected_edges)
print("Nodes with pod neighbors:", nodes_with_neighbors)
print("Isolated nodes:", isolated_nodes)

print()
print("Degree distribution:")

values, counts = np.unique(
    degrees,
    return_counts=True
)

for d, c in zip(values, counts):
    print(f"degree {d}: {c} nodes")


# Expected from the frozen rule.
assert undirected_edges == 56, (
    f"Expected 56 pod edges, got {undirected_edges}"
)

assert nodes_with_neighbors == 16, (
    f"Expected 16 pod-connected nodes, got {nodes_with_neighbors}"
)

assert isolated_nodes == 21, (
    f"Expected 21 isolated nodes, got {isolated_nodes}"
)


# ============================================================
# SAVE
# ============================================================

np.save(ADJ_FILE, A)

pd.DataFrame(edges).to_csv(
    EDGE_FILE,
    index=False,
)

metadata = {
    "n_nodes": N,
    "n_edges_undirected": undirected_edges,
    "nodes_with_pod_neighbors": nodes_with_neighbors,
    "isolated_nodes": isolated_nodes,
    "included_constraints": POD_CONSTRAINTS,
    "excluded_constraints": [
        "Secondary A",
        "Secondary B",
        "Secondary C",
        "Primary A",
        "Primary B",
        "Primary C",
    ],
    "graph_rule":
        "Undirected clique edges between selected EVSEs "
        "sharing an explicit ACN-Sim CC Pod or AV Pod constraint.",
    "self_loops_saved": False,
    "adjacency_order":
        "node_id order from caltech_station_mapping.csv",
}

with open(META_FILE, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("Saved:")
print(ADJ_FILE)
print(EDGE_FILE)
print(META_FILE)
