import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from experiments_v2.config import ROOT, DATA_DIR

EARTH_RADIUS_KM = 6371.0088


def _node_df():
    p = ROOT / 'data' / 'processed' / 'node_features.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. This standalone project ships with data/processed/node_features.csv; restore that file from the ZIP.')
    df = pd.read_csv(p).copy()
    if df[['lat', 'lon']].isna().any().any():
        raise ValueError('node_features.csv contains missing lat/lon; resolve before v2 experiments.')
    return df


def build_geographic_knn(k=6):
    df = _node_df()
    coords_rad = np.radians(df[['lat', 'lon']].to_numpy(dtype=float))
    n = len(df)
    knn = NearestNeighbors(n_neighbors=min(k + 1, n), metric='haversine', algorithm='ball_tree')
    knn.fit(coords_rad)
    distances_rad, indices = knn.kneighbors(coords_rad)

    undirected = {}
    for i in range(n):
        for dist_rad, j in zip(distances_rad[i], indices[i]):
            j = int(j)
            if i == j:
                continue
            a, b = sorted((i, j))
            d_km = float(dist_rad * EARTH_RADIUS_KM)
            if (a, b) not in undirected or d_km < undirected[(a, b)]:
                undirected[(a, b)] = d_km

    edges = []
    dists = []
    for (a, b), d in sorted(undirected.items()):
        edges.extend([(a, b), (b, a)])
        dists.extend([d, d])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    distances = torch.tensor(dists, dtype=torch.float32)
    return edge_index, distances, n


def distance_weights(distances):
    d = distances.detach().cpu().numpy()
    positive = d[d > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    w = np.exp(-d / max(scale, 1e-8)).astype(np.float32)
    return torch.tensor(w)


def build_random_graph_like(reference_edge_index, n_nodes, seed=42):
    rng = np.random.default_rng(seed)
    # reference is symmetric: number of unique undirected edges is E/2
    n_undir = reference_edge_index.shape[1] // 2
    chosen = set()
    while len(chosen) < n_undir:
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        if a == b:
            continue
        chosen.add(tuple(sorted((a, b))))
    edges = []
    for a, b in sorted(chosen):
        edges.extend([(a, b), (b, a)])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()



def _undirected_edge_set(edge_index):
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(src, dst) if int(a) != int(b)}


def edge_overlap_fraction(edge_a, edge_b):
    """Fraction of unique undirected edges in A that also appear in B."""
    a = _undirected_edge_set(edge_a)
    b = _undirected_edge_set(edge_b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def build_permuted_graph_like(reference_edge_index, n_nodes, seed=777, tries=64):
    """Node-label permutation negative control.

    This preserves the reference graph's exact topology and node-degree
    distribution, but breaks the mapping between geographic regions and graph
    positions. Among several random permutations, the one with the lowest edge
    overlap with the reference graph is returned.
    """
    if int(reference_edge_index.max()) >= n_nodes:
        raise ValueError('reference_edge_index contains node id >= n_nodes')
    rng = np.random.default_rng(seed)
    best = None
    best_perm = None
    best_overlap = float('inf')
    tries = max(1, int(tries))
    for _ in range(tries):
        perm = rng.permutation(n_nodes)
        mapped = torch.as_tensor(perm, dtype=torch.long)[reference_edge_index.cpu()]
        overlap = edge_overlap_fraction(reference_edge_index, mapped)
        if overlap < best_overlap:
            best = mapped.contiguous()
            best_perm = perm.copy()
            best_overlap = overlap
            if overlap == 0.0:
                break
    return best, best_perm, float(best_overlap)

def row_normalized_adjacency(edge_index, n_nodes):
    w = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    w[src, dst] = 1.0
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return w / row_sum


def save_graphs(k=6, seed=42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    geo, dist, n = build_geographic_knn(k=k)
    weighted = distance_weights(dist)
    random_edge = build_random_graph_like(geo, n, seed=seed)
    torch.save({'edge_index': geo, 'edge_weight': None, 'distance_km': dist}, DATA_DIR / 'graph_geo.pt')
    torch.save({'edge_index': geo, 'edge_weight': weighted, 'distance_km': dist}, DATA_DIR / 'graph_geo_weighted.pt')
    torch.save({'edge_index': random_edge, 'edge_weight': None}, DATA_DIR / f'graph_random_seed{seed}.pt')
    print(f'Geographic symmetric KNN: N={n}, directed edges={geo.shape[1]}, undirected={geo.shape[1]//2}')
    print(f'Median geographic edge distance: {float(dist.median()):.2f} km')
    return geo


if __name__ == '__main__':
    save_graphs()
