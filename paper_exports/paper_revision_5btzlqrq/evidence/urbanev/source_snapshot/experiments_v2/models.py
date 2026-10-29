import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, batch_first=True)

    def forward(self, x):
        B, N, T = x.shape
        _, h = self.gru(x.reshape(B * N, T, 1))
        return h[-1].reshape(B, N, -1)


class GRUForecast(nn.Module):
    def __init__(self, hidden_dim=64, horizon=24):
        super().__init__()
        self.temporal = TemporalEncoder(hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, x_static=None, edge_index=None, edge_weight=None):
        return self.head(self.temporal(x))


class GRUStaticForecast(nn.Module):
    def __init__(self, n_static, hidden_dim=64, horizon=24):
        super().__init__()
        self.temporal = TemporalEncoder(hidden_dim)
        self.static = nn.Sequential(nn.Linear(n_static, hidden_dim), nn.ReLU())
        self.head = nn.Linear(2 * hidden_dim, horizon)

    def forward(self, x, x_static, edge_index=None, edge_weight=None):
        B = x.shape[0]
        ht = self.temporal(x)
        hs = self.static(x_static).unsqueeze(0).expand(B, -1, -1)
        return self.head(torch.cat([ht, hs], dim=-1))


class _BatchedGraphMixin:
    def _expanded_graph(self, edge_index, edge_weight, B, N):
        key = (B, N, edge_index.device.type, edge_index.device.index, edge_index.data_ptr())
        if not hasattr(self, '_graph_cache'):
            self._graph_cache = {}
        if key not in self._graph_cache:
            offsets = (
                torch.arange(B, device=edge_index.device, dtype=edge_index.dtype) * N
            ).view(B, 1, 1)
            e = edge_index.unsqueeze(0) + offsets
            e = e.permute(1, 0, 2).reshape(2, -1).contiguous()
            ew = None if edge_weight is None else edge_weight.repeat(B)
            self._graph_cache[key] = (e, ew)
        return self._graph_cache[key]


class GRUGCNForecast(_BatchedGraphMixin, nn.Module):
    """Temporal GRU plus an *incremental* residual GCN path.

    The local GRU representation is always preserved. The graph path is learned
    as an additive correction, controlled by a learnable gate initialized small.
    This makes the architecture match the research question: does spatial
    information add value beyond local temporal history?
    """

    def __init__(self, hidden_dim=64, horizon=24, dropout=0.3):
        super().__init__()
        self.temporal = TemporalEncoder(hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.spatial_norm = nn.LayerNorm(hidden_dim)

        # Warm-start friendly gate. sigmoid(-5) ~= 0.0067, so after copying a
        # trained GRU the graph model begins extremely close to the temporal
        # baseline and learns only an incremental spatial correction.
        self.graph_gate_logit = nn.Parameter(torch.tensor(-5.0))
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, x_static=None, edge_index=None, edge_weight=None):
        B, N, _ = x.shape
        h_local = self.temporal(x)                       # [B, N, H]

        h = h_local.reshape(B * N, -1)
        edges, ew = self._expanded_graph(edge_index, edge_weight, B, N)
        h_spatial = F.relu(self.conv1(h, edges, edge_weight=ew))
        h_spatial = self.dropout(h_spatial)
        h_spatial = self.conv2(h_spatial, edges, edge_weight=ew)
        h_spatial = h_spatial.reshape(B, N, -1)
        h_spatial = self.spatial_norm(h_spatial)

        gate = torch.sigmoid(self.graph_gate_logit)
        h_fused = h_local + gate * h_spatial
        return self.head(h_fused)

    def graph_gate(self):
        return float(torch.sigmoid(self.graph_gate_logit.detach()).cpu())

    def initialize_from_gru(self, gru_model):
        """Copy the temporal encoder and forecast head from a trained GRU.

        The graph branch remains independently initialized and the graph gate
        stays near zero, so the initial warm-start prediction is effectively
        the GRU forecast plus a tiny spatial correction.
        """
        self.temporal.load_state_dict(gru_model.temporal.state_dict())
        self.head.load_state_dict(gru_model.head.state_dict())
        return self

    def set_graph_only_trainable(self):
        """Freeze the copied GRU path and train only the spatial correction."""
        for p in self.parameters():
            p.requires_grad = False
        for module in (self.conv1, self.conv2, self.spatial_norm):
            for p in module.parameters():
                p.requires_grad = True
        self.graph_gate_logit.requires_grad = True
        return self

    def set_all_trainable(self):
        for p in self.parameters():
            p.requires_grad = True
        return self

class STGNNForecastV2(_BatchedGraphMixin, nn.Module):
    """Residual graph model with static context.

    Temporal + static information forms the local representation. The GCN is an
    incremental correction rather than a replacement for that representation.
    """

    def __init__(self, n_static, hidden_dim=64, horizon=24, dropout=0.3):
        super().__init__()
        self.temporal = TemporalEncoder(hidden_dim)
        self.static = nn.Sequential(nn.Linear(n_static, hidden_dim), nn.ReLU())
        self.local_fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.graph_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, x_static, edge_index=None, edge_weight=None):
        B, N, _ = x.shape
        ht = self.temporal(x)
        hs = self.static(x_static).unsqueeze(0).expand(B, -1, -1)
        h_local = self.local_fuse(torch.cat([ht, hs], dim=-1))

        h = h_local.reshape(B * N, -1)
        edges, ew = self._expanded_graph(edge_index, edge_weight, B, N)
        h_spatial = F.relu(self.conv1(h, edges, edge_weight=ew))
        h_spatial = self.dropout(h_spatial)
        h_spatial = self.conv2(h_spatial, edges, edge_weight=ew)
        h_spatial = h_spatial.reshape(B, N, -1)
        h_spatial = self.spatial_norm(h_spatial)

        gate = torch.sigmoid(self.graph_gate_logit)
        h_fused = h_local + gate * h_spatial
        return self.head(h_fused)

    def graph_gate(self):
        return float(torch.sigmoid(self.graph_gate_logit.detach()).cpu())
