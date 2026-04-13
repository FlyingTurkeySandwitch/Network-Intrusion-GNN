"""
GNN-Based Anomaly Detection for Network Activity Graphs
========================================================
Pipeline Outline:
  1. Data Loading & Parsing
  2. Feature Engineering
  3. Graph Construction (PyG Data objects)
  4. Train/Val/Test Split
  5. Model Definition (GraphSAGE / GAT encoder + anomaly head)
  6. Training Loop
  7. Evaluation & Threshold Tuning
  8. Inference / Scoring

Dependencies:
    pip install torch torch-geometric networkx scikit-learn pandas numpy
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, GATv2Conv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             average_precision_score, classification_report)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Paths
    GRAPHML_PATH   = "Network-Intrusion-GNN/data/0.1M-Stratified-Multi.graphml"
    MODEL_SAVE_PATH = "Network-Intrusion-GNN/models/gnn_anomaly_model_1.pt"

    # Edge feature keys (from GraphML schema)
    EDGE_FEATURES = [
        "TotPkts",       # d0
        "TotBytes",      # d1
        "SrcBytes",      # d2
        "Dur",           # d3
        "Proto_encoded", # d4
        "Dir_encoded",   # d5
        "State_encoded", # d6
    ]
    LABEL_KEY = "ActivityLabel"  # d7  (0 = normal, 1 = anomaly)

    # Model hyperparameters
    NODE_FEAT_DIM   = 16    # learned node embedding dimension
    EDGE_FEAT_DIM   = len(EDGE_FEATURES)
    HIDDEN_DIM      = 64
    NUM_LAYERS      = 3
    DROPOUT         = 0.3
    HEADS           = 4

    # Training
    EPOCHS          = 1000
    BATCH_SIZE      = 32      # graphs per batch (for multi-graph settings)
    LR              = 1e-3
    WEIGHT_DECAY    = 1e-4
    CLASS_WEIGHT    = 17.0    # upweight anomaly class (imbalanced labels)

    # Anomaly threshold
    THRESHOLD       = 0.5     # tuned on validation set via F1-maximisation

    SEED            = 42
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
torch.manual_seed(cfg.SEED)


# ─────────────────────────────────────────────
# 2. DATA LOADING & PARSING
# ─────────────────────────────────────────────
def load_graphml(path: str) -> nx.Graph:
    """
    Load the .graphml file into a NetworkX graph.
    Nodes  → endpoint IPs
    Edges  → network connections with numerical features + label
    """
    G = nx.read_graphml(path)
    print(f"[Data] Loaded graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def inspect_graph(G: nx.Graph):
    """Quick sanity check: label distribution, missing values, feature ranges."""
    labels = [d.get(cfg.LABEL_KEY, np.nan) for _, _, d in G.edges(data=True)]
    labels = np.array(labels, dtype=float)

    print(f"[Inspect] Label distribution — "
          f"Normal: {(labels == 0).sum()}  "
          f"Anomaly: {(labels == 1).sum()}  "
          f"Unknown/NaN: {np.isnan(labels).sum()}")

    for feat in cfg.EDGE_FEATURES:
        vals = [d.get(feat, np.nan) for _, _, d in G.edges(data=True)]
        arr  = np.array(vals, dtype=float)
        print(f"  {feat:20s}  mean={np.nanmean(arr):.4f}  "
              f"std={np.nanstd(arr):.4f}  "
              f"missing={np.isnan(arr).sum()}")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_node_index(G: nx.Graph) -> dict:
    """Map node IDs (IP strings) → integer indices."""
    return {node: idx for idx, node in enumerate(G.nodes())}


def build_node_features(G: nx.Graph, node_index: dict) -> torch.Tensor:
    """
    Construct initial node feature matrix.

    Strategy options (choose one or combine):
      A. Degree-based stats: in-degree, out-degree, weighted degree
      B. Aggregated edge stats: mean/std of neighbour edge features
      C. Learnable embedding: start with random init, learned end-to-end
      D. Structural: PageRank, betweenness centrality (expensive on large graphs)

    Here we implement (A) + (B) as a starting point.
    """
    n = len(node_index)

    # --- A: degree features ---
    degree = dict(G.degree())
    # For directed graphs use in_degree / out_degree separately

    # --- B: aggregated edge features per node ---
    agg = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        feats = [data.get(f, 0.0) for f in cfg.EDGE_FEATURES]
        agg[node_index[u]].append(feats)
        agg[node_index[v]].append(feats)  # undirected; omit for directed

    node_feats = []
    for i in range(n):
        deg_feat  = [degree.get(list(G.nodes())[i], 0)]
        edge_agg  = np.mean(agg[i], axis=0) if agg[i] else np.zeros(cfg.EDGE_FEAT_DIM)
        node_feats.append(deg_feat + list(edge_agg))

    X = torch.tensor(node_feats, dtype=torch.float)
    print(f"[Features] Node feature matrix: {X.shape}")
    return X


def build_edge_features_and_labels(
    G: nx.Graph, node_index: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        edge_index  : (2, E) long tensor
        edge_attr   : (E, F) float tensor  — normalised edge features
        edge_labels : (E,)   long tensor   — 0=normal, 1=anomaly
    """
    src_list, dst_list, feat_list, label_list = [], [], [], []

    for u, v, data in G.edges(data=True):
        src_list.append(node_index[u])
        dst_list.append(node_index[v])
        feat_list.append([data.get(f, 0.0) for f in cfg.EDGE_FEATURES])
        label_list.append(int(data.get(cfg.LABEL_KEY, 0)))

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr   = torch.tensor(feat_list,  dtype=torch.float)
    edge_labels = torch.tensor(label_list, dtype=torch.long)

    # Normalise edge features (fit scaler on train split only — see section 4)
    print(f"[Features] Edge feature matrix : {edge_attr.shape}")
    print(f"[Features] Edge labels         : {edge_labels.shape}")
    return edge_index, edge_attr, edge_labels


def normalise_edge_features(
    edge_attr: torch.Tensor,
    train_mask: torch.Tensor
) -> tuple[torch.Tensor, StandardScaler]:
    """Fit StandardScaler on training edges, transform all edges."""
    scaler = StandardScaler()
    train_feats = edge_attr[train_mask].numpy()
    scaler.fit(train_feats)
    normed = scaler.transform(edge_attr.numpy())
    return torch.tensor(normed, dtype=torch.float), scaler


# ─────────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
def split_edges(
    num_edges: int,
    edge_labels: torch.Tensor,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratified split on edges (to preserve label ratio).
    Returns boolean masks for train / val / test.
    """
    indices = np.arange(num_edges)
    y       = edge_labels.numpy()

    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=y, random_state=cfg.SEED)
    train_idx, val_idx  = train_test_split(
        train_idx, test_size=val_ratio / (1 - test_ratio),
        stratify=y[train_idx], random_state=cfg.SEED)

    def to_mask(idx):
        m = torch.zeros(num_edges, dtype=torch.bool)
        m[idx] = True
        return m

    train_mask = to_mask(train_idx)
    val_mask   = to_mask(val_idx)
    test_mask  = to_mask(test_idx)

    print(f"[Split] Train: {train_mask.sum()}  "
          f"Val: {val_mask.sum()}  Test: {test_mask.sum()}")
    return train_mask, val_mask, test_mask


# ─────────────────────────────────────────────
# 5. MODEL DEFINITION
# ─────────────────────────────────────────────
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden, num_layers,
                 edge_dim, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATv2Conv(in_channels, hidden // heads,
                                    heads=heads, edge_dim=edge_dim,
                                    dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden, hidden // heads,
                                        heads=heads, edge_dim=edge_dim,
                                        dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden))

        self.convs.append(GATv2Conv(hidden, hidden, heads=1,
                                    concat=False, edge_dim=edge_dim,
                                    dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):  # edge_attr added here
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeAnomalyClassifier(nn.Module):
    """
    Given node embeddings z_u, z_v and edge features e_uv,
    classify each edge as normal (0) or anomalous (1).

    Input per edge: [z_u || z_v || z_u * z_v || e_uv]
    """
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden: int):
        super().__init__()
        in_dim = node_emb_dim * 3 + edge_feat_dim  # concat + Hadamard
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # binary logit
        )

    def forward(
        self,
        z: torch.Tensor,          # (N, emb_dim)
        edge_index: torch.Tensor, # (2, E)
        edge_attr: torch.Tensor,  # (E, F)
    ) -> torch.Tensor:            # (E,) logits
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        edge_repr = torch.cat([z_src, z_dst, z_src * z_dst, edge_attr], dim=-1)
        return self.mlp(edge_repr).squeeze(-1)


class GNNAnomalyDetector(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_layers, dropout, heads):
        super().__init__()
        self.encoder = GNNEncoder(
            in_channels=node_in,
            hidden=hidden,
            num_layers=num_layers,
            edge_dim=edge_in,        # pass edge_in through
            heads=heads,
            dropout=dropout
        )
        self.classifier = EdgeAnomalyClassifier(hidden, edge_in, hidden)

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)  # edge_attr flows in
        logits = self.classifier(z, edge_index, edge_attr)
        return logits


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
def build_pyg_data(
    node_features: torch.Tensor,
    edge_index:    torch.Tensor,
    edge_attr:     torch.Tensor,
    edge_labels:   torch.Tensor,
) -> Data:
    """Wrap everything into a single PyG Data object."""
    return Data(
        x          = node_features,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = edge_labels,
    )

def compute_loss(
    logits:      torch.Tensor,
    labels:      torch.Tensor,
    mask:        torch.Tensor,
    pos_weight:  float = cfg.CLASS_WEIGHT,
) -> torch.Tensor:
    """Binary cross-entropy with positive class weighting for imbalance."""
    weight = torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask].float(), pos_weight=weight
    )

def train_epoch(
    model:      nn.Module,
    data:       Data,
    train_mask: torch.Tensor,
    optimiser:  torch.optim.Optimizer,
) -> float:
    model.train()
    optimiser.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_attr)
    loss   = compute_loss(logits, data.y, train_mask)
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data:  Data,
    mask:  torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_attr)
    probs  = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    preds  = (probs >= threshold).astype(int)
    roc    = roc_auc_score(labels, probs) if labels.sum() > 0 else float("nan")
    ap     = average_precision_score(labels, probs) if labels.sum() > 0 else float("nan")

    return {"roc_auc": roc, "avg_precision": ap,
            "probs": probs, "labels": labels, "preds": preds}


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Choose threshold that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx   = np.argmax(f1_scores[:-1])
    best_t     = thresholds[best_idx]
    print(f"[Threshold] Best F1={f1_scores[best_idx]:.4f} at t={best_t:.4f}")
    return float(best_t)


def train(
    model:      nn.Module,
    data:       Data,
    train_mask: torch.Tensor,
    val_mask:   torch.Tensor,
) -> nn.Module:
    """Full training loop with early stopping on val ROC-AUC."""
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.EPOCHS)

    best_auc   = 0.0
    patience   = 15
    no_improve = 0

    data  = data.to(cfg.DEVICE)
    model = model.to(cfg.DEVICE)
    t_mask = train_mask.to(cfg.DEVICE)
    v_mask = val_mask.to(cfg.DEVICE)

    for epoch in range(1, cfg.EPOCHS + 1):
        loss = train_epoch(model, data, t_mask, optimiser)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, data, v_mask)
            auc = val_metrics["roc_auc"]
            print(f"Epoch {epoch:04d} | loss={loss:.4f} | "
                  f"val_roc_auc={auc:.4f} | val_ap={val_metrics['avg_precision']:.4f}")

            if auc > best_auc:
                best_auc   = auc
                no_improve = 0
                torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
                print(f"  ✓ Saved best model (auc={best_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[Train] Early stopping at epoch {epoch}")
                    break

    # Reload best weights
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH,
                                     map_location=cfg.DEVICE))
    return model


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def full_evaluation(
    model:     nn.Module,
    data:      Data,
    val_mask:  torch.Tensor,
    test_mask: torch.Tensor,
):
    data  = data.to(cfg.DEVICE)
    model = model.to(cfg.DEVICE)

    # Tune threshold on validation set
    val_metrics  = evaluate(model, data, val_mask.to(cfg.DEVICE))
    best_t       = find_best_threshold(val_metrics["probs"], val_metrics["labels"])

    # Final test evaluation
    test_metrics = evaluate(model, data, test_mask.to(cfg.DEVICE), threshold=best_t)
    print("\n[Test Results]")
    print(f"  ROC-AUC          : {test_metrics['roc_auc']:.4f}")
    print(f"  Avg Precision    : {test_metrics['avg_precision']:.4f}")
    print(classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["Normal", "Anomaly"]))

    return best_t


# ─────────────────────────────────────────────
# 8. INFERENCE / SCORING
# ─────────────────────────────────────────────
@torch.no_grad()
def score_edges(
    model:     nn.Module,
    data:      Data,
    threshold: float,
    G:         nx.Graph,
) -> pd.DataFrame:
    """
    Return a DataFrame of all edges with their anomaly probability scores.
    Useful for explainability, dashboards, or downstream alerting.
    """
    model.eval()
    data   = data.to(cfg.DEVICE)
    logits = model(data.x, data.edge_index, data.edge_attr)
    probs  = torch.sigmoid(logits).cpu().numpy()

    edges = list(G.edges(data=True))
    records = []
    for i, (u, v, d) in enumerate(edges):
        records.append({
            "src"          : u,
            "dst"          : v,
            "anomaly_prob" : probs[i],
            "predicted"    : int(probs[i] >= threshold),
            "true_label"   : int(d.get(cfg.LABEL_KEY, -1)),
        })

    df = pd.DataFrame(records).sort_values("anomaly_prob", ascending=False)
    print(f"\n[Inference] Top anomalous edges:")
    print(df[df["predicted"] == 1].head(10).to_string(index=False))
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── 1. Load ──────────────────────────────
    G = load_graphml(cfg.GRAPHML_PATH)
    inspect_graph(G)

    # ── 2. Feature engineering ───────────────
    node_index  = build_node_index(G)
    node_feats  = build_node_features(G, node_index)
    edge_index, edge_attr, edge_labels = build_edge_features_and_labels(
        G, node_index)

    # ── 3. Split ─────────────────────────────
    train_mask, val_mask, test_mask = split_edges(
        edge_labels.shape[0], edge_labels)

    # ── 4. Normalise edge features ───────────
    edge_attr, scaler = normalise_edge_features(edge_attr, train_mask)

    # ── 5. Build PyG Data object ─────────────
    data = build_pyg_data(node_feats, edge_index, edge_attr, edge_labels)

    # ── 6. Instantiate model ─────────────────
    node_in_dim = node_feats.shape[1]
    model = GNNAnomalyDetector(
        node_in   = node_in_dim,
        edge_in   = cfg.EDGE_FEAT_DIM,
        hidden    = cfg.HIDDEN_DIM,
        num_layers= cfg.NUM_LAYERS,
        dropout   = cfg.DROPOUT,
        heads     = cfg.HEADS
    )
    print(f"\n[Model] Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── 7. Train ─────────────────────────────
    model = train(model, data, train_mask, val_mask)

    # ── 8. Evaluate ──────────────────────────
    threshold = full_evaluation(model, data, val_mask, test_mask)

    # ── 9. Score all edges ───────────────────
    results_df = score_edges(model, data, threshold, G)
    results_df.to_csv("edge_anomaly_scores.csv", index=False)
    print("\n[Done] Scores saved to edge_anomaly_scores.csv")


if __name__ == "__main__":
    main()