"""
GNN-Based Anomaly Detection for Network Activity Graphs
========================================================
Inductive pipeline — test nodes are fully held out during training.
The model never sees test node embeddings during message passing,
ensuring evaluation reflects real-world generalisation to new endpoints.

Pipeline Outline:
  1. Data Loading & Parsing
  2. Inductive Node Split (train nodes vs test nodes)
  3. Feature Engineering (per subgraph, scaler fit on train only)
  4. Graph Construction (separate PyG Data objects per split)
  5. Model Definition (GraphSAGE encoder + edge anomaly head)
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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge

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
    MODEL_SAVE_PATH = "Network-Intrusion-GNN/models/gnn_inductive_anomaly_model_1.pt"

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

    # Training
    EPOCHS          = 500
    BATCH_SIZE      = 32
    LR              = 1e-3
    WEIGHT_DECAY    = 1e-4
    CLASS_WEIGHT    = 10.0    # tuned for 5% anomaly imbalance ratio

    # Inductive split — fraction of nodes held out entirely for test
    TEST_NODE_RATIO = 0.20    # 20% of nodes (and all their edges) unseen at train
    VAL_RATIO       = 0.15    # fraction of train edges used for validation

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
# 3. INDUCTIVE SPLIT
# ─────────────────────────────────────────────
def inductive_split(
    G: nx.Graph,
    test_node_ratio:  float = cfg.TEST_NODE_RATIO,
    calib_node_ratio: float = 0.05,
    seed: int = cfg.SEED,
) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    """
    Three-way node partition:
        G_train — seen during training and validation
        G_calib — unseen during training, used only for threshold tuning
        G_test  — fully held out, never touched until final evaluation
    """
    import random
    random.seed(seed)

    nodes   = list(G.nodes())
    n_held  = int(len(nodes) * (test_node_ratio + calib_node_ratio))
    n_calib = int(len(nodes) * calib_node_ratio)

    held_out    = set(random.sample(nodes, n_held))
    calib_nodes = set(random.sample(list(held_out), n_calib))
    test_nodes  = held_out - calib_nodes
    train_nodes = set(nodes) - held_out

    def make_subgraph(node_set, edge_filter_fn):
        H = nx.Graph()
        H.add_nodes_from(node_set)
        H.add_edges_from(
            (u, v, d) for u, v, d in G.edges(data=True) if edge_filter_fn(u, v)
        )
        return H

    G_train = make_subgraph(
        train_nodes,
        lambda u, v: u in train_nodes and v in train_nodes
    )
    G_calib = make_subgraph(
        calib_nodes,
        lambda u, v: u in calib_nodes or  v in calib_nodes
    )
    G_test  = make_subgraph(
        test_nodes,
        lambda u, v: u in test_nodes  or  v in test_nodes
    )

    def anomaly_rate(H):
        labels = [d.get(cfg.LABEL_KEY, 0) for _, _, d in H.edges(data=True)]
        return sum(labels) / len(labels) if labels else 0.0

    print(f"[Inductive Split]")
    print(f"  Train — nodes: {G_train.number_of_nodes():,}  "
          f"edges: {G_train.number_of_edges():,}  "
          f"anomaly rate: {anomaly_rate(G_train):.3f}")
    print(f"  Calib — nodes: {G_calib.number_of_nodes():,}  "
          f"edges: {G_calib.number_of_edges():,}  "
          f"anomaly rate: {anomaly_rate(G_calib):.3f}")
    print(f"  Test  — nodes: {G_test.number_of_nodes():,}  "
          f"edges: {G_test.number_of_edges():,}  "
          f"anomaly rate: {anomaly_rate(G_test):.3f}")

    return G_train, G_calib, G_test


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_node_index(G: nx.Graph) -> dict:
    """Map node IDs (IP strings) → integer indices, local to this subgraph."""
    return {node: idx for idx, node in enumerate(G.nodes())}


def build_node_features(G: nx.Graph, node_index: dict) -> torch.Tensor:
    """
    Node features: degree + mean aggregated edge features per node.
    Computed independently for each subgraph so test nodes have no
    dependency on training graph statistics.
    """
    n      = len(node_index)
    degree = dict(G.degree())
    nodes  = list(G.nodes())

    agg = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        feats = [data.get(f, 0.0) for f in cfg.EDGE_FEATURES]
        agg[node_index[u]].append(feats)
        agg[node_index[v]].append(feats)

    node_feats = []
    for i in range(n):
        deg_feat = [degree.get(nodes[i], 0)]
        edge_agg = np.mean(agg[i], axis=0) if agg[i] else np.zeros(cfg.EDGE_FEAT_DIM)
        node_feats.append(deg_feat + list(edge_agg))

    return torch.tensor(node_feats, dtype=torch.float)


def build_edge_tensors(
    G: nx.Graph, node_index: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build edge_index, edge_attr, edge_labels for a subgraph."""
    src_list, dst_list, feat_list, label_list = [], [], [], []

    for u, v, data in G.edges(data=True):
        src_list.append(node_index[u])
        dst_list.append(node_index[v])
        feat_list.append([data.get(f, 0.0) for f in cfg.EDGE_FEATURES])
        label_list.append(int(data.get(cfg.LABEL_KEY, 0)))

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr   = torch.tensor(feat_list,            dtype=torch.float)
    edge_labels = torch.tensor(label_list,           dtype=torch.long)
    return edge_index, edge_attr, edge_labels


def build_pyg_data(
    node_features: torch.Tensor,
    edge_index:    torch.Tensor,
    edge_attr:     torch.Tensor,
    edge_labels:   torch.Tensor,
) -> Data:
    return Data(x=node_features, edge_index=edge_index,
                edge_attr=edge_attr, y=edge_labels)


def build_train_val_data(
    G_train: nx.Graph,
) -> tuple[Data, Data, torch.Tensor, StandardScaler]:
    """
    Build train and val PyG Data objects from the training subgraph.
    The scaler is fit on train edges only and returned for use on test data.

    Val edges are a random stratified subset of train graph edges —
    they share the same nodes, which is fine since they're all already
    in the training node set. No test nodes are involved.
    """
    node_index = build_node_index(G_train)
    node_feats = build_node_features(G_train, node_index)
    edge_index, edge_attr, edge_labels = build_edge_tensors(G_train, node_index)

    # Stratified edge split within the training subgraph
    indices = np.arange(len(edge_labels))
    y       = edge_labels.numpy()
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.VAL_RATIO, stratify=y, random_state=cfg.SEED)

    train_mask = torch.zeros(len(edge_labels), dtype=torch.bool)
    val_mask   = torch.zeros(len(edge_labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True

    # Fit scaler on training edges only
    scaler = StandardScaler()
    scaler.fit(edge_attr[train_mask].numpy())
    edge_attr_norm = torch.tensor(
        scaler.transform(edge_attr.numpy()), dtype=torch.float)

    # Both splits share the same node set and full graph structure —
    # only the loss mask differs, which is correct for inductive training
    data = build_pyg_data(node_feats, edge_index, edge_attr_norm, edge_labels)

    print(f"[Train/Val] Train edges: {train_mask.sum()}  "
          f"Val edges: {val_mask.sum()}")
    return data, train_mask, val_mask, scaler


def build_test_data(
    G_test: nx.Graph, scaler: StandardScaler
) -> Data:
    """
    Build the test PyG Data object using held-out nodes.
    Edge features are normalised with the scaler fit on training data —
    test data never influences normalisation statistics.
    """
    node_index = build_node_index(G_test)
    node_feats = build_node_features(G_test, node_index)
    edge_index, edge_attr, edge_labels = build_edge_tensors(G_test, node_index)

    edge_attr_norm = torch.tensor(
        scaler.transform(edge_attr.numpy()), dtype=torch.float)

    print(f"[Test] Edges: {edge_labels.shape[0]}  "
          f"Anomalies: {edge_labels.sum().item()}")
    return build_pyg_data(node_feats, edge_index, edge_attr_norm, edge_labels)


# ─────────────────────────────────────────────
# 5. MODEL DEFINITION
# ─────────────────────────────────────────────
class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder producing node embeddings.
    Uses GraphSAGE layers (swap for GATConv for attention-based aggregation).
    """
    def __init__(self, in_channels: int, hidden: int, num_layers: int,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # (N, hidden)


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
    """Full model: GNN encoder + edge classifier."""
    def __init__(self, node_in: int, edge_in: int,
                 hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder    = GNNEncoder(node_in, hidden, num_layers, dropout)
        self.classifier = EdgeAnomalyClassifier(hidden, edge_in, hidden)

    def forward(self, x, edge_index, edge_attr):
        z      = self.encoder(x, edge_index)
        logits = self.classifier(z, edge_index, edge_attr)
        return logits


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
def compute_loss(
    logits:     torch.Tensor,
    labels:     torch.Tensor,
    mask:       torch.Tensor,
    pos_weight: float = cfg.CLASS_WEIGHT,
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

    # Drop 20% of edges — edge_mask is True for edges that survived
    edge_index_dropped, edge_mask = dropout_edge(data.edge_index, p=0.2, training=True)
    edge_attr_dropped = data.edge_attr[edge_mask]

    logits = model(data.x, edge_index_dropped, edge_attr_dropped)

    # Remap train_mask to only the surviving edges
    # train_mask[edge_mask] selects the mask values for edges still present
    surviving_train_mask = train_mask[edge_mask]

    loss = compute_loss(logits, data.y[edge_mask], surviving_train_mask)
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    data:      Data,
    mask:      torch.Tensor | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate on a Data object. If mask is provided, evaluate only those edges.
    If mask is None (e.g. the standalone test graph), evaluate all edges.
    """
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_attr)

    if mask is not None:
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
    else:
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = data.y.cpu().numpy()

    preds = (probs >= threshold).astype(int)
    roc   = roc_auc_score(labels, probs)           if labels.sum() > 0 else float("nan")
    ap    = average_precision_score(labels, probs)  if labels.sum() > 0 else float("nan")

    return {"roc_auc": roc, "avg_precision": ap,
            "probs": probs, "labels": labels, "preds": preds}


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Choose threshold that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx  = np.argmax(f1_scores[:-1])
    best_t    = thresholds[best_idx]
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

    data       = data.to(cfg.DEVICE)
    model      = model.to(cfg.DEVICE)
    t_mask     = train_mask.to(cfg.DEVICE)
    v_mask     = val_mask.to(cfg.DEVICE)

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

    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH,
                                     map_location=cfg.DEVICE))
    return model


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def full_evaluation(
    model:      nn.Module,
    train_data: Data,
    val_mask:   torch.Tensor,
    calib_data: Data,
    test_data:  Data,
) -> float:
    model = model.to(cfg.DEVICE)

    # Val metrics still logged for training diagnostics but NOT used for threshold
    val_metrics = evaluate(model, train_data.to(cfg.DEVICE), val_mask.to(cfg.DEVICE))
    print(f"[Val]  ROC-AUC: {val_metrics['roc_auc']:.4f}  "
          f"AP: {val_metrics['avg_precision']:.4f}")

    # Threshold tuned on calib — unseen nodes, same conditions as test
    calib_metrics = evaluate(model, calib_data.to(cfg.DEVICE), mask=None)
    best_t = find_best_threshold(calib_metrics["probs"], calib_metrics["labels"])

    # Final evaluation on fully held-out test nodes
    test_metrics = evaluate(model, test_data.to(cfg.DEVICE),
                            mask=None, threshold=best_t)
    print("\n[Test Results — Inductive (unseen nodes)]")
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
    test_data: Data,
    threshold: float,
    G_test:    nx.Graph,
) -> pd.DataFrame:
    """
    Score all edges in the test subgraph with anomaly probabilities.
    G_test is used only to recover human-readable IP addresses for the output.
    """
    model.eval()
    test_data = test_data.to(cfg.DEVICE)
    logits    = model(test_data.x, test_data.edge_index, test_data.edge_attr)
    probs     = torch.sigmoid(logits).cpu().numpy()

    edges = list(G_test.edges(data=True))
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
    print(f"\n[Inference] Top anomalous edges (unseen nodes):")
    print(df[df["predicted"] == 1].head(10).to_string(index=False))
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── 1. Load ──────────────────────────────
    G = load_graphml(cfg.GRAPHML_PATH)
    inspect_graph(G)
 
    # ── 2. Inductive split ───────────────────
    # Three-way node partition:
    #   G_train — seen during training and val
    #   G_calib — unseen during training, used only for threshold tuning
    #   G_test  — fully held out until final evaluation
    G_train, G_calib, G_test = inductive_split(G)
 
    # ── 3. Build train/val data ──────────────
    # Scaler fit on training edges only, returned for reuse on unseen subgraphs
    train_data, train_mask, val_mask, scaler = build_train_val_data(G_train)
 
    # ── 4. Build calib and test data ─────────
    # Both use the training scaler — no unseen-node statistics leak in
    calib_data = build_test_data(G_calib, scaler)
    test_data  = build_test_data(G_test,  scaler)
 
    # ── 5. Instantiate model ─────────────────
    node_in_dim = train_data.x.shape[1]
    model = GNNAnomalyDetector(
        node_in    = node_in_dim,
        edge_in    = cfg.EDGE_FEAT_DIM,
        hidden     = cfg.HIDDEN_DIM,
        num_layers = cfg.NUM_LAYERS,
        dropout    = cfg.DROPOUT,
    )
    print(f"\n[Model] Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
 
    # ── 6. Train ─────────────────────────────
    model = train(model, train_data, train_mask, val_mask)
 
    # ── 7. Evaluate ──────────────────────────
    # Threshold tuned on calib (unseen nodes), applied to test (unseen nodes)
    threshold = full_evaluation(model, train_data, val_mask, calib_data, test_data)
    #threshold = 0.5
 
    # ── 8. Score test edges ──────────────────
    results_df = score_edges(model, test_data, threshold, G_test)
    results_df.to_csv("edge_anomaly_scores.csv", index=False)
    print("\n[Done] Scores saved to edge_anomaly_scores.csv")

if __name__ == "__main__":
    main()