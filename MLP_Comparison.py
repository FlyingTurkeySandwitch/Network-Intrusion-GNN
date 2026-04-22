"""
=============================================================================
Edge-Only MLP Ablation
=============================================================================
Ablation baseline for the GNN anomaly detection pipeline.

Classifies edges as benign (0) or hostile (1) using ONLY the 7 raw edge
features — no graph structure, no node embeddings, no neighbourhood
aggregation.

Identical to the GNN script in:
  - Data loading and feature extraction
  - Inductive node-disjoint train/val/test split
  - Loss function and class imbalance handling
  - Evaluation metrics

The only thing removed is the GNN encoder. If this MLP matches the GNN's
test AUC/F1, graph structure is not contributing and the edge features alone
are sufficient. If the GNN meaningfully outperforms, neighbourhood context
is providing genuine signal.

Run both scripts and compare [Test Results] side by side.

Dependencies:
    pip install torch networkx scikit-learn pandas numpy
=============================================================================
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data


# =============================================================================
# SECTION 1: Configuration
# =============================================================================

CONFIG = {
    # Paths — keep identical to GNN script so both run on the same file
    "graphml_path":    "Network-Intrusion-GNN/data/0.1M-Stratified-Multi.graphml",
    "checkpoint_path": "Network-Intrusion-GNN/models/mlp_ablation_model.pt",

    # Must be identical to GNN script for a fair comparison
    "edge_feature_keys": [
        "TotPkts",
        "TotBytes",
        "SrcBytes",
        "Dur",
        "Proto_encoded",
        "Dir_encoded",
        "State_encoded",
    ],
    "edge_label_key": "ActivityLabel",

    # MLP architecture — matched in capacity to the GNN's MLP head
    # (the GNN head receives 2*hidden_dim + edge_feat_dim = 135 inputs;
    #  here we give the MLP a comparable first layer width)
    "hidden_dim":  128,
    "num_layers":  3,
    "dropout":     0.3,

    # Training — identical to GNN script
    "epochs":       100,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,

    # Split ratios — must be identical to GNN script
    "val_ratio":  0.15,
    "test_ratio": 0.15,

    "pos_weight": None,  # None = auto-computed

    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# =============================================================================
# SECTION 2: Data Loading & Parsing
# (identical to GNN script)
# =============================================================================

def load_graphml(path: str) -> nx.Graph:
    G = nx.read_graphml(path)
    print(f"[Data] Loaded graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def inspect_graph(G: nx.Graph):
    sample_edge = next(iter(G.edges(data=True)))
    print(f"[Data] Sample edge attributes: {sample_edge[2]}")
    labels = [d.get(CONFIG["edge_label_key"], -1)
              for _, _, d in G.edges(data=True)]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[Data] Label distribution: {dict(zip(unique, counts))}")


# =============================================================================
# SECTION 3: Feature Extraction
# (edge features only — node aggregation is intentionally excluded)
# =============================================================================

def build_edge_dataset(G: nx.Graph) -> tuple:
    """
    Extract raw edge features and labels directly from the graph.

    Unlike the GNN script, we do NOT aggregate node features — the MLP
    sees only the 7 edge-level attributes per connection. This is the
    strict ablation condition: zero graph structure information.

    Returns:
        edge_attr:   np.ndarray [num_edges, num_edge_features]
        edge_labels: np.ndarray [num_edges]
        edge_index:  torch.LongTensor [2, num_edges]  (for split logic)
        node_to_idx: dict  (for split logic)
    """
    feat_keys   = CONFIG["edge_feature_keys"]
    label_key   = CONFIG["edge_label_key"]
    node_to_idx = {n: i for i, n in enumerate(G.nodes())}

    src_list, dst_list, feat_list, label_list = [], [], [], []

    for u, v, data in G.edges(data=True):
        src_list.append(node_to_idx[u])
        dst_list.append(node_to_idx[v])
        feat_list.append([float(data.get(k, 0.0)) for k in feat_keys])
        label_list.append(int(data.get(label_key, 0)))

    edge_attr   = np.array(feat_list,  dtype=np.float32)
    edge_labels = np.array(label_list, dtype=np.int64)
    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Normalise edge features with StandardScaler
    scaler    = StandardScaler()
    edge_attr = scaler.fit_transform(edge_attr)

    print(f"[Data] Edge feature matrix: {edge_attr.shape}  "
          f"(no node features — ablation condition)")

    return edge_attr, edge_labels, edge_index, node_to_idx, scaler


# =============================================================================
# SECTION 4: Inductive Node-Disjoint Split
# (identical logic to GNN script — same seed, same ratios)
# =============================================================================

def _node_hostility_ratio(G: nx.Graph, node_to_idx: dict) -> np.ndarray:
    """
    Per-node hostile edge fraction, used to stratify the node split.
    Identical to GNN script.
    """
    label_key   = CONFIG["edge_label_key"]
    num_nodes   = len(node_to_idx)
    hostile_cnt = np.zeros(num_nodes, dtype=np.float32)
    total_cnt   = np.zeros(num_nodes, dtype=np.float32)

    for u, v, data in G.edges(data=True):
        label = int(data.get(label_key, 0))
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        hostile_cnt[u_idx] += label
        hostile_cnt[v_idx] += label
        total_cnt[u_idx]   += 1
        total_cnt[v_idx]   += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(total_cnt > 0, hostile_cnt / total_cnt, 0.0)
    return ratios


def _stratify_bins(ratios: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin hostility ratios into stratification labels. Identical to GNN script."""
    bins = np.digitize(
        ratios, bins=np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    )
    unique, counts = np.unique(bins, return_counts=True)
    for r in unique[counts < 2]:
        bins[bins == r] = 0
    return bins


def split_nodes_inductively(edge_index:   torch.Tensor,
                             edge_labels:  np.ndarray,
                             G:            nx.Graph,
                             node_to_idx:  dict) -> tuple:
    """
    Identical node-disjoint split to the GNN script.

    Returns boolean masks over edges:
        train_mask, val_mask, test_mask
    """
    val_ratio  = CONFIG["val_ratio"]
    test_ratio = CONFIG["test_ratio"]

    all_nodes = np.array(list(node_to_idx.keys()))
    ratios    = _node_hostility_ratio(G, node_to_idx)
    strata    = _stratify_bins(ratios)

    train_nodes, temp_nodes = train_test_split(
        all_nodes,
        test_size    = val_ratio + test_ratio,
        stratify     = strata,
        random_state = CONFIG["seed"],
    )
    temp_indices = np.isin(all_nodes, temp_nodes)
    temp_strata  = _stratify_bins(ratios[temp_indices])

    val_nodes, test_nodes = train_test_split(
        temp_nodes,
        test_size    = test_ratio / (val_ratio + test_ratio),
        stratify     = temp_strata,
        random_state = CONFIG["seed"],
    )

    train_idx_set = set(node_to_idx[n] for n in train_nodes)
    val_idx_set   = set(node_to_idx[n] for n in val_nodes)
    test_idx_set  = set(node_to_idx[n] for n in test_nodes)

    src       = edge_index[0].tolist()
    dst       = edge_index[1].tolist()
    num_edges = len(src)

    train_mask = np.zeros(num_edges, dtype=bool)
    val_mask   = np.zeros(num_edges, dtype=bool)
    test_mask  = np.zeros(num_edges, dtype=bool)

    for i, (s, d) in enumerate(zip(src, dst)):
        train_mask[i] = (s in train_idx_set) and (d in train_idx_set)
        val_mask[i]   = ((s in val_idx_set) or (d in val_idx_set)) \
                        and (s not in test_idx_set) \
                        and (d not in test_idx_set)
        test_mask[i]  = (s in test_idx_set) or (d in test_idx_set)

    print(f"[Split] Nodes  — train: {len(train_idx_set)} | "
          f"val: {len(val_idx_set)} | test: {len(test_idx_set)}")
    print(f"[Split] Edges  — train: {train_mask.sum()} | "
          f"val: {val_mask.sum()} | test: {test_mask.sum()}")

    return train_mask, val_mask, test_mask


# =============================================================================
# SECTION 5: Model — Edge-Only MLP
# =============================================================================

class EdgeMLP(nn.Module):
    """
    Feedforward MLP that classifies edges from raw edge features alone.

    Input:  edge_attr vector  [batch, num_edge_features]
    Output: scalar logit per edge  [batch]

    No graph structure. No node embeddings. No message passing.
    This is the null hypothesis: can the task be solved without a GNN?
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()

        layers = []
        current_dim = in_dim

        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr).squeeze(-1)  # [batch]


# =============================================================================
# SECTION 6: Training Loop
# =============================================================================

def compute_pos_weight(labels: np.ndarray) -> float:
    """pos_weight = (#benign) / (#hostile) from training labels."""
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    pw    = n_neg / max(n_pos, 1)
    print(f"[Train] pos_weight={pw:.2f}  (benign={n_neg}, hostile={n_pos})")
    return float(pw)


def make_loader(edge_attr:   np.ndarray,
                edge_labels: np.ndarray,
                mask:        np.ndarray,
                shuffle:     bool = False) -> DataLoader:
    """Wrap a masked slice of edge features/labels into a DataLoader."""
    x = torch.tensor(edge_attr[mask],   dtype=torch.float)
    y = torch.tensor(edge_labels[mask], dtype=torch.float)
    return DataLoader(
        TensorDataset(x, y),
        batch_size = CONFIG["batch_size"],
        shuffle    = shuffle,
    )


def train_one_epoch(model:     EdgeMLP,
                    loader:    DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device) -> float:
    model.train()
    total_loss    = 0.0
    total_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model:  EdgeMLP,
             loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, zero_division=0)
    return {"auc": auc, "f1": f1, "probs": probs,
            "preds": preds, "labels": labels}


def train(edge_attr:   np.ndarray,
          edge_labels: np.ndarray,
          train_mask:  np.ndarray,
          val_mask:    np.ndarray,
          device:      torch.device) -> EdgeMLP:

    in_dim = edge_attr.shape[1]
    model  = EdgeMLP(
        in_dim     = in_dim,
        hidden_dim = CONFIG["hidden_dim"],
        num_layers = CONFIG["num_layers"],
        dropout    = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    pw        = CONFIG["pos_weight"] or compute_pos_weight(edge_labels[train_mask])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float).to(device)
    )

    train_loader = make_loader(edge_attr, edge_labels, train_mask, shuffle=True)
    val_loader   = make_loader(edge_attr, edge_labels, val_mask,   shuffle=False)

    best_val_auc = 0.0
    for epoch in range(1, CONFIG["epochs"] + 1):
        loss        = train_one_epoch(model, train_loader,
                                      optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1:  {val_metrics['f1']:.4f}")

    print(f"[Train] Best Val AUC: {best_val_auc:.4f}")
    return model


# =============================================================================
# SECTION 7: Test Evaluation
# =============================================================================

def test(model:       EdgeMLP,
         edge_attr:   np.ndarray,
         edge_labels: np.ndarray,
         test_mask:   np.ndarray,
         device:      torch.device):
    """Load best checkpoint and report test-set metrics."""
    model.load_state_dict(
        torch.load(CONFIG["checkpoint_path"], map_location=device)
    )
    test_loader = make_loader(edge_attr, edge_labels, test_mask, shuffle=False)
    metrics     = evaluate(model, test_loader, device)

    print("\n[Test Results — Edge-Only MLP Ablation]")
    print(f"  AUC-ROC : {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(classification_report(
        metrics["labels"], metrics["preds"],
        target_names=["Benign", "Hostile"],
    ))
    print("\n  Compare these numbers directly against the GNN test results.")
    print("  If AUC/F1 are similar, edge features alone explain the task.")
    print("  If the GNN is higher, graph structure is providing real signal.")


# =============================================================================
# SECTION 8: Main Entry Point
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using device: {device}")

    # --- Load ---
    G = load_graphml(CONFIG["graphml_path"])
    inspect_graph(G)

    # --- Edge features only (no node aggregation) ---
    edge_attr, edge_labels, edge_index, node_to_idx, scaler = \
        build_edge_dataset(G)

    # --- Same inductive node-disjoint split as GNN script ---
    train_mask, val_mask, test_mask = split_nodes_inductively(
        edge_index, edge_labels, G, node_to_idx
    )

    # --- Train ---
    model = train(edge_attr, edge_labels, train_mask, val_mask, device)

    # --- Test ---
    test(model, edge_attr, edge_labels, test_mask, device)


if __name__ == "__main__":
    main()