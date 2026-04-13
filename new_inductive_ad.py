"""
=============================================================================
GNN-Based Anomaly Detection on Network Activity Graphs
=============================================================================
Inductive node/edge classification pipeline using PyTorch Geometric (PyG).
Goal: classify edges as benign (0) or hostile (1) on graphs containing
      nodes never seen during training.

Pipeline Overview:
  1.  Configuration
  2.  Data Loading & Parsing       — Read .graphml, extract features & labels
  3.  Feature Engineering          — Node feature aggregation from edge attrs
  4.  Dataset Construction         — Build PyG Data object
  5.  Inductive Train/Val/Test Split— Node-disjoint subgraph splitting
  6.  Model Definition             — Inductive GNN encoder (GraphSAGE)
  7.  Edge Classification Head     — MLP on concatenated node embeddings
  8.  Training Loop                — NeighborLoader, loss, class imbalance
  9.  Evaluation
  10. Inductive Inference          — Predict on new nodes/edges at runtime
  11. Main Entry Point

Dependencies:
    pip install torch torch-geometric networkx scikit-learn pandas numpy
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
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

CONFIG = {
    # Paths
    "graphml_path":    "Network-Intrusion-GNN/data/0.1M-Stratified-Multi.graphml",
    "checkpoint_path": "Network-Intrusion-GNN/models/new_ad_model.py",

    # Edge feature keys (must match .graphml <key> attr.name values)
    "edge_feature_keys": [
        "TotPkts",        # d0
        "TotBytes",       # d1
        "SrcBytes",       # d2
        "Dur",            # d3
        "Proto_encoded",  # d4
        "Dir_encoded",    # d5
        "State_encoded",  # d6
    ],
    "edge_label_key": "ActivityLabel",  # d7: 0=benign, 1=hostile

    # Model hyperparameters
    "hidden_dim":  64,
    "num_layers":  3,
    "dropout":     0.3,

    # Training
    "epochs":        100,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "batch_size":    512,
    "num_neighbors": [15, 10, 5],  # per-layer neighbor sampling

    # Split ratios (node-level)
    "val_ratio":  0.15,
    "test_ratio": 0.15,

    # Class imbalance: None = auto-computed from training edges
    "pos_weight": None,

    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# =============================================================================
# SECTION 2: Data Loading & Parsing
# =============================================================================

def load_graphml(path: str) -> nx.Graph:
    """
    Load a .graphml file into a NetworkX graph.
    Nodes are endpoints (IP addresses), edges are connections.
    """
    G = nx.read_graphml(path)
    print(f"[Data] Loaded graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def inspect_graph(G: nx.Graph):
    """Print a summary of edge attributes and label distribution."""
    sample_edge = next(iter(G.edges(data=True)))
    print(f"[Data] Sample edge attributes: {sample_edge[2]}")

    labels = [d.get(CONFIG["edge_label_key"], -1)
              for _, _, d in G.edges(data=True)]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[Data] Label distribution: {dict(zip(unique, counts))}")


# =============================================================================
# SECTION 3: Feature Engineering
# =============================================================================

def build_node_features(G: nx.Graph) -> dict:
    """
    Derive node-level feature vectors by aggregating the numeric attributes
    of each node's incident edges (mean, std, max per feature).

    Raw nodes carry no attributes of their own, so this aggregation is the
    only source of node identity information.

    Returns:
        node_feat_dict: {node_id: np.ndarray of shape (3 * num_edge_features,)}
    """
    node_ids  = list(G.nodes())
    feat_keys = CONFIG["edge_feature_keys"]

    node_edge_feats = {n: [] for n in node_ids}
    for u, v, data in G.edges(data=True):
        feat_vec = [float(data.get(k, 0.0)) for k in feat_keys]
        node_edge_feats[u].append(feat_vec)
        node_edge_feats[v].append(feat_vec)

    node_feat_dict = {}
    for node in node_ids:
        feats = node_edge_feats[node]
        if feats:
            arr = np.array(feats)
            agg = np.concatenate([arr.mean(0), arr.std(0), arr.max(0)])
        else:
            agg = np.zeros(3 * len(feat_keys))
        node_feat_dict[node] = agg

    return node_feat_dict


def build_edge_features_and_labels(G: nx.Graph,
                                   node_to_idx: dict) -> tuple:
    """
    Extract per-edge feature matrix and binary labels aligned to node_to_idx.

    Returns:
        edge_index:  torch.LongTensor [2, num_edges]
        edge_attr:   torch.FloatTensor [num_edges, num_edge_features]
        edge_labels: torch.LongTensor [num_edges]
    """
    feat_keys = CONFIG["edge_feature_keys"]
    label_key = CONFIG["edge_label_key"]

    src_list, dst_list, feat_list, label_list = [], [], [], []

    for u, v, data in G.edges(data=True):
        src_list.append(node_to_idx[u])
        dst_list.append(node_to_idx[v])
        feat_list.append([float(data.get(k, 0.0)) for k in feat_keys])
        label_list.append(int(data.get(label_key, 0)))

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr   = torch.tensor(feat_list, dtype=torch.float)
    edge_labels = torch.tensor(label_list, dtype=torch.long)

    return edge_index, edge_attr, edge_labels


# =============================================================================
# SECTION 4: Dataset Construction (PyG Data object)
# =============================================================================

def build_pyg_data(G: nx.Graph) -> tuple:
    """
    Assemble a PyTorch Geometric Data object:
      x          — node feature matrix  [num_nodes, node_feat_dim]
      edge_index — COO connectivity     [2, num_edges]
      edge_attr  — edge feature matrix  [num_edges, num_edge_features]
      edge_y     — edge labels          [num_edges]  0=benign, 1=hostile

    Returns:
        data, scaler, node_to_idx
    """
    node_to_idx    = {n: i for i, n in enumerate(G.nodes())}
    node_feat_dict = build_node_features(G)
    edge_index, edge_attr, edge_labels = \
        build_edge_features_and_labels(G, node_to_idx)

    # Build node feature matrix in consistent index order
    nodes_ordered = sorted(node_to_idx, key=node_to_idx.get)
    x = np.stack([node_feat_dict[n] for n in nodes_ordered])

    # Fit scaler on ALL node features here; only train-node rows will be
    # used for model training — we save the scaler for inference later.
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        edge_y     = edge_labels,
    )

    print(f"[Dataset] PyG Data: {data}")
    return data, scaler, node_to_idx


# =============================================================================
# SECTION 5: Inductive Train / Val / Test Split  (NODE-DISJOINT)
# =============================================================================

def _node_hostility_ratio(G: nx.Graph, node_to_idx: dict) -> np.ndarray:
    """
    Compute a per-node hostility ratio: the fraction of each node's incident
    edges that are labelled hostile.

    Used to stratify the node-level split so each partition receives a
    representative mix of high-risk and low-risk endpoints, preventing the
    model from training only on benign-heavy nodes or evaluating only on
    hostile-heavy ones.

    Returns:
        ratios: np.ndarray of shape (num_nodes,), values in [0.0, 1.0].
                Isolated nodes (no edges) are assigned 0.0.
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
    """
    Bin continuous hostility ratios into discrete strata so that
    train_test_split can stratify at the node level.

    Bins with fewer than 2 members are merged into bin 0 to satisfy
    scikit-learn's minimum-samples-per-class constraint.

    Returns:
        bins: np.ndarray of int strata labels, shape (num_nodes,)
    """
    bins = np.digitize(
        ratios, bins=np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    )
    unique, counts = np.unique(bins, return_counts=True)
    rare = unique[counts < 2]
    for r in rare:
        bins[bins == r] = 0
    return bins


def split_nodes_inductively(data:        Data,
                             G:           nx.Graph,
                             node_to_idx: dict,
                             val_ratio:   float = None,
                             test_ratio:  float = None) -> Data:
    """
    Partition the NODE set into three disjoint groups, then derive edge masks:

      train_edge_mask — both endpoints in the train node set
      val_edge_mask   — at least one endpoint in val, none in test
      test_edge_mask  — at least one endpoint in the test node set

    Val and test edge sets deliberately include cross-boundary edges
    (new node <-> train node) so the GNN can aggregate from known
    neighbours during evaluation — exactly mirroring real inference.

    Attaches to `data`:
      train_edge_mask, val_edge_mask, test_edge_mask  (bool tensors)
      train_node_idx, val_node_idx, test_node_idx     (long tensors)
    """
    val_ratio  = val_ratio  or CONFIG["val_ratio"]
    test_ratio = test_ratio or CONFIG["test_ratio"]

    all_nodes = np.array(list(node_to_idx.keys()))
    ratios    = _node_hostility_ratio(G, node_to_idx)
    strata    = _stratify_bins(ratios)

    # First split: train vs (val + test)
    train_nodes, temp_nodes = train_test_split(
        all_nodes,
        test_size    = val_ratio + test_ratio,
        stratify     = strata,
        random_state = CONFIG["seed"],
    )

    # Second split: val vs test from the temp pool
    temp_indices  = np.isin(all_nodes, temp_nodes)
    temp_ratios   = ratios[temp_indices]
    temp_strata   = _stratify_bins(temp_ratios)

    val_nodes, test_nodes = train_test_split(
        temp_nodes,
        test_size    = test_ratio / (val_ratio + test_ratio),
        stratify     = temp_strata,
        random_state = CONFIG["seed"],
    )

    train_idx_set = set(node_to_idx[n] for n in train_nodes)
    val_idx_set   = set(node_to_idx[n] for n in val_nodes)
    test_idx_set  = set(node_to_idx[n] for n in test_nodes)

    src       = data.edge_index[0].tolist()
    dst       = data.edge_index[1].tolist()
    num_edges = len(src)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask   = torch.zeros(num_edges, dtype=torch.bool)
    test_mask  = torch.zeros(num_edges, dtype=torch.bool)


    for i, (s, d) in enumerate(zip(src, dst)):
        in_train = (s in train_idx_set) and (d in train_idx_set)
        in_val   = (((s in val_idx_set) and (s not in train_idx_set and d  in train_idx_set)) or \
                    ((d in val_idx_set) and (d not in train_idx_set and s in train_idx_set))) and\
                    (s not in test_idx_set)
        in_test  = (((s in test_idx_set) and (s not in train_idx_set and d in train_idx_set)) or \
                    ((d in test_idx_set) and (d not in train_idx_set and s in train_idx_set))) and\
                    (s not in val_idx_set)

        if in_train and in_val:
            raise(Exception('val train overlap detected'))
        if in_train and in_test:
            raise(Exception('test train overlap detected'))
        if in_val and in_test:
            raise(Exception('val test overlap detected'))

        train_mask[i] = in_train
        val_mask[i]   = in_val
        test_mask[i]  = in_test

    data.train_edge_mask = train_mask
    data.val_edge_mask   = val_mask
    data.test_edge_mask  = test_mask

    # Node index tensors used to scope NeighborLoader to training nodes only
    data.train_node_idx = torch.tensor(sorted(train_idx_set), dtype=torch.long)
    data.val_node_idx   = torch.tensor(sorted(val_idx_set),   dtype=torch.long)
    data.test_node_idx  = torch.tensor(sorted(test_idx_set),  dtype=torch.long)

    print(f"[Split] Nodes  — train: {len(train_idx_set)} | "
          f"val: {len(val_idx_set)} | test: {len(test_idx_set)}")
    print(f"[Split] Edges  — train: {train_mask.sum()} | "
          f"val: {val_mask.sum()} | test: {test_mask.sum()}")
    return data


# =============================================================================
# SECTION 6: Model Definition — Inductive GNN Encoder (GraphSAGE)
# =============================================================================

class GNNEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder.

    SAGEConv aggregates from sampled local neighbourhoods, so it produces
    valid embeddings for nodes it has never seen — the core requirement for
    inductive generalisation.
    """
    def __init__(self, in_channels: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [num_nodes, hidden_dim]


# =============================================================================
# SECTION 7: Edge Classification Head
# =============================================================================

class EdgeClassifier(nn.Module):
    """
    Full model: GNNEncoder + MLP edge classifier.

    Edge representation = concat(src_embed, dst_embed, edge_attr).
    Output: scalar logit per edge (hostile probability after sigmoid).
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = GNNEncoder(node_feat_dim, hidden_dim,
                                  num_layers, dropout)

        mlp_in = 2 * hidden_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        node_emb  = self.encoder(x, edge_index)            # [N, H]
        src_emb   = node_emb[edge_index[0]]                # [E, H]
        dst_emb   = node_emb[edge_index[1]]                # [E, H]
        edge_repr = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.mlp(edge_repr).squeeze(-1)             # [E]


# =============================================================================
# SECTION 8: Training Loop
# =============================================================================

def compute_pos_weight(data: Data) -> float:
    """
    Auto-compute BCEWithLogitsLoss pos_weight from training edge labels.
    pos_weight = (#benign train edges) / (#hostile train edges)
    """
    train_labels = data.edge_y[data.train_edge_mask].float()
    n_neg = (train_labels == 0).sum().item()
    n_pos = (train_labels == 1).sum().item()
    pw = n_neg / max(n_pos, 1)
    print(f"[Train] pos_weight={pw:.2f}  (benign={n_neg}, hostile={n_pos})")
    return pw


def _build_train_loader(data: Data) -> NeighborLoader:
    """
    Build a NeighborLoader scoped strictly to training nodes.

    Setting input_nodes=data.train_node_idx means mini-batch seeds are
    drawn only from the train partition. The sampler still follows edges
    in the full graph structure, but no val/test node will ever be a seed,
    preventing their neighbourhood structure from influencing learned weights.
    """
    return NeighborLoader(
        data,
        num_neighbors = CONFIG["num_neighbors"],
        input_nodes   = data.train_node_idx,
        batch_size    = CONFIG["batch_size"],
        shuffle       = True,
    )


def train_one_epoch(model:     EdgeClassifier,
                    loader:    NeighborLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device) -> float:
    """
    One training epoch over NeighborLoader mini-batches.

    Each mini-batch is a sampled subgraph. We only compute loss on edges
    whose train_edge_mask is True within the batch, ensuring the model
    is supervised exclusively on train-partition edges.
    """
    model.train()
    total_loss    = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr)

        if not hasattr(batch, "train_edge_mask") \
                or batch.train_edge_mask.sum() == 0:
            continue

        loss = criterion(
            logits[batch.train_edge_mask],
            batch.edge_y[batch.train_edge_mask].float(),
        )
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model:  EdgeClassifier,
             data:   Data,
             mask:   torch.Tensor,
             device: torch.device) -> dict:
    """
    Full-graph evaluation on val or test edges.
    Uses the complete graph so neighbourhood context is maximally rich,
    matching the conditions under which inductive inference will be run.
    """
    model.eval()
    logits = model(
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device),
    )
    probs  = torch.sigmoid(logits[mask.to(device)]).cpu().numpy()
    preds  = (probs >= 0.5).astype(int)
    labels = data.edge_y[mask].numpy()

    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, zero_division=0)
    return {"auc": auc, "f1": f1, "probs": probs,
            "preds": preds, "labels": labels}


def train(data: Data, device: torch.device) -> EdgeClassifier:
    node_feat_dim = data.x.size(1)
    edge_feat_dim = data.edge_attr.size(1)

    model = EdgeClassifier(
        node_feat_dim = node_feat_dim,
        edge_feat_dim = edge_feat_dim,
        hidden_dim    = CONFIG["hidden_dim"],
        num_layers    = CONFIG["num_layers"],
        dropout       = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    pw        = CONFIG["pos_weight"] or compute_pos_weight(data)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float).to(device)
    )

    train_loader = _build_train_loader(data)

    best_val_auc = 0.0
    for epoch in range(1, CONFIG["epochs"] + 1):
        loss        = train_one_epoch(model, train_loader,
                                      optimizer, criterion, device)
        val_metrics = evaluate(model, data, data.val_edge_mask, device)
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
# SECTION 9: Evaluation on Test Set
# =============================================================================

def test(model: EdgeClassifier, data: Data, device: torch.device):
    """Load best checkpoint and report test-set metrics."""
    model.load_state_dict(
        torch.load(CONFIG["checkpoint_path"], map_location=device)
    )
    metrics = evaluate(model, data, data.test_edge_mask, device)

    print("\n[Test Results]")
    print(f"  AUC-ROC : {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(classification_report(
        metrics["labels"], metrics["preds"],
        target_names=["Benign", "Hostile"],
    ))


# =============================================================================
# SECTION 10: Inductive Inference on New Nodes & Edges
# =============================================================================

def inductive_inference(
    model:             EdgeClassifier,
    new_graphml_path:  str,
    scaler:            StandardScaler,
    node_to_idx_train: dict,
    device:            torch.device,
) -> pd.DataFrame:
    """
    Classify edges in a new .graphml that may contain previously unseen nodes.

    Inductive contract:
      - Known nodes reuse their training integer index for feature-matrix
        position consistency.
      - Brand-new nodes are appended beyond the training index range; their
        features are derived purely from incident edges in the new graph.
      - The TRAINING scaler normalises new features — never re-fit on new data.
      - No re-training or fine-tuning is required.

    Returns:
        DataFrame [src, dst, prob_hostile, prediction, true_label]
    """
    model.eval()
    G_new = load_graphml(new_graphml_path)

    node_feat_dict_new = build_node_features(G_new)

    # Extend training index map with any brand-new nodes
    extended_idx = dict(node_to_idx_train)
    next_idx     = max(node_to_idx_train.values()) + 1
    for node in G_new.nodes():
        if node not in extended_idx:
            extended_idx[node] = next_idx
            next_idx += 1

    # Build feature matrix: rows for training nodes default to zero
    # (they have no edges in the new graph to aggregate from)
    feat_dim = next(iter(node_feat_dict_new.values())).shape[0]
    x_new    = np.zeros((next_idx, feat_dim), dtype=np.float32)
    for node, feat in node_feat_dict_new.items():
        x_new[extended_idx[node]] = feat

    x_new = scaler.transform(x_new)  # normalise with TRAINING scaler

    feat_keys   = CONFIG["edge_feature_keys"]
    label_key   = CONFIG["edge_label_key"]
    src_list, dst_list, feat_list, true_labels = [], [], [], []

    for u, v, edata in G_new.edges(data=True):
        src_list.append(extended_idx[u])
        dst_list.append(extended_idx[v])
        feat_list.append([float(edata.get(k, 0.0)) for k in feat_keys])
        true_labels.append(int(edata.get(label_key, -1)))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)
    edge_attr  = torch.tensor(feat_list, dtype=torch.float).to(device)
    x_tensor   = torch.tensor(x_new, dtype=torch.float).to(device)

    with torch.no_grad():
        logits = model(x_tensor, edge_index, edge_attr)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

    idx_to_node = {v: k for k, v in extended_idx.items()}
    results = pd.DataFrame({
        "src":          [idx_to_node.get(s, s) for s in src_list],
        "dst":          [idx_to_node.get(d, d) for d in dst_list],
        "prob_hostile": probs,
        "prediction":   ["hostile" if p else "benign" for p in preds],
        "true_label":   true_labels,
    })

    print(f"[Inference] {results['prediction'].value_counts().to_dict()}")
    return results


# =============================================================================
# SECTION 11: Main Entry Point
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using device: {device}")

    # --- Load & Inspect ---
    G = load_graphml(CONFIG["graphml_path"])
    inspect_graph(G)

    # --- Build PyG Dataset ---
    data, scaler, node_to_idx = build_pyg_data(G)

    # --- Inductive Node-Disjoint Split ---
    data = split_nodes_inductively(data, G, node_to_idx)

    # --- Train ---
    model = train(data, device)

    # --- Test ---
    test(model, data, device)

    # --- Inductive Inference Example ---
    # Point to a new .graphml to classify unseen nodes/edges without retraining:
    #
    # results = inductive_inference(
    #     model             = model,
    #     new_graphml_path  = "new_network_capture.graphml",
    #     scaler            = scaler,
    #     node_to_idx_train = node_to_idx,
    #     device            = device,
    # )
    # results.to_csv("inference_results.csv", index=False)
    # print(results.head(20))


if __name__ == "__main__":
    main()