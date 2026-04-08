import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


# ----------------------------
# 0. Strip XML Namespace
# ----------------------------
def strip_namespace(root):
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return root

# ----------------------------
# 1. Parse GraphML
# ----------------------------
def parse_graphml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    root = strip_namespace(root)

    node_map = {}
    edge_index = []
    edge_attr = []
    edge_labels = []

    def get_node_id(node):
        if node not in node_map:
            node_map[node] = len(node_map)
        return node_map[node]

    for edge in root.iter('edge'):
        src = edge.attrib['source']
        tgt = edge.attrib['target']
        src_id = get_node_id(src)
        tgt_id = get_node_id(tgt)

        # Extract features safely
        feature_dict = {}
        for data in edge.iter('data'):
            key = data.attrib['key']
            feature_dict[key] = float(data.text)
        if not all(f'd{i}' in feature_dict for i in range(8)):
            continue

        features = [feature_dict[f'd{i}'] for i in range(8)]
        edge_labels.append(int(features[7]))
        edge_attr.append(features[:7])
        edge_index.append([src_id, tgt_id])

    print(f"Parsed nodes: {len(node_map)}, edges: {len(edge_index)}")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    return node_map, edge_index, edge_attr, edge_labels

# ----------------------------
# 2. Edge Classification Model with plain nn.Embedding
# ----------------------------
class IntrusionEdgeClassifier(nn.Module):
    def __init__(self, num_nodes, node_dim=32, edge_dim=7, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, node_dim)  # plain embedding
        self.nnconv = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=nn.Sequential(
                nn.Linear(edge_dim, node_dim * hidden_dim),
                nn.ReLU()
            )
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # benign vs attack
        )

    def forward(self, edge_index, edge_attr):
        x = self.node_emb.weight           # [num_nodes, node_dim]
        x = F.relu(self.nnconv(x, edge_index, edge_attr))
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.edge_mlp(edge_repr)

# ----------------------------
# 3. Build PyG Data
# ----------------------------
def build_data(file_path, val_ratio=0.2, random_state=42):
    node_map, edge_index, edge_attr, edge_labels = parse_graphml(file_path)

    # Total number of edges
    num_edges = edge_index.size(1)
    edge_indices = torch.arange(num_edges)

    # Split edges into train/val
    train_idx, val_idx = train_test_split(
        edge_indices.numpy(),
        test_size=val_ratio,
        random_state=random_state,
        stratify=edge_labels.numpy()  # keep label distribution
    )

    # Create train Data
    train_data = Data(
        edge_index=edge_index[:, train_idx],
        edge_attr=edge_attr[train_idx],
        edge_label=edge_labels[train_idx]
    )

    # Create validation Data
    val_data = Data(
        edge_index=edge_index[:, val_idx],
        edge_attr=edge_attr[val_idx],
        edge_label=edge_labels[val_idx]
    )

    print(f"Train edges: {train_data.edge_index.size(1)}, "
          f"Validation edges: {val_data.edge_index.size(1)}")

    return train_data, val_data, len(node_map)

# ----------------------------
# 4. Training Loop
# ----------------------------
def train(model, train_data, val_data, epochs=20, lr=1e-3, batch_size=256):
    # Compute class weights from training labels
    labels = train_data.edge_label
    num_classes = int(labels.max().item()) + 1
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-6)  # avoid division by zero
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Helper function to evaluate loss & accuracy on a Data object
    @torch.no_grad()
    def evaluate_data(data):
        model.eval()
        logits = model(data.edge_index, data.edge_attr)
        loss = criterion(logits, data.edge_label)
        pred = logits.argmax(dim=1)
        acc = (pred == data.edge_label).float().mean()
        return loss.item(), acc.item()

    # Edge DataLoader
    edge_dataset = torch.utils.data.TensorDataset(
        train_data.edge_index.t(), train_data.edge_attr, train_data.edge_label
    )
    loader = DataLoader(edge_dataset, batch_size=batch_size, shuffle=True) # type: ignore

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for edge_idx_batch, edge_attr_batch, label_batch in loader:
            optimizer.zero_grad()
            edge_index_batch = edge_idx_batch.t().contiguous()
            logits = model(edge_index_batch, edge_attr_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * edge_idx_batch.size(0)

        train_loss = total_loss / train_data.edge_label.size(0)
        train_loss_val, train_acc = evaluate_data(train_data)
        val_loss, val_acc = evaluate_data(val_data)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ----------------------------
# 5. Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.edge_index, data.edge_attr)
    pred = logits.argmax(dim=1)
    acc = (pred == data.edge_label).float().mean()
    print(f"Accuracy: {acc:.4f}")

# ----------------------------
# 6. Example Usage
# ----------------------------
if __name__ == "__main__":
    file_path = 'data/0.1M-Stratified-Multi.graphml'
    train_data, val_data, num_nodes = build_data(file_path)
    model = IntrusionEdgeClassifier(num_nodes)

    train(model, train_data=train_data, val_data=val_data, epochs=10, lr=1e-3)
    evaluate(model, val_data)