import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import json
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def print_graph_stats(data):
    print("\n[GRAPH STATS]")
    for node_type in data.node_types:
        x = data[node_type].x
        y = data[node_type].get('y', None)
        print(f"  {node_type}: {data[node_type].num_nodes} nodes | features: {x.shape if x is not None else 'None'}")
        if y is not None:
            unique, counts = torch.unique(y, return_counts=True)
            print(f"    -> Labels: {dict(zip(unique.tolist(), counts.tolist()))}")
    for et in data.edge_types:
        print(f"  Edge {et}: {data[et].edge_index.shape[1]} edges")


# ---------------------------------------------------------
# Feature Encoder (inductive — Linear on real features)
# ---------------------------------------------------------
class FeatureEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        pat_in = data['Patient'].x.shape[1]
        adm_in = data['Admission'].x.shape[1]
        self.pat_lin = torch.nn.Linear(pat_in, hidden_channels)
        self.adm_lin = torch.nn.Linear(adm_in, hidden_channels)

        # Concept nodes: shared medical entities → learnable embeddings
        self.diag_emb = torch.nn.Embedding(data['Diagnosis'].num_nodes,  hidden_channels)
        self.med_emb  = torch.nn.Embedding(data['Medication'].num_nodes, hidden_channels)
        self.proc_emb = torch.nn.Embedding(data['Procedure'].num_nodes,  hidden_channels)  # NEW
        self.lab_emb  = torch.nn.Embedding(data['LabTest'].num_nodes,    hidden_channels)  # NEW

        self.pat_bn = torch.nn.BatchNorm1d(hidden_channels)
        self.adm_bn = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x_dict):
        return {
            'Patient':    self.pat_bn(self.pat_lin(x_dict['Patient'])),
            'Admission':  self.adm_bn(self.adm_lin(x_dict['Admission'])),
            'Diagnosis':  self.diag_emb.weight,
            'Medication': self.med_emb.weight,
            'Procedure':  self.proc_emb.weight,  # NEW
            'LabTest':    self.lab_emb.weight,    # NEW
        }


# ---------------------------------------------------------
# GNN Backbone — 3 SAGEConv layers
# ---------------------------------------------------------
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.bn1(self.conv1(x, edge_index)).relu()
        x = self.dropout(x)
        x = self.bn2(self.conv2(x, edge_index)).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        return self.lin(x)


class MedicalKnowledgeGraphModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, data):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_channels, data)
        self.gnn = to_hetero(
            BaseGNN(hidden_channels, out_channels),
            data.metadata(),
            aggr='max'
        )

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(self.encoder(x_dict), edge_index_dict)


# ---------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------
def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        out   = model(data.x_dict, data.edge_index_dict)['Admission'][mask]
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        tgts  = data['Admission'].y[mask].cpu().numpy()
    return probs, preds, tgts


def print_metrics(label, tgts, probs, preds):
    auc  = roc_auc_score(tgts, probs)
    f1   = f1_score(tgts, preds, zero_division=0)
    prec = precision_score(tgts, preds, zero_division=0)
    rec  = recall_score(tgts, preds, zero_division=0)
    cm   = confusion_matrix(tgts, preds)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*50}")
    return {'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec}


def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as file:
        return yaml.safe_load(file)

def train_and_evaluate():
    config = load_config()
    proc_dir = config['paths']['processed_data']


    print("1. Loading PyTorch Graph...")
    data = torch.load(os.path.join(proc_dir, 'mimic_graph.pt'), weights_only=False)
    data = T.ToUndirected()(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   -> Device: {device}")
    print_graph_stats(data)

    # Masks are now loaded from the saved .pt (not recreated randomly)
    train_mask = data['Admission'].train_mask
    val_mask   = data['Admission'].val_mask
    test_mask  = data['Admission'].test_mask
    print(f"\n   Masks loaded from graph: Train={train_mask.sum()} | Val={val_mask.sum()} | Test={test_mask.sum()}")

    data = data.to(device)

    # ---------------------------------------------------------
    # Undersampling on train split
    # ---------------------------------------------------------
    labels_train = data['Admission'].y[train_mask.to(device)]
    train_indices = train_mask.to(device).nonzero(as_tuple=False).squeeze()
    died_idx = train_indices[labels_train == 1]
    surv_idx = train_indices[labels_train == 0]
    sampled_surv = surv_idx[torch.randperm(len(surv_idx), device=device)[:len(died_idx) * 2]]
    balanced_idx = torch.cat([died_idx, sampled_surv])
    balanced_mask = torch.zeros(data['Admission'].num_nodes, dtype=torch.bool, device=device)
    balanced_mask[balanced_idx] = True
    data['Admission'].train_mask = balanced_mask

    print(f"   Train (balanced): {balanced_mask.sum().item()} | "
          f"{len(died_idx)} died / {len(sampled_surv)} survived")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    print("\n2. Initializing model...")
    HIDDEN = 64
    model = MedicalKnowledgeGraphModel(HIDDEN, 2, data).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], dtype=torch.float).to(device))

    # ---------------------------------------------------------
    # Training loop (full-batch)
    # ---------------------------------------------------------
    print("\n3. Training...")
    epochs = 300
    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out    = model(data.x_dict, data.edge_index_dict)
        pred   = out['Admission'][data['Admission'].train_mask]
        target = data['Admission'].y[data['Admission'].train_mask]
        loss   = criterion(pred, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            probs, preds, tgts = evaluate(model, data, data['Admission'].val_mask, device)
            val_auc = roc_auc_score(tgts, probs)
            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), os.path.join(proc_dir, 'heterosage_mortality.pth'))
                marker = " *** Best ***"
            else:
                marker = ""

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}{marker}")

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------
    print(f"\n4. Best Val AUC: {best_val_auc:.4f} — evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(proc_dir, 'heterosage_mortality.pth'), weights_only=True))

    probs, preds, tgts = evaluate(model, data, data['Admission'].test_mask, device)
    metrics = print_metrics("FINAL TEST METRICS (Unseen Data)", tgts, probs, preds)

    with open(os.path.join(proc_dir, 'stats_phase8_gnn_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # ---------------------------------------------------------
    # Save Admission embeddings for RAG + FAISS (script 07)
    # ---------------------------------------------------------
    print("\n5. Saving Admission embeddings for RAG/FAISS...")
    model.eval()
    with torch.no_grad():
        all_embeddings = model.encoder(data.x_dict)['Admission'].cpu()
    torch.save(all_embeddings, os.path.join(proc_dir, 'admission_embeddings.pt'))
    print(f"   Embeddings saved: {all_embeddings.shape}  (n_admissions x hidden_dim)")

    print("\nPhases 8 & 9 prep complete.")


if __name__ == "__main__":
    train_and_evaluate()