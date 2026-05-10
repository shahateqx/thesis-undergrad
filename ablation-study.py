import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             accuracy_score, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import copy


try:
    from adhd_reporting_utils import print_detailed_confusion_matrix
except ImportError:
    print("Warning: adhd_reporting_utils.py not found. Some detailed tables may be skipped.")


DATA_PATH = '/content/drive/MyDrive/adhd_deepfmri_aal.npy'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 60 
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.5
HIDDEN_CHANNELS = 32
NUM_HEADS = 4
CORRELATION_THRESHOLD = 0.6
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_graph_data(npy_path, threshold=0.5):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found at {npy_path}")
    raw_data = np.load(npy_path, allow_pickle=True)

    pheno_matrix = np.array([item['pheno'] for item in raw_data])
    scaler = StandardScaler()
    pheno_normalized = scaler.fit_transform(pheno_matrix)

    data_list = []
    for idx, item in enumerate(raw_data):
        ts = item['time_series'].T
        correlation_matrix = np.corrcoef(ts)
        np.nan_to_num(correlation_matrix, copy=False, nan=0.0)

        x = torch.tensor(correlation_matrix, dtype=torch.float)

        adj = np.abs(correlation_matrix) - np.eye(correlation_matrix.shape[0])
        edge_indices = np.where(adj > threshold)
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        edge_attr = torch.tensor(adj[edge_indices], dtype=torch.float)

        y = torch.tensor([item['label']], dtype=torch.long)
        pheno = torch.tensor([pheno_normalized[idx]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pheno=pheno)
        data_list.append(data)
    return data_list

class AblationModel(nn.Module):
    def __init__(self, use_gat, use_pheno, num_node_features, num_pheno_features, hidden_channels, num_heads, dropout):
        super(AblationModel, self).__init__()
        self.use_gat = use_gat
        self.use_pheno = use_pheno
        self.dropout_val = dropout

        if self.use_gat:
            self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout)


        if self.use_pheno:
            self.pheno_mlp = nn.Sequential(
                nn.Linear(num_pheno_features, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        combined_dim = 0
        if self.use_gat:
            combined_dim += hidden_channels
        if self.use_pheno:
            combined_dim += 16

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        features = []

        # 1. GAT Forward
        if self.use_gat:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.dropout(x, p=self.dropout_val, training=self.training)
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_val, training=self.training)
            x = self.gat2(x, edge_index)
            x = global_mean_pool(x, batch)
            features.append(x)

        # 2. Pheno Forward
        if self.use_pheno:
            p = data.pheno.squeeze(1)
            p = self.pheno_mlp(p)
            features.append(p)

        # 3. Combine
        combined = torch.cat(features, dim=1)
        out = self.classifier(combined)
        return out


def run_experiment(name, use_gat, use_pheno, loaders, pos_weight):
    print(f"\n{'='*20} Running Ablation: {name} {'='*20}")
    train_loader, val_loader, test_loader = loaders

    model = AblationModel(
        use_gat=use_gat, use_pheno=use_pheno,
        num_node_features=90, num_pheno_features=3,
        hidden_channels=HIDDEN_CHANNELS, num_heads=NUM_HEADS, dropout=DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y.float().unsqueeze(1))
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break


    model.load_state_dict(best_model_state)
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'AUC': roc_auc_score(all_labels, all_probs),
        'F1-Score': f1_score(all_labels, all_preds),
        'Precision': average_precision_score(all_labels, all_probs)
    }

    print(f"Result [{name}]: Accuracy={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}")
    return metrics, (all_labels, all_preds)

# MAIN ABLATION LOOP
def main():
    # 1. Setup Data
    try:
        graph_list = create_graph_data(DATA_PATH, threshold=CORRELATION_THRESHOLD)
    except Exception as e:
        print(f"Data load error: {e}. generating dummy data for structure check.")
       
        graph_list = []
        for _ in range(100):
            graph_list.append(Data(
                x=torch.randn(90, 90),
                edge_index=torch.randint(0, 90, (2, 200)),
                y=torch.tensor([np.random.randint(0,2)]),
                pheno=torch.randn(1,3)
            ))

    labels = [data.y.item() for data in graph_list]
    train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=[labels[i] for i in train_idx], random_state=SEED)

    train_loader = DataLoader([graph_list[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([graph_list[i] for i in val_idx], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader([graph_list[i] for i in test_idx], batch_size=BATCH_SIZE, shuffle=False)
    loaders = (train_loader, val_loader, test_loader)


    n_neg = labels.count(0)
    n_pos = labels.count(1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)


    results = {}
    predictions = {}

    # Exp 1: Full Model
    results['Full (GAT+Pheno)'], predictions['Full'] = run_experiment(
        "Full Hybrid Model", use_gat=True, use_pheno=True, loaders=loaders, pos_weight=pos_weight)

    # Exp 2: GAT Only
    results['Brain Only (GAT)'], predictions['GAT'] = run_experiment(
        "Brain Only (GAT)", use_gat=True, use_pheno=False, loaders=loaders, pos_weight=pos_weight)

    # Exp 3: Pheno Only
    results['Pheno Only (MLP)'], predictions['Pheno'] = run_experiment(
        "Pheno Only (MLP)", use_gat=False, use_pheno=True, loaders=loaders, pos_weight=pos_weight)

    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS (Copy to Paper)")
    print("="*60)

    headers = ["Model Variant", "Accuracy", "AUC-ROC", "F1-Score", "Precision"]
    row_format = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(row_format.format(*headers))
    print("-" * 70)

    for name, metrics in results.items():
        print(row_format.format(
            name,
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['AUC']:.4f}",
            f"{metrics['F1-Score']:.4f}",
            f"{metrics['Precision']:.4f}"
        ))
    print("-" * 70)

    # B. Comparison Plot
    metrics_df = pd.DataFrame(results).T

    plt.figure(figsize=(10, 6))
    metrics_df[['Accuracy', 'AUC']].plot(kind='bar', rot=0, color=['#4c72b0', '#55a868'], figsize=(10, 6))
    plt.title('Ablation Study: Component Contribution', fontsize=14)
    plt.ylabel('Score')
    plt.ylim(0.4, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('ablation_comparison.png')
    print("\n[INFO] Comparison chart saved to 'ablation_comparison.png'")

    # C. Print detailed confusion matrix for the best model (Full)
    print("\n--- Detailed Confusion Matrix (Full Model) ---")
    y_true, y_pred = predictions['Full']
    try:
        from adhd_reporting_utils import print_detailed_confusion_matrix
        print_detailed_confusion_matrix(y_true, y_pred)
    except:
        print(classification_report(y_true, y_pred, target_names=['Healthy', 'ADHD']))

if __name__ == "__main__":
    main()
