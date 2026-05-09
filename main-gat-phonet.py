import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


DATA_PATH = '/content/drive/MyDrive/adhd_deepfmri_aal.npy'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4  
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.5
HIDDEN_CHANNELS = 32
NUM_HEADS = 4
CORRELATION_THRESHOLD = 0.6 
SEED = 42


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ADHDGraphDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super(ADHDGraphDataset, self).__init__('.', transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass

def create_graph_data(npy_path, threshold=0.5):
    print(f"Loading data from {npy_path}...")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found at {npy_path}")

    raw_data = np.load(npy_path, allow_pickle=True)


    pheno_matrix = np.array([item['pheno'] for item in raw_data])
    scaler = StandardScaler()
    pheno_normalized = scaler.fit_transform(pheno_matrix)

    data_list = []

    print("Converting Time-Series to Graphs...")
    for idx, item in enumerate(raw_data):
        ts = item['time_series'].T
        correlation_matrix = np.corrcoef(ts)
        np.nan_to_num(correlation_matrix, copy=False, nan=0.0)
        x = torch.tensor(correlation_matrix, dtype=torch.float)
        adj = np.abs(correlation_matrix)
        adj = adj - np.eye(adj.shape[0])
        edge_indices = np.where(adj > threshold)
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        edge_attr = torch.tensor(adj[edge_indices], dtype=torch.float)
        y = torch.tensor([item['label']], dtype=torch.long)
        pheno = torch.tensor([pheno_normalized[idx]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pheno=pheno)
        data_list.append(data)

    return data_list

#MODEL DEFINITION (GAT + MLP)

class GAT_Pheno_Model(nn.Module):
    def __init__(self, num_node_features, num_pheno_features, hidden_channels, num_heads, dropout):
        super(GAT_Pheno_Model, self).__init__()
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.pheno_mlp = nn.Sequential(
            nn.Linear(num_pheno_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_dim = hidden_channels + 16

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.dropout_val = dropout

    def forward(self, data):
        x, edge_index, batch, pheno = data.x, data.edge_index, data.batch, data.pheno
        x = F.dropout(x, p=self.dropout_val, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_val, training=self.training)
        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, batch) 
        pheno = pheno.squeeze(1)
        p = self.pheno_mlp(pheno)
        combined = torch.cat([x, p], dim=1)
        out = self.classifier(combined)
        return out

# TRAINING

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y.float().unsqueeze(1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.float().unsqueeze(1))
            total_loss += loss.item() * data.num_graphs

            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    labels_np = np.array(all_labels).flatten()
    probs_np = np.array(all_probs).flatten()


    if len(np.unique(labels_np)) > 1:
        auc = roc_auc_score(labels_np, probs_np)
    else:
        auc = np.nan

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(labels_np, np.array(all_preds).flatten()),
        'auc': auc,
        'probs': probs_np,
        'preds': np.array(all_preds).flatten(),
        'labels': labels_np
    }

    return metrics

def plot_learning_curves(history, save_path="learning_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Learning curves saved to {save_path}")



def main():
   
    try:
        graph_list = create_graph_data(DATA_PATH, threshold=CORRELATION_THRESHOLD)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    labels = [data.y.item() for data in graph_list]
    train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=[labels[i] for i in train_idx], random_state=SEED)

    train_dataset = [graph_list[i] for i in train_idx]
    val_dataset = [graph_list[i] for i in val_idx]
    test_dataset = [graph_list[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nData Split: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")

    train_labels = [data.y.item() for data in train_dataset]
    n_neg = train_labels.count(0)
    n_pos = train_labels.count(1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
    print(f"Class Imbalance Correction: pos_weight = {pos_weight.item():.2f}")

    model = GAT_Pheno_Model(
        num_node_features=90,
        num_pheno_features=3,
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0
    best_epoch = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}


    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)
        history['val_auc'].append(val_metrics['auc'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])

        scheduler.step(val_metrics['loss'])

        print(
            f"Epoch {epoch+1:03d}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )



        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_adhd_gat_model.pth')
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


    plot_learning_curves(history, save_path="adhd_learning_curves.png")



    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('best_adhd_gat_model.pth'))
    test_metrics = evaluate(model, test_loader, criterion)

    test_auc = roc_auc_score(test_metrics['labels'], test_metrics['probs'])
    test_ap = average_precision_score(test_metrics['labels'], test_metrics['probs'])
    test_f1 = f1_score(test_metrics['labels'], test_metrics['preds'])



    y_true = test_metrics['labels']
    y_pred = test_metrics['preds']
    y_probs = test_metrics['probs']


    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'ADHD']))


    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'ADHD'], yticklabels=['Healthy', 'ADHD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')


    auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('adhd_evaluation_results.png')
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(history['val_auc']) + 1), history['val_auc'])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("val_auc_curve.png")
    plt.show()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nSUMMARY REPORT")
    print("=" * 80)
    print(f"Configuration: Fixed Correlation Threshold")
    print(f"Graph Construction: |r| > {CORRELATION_THRESHOLD}")
    print(f"Data Augmentation: Disabled")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"ROC-AUC Score: {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Total Training Graphs: {len(train_dataset)}")
    print(f"Total Test Graphs: {len(test_dataset)}")
    print(f"Model Parameters: {num_params:,}")
    print(f"Training Epochs Completed: {best_epoch}")
    print("=" * 80)



    print(f"Results saved to adhd_evaluation_results.png")

if __name__ == "__main__":
    main()
