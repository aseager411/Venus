# Author: Alex Seager
# Last Version: 7/18/25
#
# Description: I am attempting to build a neural network which performs 
#              Spectral deconvolution with a machine specific library based 
#              approach. 

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.colors as mcolors

from MS_Inversion_Toy import (
    strict_recall_score
)

# -----------------------------
# Step 2: Load and Prepare Data
# -----------------------------
# Goal: upload spectra (averaged across replicates for each molecule).
# Then create artificial mixtures of size 1 to 25 (equal weight by default),
# each labeled with a binary vector indicating which molecules are present.

def load_and_prepare_data(spectra_path, metadata_path, N_Mixtures=50, max_complexity=25, seed=42):
    # Load and average spectra by molecule
    X_raw = pd.read_csv(spectra_path, index_col=0).T  # shape: (samples × m/z bins)
    meta = pd.read_csv(metadata_path).set_index("molecule").loc[X_raw.index]

    # Average replicates by molecule name
    X_avg = X_raw.groupby(X_raw.index).mean()
    molecule_names = list(X_avg.index)
    spectral_matrix = X_avg.values.T  # shape: (num_bins, num_molecules)

    # Generate synthetic mixtures
    X_mixed, Y_mixed = generate_mixture_dataset(
        spectral_matrix,
        N=N_Mixtures,
        min_complexity=1,
        max_complexity=max_complexity,
        equal_weights=True,
        seed=seed
    )

    print(f"✅ Molecules in library: {spectral_matrix.shape[1]}")
    print(f"📦 Total generated mixtures: {len(X_mixed)}")

    return X_mixed, Y_mixed, molecule_names




#mix generating helper
def generate_mixture_dataset(spectral_matrix, N, min_complexity=1, max_complexity=25, equal_weights=True, seed=None):
    """
    Generate synthetic mixtures using a consistent pipeline for training or testing.

    Args:
        spectral_matrix: (num_bins × num_molecules)
        N: number of mixtures per complexity level
        min_complexity, max_complexity: range of mixture sizes
        equal_weights: if True, all molecules have equal concentration
        seed: for reproducibility

    Returns:
        X: spectra matrix [num_samples × num_bins]
        Y: binary labels [num_samples × num_molecules]
    """
    if seed is not None:
        np.random.seed(seed)

    num_bins, num_molecules = spectral_matrix.shape
    mixtures = []
    labels = []

    for complexity in range(min_complexity, max_complexity + 1):
        current_N = min(N, num_molecules) if complexity == 1 else N

        for _ in range(current_N):
            mol_indices = np.random.choice(num_molecules, size=complexity, replace=False)

            if equal_weights:
                weights = np.ones(complexity) / complexity
            else:
                weights = np.random.dirichlet(np.ones(complexity))

            spectrum = np.sum(spectral_matrix[:, mol_indices] * weights, axis=1)

            label = np.zeros(num_molecules)
            label[mol_indices] = 1.0

            mixtures.append(spectrum)
            labels.append(label)

    return np.stack(mixtures), np.stack(labels)

# -----------------------------
# Step 3: PyTorch Dataset Class
# -----------------------------
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Step 4: Neural Network Model
# -----------------------------
class SpectraClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Extract first-level features
        x = F.relu(self.fc2(x))  # Learn abstract patterns
        return self.out(x)       # Output raw scores (logits)

# -----------------------------
# Step 4.5: Loss helper functions
# -----------------------------
def strict_recall_loss(y_pred, y_true, precision_threshold=0.95, alpha=5.0, eps=1e-6):
    """
    Soft version of strict recall penalty.
    Penalizes predictions if precision is below the threshold.
    """
    TP = torch.sum(y_true * y_pred, dim=1)
    FP = torch.sum((1 - y_true) * y_pred, dim=1)
    FN = torch.sum(y_true * (1 - y_pred), dim=1)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)

    # Add penalty when precision < threshold
    penalty = torch.where(
        precision < precision_threshold,
        alpha * (precision_threshold - precision),
        torch.zeros_like(precision)
    )

    return (1 - recall + penalty).mean()

def hybrid_loss(y_pred, y_true, lambda_weight=1.0):
    """
    Combines standard binary cross-entropy with precision-aware recall loss.
    lambda_weight controls the influence of the custom loss.
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    strict = strict_recall_loss(y_pred, y_true)
    return bce + lambda_weight * strict


# -----------------------------
# Step 5: Training Function
# -----------------------------
def train_model(model, dataloader, loss_fn, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)  # logits, not probs
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")



# -----------------------------
# Step 6: Evaluation Function
# -----------------------------
def evaluate_model_vs_complexity(model, spectral_matrix, molecule_names, device,
                                 max_complexity=25, N_per_complexity=20,
                                 threshold=0.5, score_fn=strict_recall_score):
    import matplotlib.pyplot as plt

    model.eval()
    scores = []

    for complexity in range(1, max_complexity + 1):
        # Generate N synthetic mixtures of this complexity
        X_eval, Y_eval = generate_mixture_dataset(
            spectral_matrix,
            N=N_per_complexity,
            min_complexity=complexity,
            max_complexity=complexity,
            equal_weights=True,
            seed=complexity * 13 + 42  # unique seed per complexity
        )

        with torch.no_grad():
            inputs = torch.tensor(X_eval, dtype=torch.float32).to(device)
            outputs = torch.sigmoid(model(inputs)).cpu().numpy()
            preds = (outputs >= threshold).astype(int)

        # Score each sample individually
        complexity_scores = []
        for true_vec, pred_vec in zip(Y_eval, preds):
            true_idxs = np.where(true_vec == 1)[0]
            pred_idxs = np.where(pred_vec == 1)[0]
            complexity_scores.append(score_fn(true_idxs, pred_idxs))

        scores.append((complexity, complexity_scores))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 4))

    xs = []
    ys = []

    for complexity, scs in scores:
        for score in scs:
            x_jitter = np.random.uniform(-0.2, 0.2)
            xs.append(complexity + x_jitter)
            ys.append(score)

    ax.scatter(xs, ys, color='C0', s=50, edgecolors='k', linewidths=0.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title("Strict Recall vs. Mixture Complexity")
    ax.set_xlabel("Number of Molecules in Mixture")
    ax.set_ylabel("Strict Recall Score")

    ax.set_xlim(0.5, max_complexity + 0.5)
    ax.set_ylim(min(-0.1, min(ys)-0.1), max(1.05, max(ys)+0.05))
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution Block
# -----------------------------
def main():
    spectra_file = "mass_spectra_individual.csv"
    metadata_file = "mass_spectra_metadata_individual.csv"

    # === Data Generation ===
    N_MIXTURES = 50
    MAX_COMPLEXITY = 25
    X_train, Y_train, molecule_names = load_and_prepare_data(
        spectra_file,
        metadata_file,
        N_Mixtures=N_MIXTURES,
        max_complexity=MAX_COMPLEXITY
    )

    # === Dataloader Setup ===
    train_ds = SpectraDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # === Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraClassifier(input_dim=X_train.shape[1], num_classes=Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # define loss fn here
    # In main()
    label_freq = Y_train.mean(axis=0)  # frequency of each class
    pos_weight = 1.0 / (label_freq + 1e-6)  # avoid div-by-zero
    pos_weight = torch.tensor(pos_weight).float().clamp(max=20.0).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # === Training ===
    print("🚀 Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, device, epochs=100) # using custom loss

    # === Evaluation ===
    evaluate_model_vs_complexity(model, spectral_matrix=X_train.T, molecule_names=molecule_names, device=device)

if __name__ == "__main__":
    main()

