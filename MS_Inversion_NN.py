# Author: Alex Seager
# Last Version: 7/18/25
#
# Description: I am attempting to build a neural network which performs 
#              Spectral deconvolution with a machine specific library based 
#              approach. 


#save the distribution of log probs
# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

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

def load_and_prepare_data(spectra_path, metadata_path, N_Mixtures=100, max_complexity=25, seed=42, noise=True):
    # Load and average spectra by molecule
    X_raw = pd.read_csv(spectra_path, index_col=0).T  # shape: (samples × m/z bins)
    meta = pd.read_csv(metadata_path).set_index("molecule").loc[X_raw.index]

    # Average replicates by molecule name
    X_avg = X_raw.groupby(X_raw.index).mean()
    X_avg = X_avg.sort_index()
    molecule_names = list(X_avg.index)
    spectral_matrix = X_avg.values.T  # shape: (num_bins, num_molecules)

    # Generate synthetic mixtures
    X_mixed, Y_mixed = generate_mixture_dataset(
        spectral_matrix,
        N=N_Mixtures,
        min_complexity=1,
        max_complexity=max_complexity,
        equal_weights=True,
        seed=seed,
        add_noise=noise
    )

    print(f" Molecules in library: {spectral_matrix.shape[1]}")
    print(f" Total generated mixtures: {len(X_mixed)}")

    return X_mixed, Y_mixed, molecule_names, spectral_matrix


#mix generating helper
def generate_mixture_dataset(spectral_matrix, N, min_complexity=1, max_complexity=25,
                              equal_weights=True, snr=None, add_noise=True, seed=None):

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

        # Ensure coverage of all molecules when complexity == 1
        if complexity == 1:
            mol_indices_list = np.arange(num_molecules)
            np.random.shuffle(mol_indices_list)
            for idx in mol_indices_list:
                spectrum = spectral_matrix[:, idx]
                label = np.zeros(num_molecules)
                label[idx] = 1.0
                mixtures.append(spectrum)
                labels.append(label)

            # Add extra random ones if N > num_molecules
            for _ in range(N - num_molecules):
                idx = np.random.choice(num_molecules)
                spectrum = spectral_matrix[:, idx]
                if add_noise and snr is not None:
                    spectrum = AddNoise(snr, spectrum)
                label = np.zeros(num_molecules)
                label[idx] = 1.0
                mixtures.append(spectrum)
                labels.append(label)

            continue  # skip rest of loop for complexity == 1

        # Standard random mixtures for complexity > 1
        for _ in range(N):
            mol_indices = np.random.choice(num_molecules, size=complexity, replace=True)

            if equal_weights:
                weights = np.ones(complexity) / complexity
            else:
                weights = np.random.dirichlet(np.ones(complexity))

            spectrum = np.sum(spectral_matrix[:, mol_indices] * weights, axis=1)
            if add_noise and snr is not None:
                spectrum = AddNoise(snr, spectrum)
            label = np.zeros(num_molecules)
            label[mol_indices] = 1.0

            mixtures.append(spectrum)
            labels.append(label)

    return np.stack(mixtures), np.stack(labels)

#noise helper
def AddNoise(snr, spectra):
    mask = (spectra != 0)
    sigma = 1.0 / np.sqrt(snr)
    factors = np.random.normal(loc=1.0, scale=sigma, size=spectra.shape)
    noisy = spectra.copy()
    noisy[mask] = spectra[mask] * factors[mask]
    return noisy


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
    def __init__(self, input_dim, num_classes, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.out(x)
    
#bigger model
class BiggerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.out(x)
    
#simpler model
class SparseLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.out(x)


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

def hybrid_loss_with_L1(y_pred, y_true,
                        lambda_weight=1.0,
                        l1_weight=0.01):
    """
    Hybrid loss using:
    - Binary Cross-Entropy
    - Strict recall penalty (penalizes low precision)
    - L1 penalty to promote sparse outputs
    """

    # Binary cross-entropy
    bce = F.binary_cross_entropy(y_pred, y_true)

    # Strict recall penalty
    strict = strict_recall_loss(y_pred, y_true)

    # L1 norm of predicted probabilities
    l1 = torch.sum(y_pred, dim=1).mean()

    return bce + lambda_weight * strict + l1_weight * l1




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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # to reduce loss jumps
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# -----------------------------
# Step 6: Evaluation Function
# -----------------------------
def evaluate_model_with_noise_levels(model, spectral_matrix, molecule_names, device,
                                     snr_values=[3, 5, 8], max_complexity=25, N_per_complexity=20,
                                     threshold=0.7, score_fn=strict_recall_score,
                                     noise=True):

    snr_colors = {3: 'C0', 5: 'C1', 8: 'C2', None: 'black'}
    marker_map = {3: 'o', 5: 's', 8: '^', None: 'x'}
    offsets    = {3: -0.2, 5: 0.0, 8: +0.2, None: 0.0}
    jitter_amp = 0.05

    fig, ax = plt.subplots(figsize=(7, 5))
    results = []

    for snr in snr_values:
        label = f"SNR = {snr}" if noise else "No Noise"
        print(f"\n--- Evaluating at SNR = {snr if noise else 'None'} ---")
        for complexity in range(1, max_complexity + 1):
            # Generate test set with or without noise
            X_eval, Y_eval = generate_mixture_dataset(
                spectral_matrix,
                N=N_per_complexity,
                min_complexity=complexity,
                max_complexity=complexity,
                equal_weights=True,
                snr=snr,
                add_noise=noise,
                seed=(snr if noise else 0) * 100 + complexity  # unique seed
            )

            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_eval, dtype=torch.float32).to(device)
                outputs = torch.sigmoid(model(inputs)).cpu().numpy()
                preds = (outputs >= threshold).astype(int)

            # Score each sample
            for i in range(len(Y_eval)):
                true_idxs = np.where(Y_eval[i] == 1)[0]
                pred_idxs = np.where(preds[i] == 1)[0]
                score = score_fn(true_idxs, pred_idxs)
                x_jittered = complexity + offsets.get(snr, 0.0) + np.random.uniform(-jitter_amp, jitter_amp)
                results.append((snr, x_jittered, score))

            # Print example
            i = 0
            true_names = [molecule_names[j] for j in np.where(Y_eval[i] == 1)[0]]
            pred_names = [molecule_names[j] for j in np.where(preds[i] == 1)[0]]
            print(f"\nComplexity {complexity} | {'No Noise' if not noise else f'SNR {snr}'}")
            print(f" True: {true_names}")
            print(f" Pred: {pred_names}")

    # --- Plot ---
    if noise:
        for snr in snr_values:
            xs = [x for s, x, y in results if s == snr]
            ys = [y for s, x, y in results if s == snr]
            ax.scatter(xs, ys,
                       color=snr_colors[snr],
                       marker=marker_map[snr],
                       label=f'SNR = {snr}',
                       s=60, alpha=0.8,
                       edgecolors='k', linewidths=0.5)
    else:
        # All points same appearance when noise=False
        xs = [x for _, x, _ in results]
        ys = [y for _, _, y in results]
        ax.scatter(xs, ys,
                   color='blue',
                   marker='o',
                   label='No Noise',
                   s=60, alpha=0.8,
                   edgecolors='k', linewidths=0.5)

    ax.set_xlabel("Number of Molecules in Mixture")
    ax.set_ylabel("Strict Recall Score")
    ax.set_title("Model Performance vs. Mixture Complexity" + (" (Noisy)" if noise else " (Clean)"))
    ax.set_xlim(0.5, max_complexity + 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title='Noise Level' if noise else 'Evaluation')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("NN_Test_NoiseLevels.png", dpi=300)
    plt.show()

def evaluate_on_fixed_test_set(model, X_test, Y_test, molecule_names, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        # Average probability per molecule (axis 0 = over samples)
        mean_probs = probs.mean(axis=0)

        # Plot
        plt.figure(figsize=(12, 4))
        sns.barplot(x=molecule_names, y=mean_probs, color="steelblue")
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("Avg Predicted Probability")
        plt.title("Average Prediction per Molecule Across Test Set")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    print("\n Evaluation on held-out synthetic test set")
    correct = 0
    total = 0
    for i in range(len(Y_test)):
        true_idxs = np.where(Y_test[i] == 1)[0]
        pred_idxs = np.where(preds[i] == 1)[0]
        true_names = [molecule_names[j] for j in true_idxs]
        pred_names = [molecule_names[j] for j in pred_idxs]

        if set(true_idxs) == set(pred_idxs):
            correct += 1
        total += 1

        if i < 5:  # print only first 5 test cases
            print(f"\nExample {i+1}")
            print(f" True: {true_names}")
            print(f" Pred: {pred_names}")

    print(f"\n Exact match accuracy on test set: {correct}/{total} = {correct/total:.2f}")


# -----------------------------
# Main Execution Block
# -----------------------------
def train_model_with_curriculum(model, spectral_matrix, loss_fn, optimizer, device,
                                molecule_names, epochs=100, batch_size=256,
                                snr_schedule=[None, 8, 5, 3], mixtures_per_epoch=1000,
                                smoothing_epsilon=0.05):

    from torch.utils.data import DataLoader

    model.train()
    num_classes = spectral_matrix.shape[1]

    for epoch in range(epochs):
        # Choose noise level based on schedule
        snr = snr_schedule[min(epoch // (epochs // len(snr_schedule)), len(snr_schedule) - 1)]
        print(f"\n🌀 Epoch {epoch+1}/{epochs} | SNR = {snr}")

        # Generate noisy mixtures on-the-fly
        X_train, Y_train = generate_mixture_dataset(
            spectral_matrix,
            N=mixtures_per_epoch,
            min_complexity=1,
            max_complexity=50,
            snr=snr,
            add_noise=True,
            seed=epoch + 42
        )

        # Apply label smoothing
        Y_train = Y_train * (1 - smoothing_epsilon) + 0.5 * smoothing_epsilon

        # Prepare dataloader
        train_ds = SpectraDataset(X_train, Y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Loss: {total_loss:.4f}")




def main():
    spectra_file = "mass_spectra_individual.csv"
    metadata_file = "mass_spectra_metadata_individual.csv"

        # === Training Data ===
    N_MIXTURES = 1000
    MAX_COMPLEXITY = 25


    X_train, Y_train, molecule_names, spectral_matrix = load_and_prepare_data(
        spectra_file,
        metadata_file,
        N_Mixtures=N_MIXTURES,
        max_complexity=MAX_COMPLEXITY,
        seed=42,
        noise = True
    )


    # === Test Data (new seed) ===
    X_test, Y_test = generate_mixture_dataset(
        spectral_matrix,
        N=N_MIXTURES,
        min_complexity=1,
        max_complexity=MAX_COMPLEXITY,
        equal_weights=True,
        seed=1337  # different from training
    )


    # === Dataloader Setup ===
    train_ds = SpectraDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

   # === Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseLinearClassifier(input_dim=X_train.shape[1], num_classes=Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # === Loss function: BCEWithLogitsLoss + pos_weight to handle class imbalance ===
    label_freq = Y_train.mean(axis=0)  # fraction of samples each label appears in
    pos_weight = (1.0 / (label_freq + 1e-6)).clip(max=20.0)  # avoid huge weights
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # define loss fn here
    # 
    # label_freq = Y_train.mean(axis=0)  # frequency of each class
    # pos_weight = 1.0 / (label_freq + 1e-6)  # avoid div-by-zero
    # pos_weight = torch.tensor(pos_weight).float().clamp(max=20.0).to(device)

#     loss_fn = lambda logits, y: hybrid_loss_with_L1(
#         torch.sigmoid(logits), y,
#         lambda_weight=0,
#         l1_weight=0.01
# )
    # loss_fn = lambda logits, y_true: F.binary_cross_entropy(torch.sigmoid(logits), y_true)

    # === Training ===
    print("Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, device, epochs=100) 
    # train_model_with_curriculum(
    #     model=model,
    #     spectral_matrix=spectral_matrix,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     device=device,
    #     molecule_names=molecule_names,
    #     epochs=100,
    #     batch_size=256,
    #     snr_schedule=[None, 8, 5, 3],   # Gradually add noise
    #     mixtures_per_epoch=1000,
    #     smoothing_epsilon=0.05
    # )

    # === Evaluation ===
    #evaluate_on_fixed_test_set(model, X_test, Y_test, molecule_names, device=device)
    evaluate_model_with_noise_levels(model, spectral_matrix=spectral_matrix, molecule_names=molecule_names, device=device, max_complexity=25, noise = False)

if __name__ == "__main__":
    main()

