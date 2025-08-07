# Author: Alex Seager
# Last Version: 7/18/25
#
# Description: I am attempting to build a neural network which performs 
#              Spectral deconvolution with a machine specific library based 
#              approach. 


#should be noramlizing mixtures otherwise easy to tell complexity by signal strength

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
import matplotlib.pyplot as plt

from MS_Inversion import (
    strict_recall_score,
    f_beta
)

# -----------------------------
# Step 2: Load and Prepare Data
# -----------------------------
# Goal: upload spectra (averaged across replicates for each molecule).
# Then create artificial mixtures of size 1 to 25 (equal weight by default),
# each labeled with a binary vector indicating which molecules are present.

def load_and_prepare_data(
    spectra_path,
    metadata_path,
    N_Mixtures=100,
    max_complexity=25,
    seed=42,
    noise=True,
    normalize_library=True,
    mz_min=50,
    mz_max=450,
):
    # 1) Read in the full library matrix (bins × runs)
    df_full = pd.read_csv(spectra_path, index_col=0)  # index = m/z

    # 2) Truncate the m/z window
    #    keeps only rows with 50 ≤ m/z ≤ 450
    df_window = df_full.loc[mz_min:mz_max]

    # 3) Transpose: now rows are runs (e.g. "alanine_1"), columns are bins 50–450
    X_raw = df_window.T  # shape: (runs × selected_bins)

    # 4) Load metadata and align to the runs
    meta = pd.read_csv(metadata_path).set_index("molecule")
    meta = meta.reindex(X_raw.index)

    # 5) Build grouping key (short_molecule or strip "_N")
    if "short_molecule" in meta.columns:
        grouping = meta["short_molecule"]
    else:
        grouping = X_raw.index.str.rsplit("_", 1).str[0]

    # 6) Average replicates per molecule
    X_avg = X_raw.groupby(grouping).mean().sort_index()  
      # shape: (unique_molecules × selected_bins)
    molecule_names = X_avg.index.tolist()

    # 7) Convert to numpy matrix (bins × molecules)
    spectral_matrix = X_avg.values.T  # shape: (num_selected_bins, num_molecules)

    # 8) (Optional) Normalize each column to unit ℓ₂ norm
    if normalize_library:
        norms = np.linalg.norm(spectral_matrix, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        spectral_matrix = spectral_matrix / norms

    # 9) Build your synthetic mixtures
    X_mixed, Y_mixed = generate_mixture_dataset(
        spectral_matrix,
        N=N_Mixtures,
        min_complexity=1,
        max_complexity=max_complexity,
        equal_weights=True,
        seed=seed,
        add_noise=noise,
        snr=5
    )

    print(f" Molecules in library: {spectral_matrix.shape[1]}")
    print(f" Selected bins: {spectral_matrix.shape[0]} (m/z {mz_min}–{mz_max})")
    print(f" Total generated mixtures: {len(X_mixed)}")

    return X_mixed, Y_mixed, molecule_names, spectral_matrix



#mix generating helper
def generate_mixture_dataset(
    spectral_matrix,
    N,
    min_complexity=1,
    max_complexity=25,
    equal_weights=True,
    snr=None,
    add_noise=True,
    seed=None
):
    """
    Generate synthetic mixtures using a consistent pipeline for training or testing.

    Now with per‐sample L2 normalization of each mixture spectrum.
    """
    if seed is not None:
        np.random.seed(seed)

    num_bins, num_molecules = spectral_matrix.shape
    mixtures = []
    labels = []

    def normalize(s):
        norm = np.linalg.norm(s)
        return s / norm if norm > 0 else s

    for complexity in range(min_complexity, max_complexity + 1):

        # Complexity == 1: ensure coverage of every singleton
        if complexity == 1:
            mol_indices_list = np.arange(num_molecules)
            np.random.shuffle(mol_indices_list)
            for idx in mol_indices_list:
                s = spectral_matrix[:, idx].copy()
                if add_noise and snr is not None:
                    s = AddNoise(snr, s)
                mixtures.append(normalize(s))
                lbl = np.zeros(num_molecules)
                lbl[idx] = 1.0
                labels.append(lbl)

            # Extra random singletons if N > num_molecules
            for _ in range(N - num_molecules):
                idx = np.random.choice(num_molecules)
                s = spectral_matrix[:, idx].copy()
                if add_noise and snr is not None:
                    s = AddNoise(snr, s)
                mixtures.append(normalize(s))
                lbl = np.zeros(num_molecules)
                lbl[idx] = 1.0
                labels.append(lbl)

            continue

        # Complexity > 1: random mixtures
        for _ in range(N):
            mol_indices = np.random.choice(num_molecules, size=complexity, replace=True)
            if equal_weights:
                weights = np.ones(complexity) / complexity
            else:
                weights = np.random.dirichlet(np.ones(complexity))

            s = np.sum(spectral_matrix[:, mol_indices] * weights, axis=1)
            if add_noise and snr is not None:
                s = AddNoise(snr, s)

            mixtures.append(normalize(s))

            lbl = np.zeros(num_molecules)
            lbl[mol_indices] = 1.0
            labels.append(lbl)

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
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        x = self.act2(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)  # logits


    
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

import torch.nn as nn
import torch.nn.functional as F

def recon_class_loss(
    logits,       # [B x M]
    y_true,       # [B x M]
    s_true,       # [B x N_bins]
    A_T,          # [M x N_bins] library matrix transposed
    pos_weight,   # [M] tensor of positive-class weights
    alpha=1.0,
    beta=0.01
):
    # 1) classification with positive‐class weighting
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce    = bce_fn(logits, y_true)

    # 2) get probabilities
    p      = torch.sigmoid(logits)               # [B x M]

    # 3) reconstruction MSE
    s_hat  = p @ A_T                             # [B x N_bins]
    mse    = F.mse_loss(s_hat, s_true)

    # 4) sparsity penalty
    l1     = p.abs().sum(dim=1).mean()           # mean ℓ₁ per sample

    return bce + alpha * mse + beta * l1

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

def train_with_recon(model, dataloader, optimizer, device,
                     A_T, pos_weight, alpha=1.0, beta=0.01, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for s_batch, y_batch in dataloader:
            # s_batch: [B x N_bins], y_batch: [B x M]
            s_batch = s_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(s_batch)   # [B x M]
            loss   = recon_class_loss(
                logits,
                y_batch,
                s_batch,
                A_T,
                pos_weight=pos_weight,
                alpha=alpha,
                beta=beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}")
# -----------------------------
# Step 6: Evaluation Function
# -----------------------------
def evaluate_model_with_noise_levels(model, spectral_matrix, molecule_names, device,
                                     snr_values=[3, 5, 8], max_complexity=25, N_per_complexity=20,
                                     threshold=0.8, score_fn=None,
                                     noise=True, equal_weights=True):
    if score_fn is None:
        score_fn = lambda true_idxs, pred_idxs: f_beta(true_idxs, pred_idxs, beta=1)

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
            # generate test mixtures
            X_eval, Y_eval = generate_mixture_dataset(
                spectral_matrix,
                N=N_per_complexity,
                min_complexity=complexity,
                max_complexity=complexity,
                equal_weights=equal_weights,
                snr=snr,
                add_noise=noise,
                seed=None
            )

            model.eval()
            with torch.no_grad():
                inputs  = torch.tensor(X_eval, dtype=torch.float32).to(device)
                logits  = model(inputs)
                probs   = torch.sigmoid(logits).cpu().numpy()
                preds   = (probs >= threshold).astype(int)

            # Score all samples
            for i in range(len(Y_eval)):
                true_idxs = np.where(Y_eval[i] == 1)[0]
                pred_idxs = np.where(preds[i]   == 1)[0]
                score     = score_fn(true_idxs, pred_idxs)
                x_jit     = complexity + offsets.get(snr,0) + np.random.uniform(-jitter_amp, jitter_amp)
                results.append((snr, x_jit, score))

            # Print one example *with probabilities*
            i = 0
            true_idxs = np.where(Y_eval[i] == 1)[0]
            pred_idxs = np.where(preds[i]   == 1)[0]
            true_names = [molecule_names[j] for j in true_idxs]
            print(f"\nComplexity {complexity} | {'No Noise' if not noise else f'SNR {snr}'}")
            print(f" True: {true_names}")
            print(" Predicted (name: probability):")
            for j in pred_idxs:
                print(f"  • {molecule_names[j]}: {probs[i][j]:.6f}")


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
        xs = [x for _, x, _ in results]
        ys = [y for _, _, y in results]
        ax.scatter(xs, ys,
                   color='blue',
                   marker='o',
                   label='No Noise',
                   s=60, alpha=0.8,
                   edgecolors='k', linewidths=0.5)

    ax.set_xlabel("Number of Molecules in Mixture")
    ax.set_ylabel("F₁ Score" if score_fn.__name__ == "<lambda>" else "Score")
    ax.set_title("Model Performance vs. Mixture Complexity" + (" (Noisy)" if noise else " (Clean)"))
    ax.set_xlim(0.5, max_complexity + 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title='Noise Level' if noise else 'Evaluation')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("NN_Test_Noise_Levels.png", dpi=300)
    plt.show()


def evaluate_on_fixed_test_set(model, X_test, Y_test, molecule_names, device, threshold=0.01):
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

def main():
    spectra_file = "mass_spectra_individual.csv"
    metadata_file = "mass_spectra_metadata_individual.csv"

        # === Training Data ===
    N_MIXTURES = 50000
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
        equal_weights=False,
        seed=1337  # different from training
    )

    # === Dataloader Setup ===
    train_ds = SpectraDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

   # === Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraClassifier(input_dim=X_train.shape[1], num_classes=Y_train.shape[1]).to(device)
    
    #trying new optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # === Loss function: BCEWithLogitsLoss + pos_weight to handle class imbalance ===
    # label_freq = Y_train.mean(axis=0)  # fraction positive per molecule
    # pos_weight_vec = torch.tensor(
    #     np.minimum(1.0 / (label_freq + 1e-6), 20.0),
    #     dtype=torch.float32
    # ).to(device)  # cap to avoid extreme scaling

    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec)


    # define loss fn here
    # 
    # label_freq = Y_train.mean(axis=0)  # frequency of each class
    # pos_weight = 1.0 / (label_freq + 1e-6)  # avoid div-by-zero
    # pos_weight = torch.tensor(pos_weight).float().clamp(max=20.0).to(device)

    # loss_fn = lambda logits, y: hybrid_loss_with_L1(
    #     torch.sigmoid(logits), y,
    #     lambda_weight=0,
    #     l1_weight=0.000001
    # )

    # === Training ===
    # print("Starting training...")
    # train_model(model, train_loader, loss_fn, optimizer, device, epochs=20) 
    # build A_T for reconstruction
    A_T = torch.tensor(
        spectral_matrix.T,
        dtype=torch.float32,
        device=device
    )

    # compute pos_weight vector exactly as before
    label_freq    = Y_train.mean(axis=0)  # [M]
    pos_weight_np = np.minimum(1.0 / (label_freq + 1e-6), 20.0)
    pos_weight_vec= torch.tensor(
        pos_weight_np,
        dtype=torch.float32,
        device=device
    )

    # now train with the reconstruction‐regularized loss
    print("Starting training with reconstruction loss…")
    train_with_recon(
        model,
        train_loader,
        optimizer,
        device,
        A_T,
        pos_weight=pos_weight_vec,
        alpha=10.0,
        beta=0.01,
        epochs=100
    )

    # after training completes
    save_path = "spectra_classifier_recon.pth"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "molecule_names": molecule_names,
    }, save_path)
    print(f"✅ Saved checkpoint to {save_path}")

    # === Evaluation ===
    #evaluate_on_fixed_test_set(model, X_test, Y_test, molecule_names, device=device)
    evaluate_model_with_noise_levels(model, spectral_matrix=spectral_matrix, molecule_names=molecule_names, device=device, max_complexity=25, noise = True, equal_weights=False)

if __name__ == "__main__":
    main()

