# Author: Alex Seager
# Last Version: 7/16/25
#
# Description: I am attempting to build a neural network which classifies 
# molecules into molecular groups based on their spectra. I am wokring 
# with data from the MIT accuTOF MS instrument which is manually labelled
# and I am using pytorch to create the NN pipeline
# note: this is exploratory and as much for my lerning as anything else

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

# -----------------------------
# Step 2: Load and Preprocess Data
# -----------------------------
def load_and_prepare_data(spectra_path, metadata_path, min_group_size=5, exclude_groups=["Misc"]):
    # Load spectra and metadata
    X = pd.read_csv(spectra_path, index_col=0).T
    meta = pd.read_csv(metadata_path)

    # Align metadata to spectra
    meta = meta.set_index("molecule").loc[X.index]
    y_raw = meta["group"]

    # Filter out unwanted groups (e.g., "Misc")
    if exclude_groups:
        mask_exclude = ~y_raw.isin(exclude_groups)
        X = X[mask_exclude]
        y_raw = y_raw[mask_exclude]

    # Group count filtering (ensure enough total samples per class)
    group_counts = y_raw.value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index
    mask_valid = y_raw.isin(valid_groups)

    X_filtered = X[mask_valid]
    y_filtered = y_raw[mask_valid]

    # Encode group labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filtered)

    # Normalize each spectrum row-wise (preserve relative peak heights)
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_filtered, norm='l2')

    print(f"✅ Final groups used: {list(le.classes_)}")
    print(f"🧪 Total molecules used: {X_filtered.shape[0]}")

    from collections import Counter
    print("📊 Group distribution:", dict(Counter(y_encoded)))

    return X_norm, y_encoded, le

#for lumping extras together
def load_and_prepare_data_lump_small_groups(spectra_path, metadata_path, min_group_size=5, exclude_groups=["Misc"], lump_label="Other"):
    # Load spectra and metadata
    X = pd.read_csv(spectra_path, index_col=0).T
    meta = pd.read_csv(metadata_path)

    # Align metadata to spectra
    meta = meta.set_index("molecule").loc[X.index]
    y_raw = meta["group"]

    # Count group sizes
    group_counts = y_raw.value_counts()

    # Identify groups to lump
    small_groups = group_counts[group_counts < min_group_size].index.tolist()
    lump_groups = set(small_groups) | set(exclude_groups)

    # Replace small or excluded groups with lump_label
    y_lumped = y_raw.apply(lambda g: g if g not in lump_groups else lump_label)

    # Normalize spectra
    X_norm = normalize(X, norm='l2')

    # Encode group labels and manually place 'Other' last
    unique_labels = sorted(set(y_lumped) - {lump_label}) + [lump_label]
    le = LabelEncoder()
    le.classes_ = np.array(unique_labels)
    y_encoded = le.transform(y_lumped)

    # Confirm label counts
    from collections import Counter
    print(f"✅ Final groups used: {list(le.classes_)}")
    print(f"🧪 Total molecules used: {X_norm.shape[0]}")
    print("📊 Group distribution:", dict(Counter(y_encoded)))

    return X_norm, y_encoded, le

#for diff train and test sets
def load_and_split_train_test(
    spectra_path, 
    metadata_path, 
    min_group_size=5, 
    exclude_groups=["Misc"], 
    lump_label="Other", 
    test_size_valid=0.2, 
    test_size_other=0.2,
    binarize=False,
    intensity_threshold_frac=0.05
):
    # Load all data
    X_all = pd.read_csv(spectra_path, index_col=0).T
    meta_all = pd.read_csv(metadata_path).set_index("molecule").loc[X_all.index]
    y_raw = meta_all["group"]

    # Count group sizes
    group_counts = y_raw.value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index
    lump_groups = set(group_counts[group_counts < min_group_size].index).union(exclude_groups)

    # === Split valid group samples ===
    valid_mask = y_raw.isin(valid_groups)
    X_valid = X_all[valid_mask]
    y_valid = y_raw[valid_mask]
    mols_valid_train, mols_valid_test = train_test_split(X_valid.index, stratify=y_valid, test_size=test_size_valid, random_state=42)

    X_train = X_valid.loc[mols_valid_train]
    y_train = y_valid.loc[mols_valid_train]

    # Normalize
    X_train_norm = normalize(X_train, norm='l2')

    # Binarize if requested
    if binarize:
        X_train_bin = np.zeros_like(X_train_norm)
        for i in range(X_train_norm.shape[0]):
            max_val = np.max(X_train_norm[i])
            threshold = intensity_threshold_frac * max_val
            X_train_bin[i] = np.where(X_train_norm[i] >= threshold, 1.0, 0.0)
        X_train_norm = X_train_bin

    le_train = LabelEncoder()
    y_train_encoded = le_train.fit_transform(y_train)

    # === Split "Other" samples ===
    other_mask = y_raw.isin(lump_groups)
    X_other = X_all[other_mask]
    y_other = pd.Series([lump_label] * X_other.shape[0], index=X_other.index)
    mols_other_test = X_other.sample(frac=test_size_other, random_state=42).index

    # === Combine test set ===
    mols_test = mols_valid_test.union(mols_other_test)
    X_test = X_all.loc[mols_test]
    y_test_raw = y_raw.loc[mols_test].apply(lambda g: lump_label if g in lump_groups else g)
    
    X_test_norm = normalize(X_test, norm='l2')

    if binarize:
        X_test_bin = np.zeros_like(X_test_norm)
        for i in range(X_test_norm.shape[0]):
            max_val = np.max(X_test_norm[i])
            threshold = intensity_threshold_frac * max_val
            X_test_bin[i] = np.where(X_test_norm[i] >= threshold, 1.0, 0.0)
        X_test_norm = X_test_bin

    # Label encode test set with "Other" last
    test_labels = sorted(set(y_test_raw) - {lump_label}) + [lump_label]
    le_eval = LabelEncoder()
    le_eval.classes_ = np.array(test_labels)
    y_test_encoded = le_eval.transform(y_test_raw)

    return X_train_norm, y_train_encoded, le_train, X_test_norm, y_test_encoded, le_eval

# -----------------------------
# Step 3: PyTorch Dataset Class
# -----------------------------
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

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
# Step 5: Training Function
# -----------------------------
def train_model(model, dataloader, loss_fn, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() # Reset Gradients
            logits = model(X_batch)  #forward pass
            loss = loss_fn(logits, y_batch) # calc loss
            loss.backward() #backwards pass 
            optimizer.step() #update weights and biases

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------------
# Step 6: Evaluation Function
# -----------------------------
def evaluate_model(model, dataloader, label_encoder, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_))

    # generate confusion matrix
    # Get class labels
    # Assume label_encoder and all_preds / all_targets already defined
    class_labels = list(label_encoder.classes_)
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(len(class_labels)))

    # Reverse rows to have bottom-left to top-right diagonal
    cm = cm[::-1, :]
    y_labels = class_labels[::-1]
    x_labels = class_labels

    vmax = cm.max() if cm.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = val / vmax

            is_diagonal = (i + j) == (cm.shape[0] - 1)

            if val == 0:
                color = "white"
                text_color = "black"
            elif is_diagonal:
                color = plt.cm.Greens(norm_val)
                text_color = "black" if norm_val < 0.5 else "white"
            else:
                color = plt.cm.Reds(norm_val)
                text_color = "black" if norm_val < 0.5 else "white"

            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            ax.text(j + 0.5, i + 0.5, str(val), ha="center", va="center", color=text_color, fontsize=9)

    # Set centered ticks
    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=30, ha="center", va="top", fontsize=9)
    ax.tick_params(axis='x', pad=10)  # increase space between ticks and labels
    ax.set_yticklabels(y_labels, va="center", fontsize=9)

    # Label and format
    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xlim(0, cm.shape[1])
    plt.ylim(0, cm.shape[0])
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# With low confidence throw into other bin
def evaluate_model_with_threshold(model, dataloader, label_encoder, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []

    # Ensure "Other" is added to label_encoder classes for evaluation
    class_list = list(label_encoder.classes_)
    if "Other" not in class_list:
        class_list.append("Other")
    label_classes_extended = np.array(class_list)
    other_index = class_list.index("Other")


    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)

            max_probs, raw_preds = torch.max(probs, dim=1)

            # Replace low-confidence predictions with 'Other'
            adjusted_preds = torch.where(
                max_probs < threshold,
                torch.tensor(other_index).to(device),
                raw_preds
            )

            all_preds.extend(adjusted_preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    # Classification Report
    print(f"\n📊 Classification Report (threshold = {threshold:.2f}):")
    print(classification_report(
        all_targets, all_preds, target_names=label_encoder.classes_
    ))

    # Confusion Matrix
    class_labels = list(label_encoder.classes_)
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(len(class_labels)))

    # Flip for visual style
    cm = cm[::-1, :]
    y_labels = class_labels[::-1]
    x_labels = class_labels

    vmax = cm.max() if cm.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = val / vmax

            is_diagonal = (i + j) == (cm.shape[0] - 1)

            if val == 0:
                color = "white"
                text_color = "black"
            elif is_diagonal:
                color = plt.cm.Greens(norm_val)
                text_color = "black" if norm_val < 0.5 else "white"
            else:
                color = plt.cm.Reds(norm_val)
                text_color = "black" if norm_val < 0.5 else "white"

            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            ax.text(j + 0.5, i + 0.5, str(val), ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=30, ha="center", va="top", fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.set_yticklabels(y_labels, va="center", fontsize=9)

    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xlim(0, cm.shape[1])
    plt.ylim(0, cm.shape[0])
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.show()



# -----------------------------
# Main Execution Block
# -----------------------------
def main():
    spectra_file = "mass_spectra_individual.csv"
    metadata_file = "mass_spectra_metadata_individual.csv"

    X_train, y_train, le_train, X_test, y_test, le_eval = load_and_split_train_test(
    spectra_file, metadata_file,
    min_group_size=5,
    exclude_groups=["Misc"],
    binarize=False,
    intensity_threshold_frac=0
)
    # === Loaders, model, training, evaluation ===
    train_ds = SpectraDataset(X_train, y_train)
    test_ds = SpectraDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraClassifier(input_dim=X_train.shape[1], num_classes=len(le_train.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("🚀 Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, device, epochs=199)

    evaluate_model_with_threshold(model, test_loader, le_eval, device, 0.5)


if __name__ == "__main__":
    main()
