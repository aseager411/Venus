#Testing the saved neural net

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from MS_Neural_Net import (
    load_and_prepare_data,
    generate_mixture_dataset,
    SpectraClassifier,
    evaluate_model_with_noise_levels
)

from MS_Inversion import (
    strict_recall_score,
    f_beta
)

# -----------------------------
# Helper: Load library only
# -----------------------------
def load_library(spectra_path, metadata_path, normalize_library=True):
    """
    Reads the averaged library (no mixtures), returns molecule_names and spectral_matrix.
    """
    # Use load_and_prepare_data with zero mixtures to get library
    X_dummy, Y_dummy, molecule_names, spectral_matrix = load_and_prepare_data(
        spectra_path,
        metadata_path,
        N_Mixtures=0,
        max_complexity=1,
        seed=0,
        noise=False,
        normalize_library=normalize_library
    )
    return molecule_names, spectral_matrix

# -----------------------------
# Main test script
# -----------------------------
def main():
    # Paths
    spectra_file  = "mass_spectra_individual.csv"
    metadata_file = "mass_spectra_metadata_individual.csv"
    checkpoint    = "spectra_classifier_recon.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load library
    molecule_names, spectral_matrix = load_library(
        spectra_file,
        metadata_file,
        normalize_library=True
    )

    # Recreate and load model
    num_bins, num_mols = spectral_matrix.shape[0], spectral_matrix.shape[1]
    model = SpectraClassifier(input_dim=num_bins, num_classes=num_mols).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Noise evaluation
    evaluate_model_with_noise_levels(
        model,
        spectral_matrix=spectral_matrix,
        molecule_names=molecule_names,
        device=device,
        snr_values=[3,5,8],
        max_complexity=25,
        N_per_complexity=3,
        threshold=0.8,
        noise=True,
        score_fn=strict_recall_score,
        equal_weights=True
    )

if __name__ == "__main__":
    main()
