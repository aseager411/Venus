# Author: Alex Seager
# Last Version: 7/8/25
#
# Description: I am attempting to invert MS data from combinations and recover which molecules
# from a library contributed to the spectral signal. I explore many methods including Lasso,
# ABESS, L0, and a trained neural network.

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from MS_Inversion_Toy import (
    Lasso_L1,
    f_beta,
    L_Zero,
    OneSampleTest,
    NSampleTest
)
from ABESS import ABESS

# Import the neural net classifier and loader
from MS_Inversion_NN import SpectraClassifier

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
    df_full = pd.read_csv(csv_path, index_col=0)
    df = df_full.loc[50:450]
    short_names = df.columns.str.rsplit("_", n=1).str[0]
    grouped_df = df.T.groupby(short_names).mean().T
    if numWavelengths is not None:
        grouped_df = grouped_df.iloc[:numWavelengths, :]
    if numMolecules is not None:
        grouped_df = grouped_df.iloc[:, :numMolecules]
    A = grouped_df.values
    if normalize:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1
        A = A / norms
    return A, grouped_df

# -----------------------------
# Sample Generation
# -----------------------------
def GetSample(molecule_names, spectral_df):
    spectra = np.zeros(spectral_df.shape[0])
    used = []
    for name in molecule_names:
        if name in spectral_df.columns:
            spectra += spectral_df[name].values
            used.append(name)
        else:
            cols = [c for c in spectral_df.columns if c.startswith(name)]
            if not cols:
                raise ValueError(f"Molecule '{name}' not found.")
            avg = spectral_df[cols].mean(axis=1).values
            spectra += avg
            used.append(f"{name}({len(cols)})")
    return spectra, used

# -----------------------------
# Method Tests
# -----------------------------
def Lasso_Test(matrix, spectra, alpha, df):
    x_hat = Lasso_L1(matrix, spectra, alpha)
    thresh = 1e-3
    return [(df.columns[i], x_hat[i]) for i in range(len(x_hat)) if abs(x_hat[i])>thresh]


def ABESS_Test(matrix, spectra, sMax, df):
    x_hat = ABESS(matrix, spectra, sMax)
    thresh = 1e-4
    if isinstance(x_hat[0], tuple):
        return {df.columns[i]:coef for i,coef in x_hat if abs(coef)>thresh}
    return {df.columns[i]:x_hat[i] for i in range(len(x_hat)) if abs(x_hat[i])>thresh}


def L_Zero_Test(matrix, spectra, df):
    x_hat = L_Zero(matrix, spectra)
    thresh = 1e-4
    if isinstance(x_hat[0], tuple):
        return {df.columns[i]:coef for i,coef in x_hat if abs(coef)>thresh}
    return {df.columns[i]:x_hat[i] for i in range(len(x_hat)) if abs(x_hat[i])>thresh}

# -----------------------------
# Neural Network Test
# -----------------------------
def NN_Test(matrix, spectra, df, checkpoint_path, device, threshold=0.5):
    # spectra: 1D numpy array
    # matrix unused here except for shape
    model = SpectraClassifier(input_dim=matrix.shape[0], num_classes=matrix.shape[1]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(spectra, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(inp)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    results = [(df.columns[i], float(probs[i])) for i in range(len(probs)) if probs[i]>=threshold]
    return results

# -----------------------------
# Main Execution
# -----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load the trained library matrix from CSV ---
    df1 = pd.read_csv('mass_spectra_individual.csv', index_col=0)
    A_ind = df1.values  # shape: (num_bins, num_molecules)

    # --- Load mixture grouping for sampling ---
    A_mix, df2 = LoadRealMatrix('mass_spectra_mixtures.csv')

    # Create an example synthetic sample from the mixtures DataFrame
    spectra2, names2 = GetSample(['B6M3'], df2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load matrices
    A_ind, df1 = LoadRealMatrix('mass_spectra_individual.csv')
    A_mix, df2 = LoadRealMatrix('mass_spectra_mixtures.csv')

    # Example mix
    spectra2, names2 = GetSample(['B6M3'], df2)

    print('Lasso predictions:')
    print(Lasso_Test(A_ind, spectra2, alpha=1e5, df=df1))

    print('ABESS predictions:')
    print(ABESS_Test(A_ind, spectra2, sMax=10, df=df1))

    print('NN predictions:')
    nn_results = NN_Test(A_ind, spectra2, df1, 'spectra_classifier_recon.pth', device, threshold=0.8)
    for name, prob in nn_results:
        print(f"  {name}: {prob:.3f}")

if __name__=='__main__':
    main()
