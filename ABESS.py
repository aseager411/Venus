# Author: Alex Seager
# Last Version: 7/10/25
#
# Description: I am attempting to implement the ABESS algorithm from zhu et al 2020 to have control over 
# specific implementation and debugging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear

from MS_Model_Current import (
    spectralMatrix,
    GetSampleSpectrum,
    AddNoise
)

# import data from csv
def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=False):
    df_full = pd.read_csv(csv_path, index_col=0)
    #truncate
    df = df_full.loc[50:787]

    A = df.values  # shape: (numWavelengths, numMolecules)

    if normalize:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1  # avoid division by zero
        A = A / norms

    return A, df


def compute_sic(y, y_pred, s, p, alpha=2):
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    if rss == 0:
        rss = 1e-10
    sic = n * np.log(rss / n) + alpha * s * np.log(p) * np.log(np.log(n))
    return sic

# Outer most call 
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
#            sMax -> the largest guess of how many molecules could make up the sample
#            df -> the data frame for name purposes 
# Returns:   vector prediction -> the guess for which molecules at what concentrations made up the sample
def ABESS(matrix, spectra, sMax, df=None, k=1, exhaustive_k=False):
    best_sic = float('inf')
    best_molecules = None
    best_coefficients = None
    best_s = None
    best_k = None

    p = matrix.shape[1]  # number of molecules

    for s in range(1, sMax + 1):
        k_values = range(1, s + 1) if exhaustive_k else [k]

        for current_k in k_values:
            #debug
            if not np.all(np.isfinite(spectra)):
                print(f"⚠️ Non-finite values in input spectra. Skipping s={s}, k={current_k}")
                continue
            selected_indices, coefficients = Splice(matrix, spectra, s, current_k)
            y_pred = matrix[:, selected_indices] @ coefficients
            sic = compute_sic(spectra, y_pred, s, p)

            if sic < best_sic:
                best_sic = sic
                best_molecules = selected_indices
                best_coefficients = coefficients
                best_s = s
                best_k = current_k

    # Retrieve names from dataframe if provided, else just return indices
    if df is not None:
        names = df.columns[best_molecules].tolist()
    else:
        names = best_molecules

    result = list(zip(names, best_coefficients))
    print(f"Best model found at s = {best_s}, k = {best_k} with SIC = {best_sic:.2f}")
    return result

# The splicing algorithm to find the best s molecules  
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
#            int s -> how many molecules to try to make up the sample
#            int k -> how many to splice at once
# Returns:   vector prediction -> the guess for which molecules at what concentrations made up the sampl


def Splice(matrix, spectra, s, k, max_iter=100, tol=1e-6):
    n, p = matrix.shape

    # Step 1: Initialize active set with s most correlated variables (magnitude-aware)
    col_norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.copy(col_norms)
    safe_norms[safe_norms < 1e-12] = 1e-12  # Avoid divide-by-zero
    normalized = matrix / safe_norms

    correlations = normalized.T @ spectra
    initial_indices = np.argsort(np.abs(correlations))[-s:]

    active_set = set(initial_indices)
    inactive_set = set(range(p)) - active_set

    for iteration in range(max_iter):
        A = sorted(active_set)
        I = sorted(inactive_set)

        X_A = matrix[:, A]

        if not np.all(np.isfinite(X_A)) or not np.all(np.isfinite(spectra)):
            print(f"⚠️ Iteration {iteration}: NaNs/Infs in X_A or spectra. Skipping.")
            continue

        lsq_result = lsq_linear(X_A, spectra, bounds=(0, np.inf), method='trf')
        beta_A = lsq_result.x

        if not np.all(np.isfinite(beta_A)):
            print(f"⚠️ Iteration {iteration}: NaNs/Infs in beta_A. Skipping.")
            continue

        residual = spectra - X_A @ beta_A

        loss_prev = 0.5 * np.mean(residual**2)

        # Backward sacrifices
        xi = {}
        for idx, j in enumerate(A):
            xj = matrix[:, j]
            bj = beta_A[idx]
            xi[j] = (xj @ xj) / (2 * n) * bj**2

        # Forward sacrifices
        zeta = {}
        for j in I:
            xj = matrix[:, j]
            xj_norm2 = xj @ xj
            if xj_norm2 < 1e-12:
                zeta[j] = 0
                continue
            dj = (xj @ residual) / n
            zeta[j] = xj_norm2 / (2 * n) * (dj / (xj_norm2 / n))**2

        A_k = sorted(xi, key=xi.get)[:min(k, len(A))]
        I_k = sorted(zeta, key=zeta.get, reverse=True)[:min(k, len(I))]

        new_active = (active_set - set(A_k)) | set(I_k)
        new_inactive = set(range(p)) - new_active

        A_new = sorted(new_active)
        X_new = matrix[:, A_new]

        if not np.all(np.isfinite(X_new)):
            print(f"⚠️ Iteration {iteration}: NaNs/Infs in X_new. Skipping.")
            continue

        lsq_result_new = lsq_linear(X_new, spectra, bounds=(0, np.inf), method='trf')
        beta_new = lsq_result_new.x

        if not np.all(np.isfinite(beta_new)):
            print(f"⚠️ Iteration {iteration}: NaNs/Infs in beta_new. Skipping.")
            continue

        residual_new = spectra - X_new @ beta_new
        loss_new = 0.5 * np.mean(residual_new**2)

        if loss_prev - loss_new > tol:
            active_set = new_active
            inactive_set = new_inactive
        else:
            break

    # Final output
    A_final = sorted(active_set)
    X_final = matrix[:, A_final]

    if not np.all(np.isfinite(X_final)):
        print("⚠️ Final fit: NaNs/Infs in X_final. Returning empty result.")
        return np.array([], dtype=int), np.array([])

    final_result = lsq_linear(X_final, spectra, bounds=(0, np.inf), method='trf')
    beta_final = final_result.x

    if not np.all(np.isfinite(beta_final)):
        print("⚠️ Final fit: NaNs/Infs in beta_final. Returning empty result.")
        return np.array([], dtype=int), np.array([])

    return np.array(A_final), beta_final



def main():
    s, molecules = GetSampleSpectrum(10, spectralMatrix)
    noisySpectra = AddNoise(10, s)
    print("true molecules: ", molecules)
    guess = ABESS(spectralMatrix, noisySpectra, 30)
    print("guesses: ", guess)

    # individual = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_individual.csv"
    # spectralMatrix, df1 = LoadRealMatrix(individual)
    # individual_names = df1.columns.tolist()

    # mixtures = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_mixtures.csv"
    # samples, df2 = LoadRealMatrix(mixtures)
    # mixture_names = df2.columns.tolist()

if __name__ == "__main__":
    main()