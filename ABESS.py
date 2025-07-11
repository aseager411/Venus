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


# Outer most call 
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
#            sMax -> the largest guess of how many molecules could make up the sample
#            df -> the data frame for name purposes 
# Returns:   vector prediction -> the guess for which molecules at what concentrations made up the sample
def compute_sic(y, y_pred, s, p, alpha=2):
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    if rss == 0:
        rss = 1e-10
    sic = n * np.log(rss / n) + alpha * s * np.log(p) * np.log(np.log(n))
    return sic

def ABESS(matrix, spectra, sMax, df=None):
    best_sic = float('inf')
    best_molecules = None
    best_coefficients = None
    best_s = None

    p = matrix.shape[1]  # number of molecules

    for s in range(1, sMax + 1):
        selected_indices, coefficients = Splice(matrix, spectra, s, k=1)
        y_pred = matrix[:, selected_indices] @ coefficients
        sic = compute_sic(spectra, y_pred, s, p)

        if sic < best_sic:
            best_sic = sic
            best_molecules = selected_indices
            best_coefficients = coefficients
            best_s = s

    # Retrieve names from dataframe if provided, else just return indices
    if df is not None:
        names = df.columns[best_molecules].tolist()
    else:
        names = best_molecules

    result = list(zip(names, best_coefficients))

    print(f"Best model found at s = {best_s} with SIC = {best_sic:.2f}")
    return result

# The splicing algorithm to find the best s molecules  
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
#            int s -> how many molecules to try to make up the sample
#            int k -> how many to splice at once
# Returns:   vector prediction -> the guess for which molecules at what concentrations made up the sample
import numpy as np

from scipy.optimize import lsq_linear

def Splice(matrix, spectra, s, k, max_iter=100, tol=1e-6):
    """
    Perform best-subset selection using the splicing algorithm with non-negative coefficients.
    
    Arguments:
    matrix  -- (n_samples, n_molecules) 2D numpy array. Each column is a molecule's spectrum.
    spectra -- (n_samples,) target spectrum to fit as linear combination of matrix columns.
    s       -- number of molecules to select in the subset.
    k       -- number of variables to splice (swap in/out) per iteration.
    max_iter-- maximum number of splicing iterations.
    tol     -- improvement threshold to continue splicing (stopping criterion).
    
    Returns:
    selected_indices -- list of indices of selected columns.
    coefficients     -- corresponding non-negative least-squares coefficients (same order as indices).
    """
    n, p = matrix.shape

    # Step 1: Initialize active set with s most correlated variables (magnitude-aware)
    norms = np.linalg.norm(matrix, axis=0) # Get norms to account for diff magnitudes
    safe_norms = norms.copy()
    safe_norms[safe_norms == 0] = 1e-10  # prevent divide-by-zero
    normalized = matrix / safe_norms

    correlations = normalized.T @ spectra
    initial_indices = np.argsort(np.abs(correlations))[-s:]

    active_set = set(initial_indices)
    inactive_set = set(range(p)) - active_set

    for iteration in range(max_iter):
        A = sorted(active_set)
        I = sorted(inactive_set)

        X_A = matrix[:, A] #sub matrix of current active set
        lsq_result = lsq_linear(X_A, spectra, bounds=(0, np.inf), method='trf') # non negative ls
        beta_A = lsq_result.x # get result from fit
        residual = spectra - X_A @ beta_A # compute residual
        loss_prev = 0.5 * np.mean(residual**2) # compute loss

        # Backward sacrifices - black box
        xi = {}
        for idx, j in enumerate(A):
            xj = matrix[:, j]
            bj = beta_A[idx]
            xi[j] = (xj @ xj) / (2 * n) * bj**2

        # Forward sacrifices - black box
        zeta = {}
        for j in I:
            xj = matrix[:, j]
            xj_norm2 = xj @ xj
            if xj_norm2 == 0:
                zeta[j] = 0
                continue
            dj = (xj @ residual) / n
            zeta[j] = xj_norm2 / (2 * n) * (dj / (xj_norm2 / n))**2

        A_k = sorted(xi, key=xi.get)[:min(k, len(A))] # select k active molecules with least sacrifice
        I_k = sorted(zeta, key=zeta.get, reverse=True)[:min(k, len(I))] # k non active with largest sacrifice

        # form new active and inactive sets
        new_active = (active_set - set(A_k)) | set(I_k)
        new_inactive = set(range(p)) - new_active

        A_new = sorted(new_active)
        X_new = matrix[:, A_new]
        lsq_result_new = lsq_linear(X_new, spectra, bounds=(0, np.inf), method='trf')
        beta_new = lsq_result_new.x
        residual_new = spectra - X_new @ beta_new
        loss_new = 0.5 * np.mean(residual_new**2)

        if loss_prev - loss_new > tol: # if no convergence repeat with new active set
            active_set = new_active
            inactive_set = new_inactive
        else:
            break

    A_final = sorted(active_set)
    X_final = matrix[:, A_final]
    final_result = lsq_linear(X_final, spectra, bounds=(0, np.inf), method='trf')
    beta_final = final_result.x
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