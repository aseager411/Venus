# Author: Alex Seager
# Last Version: 6/17/25
#
# Description: I am attempting to invert simulated MS data and recover which molecules 
# contributed to the spectral signal. I explore many ways at attempting to do this.

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
# importing generated data from teh data generation file
from MS_Model_Current import (
    spectralMatrix,
    GetSampleSpectrum,
    AddNoise,
    PlotSpectra,
    SAMPLECOMPLEXITY,
    NUMMOLECULES,
    NUMWAVELENGTHS,
    SNR
)

# Tasks
# add cost to L0
# normalize guesses ie guessed concentrations add to 1
# consider concentrations in model scoring?
# confusion matrix - false neg false pos etc

# Questions
# is it worth it to implement concentrations adding to one optimization
# ask how to optimize the instrument - how many wavelengths, what resolution


# Generate all possible molecule combinations and choose the best linear fit
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
# Returns:   vector -> the guess for which molecules at what concentrations made up the sample
def Brute_Force(matrix, spectra):
    # store parameters of the best solution
    best_combo   = None
    best_R2      = -np.inf
    best_factors = None

    n = matrix.shape[1]
    ss_tot = np.sum((spectra - np.mean(spectra))**2)

    # loop over all non-empty subsets of columns
    for r in range(1, n+1):
        for cols in combinations(range(n), r):
            subM = matrix[:, cols]  # shape (m, r)

            # least-squares fit of subM @ x ≈ spectra
            x_hat, residuals, rank, s = np.linalg.lstsq(subM, spectra, rcond=None)

            # get the sum of squared residuals if the system is full rank (dont get it fully)
            if residuals.size > 0:
                ss_res = residuals[0]
            else:
                # if no residuals returned (e.g. underdetermined),
                # compute it explicitly:
                y_hat = subM.dot(x_hat)
                ss_res = np.sum((spectra - y_hat)**2)

            # compute R^2
            R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            # update best if this is a better fit
            if R2 > best_R2:
                best_R2      = R2
                best_combo   = cols
                best_factors = x_hat

    return best_combo, best_factors, best_R2


# Linear fit with L1 term which pulls coefficients to zero
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
# Returns:   vector -> the guess for which molecules at what concentrations made up the sample
def Lasso_L1(matrix, spectra):
    # Pick the L1 penalty parameter α
    alpha = 0.00001

    # Use scikit-learn model
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

    # Run the fit
    lasso.fit(matrix, spectra)

    # extract solution 
    x_hat = lasso.coef_

    # get R^2
    # Predict
    y_hat = matrix.dot(x_hat)

    # Compute R^2 manually
    r2 = r2_score(spectra, y_hat)

    return x_hat, alpha, r2


# right now this often selects the lowest alpha in the range so it's p useless
def Lasso_L1_With_Cv(matrix, spectra):
    # Generate alphas to try
    alphas = np.logspace(-4, 0, 50)
    # Set up LassoCV
    lasso_cv = LassoCV(alphas=alphas, cv=5, fit_intercept=False, max_iter=10000)
    # Fit to data, selecting the alpha that minimizes CV error
    lasso_cv.fit(matrix, spectra)
    # Extract the sparse solution and best alpha
    x_hat = lasso_cv.coef_
    best_alpha = lasso_cv.alpha_

    # Compute R² on the full dataset:
    #    R² = 1 - SS_res / SS_tot
    y_hat = matrix.dot(x_hat)
    ss_res = np.sum((spectra - y_hat) ** 2)
    ss_tot = np.sum((spectra - np.mean(spectra)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return x_hat, best_alpha, r2


##scoring model using binary prediction
def f_beta(trueSet, predSet, beta=0.5):
    true_set = set(trueSet)
    pred_set = set(predSet)

    tp = len(pred_set & true_set) # True positives
    fp = len(pred_set - true_set) # False positives
    fn = len(true_set - pred_set) # False negatives
    print("in common: ", tp)

    # precision & recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Generalized F score favoring precision 
    β2 = beta**2
    num = (1 + β2) * precision * recall
    den = β2 * precision + recall
    return num/den if den>0 else 0.0

# Testing L1 for SNR and sample complexity
def Model_Test():  #change to test function
    # spectralMatrix is the same every run
    #print("Fixed spectralMatrix:\n", spectralMatrix)
    snr_colors = {3: 'C0', 5: 'C1', 8: 'C2'}
    marker_map  = {3: 'o', 5: 's', 8: '^'}
    offsets     = {3:-0.2, 5:0.0, 8:+0.2}
    jitter_amp  = 0.05

    # draw a fresh noiseless sample (new random mixture)
    fig, ax = plt.subplots(figsize=(6,4))

    # loop over the three SNRs
    for snr in (3, 5, 8):
        xs = []  # sample complexities
        ys = []  # scores
        # loop over sample complexities
        for j in range(1, 11):
            SAMPLECOMPLEXITY = j
            for k in range (5):
                s, trueMolecules = GetSampleSpectrum(
                    SAMPLECOMPLEXITY,
                    spectralMatrix,
                    NUMMOLECULES,
                    NUMWAVELENGTHS
                )
                #print("New sample spectrum:", s)

                # add noise
                noisySpectra = AddNoise(snr, s)
                #print("New noisy spectrum:", noisySpectra)

                # Run fit
                x_sol, alpha, R2 = Lasso_L1(spectralMatrix, noisySpectra)

                ## Evaluate model efficiency
                # Disregard predictions below this concentration
                gamma = 0.001 #* max(abs(x_sol))
                # take our solution and pull out which molecules were selected (low out low concentration predictions)
                predictedMolecules = [i for i, v in enumerate(x_sol) if abs(v) > gamma] 
                print("molecules chosen by model: ", predictedMolecules)
                print("true molecules: ", trueMolecules)
                # score the models choices favoring precision over recall
                score = f_beta(trueMolecules, predictedMolecules)
                print("score: ", score) 
                
                # collect for plotting
                xs.append(j)
                ys.append(score)

        # scatter for this SNR
        # apply fixed offset + random jitter for each point
        x_plot = [x + offsets[snr] + np.random.uniform(-jitter_amp, jitter_amp)
                  for x in xs]

        ax.scatter(x_plot, ys,
                   color=snr_colors[snr],
                   marker=marker_map[snr],
                   label=f'SNR = {snr}',
                   s=60, alpha=0.8,
                   edgecolors='k', linewidths=0.5)

    ax.set_xlabel('Sample Complexity (number of molecules mixed)')
    ax.set_ylabel('Weighted F-Score')
    ax.set_title('Lasso Recovery Score vs. Sample Complexity')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Noise level')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def main():
    Model_Test()

if __name__ == "__main__":
    main()

