# Author: Alex Seager
# Last Version: 6/17/25
#
# Description: I am attempting to invert simulated MS data and recover which molecules 
# contributed to the spectral signal. I explore many ways at attempting to do this.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from joblib import Parallel, delayed
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from abess.linear import LinearRegression
from scipy.optimize import lsq_linear


# importing generated data from the data generation file
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
from ABESS import (
    ABESS
)


# Tasks
# add cost to L0
# normalize guesses ie guessed concentrations add to 1?
# consider concentrations in model scoring?

# Questions
# is it worth it to implement concentrations adding to one optimization
# ask how to optimize the instrument - how many wavelengths, what resolution




from scipy.optimize import lsq_linear
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np

# Helper for evaluating one combination
def evaluate_combo(cols, matrix, spectra, criterion, n_obs, n_vars):
    subM = matrix[:, cols]

    if np.any(~np.isfinite(subM)) or np.any(~np.isfinite(spectra)):
        print("Warning: NaN or Inf detected in inputs!")

    # Non-negative least squares fit
    result = lsq_linear(subM, spectra, bounds=(0, np.inf), method='trf')
    x_hat = result.x
    y_hat = subM @ x_hat
    rss = np.sum((spectra - y_hat) ** 2)

    if rss <= 0:
        return None  # skip degenerate fit

    k = len(cols)

    if criterion == 'AIC':
        score = 2 * k + n_obs * np.log(rss / n_obs)
    elif criterion == 'BIC':
        score = k * np.log(n_obs) + n_obs * np.log(rss / n_obs)
    elif criterion == 'MDL':
        score = k * np.log(n_vars) + np.log(rss + 1e-8)
    else:
        raise ValueError("Criterion must be 'AIC', 'BIC', or 'MDL'")

    return (score, cols, x_hat)

# L_Zero function with parallelized evaluation
def L_Zero(matrix, spectra, criterion='AIC', max_support=4, n_jobs=-1):
    best_score = np.inf
    best_combo = None
    best_factors = None

    n_obs, n_vars = matrix.shape

    for r in range(1, max_support + 1):
        combos = list(combinations(range(n_vars), r))

        # Parallel evaluation
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_combo)(cols, matrix, spectra, criterion, n_obs, n_vars)
            for cols in combos
        )

        for res in results:
            if res is not None:
                score, cols, x_hat = res
                if score < best_score:
                    best_score = score
                    best_combo = cols
                    best_factors = x_hat

    # Format output to match Lasso: full-length coefficient vector
    x_full = np.zeros(n_vars)
    if best_combo is not None:
        x_full[np.array(best_combo)] = best_factors

    print("L0 ran")

    return x_full




# Linear fit with L1 term which pulls coefficients to zero
#
# Arguments: matrix -> the matrix of molecular spectra
#            vector spectra -> the sample spectrum which in a linear combo of matrix columns
# Returns:   vector -> the guess for which molecules at what concentrations made up the sample
def Lasso_L1(matrix, spectra, alpha):

    # Use scikit-learn model
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, positive = True)

    # Run the fit
    lasso.fit(matrix, spectra)

    # extract solution 
    x_hat = lasso.coef_

    # get R^2
    # Predict
    y_hat = matrix.dot(x_hat)

    # Compute R^2 manually
    r2 = r2_score(spectra, y_hat)

    print("Lasso ran")

    return x_hat


# find a good alpha via cross-validation with many samples
def Lasso_L1_With_Cv(matrix, spectra, alphas=np.logspace(-5, 0, 50)):
    """
    Fits a Lasso regression with cross-validation to select the best alpha.

    Parameters:
        matrix (ndarray): 2D array where each column is a reference molecular spectrum
        spectra (ndarray): 1D array of the mixed/observed spectrum
        alphas (array-like): List of alpha values to search over (default: logspace)

    Returns:
        x_hat (ndarray): Coefficient vector indicating contributions of each reference
        best_alpha (float): Chosen alpha value
        r2 (float): R^2 score of the fit
    """
    lasso_cv = LassoCV(alphas=alphas, fit_intercept=False, max_iter=10000, positive=True, cv=5)
    lasso_cv.fit(matrix, spectra)

    x_hat = lasso_cv.coef_
    y_hat = matrix.dot(x_hat)
    r2 = r2_score(spectra, y_hat)

    print("Best alpha selected by cross-validation:", lasso_cv.alpha_)

    return x_hat
    


# evaluating model predictions using F Beta
#
# Arguments: vector trueSet -> The true molecules in the signal we want to recover 
#            vector predSet -> The molecules predicted by the model
#            float beta     -> below one favors retrieval over ... 
# Returns:   float -> The F score value between zero and one 
def f_beta(trueSet, predSet, beta=1):
    def extract_name(x):
        # If it's a string or int-like, return as is
        if isinstance(x, (str, int, np.integer)):
            return x
        # If it's a tuple or list, return the first element
        elif isinstance(x, (tuple, list)):
            return x[0]
        # Otherwise assume it's a scalar (e.g., float or np.float32) and return it
        else:
            return int(x)  # assumes float is a molecule index (e.g., 244.0)

    true_set = set(extract_name(t) for t in trueSet)
    pred_set = set(extract_name(p) for p in predSet)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    β2 = beta**2
    num = (1 + β2) * precision * recall
    den = β2 * precision + recall
    return num / den if den > 0 else 0.0

#scoring metric based on strict precision
def strict_recall_score(trueSet, predSet, precision_threshold=0.95):
    def extract_name(x):
        if isinstance(x, (str, int, np.integer)):
            return x
        elif isinstance(x, (tuple, list)):
            return x[0]
        else:
            return int(x)

    true_set = set(extract_name(t) for t in trueSet)
    pred_set = set(extract_name(p) for p in predSet)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision < precision_threshold:
        return -recall  # penalize if below precision threshold
    else:
        return recall   # valid only if precision ≥ threshold




###########
#  Tests  #
###########

def OneSampleTest(sampleComplexity, spectralMatrix, snr):
    s, trueMolecules = GetSampleSpectrum(sampleComplexity, spectralMatrix,)
    noisySpec = AddNoise(snr, s)
    predicted = Lasso_L1(spectralMatrix, noisySpec, 0.00001)
    print("true molecules: ", trueMolecules)
    print("predicted molecules: ", predicted)
    f_score = f_beta(trueMolecules, predicted)
    print ("f score: ", f_score)

# Testing L-zero methods with ten samples of diff complexity
def NSampleTest(spectralMatrix, snr):
    complexities = []
    f_scores = []

    for sampleComplexity in range(1, 300):

        s, trueMolecules = GetSampleSpectrum(sampleComplexity, spectralMatrix)
        noisySpec = AddNoise(snr, s)
    
        x_hat = Lasso_L1(spectralMatrix, noisySpec, 0.0001)

        predicted = [i for i, val in enumerate(x_hat) if val > 1e-4]

        print("true molecules: ", trueMolecules)
        print("predicted molecules: ", predicted)

        f_score = f_beta(trueMolecules, predicted)
        print("f score: ", f_score)

        complexities.append(sampleComplexity)
        f_scores.append(f_score)

    # Plot
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(complexities, f_scores,
                      s=100, edgecolors='k')
    plt.xlabel(f"Sample Complexity (Number of Molecules Mixed) out of {spectralMatrix.shape[1]}")
    plt.ylabel("Fβ Score")
    plt.title(f"L0 Recovery (SNR = {snr})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xlim(0.5, spectralMatrix.shape[1] + 0.5)
    plt.xticks(range(1, spectralMatrix.shape[1] + 1))

    plt.tight_layout()
    plt.show()

# testing inverse models
def Model_Test(spectralMatrix, a, noise=True, score_fn=f_beta, sampleRange = 20):
    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Parallel, delayed

    snr_values = [3, 5, 8] if noise else [None]
    snr_colors = {3: 'C0', 5: 'C1', 8: 'C2', None: 'C0'}
    marker_map  = {3: 'o', 5: 's', 8: '^', None: 'o'}
    offsets     = {3:-0.2, 5:0.0, 8:+0.2, None: 0.0}
    jitter_amp  = 0.05

    fig, ax = plt.subplots(figsize=(6, 4))

    # Define tasks
    tasks = [
        (snr, j, k)
        for snr in snr_values
        for j in range(1, sampleRange + 1)
        for k in range(1)  # increase for repeats
    ]

    # Helper function
    def run_single_test(spectralMatrix, a, snr, j, k, offsets, jitter_amp, noise):
        sampleComplexity = j
        s, trueMolecules = GetSampleSpectrum(sampleComplexity, spectralMatrix)

        noisySpectra = AddNoise(snr, s) if noise else s
        #Change model here
        x_sol = Lasso_L1_With_Cv(spectralMatrix, noisySpectra) #, 50, exhaustive_k = True)

        #change required concentration here
        gamma = 0.001
        if isinstance(x_sol[0], tuple):
            predictedMolecules = [name for (name, coef) in x_sol if abs(coef) > gamma]
        else:
            predictedMolecules = [i for i, v in enumerate(x_sol) if abs(v) > gamma]

        print(f"SNR={snr} | SampleComplexity={j} | True={sorted(trueMolecules)} | Predicted={sorted(predictedMolecules)}")
        score = score_fn(trueMolecules, predictedMolecules)
        x_val = j + offsets[snr] + np.random.uniform(-jitter_amp, jitter_amp)
        return snr, x_val, score

    # Run in parallel
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_single_test)(spectralMatrix, a, snr, j, k, offsets, jitter_amp, noise)
        for (snr, j, k) in tasks
    )

    # Plotting
    for snr in snr_values:
        xs = [x for s, x, y in results if s == snr]
        ys = [y for s, x, y in results if s == snr]

        ax.scatter(xs, ys,
                   color=snr_colors[snr],
                   marker=marker_map[snr],
                   label=f'SNR = {snr}' if noise else None,
                   s=60, alpha=0.8,
                   edgecolors='k', linewidths=0.5)

    ax.set_xlabel('Sample Complexity (number of molecules mixed)')
    ax.set_ylabel('Recall')
    ax.set_title('Lasso with CV Recovery Score vs. Sample Complexity')

    all_j_values = [j for (_, j, _) in tasks]
    max_j = max(all_j_values)

    # Dynamic tick spacing
    if max_j <= 20:
        xtick_step = 1
        fontsize = 8
    elif max_j <= 40:
        xtick_step = 2
        fontsize = 6
    elif max_j <= 80:
        xtick_step = 5
        fontsize = 5
    else:
        xtick_step = 10
        fontsize = 4

    ax.set_xticks(range(0, max_j + xtick_step, xtick_step))
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_xlim(0.5, max_j + 0.5)

    # Determine dynamic Y-axis limits based on score range
    all_scores = [y for (_, _, y) in results]
    y_min = min(min(all_scores), -0.05)
    y_max = max(max(all_scores), 1.05)
    ax.set_ylim(y_min - 0.05, y_max + 0.05)

    # Add horizontal line at 0 if negatives exist
    if y_min < 0:
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')

    if noise:
        ax.legend(title='Noise level')

    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("Test.png", dpi=300)





def main():
   s, molecules = GetSampleSpectrum(3, spectralMatrix)
   print("true molecules: ", molecules)
   guess = L_Zero(spectralMatrix, s, "BIC")
   print("guesses: ", guess)

if __name__ == "__main__":
    main()

