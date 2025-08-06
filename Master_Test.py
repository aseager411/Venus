# Author: Alex Seager
# Last Version: 6/23/25
#
# Description: Plotting model performance for different parameters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from itertools import combinations
from joblib import Parallel, delayed
import pandas as pd

from MS_Model_Current import (
    GetSampleSpectrum,
    AddNoise,
)

from MS_Inversion_Toy import (
    Lasso_L1,
    f_beta,
    L_Zero,
)

from ABESS import (
    ABESS
)

# this is the biggest size of combinations the L0 method will try
# if running locally size 2 or 3 is the maximum but higher is possible
# on a server with more compute
MAX_SUPPORT_L0 = 3 

# Import and process spectral matrix, averaging duplicate molecule columns
def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
    # Read the full matrix; first column is the bin index
    df_full = pd.read_csv(csv_path, index_col=0)

    # Truncate the mz/wavelength range
    df = df_full.loc[51:450]

    # Derive “short” names by stripping off the trailing “_1”, “_2”, etc.
    # e.g. “alanine_1” → “alanine”
    short_names = df.columns.str.rsplit("_", n=1).str[0]

    # Group columns by that short name and take the mean
    grouped_df = df.T.groupby(short_names).mean().T

    # Optional truncation of rows/columns
    if numWavelengths is not None:
        grouped_df = grouped_df.iloc[:numWavelengths, :]
    if numMolecules is not None:
        grouped_df = grouped_df.iloc[:, :numMolecules]

    # Convert to numpy array
    A = grouped_df.values

    # Column‐wise normalization (if requested)
    if normalize:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1
        A = A / norms

    return A, grouped_df


# Define wrappers for each model to account for individual settings
def safe_ABESS(A, b, threshold=1e-4, sMax=25):
    result = ABESS(A, b, sMax=5, exhaustive_k=True)  # for small lib
    
    # Result: list of (name or index, coef)
    if isinstance(result[0][0], str):
        # If molecule names are returned, convert back to index assuming A columns are ordered
        raise ValueError("ABESS returned names, but indexing is expected. Provide df to ABESS or remap here.")
    
    return [i for i, coef in result if abs(coef) > threshold]

def safe_L0(A, b, threshold=1e-4, max_support=MAX_SUPPORT_L0):
    x_hat = L_Zero(A, b, criterion='AIC', max_support=max_support)
    return [i for i, coef in enumerate(x_hat) if abs(coef) > threshold]

def safe_Lasso(A, b, threshold=1e-4, alpha=1e-5):
    x_hat = Lasso_L1(A, b, alpha=alpha)
    return [i for i, coef in enumerate(x_hat) if abs(coef) > threshold]

# Generate a mixture based on the mode and complexity and run the inverse fit with the given method
def run_single_trial(func, A, mode, x, COMPLEXITY, method_name="Unknown"):
    print(f"{method_name} ran") 

    if mode == 'complexity':
        s, trueMols = GetSampleSpectrum(x, A)
        pred = func(A, s)
        # print the name of the fucntion
        # debug output
        if method_name == "L0":
            print(f"[{method_name}] mode=complexity | x={x} | trueMols={trueMols} | pred={pred}")

    elif mode == 'snr':
        s, trueMols = GetSampleSpectrum(COMPLEXITY, A)
        b = AddNoise(x, s)
        pred = func(A, b)

    elif mode == 'library_size':
        full_indices = np.random.choice(A.shape[1], size=x, replace=False)
        A_crop = A[:, full_indices]
        s, trueMols = GetSampleSpectrum(COMPLEXITY, A_crop)
        pred = func(A_crop, s)
        trueMols = [full_indices[i] for i in trueMols]
        pred = [full_indices[i] for i in pred]

    elif mode == 'concentrations':
        indices = np.random.choice(A.shape[1], size=COMPLEXITY, replace=False)
        weights = np.geomspace(1, 1/x, num=COMPLEXITY)
        weights /= weights.sum()
        s = A[:, indices] @ weights
        pred = func(A, s)
        trueMols = indices.tolist()

    elif mode == 'known_proportion':
        total_mols = A.shape[1]
        known_size = int(x * total_mols)
        unknown_size = total_mols - known_size

        # Split full index list into known and unknown
        all_indices = np.arange(total_mols)
        known_indices = np.random.choice(all_indices, size=known_size, replace=False)
        unknown_indices = np.setdiff1d(all_indices, known_indices, assume_unique=True)

        # Choose molecules for the sample
        num_known = int(round(x * COMPLEXITY))
        num_unknown = COMPLEXITY - num_known

        true_known = np.random.choice(known_indices, size=num_known, replace=False) if num_known > 0 else np.array([], dtype=int)
        true_unknown = np.random.choice(unknown_indices, size=num_unknown, replace=False) if num_unknown > 0 else np.array([], dtype=int)

        trueMols = np.concatenate([true_known, true_unknown])
        weights = np.ones(COMPLEXITY) / COMPLEXITY
        s = A[:, trueMols] @ weights

        # Run func only on known portion of A
        A_known = A[:, known_indices]
        pred = func(A_known, s)

        # Adjust prediction indices back to full space
        pred = [known_indices[i] for i in pred]
        trueMols = true_known.tolist()  # only evaluate known ones

    # All we return is the f-score of the models guess
    return f_beta(trueMols, pred)


def master_plot(spectralMatrix, mode='complexity', x_values=None, num_trials=5, max_support = MAX_SUPPORT_L0):

    methods = {
        "Lasso": safe_Lasso,
        "L0": safe_L0,
        "ABESS": safe_ABESS
    }

    # CHANGE COMPLEXITY HERE for methods other than complexity
    COMPLEXITY = 5

    # define values for each mode
    if x_values is None:
        if mode == 'complexity':
            x_values = list(range(1, 25))
        elif mode == 'snr':
            x_values = [1000000, 100000, 10000, 1000, 100, 10, 5, 2, 1, 0.5, 0.2, 0.1]
        elif mode == 'concentrations':
            x_values = [1, 10, 100, 1000, 10000, 100000, 1000000]
        elif mode == 'library_size':
            x_values = list(range(25, spectralMatrix.shape[1] + 1, 5))
        elif mode == 'known_proportion':
            x_values = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        else:
            raise ValueError("Invalid mode")


    # paralellization for efficient computing
    # === 1. Dispatcher ===
    def run_dispatch(method, func, A, mode, x, COMPLEXITY):
        if method == "L0":
            if (mode == "complexity" and x > max_support) or (mode != "complexity" and COMPLEXITY > max_support):
                #print(f">>> SKIP {method} | x = {x} due to max_support")
                return method, x, np.nan

        try:
            score = run_single_trial(func, A, mode, x, COMPLEXITY, method_name=method)
            return method, x, score
        except Exception as e:
            return method, x, np.nan


    # === 2. Create jobs ===
    jobs = [
        (method, func, spectralMatrix, mode, x, COMPLEXITY)
        for x in x_values
        for method, func in methods.items()
        for _ in range(num_trials)
    ]

    # === 3. Run jobs in parallel ===
    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_dispatch)(method, func, A, mode, x, COMPLEXITY)
        for (method, func, A, mode, x, COMPLEXITY) in jobs
    )

    # === Count how many times each method ran ===
    from collections import Counter
    method_counter = Counter((m for (m, _, _) in raw_results))
    print("Run counts:", dict(method_counter))


    # === 4. Aggregate ===
    results = {method: [] for method in methods}
    for method in methods:
        for x in x_values:
            scores = [
                score for (m, xv, score) in raw_results
                if m == method and xv == x
            ]
            avg_score = np.mean(scores) if scores else np.nan
            results[method].append(avg_score)

    # === Plotting ===
    xlabel = {
        'complexity': "Number of Molecules in Mixture",
        'snr': "Signal-to-Noise Ratio",
        'library_size': "Number of Molecules in Library",
        'concentrations': "Ratio of Most to Least Concentrated Molecule",
        'known_proportion': "Proportion of Known Molecules in Library"
    }[mode]

    plot_styles = {
        "Lasso": {'color': 'blue', 'marker': 'o', 'markersize': 6},
        "ABESS": {'color': 'green', 'marker': '^', 'markersize': 6},
        "L0": {'color': 'orange', 'marker': 's', 'markersize': 6}
    }

    method_order = ["Lasso", "ABESS", "L0"]
    for method in method_order:
        y = results[method]
        x_filtered = [x for x, v in zip(x_values, y) if not np.isnan(v)]
        y_filtered = [v for v in y if not np.isnan(v)]

        style = plot_styles[method]
        plt.plot(
            x_filtered, y_filtered,
            label=method,
            linestyle='-' if len(x_filtered) > 1 else 'None',
            zorder=3 if method == "L0" else 2,
            **style
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.xlabel(xlabel)
    plt.ylabel("Average F Score (10 samples)")
    plt.ylim(-0.05, 1.05)  # fixed y-axis range for all plots
    plt.title(f"Model Comparison: {mode}")
    plt.grid(True)

    if mode == 'snr':
        plt.xscale('log')
        plt.gca().invert_xaxis()

    if mode == 'concentrations':
        plt.xscale('log')

    if mode == 'known_proportion':
        plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig("Models_test.png", dpi=300) #use this if running on server
    #plt.show() #use this if running locally

def main():
    print("starting...")
    file = "mass_spectra_individual.csv"
    A, df = LoadRealMatrix(file)
    master_plot(A, mode='known_proportion', x_values=None, num_trials= 1)

if __name__ == "__main__":
    main()