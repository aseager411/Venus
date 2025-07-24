# Author: Alex Seager
# Last Version: 6/23/25
#
# Description: Plotting model performance for different parameters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
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

MAX_SUPPORT_L0 = 3  # Define L0 support limit

# Import and process spectral matrix, averaging duplicate molecule columns
def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
    df_full = pd.read_csv(csv_path, index_col=0)

    # Truncate wavelength range
    df = df_full.loc[50:787]

    # Average duplicate molecule columns (by name), using updated groupby syntax
    grouped_df = df.T.groupby(df.columns).mean().T

    # Optional truncation
    if numWavelengths is not None:
        grouped_df = grouped_df.iloc[:numWavelengths, :]
    if numMolecules is not None:
        grouped_df = grouped_df.iloc[:, :numMolecules]

    # Convert to matrix
    A = grouped_df.values

    if normalize:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1
        A = A / norms

    #print("Molecules used:", list(grouped_df.columns))
    return A, grouped_df

def safe_ABESS(A, b, threshold=1e-4):
    result = ABESS(A, b, sMax=25, exhaustive_k= True)
    return [i for i, coef in result if abs(coef) > threshold]


def run_single_trial(func, A, mode, x, COMPLEXITY):
    if mode == 'complexity':
        s, trueMols = GetSampleSpectrum(x, A)
        pred = func(A, s)

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
        known_size = int(x * A.shape[1])
        known_indices = np.random.choice(A.shape[1], size=int(x * A.shape[1]), replace=False)
        A_known = A[:, known_indices]
        s, trueMols = GetSampleSpectrum(COMPLEXITY, A)
        pred = func(A_known, s)
        trueMols_in_known = [i for i in trueMols if i in known_indices]
        pred = [known_indices[i] for i in pred]
        trueMols = trueMols_in_known

    return f_beta(trueMols, pred)



def master_plot(spectralMatrix, mode='complexity', x_values=None, num_trials=5):
    methods = {
        "Lasso": lambda A, b: [i for i, v in enumerate(Lasso_L1(A, b, alpha=0.00001)) if v > 1e-4],
        "L0": lambda A, b: [i for i, v in enumerate(L_Zero(A, b, criterion='AIC', max_support=MAX_SUPPORT_L0)) if v > 1e-4],
        "ABESS": safe_ABESS
    }

    # Default parameters
    SNR = 10000
    COMPLEXITY = 5
    LIBRARYSIZE = spectralMatrix.shape[1]

    if x_values is None:
        if mode == 'complexity':
            x_values = list(range(1, 25))
        elif mode == 'snr':
            x_values = [10000, 1000, 100, 10, 8, 5, 3, 2, 1]
        elif mode == 'concentrations':
            x_values = [1, 5, 10, 50, 100, 1000, 10000]
        elif mode == 'library_size':
            x_values = list(range(20, spectralMatrix.shape[1] + 1, 5))
        elif mode == 'known_proportion':
            x_values = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        else:
            raise ValueError("Invalid mode")

    results = {method: [] for method in methods}

    for x in x_values:
        for method, func in methods.items():
            # Skip L0 if mixture complexity exceeds its support size
            if method == "L0" and mode == "complexity" and x > MAX_SUPPORT_L0:
                results[method].append(np.nan)
                continue

            if method in ["ABESS", "L0"]:
                scores = Parallel(n_jobs=-1)(
                    delayed(run_single_trial)(func, spectralMatrix, mode, x, COMPLEXITY)
                    for _ in range(num_trials)
                )
            else:  # Lasso or any fast method
                scores = [
                    run_single_trial(func, spectralMatrix, mode, x, COMPLEXITY)
                    for _ in range(num_trials)
                ]

            results[method].append(np.mean(scores))
# plotting
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

    # Ensure L0 is plotted last for visibility
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
    # Create clean legend from unique method labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # Removes duplicates while preserving last handle
    plt.legend(unique.values(), unique.keys())
    plt.xlabel(xlabel)
    plt.ylabel("Average Fβ Score")
    plt.title(f"Model Comparison: {mode}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Models_test.png", dpi=300)
    plt.show()


def main():
    print("hello")
    file = "mass_spectra_individual.csv"
    A, df = LoadRealMatrix(file)
    master_plot(A, mode='complexity', x_values=None, num_trials=1)


if __name__ == "__main__":
    main()