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

MAX_SUPPORT_L0 = 2 # Define your L0 support here

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
    result = ABESS(A, b, sMax=25, exhaustive_k=True)
    
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


def run_abess_trial(spectralMatrix, mode, x, COMPLEXITY):
    return run_single_trial(safe_ABESS, spectralMatrix, mode, x, COMPLEXITY, method_name="ABESS")

def run_l0_trial(spectralMatrix, mode, x, COMPLEXITY):
    return run_single_trial(safe_L0, spectralMatrix, mode, x, COMPLEXITY, method_name="L0")

def run_lasso_trial(spectralMatrix, mode, x, COMPLEXITY):
    return run_single_trial(safe_Lasso, spectralMatrix, mode, x, COMPLEXITY, method_name="Lasso")



def run_single_trial(func, A, mode, x, COMPLEXITY, method_name="Unknown"):
    print(f"{method_name} ran") 

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


from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

def master_plot(spectralMatrix, mode='complexity', x_values=None, num_trials=5, max_support = MAX_SUPPORT_L0):

    methods = {
        "Lasso": safe_Lasso,
        "L0": safe_L0,
        "ABESS": safe_ABESS
    }


    COMPLEXITY = 10
    if x_values is None:
        if mode == 'complexity':
            x_values = list(range(1, 25))
        elif mode == 'snr':
            x_values = [1000000, 100000, 10000, 1000, 100, 10, 5, 2, 1, 0.5, 0.2, 0.1]
        elif mode == 'concentrations':
            x_values = [1, 5, 10, 50, 100, 1000, 10000]
        elif mode == 'library_size':
            x_values = list(range(20, spectralMatrix.shape[1] + 1, 5))
        elif mode == 'known_proportion':
            x_values = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        else:
            raise ValueError("Invalid mode")

    # === 1. Dispatcher ===
    def run_dispatch(method, func, A, mode, x, COMPLEXITY):
        if method == "L0":
            if (mode == "complexity" and x > max_support) or (mode != "complexity" and COMPLEXITY > max_support):
                print(f">>> SKIP {method} | x = {x} due to max_support")
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

    plt.tight_layout()
    plt.savefig("Models_test.png", dpi=300)
    #plt.show()



def main():
    print("starting...")
    file = "mass_spectra_individual.csv"
    A, df = LoadRealMatrix(file)
    master_plot(A, mode='snr', x_values=None, num_trials=10)


if __name__ == "__main__":
    main()