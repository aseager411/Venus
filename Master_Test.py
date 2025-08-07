# Author: Alex Seager
# Last Version: 8/7/25
#
# Description: Plotting model performance for different parameters,
# now including a trained neural net alongside Lasso, L0 and ABESS.

#NN is not well incorporated i beleive due to a disparity in the data format

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import torch

from MS_Model import GetSampleSpectrum, AddNoise
from MS_Inversion    import Lasso_L1, f_beta, L_Zero
from ABESS               import ABESS

# import your trained SpectraClassifier
from MS_Neural_Net import SpectraClassifier

# maximum support for L0
MAX_SUPPORT_L0 = 1

# path to your saved NN checkpoint
checkpoint_path = "spectra_classifier_recon.pth"

def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
    df_full = pd.read_csv(csv_path, index_col=0)
    df = df_full.loc[51:451]
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

# safe wrappers for existing methods
def safe_Lasso(A, b, threshold=1e-4, alpha=1e-5):
    x_hat = Lasso_L1(A, b, alpha=alpha)
    return [i for i,coef in enumerate(x_hat) if abs(coef) > threshold]

def safe_L0(A, b, threshold=1e-4, max_support=MAX_SUPPORT_L0):
    x_hat = L_Zero(A, b, criterion='AIC', max_support=max_support)
    return [i for i,coef in enumerate(x_hat) if abs(coef) > threshold]

def safe_ABESS(A, b, threshold=1e-4, sMax=25):
    result = ABESS(A, b, sMax=25, exhaustive_k=True)
    if isinstance(result[0][0], str):
        raise ValueError("ABESS returned names; expected indices.")
    return [i for i,coef in result if abs(coef) > threshold]

def master_plot(spectralMatrix, mode='complexity', x_values=None, num_trials=5, max_support=MAX_SUPPORT_L0):
    # === 0. Setup NN model once ===
    device = torch.device("cpu")
    num_bins, num_mols = spectralMatrix.shape
    model_nn = SpectraClassifier(input_dim=num_bins, num_classes=num_mols).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_nn.load_state_dict(ckpt["model_state"])
    model_nn.eval()

    def safe_NN(A, b, threshold=0.8):
        # b: 1D numpy of length=num_bins
        inp = torch.tensor(b, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model_nn(inp)                  # [1 x M]
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        return [i for i,p in enumerate(probs) if p >= threshold]

    # === 1. Select which methods to run ===
    methods = {
        "Lasso": safe_Lasso,
        "L0":    safe_L0,
        "ABESS": safe_ABESS,
        "NN":    safe_NN
    }

    # default complexity / parameters
    COMPLEXITY = 5
    if x_values is None:
        if mode == 'complexity':
            x_values = list(range(1, 25))
        elif mode == 'snr':
            x_values = [1e6,1e5,1e4,1e3,1e2,10,5,2,1,0.5,0.2,0.1]
        elif mode == 'concentrations':
            x_values = [1,10,100,1e3,1e4,1e5,1e6]
        elif mode == 'library_size':
            x_values = list(range(25, num_mols+1, 5))
        elif mode == 'known_proportion':
            x_values = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]
        else:
            raise ValueError("Invalid mode")

    # dispatcher to call each method
    def run_dispatch(method, func, A, mode, x, COMPLEXITY):
        if method=="L0" and ((mode=="complexity" and x>max_support) or (mode!="complexity" and COMPLEXITY>max_support)):
            return method, x, np.nan
        try:
            score = run_single_trial(func, A, mode, x, COMPLEXITY, method_name=method)
        except:
            score = np.nan
        return method, x, score

    # build jobs
    jobs = [
        (m, f, spectralMatrix, mode, x, COMPLEXITY)
        for x in x_values
        for m,f in methods.items()
        for _ in range(num_trials)
    ]

    # run in parallel
    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_dispatch)(m,f,A,mode,x,COMPLEXITY)
        for (m,f,A,mode,x,COMPLEXITY) in jobs
    )

    # aggregate
    results = {m:[] for m in methods}
    for m in methods:
        for x in x_values:
            scores = [s for (mm, xx, s) in raw_results if mm==m and xx==x]
            results[m].append(np.nanmean(scores) if scores else np.nan)

    # === Plot ===
    plt.figure(figsize=(8,5))
    plot_styles = {
        "Lasso": {'marker':'o','color':'C0'},
        "ABESS": {'marker':'^','color':'C1'},
        "L0":    {'marker':'s','color':'C2'},
        "NN":    {'marker':'d','color':'C3'}
    }
    method_order = ["Lasso","ABESS","L0","NN"]
    for m in method_order:
        y = results[m]
        xs = [x for x,yv in zip(x_values,y) if not np.isnan(yv)]
        ys = [yv for yv in y if not np.isnan(yv)]
        style = plot_styles[m]
        plt.plot(xs, ys, label=m, **style)

    plt.xlabel({
        'complexity':"Mixture Complexity",
        'snr':"SNR",
        'library_size':"Library Size",
        'concentrations':"Concentration Ratio",
        'known_proportion':"Known Proportion"
    }[mode])
    plt.ylabel("Average F₁ Score")
    plt.title(f"Method comparison ({mode})")
    plt.legend()
    plt.grid(True)
    if mode=='snr': plt.xscale('log'); plt.gca().invert_xaxis()
    if mode=='concentrations': plt.xscale('log')
    if mode=='known_proportion': plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig("Models_test_with_NN.png", dpi=300)


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


def main():
    print("starting...")
    A, df = LoadRealMatrix("mass_spectra_individual.csv")
    master_plot(A, mode='snr', num_trials=5)


if __name__=="__main__":
    main()





