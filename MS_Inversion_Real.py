# Author: Alex Seager
# Last Version: 7/6/25
#
# Description: I am attempting to invert real MS data in simulated combinations and recover which molecules 
# contributed to the spectral signal. I explore many ways at attempting to do this.


# Tasks
# get access to compute
# get better plots of aic vs bic 
# tune alpha for Lasso?
# get plots for diff divisions of data 
# refactor test code
#
# try to learn on data from our instrumenty to divide into groups and see if we can recognize 
# others in the saem class. can try this with other data sets too but harder to classify 
# does the database divide into molecule classes?
# could try to generate more molecules with said parameters 

# Questions
# is there some way to do brute force without trying every combo?
# Why does trucating values below 50 break the L1

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
import matplotlib.pyplot as plt
from abess.linear import LinearRegression

#np.random.seed(42)  # Seed the random number generator for repropducability 

from MS_Model_Current import (
    GetSampleSpectrum,
    AddNoise,
    PlotSpectra,
)

from MS_Inversion_Toy import (
    Lasso_L1,
    f_beta,
    Model_Test,
    L_Zero,
    OneSampleTest,
    NSampleTest,
    strict_recall_score
)

from ABESS import (
    ABESS
)

# Global parameters
SAMPLECOMPLEXITY = 5
SNR = 10
ALPHA = 0.00001
SPLITPROPORTION = 1  # 1 -> we have data for all possible molecules, 0 -> we have no data at all 

# import data from csv
import pandas as pd
import numpy as np

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

    print("Molecules used:", list(grouped_df.columns))
    return A, grouped_df


# split data into known and unknown compounds
def Data_Split(splitProportion, spectralMatrix):
    n_cols = spectralMatrix.shape[1]
    split_idx = int(np.floor(splitProportion * n_cols))

    known = spectralMatrix[:, :split_idx]
    unknown = spectralMatrix[:, split_idx:]

    return known, unknown
    
###################
# INVERSE METHODS #
###################

# ??(Bayesian sparse regression)

# Bayesian sparse regression using ARD
# failed miserably with 5 molecules
# better with more? check
def ARD_SparseRegression(matrix, spectra):
    model = ARDRegression(fit_intercept=False)
    model.fit(matrix, spectra)
    x_ard = model.coef_
    return x_ard


##################
#     Tests      #
##################

# Testing L1 for SNR and sample complexity
def Split_Model_Test(spectralMatrix, known, a):  
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
        for j in range(1, 51):
            sampleComplexity = j
            for k in range (5):
                s, trueMolecules = GetSampleSpectrum(
                    sampleComplexity,
                    spectralMatrix,
                )
                #print("New sample spectrum:", s)

                # add noise
                noisySpectra = AddNoise(snr, s)
                #print("New noisy spectrum:", noisySpectra)

                # Run fit
                x_sol, alpha_copy, R2 = Lasso_L1(known, noisySpectra, a)

                ## Evaluate model efficiency
                # Disregard predictions below this concentration
                gamma = 0.001 #* max(abs(x_sol))
                # take our solution and pull out which molecules were selected (low out low concentration predictions)
                predictedMolecules = [i for i, v in enumerate(x_sol) if abs(v) > gamma] 
                print("molecules chosen by model: ", predictedMolecules)

                # we only care about selecting all of the molecules we know
                # Define known molecule index set (e.g. first k columns)
                known_indices = set(range(known.shape[1]))

                # Filter trueMolecules to only those that are in known set
                trueKnownMolecules = [m for m in trueMolecules if m in known_indices]


                print("true molecules in known set: ", trueKnownMolecules)
                # score the models choices favoring precision over recall
                score = f_beta(trueKnownMolecules, predictedMolecules)
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
    max_j = max(xs)

    # Use sparse ticks if too many
    xtick_spacing = 5 if max_j > 50 else 1
    ax.set_xticks(range(0, max_j, xtick_spacing))

    ax.set_xlim(0.5, max_j + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Noise level')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

###################
#  MAIN FUNCTION  #
###################

def main():
    file = "mass_spectra_individual.csv"
    A, df = LoadRealMatrix(file)

    Model_Test(A, 50, noise = False, score_fn=f_beta, sampleRange = 50)
    # spectra, trueMolecules = GetSampleSpectrum(2, A)
    # print("true molecules: ", trueMolecules)
    # predictedMolecules = ABESS(A, spectra, 10)
    # print("predicted molecules: ", predictedMolecules)
    # print("f-score: ", f_beta(trueMolecules, predictedMolecules))

    # known, unknown = Data_Split(SPLITPROPORTION, A)
    # Split_Model_Test(A, known, ALPHA)

    # croppedMatrix = A[:, :50]
    # print("Matrix shape:", croppedMatrix.shape)  # Should be (numWavelengths, numMolecules)
    #OneSampleTest(5, A, SNR)
    # s, molecules = GetSampleSpectrum(3, A)
    # print("true molecules: ", molecules)
    # guess = ABESS(A, s, 10)
    # print("guesses: ", guess)

    

if __name__ == "__main__":
    main()