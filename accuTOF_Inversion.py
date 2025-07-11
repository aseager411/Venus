# Author: Alex Seager
# Last Version: 7/8/25
#
# Description: I am attempting to invert MS data from combinationsand recover which molecules 
# from a library contributed to the spectral signal. I explore many ways at attempting to do 
# this.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

from MS_Inversion_Toy import (
    Lasso_L1,
    f_beta,
    Model_Test,
    L_Zero,
    OneSampleTest,
    NSampleTest
)

from ABESS import (
    ABESS
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

# construct a "sample" from the individual molecule spectra
# used to examine mixing
def GetSample(molecule_names, spectral_df):
    spectra = np.zeros(spectral_df.shape[0])
    for name in molecule_names:
        if name not in spectral_df.columns:
            raise ValueError(f"Molecule '{name}' not found in spectral data.")
        spectra += spectral_df[name].values
    return spectra, molecule_names

##########
#METHODS
##########

#testing lasso with name retrieval
def Lasso_Test(matrix, spectra, alpha, df):
    #fit non-negatuve lasso
    x_hat = Lasso_L1(matrix, spectra, alpha)

    # Threshold for numerical noise
    threshold = 1e-3
    selected = [
        (df.columns[i], x_hat[i])
        for i in range(len(x_hat))
        if abs(x_hat[i]) > threshold
    ]

    return selected

#testing ABESS with name retrieval
# Testing ABESS with name retrieval, supports both vector and tuple list output
def ABESS_Test(matrix, spectra, sMax, df):
    x_hat = ABESS(matrix, spectra, sMax)
    threshold = 1e-4  # Threshold to suppress numerical noise

    # Check if ABESS returned tuple list [(index, coef), ...]
    if isinstance(x_hat, list) and all(isinstance(x, tuple) for x in x_hat):
        results = {
            df.columns[i]: coef
            for i, coef in x_hat
            if abs(coef) > threshold
        }
    else:
        # Assume x_hat is a full-length vector
        results = {
            df.columns[i]: x_hat[i]
            for i in range(len(x_hat))
            if abs(x_hat[i]) > threshold
        }

    return results



def main():
    individual = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_individual.csv"
    spectralMatrix, df1 = LoadRealMatrix(individual)
    individual_names = df1.columns.tolist()

    mixtures = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_mixtures.csv"
    samples, df2 = LoadRealMatrix(mixtures)
    mixture_names = df2.columns.tolist()

    print("Test with ABESS")
    print("")
    print("True molecules: Pro, Ser, Thr")

    spectra1, _ = GetSample(["Pro", "Ser", "Thr"], df1)
    predicted = ABESS_Test(spectralMatrix, spectra1, 5, df1)
    print("predictions given fake mix: ", predicted)

    spectra2, _ = GetSample(["ProSerThr"], df2)
    predicted = ABESS_Test(spectralMatrix, spectra2, 5, df1)
    print("predictions given real mix: ", predicted)

    # Test on Ala, Arg, Glu, Gly mix with ABESS
    print("")
    print("True molecules: Ala, Arg, Glu, Gly")

    spectra3, _ = GetSample(["Ala", "Arg", "Glu", "Gly"], df1)
    predicted = ABESS_Test(spectralMatrix, spectra3, 5, df1)
    print("predictions given fake mix: ", predicted)

    spectra4, _ = GetSample(["AlaArgGluGly"], df2)
    predicted = ABESS_Test(spectralMatrix, spectra4, 5, df1)
    print("predictions given real mix: ", predicted)

    print("")
    print("True molecules: Benzenesulfonic acid + 16-diphenyl-135-hexatriene + N-methylpyrrole + Pyrene")

    spectra1, _ = GetSample(["Benzenesulfonic acid", "16-diphenyl-135-hexatriene", "N-methylpyrrole", "Pyrene"], df1)
    predicted = ABESS(spectralMatrix, spectra1, 5, df1)
    print("predictions given fake mix: ", predicted)

    spectra2, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
    predicted = ABESS(spectralMatrix, spectra2, 5, df1)
    print("predictions given real mix: ", predicted)

    # print("Testing with lasso: ")
    # print("True molecules: 1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide")
    # print("")

    # spectra1, _ = GetSample(["1-chloro-3-methoxybenzene", "Benzenesulfonic acid", "Dodecyltrimethylammonium bromide"], df1)
    # predicted = Lasso_Test(spectralMatrix, spectra1, 1000000, df1)
    # print("predictions given fake mix: ", predicted)

    # spectra2, _ = GetSample(["1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide"], df2)
    # predicted = Lasso_Test(spectralMatrix, spectra2, 1000000000, df1)
    # print("predictions given real mix: ", predicted)


    # print("Testing with lasso: ")
    # print("True molecules: Benzenesulfonic acid + 16-diphenyl-135-hexatriene + N-methylpyrrole + Pyrene")
    # print("")

    # spectra1, _ = GetSample(["Benzenesulfonic acid", "16-diphenyl-135-hexatriene", "N-methylpyrrole", "Pyrene"], df1)
    # predicted = Lasso_Test(spectralMatrix, spectra1, 1000000, df1)
    # print("predictions given fake mix: ", predicted)

    # spectra2, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
    # predicted = Lasso_Test(spectralMatrix, spectra2, 1000000000, df1)
    # print("predictions given real mix: ", predicted)

    print("Testing with lasso: ")
    print("")
    print("True molecules: Pro, Ser, Thr")

    spectra1, _ = GetSample(["Pro", "Ser", "Thr"], df1)
    predicted = Lasso_Test(spectralMatrix, spectra1, 1000000, df1)
    print("predictions given fake mix: ", predicted)

    spectra2, _ = GetSample(["ProSerThr"], df2)
    predicted = Lasso_Test(spectralMatrix, spectra2, 1000000000, df1)
    print("predictions given real mix: ", predicted)

    # Test on Ala, Arg, Glu, Gly mix with lasso
    print("")
    print("True molecules: Ala, Arg, Glu, Gly")

    spectra3, _ = GetSample(["Ala", "Arg", "Glu", "Gly"], df1)
    predicted = Lasso_Test(spectralMatrix, spectra3, 100, df1)
    print("predictions given fake mix: ", predicted)

    spectra4, _ = GetSample(["AlaArgGluGly"], df2)
    predicted = Lasso_Test(spectralMatrix, spectra4, 1, df1)
    print("predictions given real mix: ", predicted)
    
    print("")
    print("True molecules: Benzenesulfonic acid + 16-diphenyl-135-hexatriene + N-methylpyrrole + Pyrene")

    spectra1, _ = GetSample(["Benzenesulfonic acid", "16-diphenyl-135-hexatriene", "N-methylpyrrole", "Pyrene"], df1)
    predicted = Lasso_Test(spectralMatrix, spectra1, 1000000, df1)
    print("predictions given fake mix: ", predicted)

    spectra2, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
    predicted = Lasso_Test(spectralMatrix, spectra2, 1000000000, df1)
    print("predictions given real mix: ", predicted)

if __name__ == "__main__":
    main()