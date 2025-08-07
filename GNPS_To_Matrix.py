# Author: Alex Seager
# Last Version: 6/17/25
#
# Description: This is the second step in importing new GNPS data to a local csv file

import numpy as np
import pandas as pd
import pickle

def build_transposed_matrix(spectrum_dict, bin_width=1.0, mz_min=50, mz_max=1000, method='sum'):
    """
    Builds a matrix where rows are m/z bins and columns are spectra.
    Bin values are summed or averaged over the bin.
    """
    bins = np.arange(mz_min, mz_max + bin_width, bin_width)
    bin_labels = [f"{b:.1f}" for b in bins[:-1]]
    matrix = {}

    for accession, df in spectrum_dict.items():
        binned = pd.cut(df["mz"], bins=bins, labels=bin_labels, include_lowest=True)
        if method == 'mean':
            values = df.groupby(binned)["intensity"].mean()
        else:
            values = df.groupby(binned, observed=False)["intensity"].sum()

        col = pd.Series(0, index=bin_labels, dtype=float)
        col.loc[values.index] = values.values
        matrix[accession] = col

    result_df = pd.DataFrame(matrix)
    result_df.index.name = "mz"
    return result_df

def main():
    # Load data
    with open("spectrum_matrix.pkl", "rb") as f:
        spectrum_matrix = pickle.load(f)
    with open("spectrum_names.pkl", "rb") as f:
        name_lookup = pickle.load(f)

    # Build matrix
    df = build_transposed_matrix(spectrum_matrix, bin_width=1.0, mz_min=0, mz_max=1000)

    # Rename columns to compound names
    df.columns = [name_lookup.get(acc, acc) for acc in df.columns]

    #Save
    print(df.shape)
    df.to_csv("/Users/alexseager/Desktop/Summer Work 2025/Code/Full_GNPS_matrix.csv")
    print("✅ Saved Full_GNPS_matrix.csv")

if __name__ == "__main__":
    main()
