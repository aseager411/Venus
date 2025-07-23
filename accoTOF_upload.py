# Author: Alex Seager
# Last Version: 7/22/25
#
# Description: Upload and process raw txt data from MIT accuTOF
# Tasks: Split the data into two matrices for Individual and Mixtures

import numpy as np
import pandas as pd
from pathlib import Path

# Adjust this to the folder where you put the data
base_dir = Path("/Users/alexseager/Desktop/Summer_Work_2025/MS_Data")


# M/Z Bin width (assumed fixed at 1.0)
bin_width = 1.0

# Containers for spectra and metadata
spectra_individual = {}
spectra_mixtures = {}
metadata_individual = []
metadata_mixtures = []

# Load files
for file in base_dir.rglob("*.txt"):
    try:
        relative_parts = file.relative_to(base_dir).parts
    except ValueError:
        continue  # skip files that can't be resolved

    if len(relative_parts) < 2:
        continue  # not enough directory info to classify

    top_group = relative_parts[0]  # 'Individual' or 'Mixtures'
    filename = file.stem
    mol_name = filename.split("_")[0]

    # Read the data
    df = pd.read_csv(file, sep="\t", header=None, names=["mz", "intensity"])

    if top_group == "Individual" and len(relative_parts) >= 3:
        group_name = relative_parts[1]  # e.g. 'Amino Acids'
        spectra_individual[mol_name] = df
        metadata_individual.append({
        "molecule": mol_name,
        "group": group_name,
        "file": str(file), 
        "short_molecule": mol_name
    })


    elif top_group == "Mixtures":
        if len(relative_parts) >= 2:
            group_name = relative_parts[1]  # e.g., 'B5M1' or 'Ungrouped'
        else:
            group_name = "Ungrouped"
            
        spectra_mixtures[mol_name] = df
        metadata_mixtures.append({
        "molecule": mol_name,
        "group": group_name,
        "file": str(file)
    })


# Determine common mz bin range across both datasets
all_mz = pd.concat([
    df["mz"] for df in list(spectra_individual.values()) + list(spectra_mixtures.values())
])
min_mz = np.floor(all_mz.min())
max_mz = np.ceil(all_mz.max())
bin_edges = np.arange(min_mz - 0.5, max_mz + 0.5 + 1e-6, bin_width)
bin_centers = np.arange(min_mz, max_mz + 1)

# Function to build matrix
def build_matrix(spectra_dict):
    matrix = pd.DataFrame(index=bin_centers)
    for mol, df in spectra_dict.items():
        binned, _ = np.histogram(
            df["mz"],
            bins=bin_edges,
            weights=df["intensity"]
        )
        matrix[mol] = binned.astype(int)
    matrix.columns = [col.split("_")[0] for col in matrix.columns]
    return matrix

# Build matrices
matrix_individual = build_matrix(spectra_individual)
matrix_mixtures = build_matrix(spectra_mixtures)

# Convert metadata to DataFrames
meta_individual = pd.DataFrame(metadata_individual)
meta_mixtures = pd.DataFrame(metadata_mixtures)

# Save to CSV
matrix_individual.to_csv("mass_spectra_individual.csv")
meta_individual.to_csv("mass_spectra_metadata_individual.csv", index=False)

matrix_mixtures.to_csv("mass_spectra_mixtures.csv")
meta_mixtures.to_csv("mass_spectra_metadata_mixtures.csv", index=False)

# Show quick preview
if not matrix_individual.empty:
    first_cols = matrix_individual.columns[:10]
    midpoint = len(matrix_individual) // 2
    half_window = 10
    print("Preview of Individual Matrix:")
    print(matrix_individual.loc[midpoint - half_window: midpoint + half_window, first_cols])
    print(f"✅ Loaded {len(spectra_individual)} individual molecule spectra")
else:
    print("⚠️ No Individual matrix data found.")
