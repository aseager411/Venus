###
#upload accutof data from local machine
###

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

### READ
# Adjust this to the folder where you put the data on your local machine!!!
base_dir = Path("/Users/alexseager/Desktop/Summer_Work_2025/MS_Data")

# M/Z Bin width (assumed fixed at 1.0)
bin_width = 1.0

# Temporary raw storage: now each mol_name maps to *a list* of entries
raw_individual = defaultdict(list)
raw_mixtures   = defaultdict(list)

# Load files
for file in base_dir.rglob("*.txt"):
    try:
        relative_parts = file.relative_to(base_dir).parts
    except ValueError:
        continue

    if len(relative_parts) < 2:
        continue

    top_group = relative_parts[0]  # 'Individual' or 'Mixtures'
    filename  = file.stem
    mol_name  = filename.split("_")[0]
    df        = pd.read_csv(file, sep="\t", header=None,
                            names=["mz", "intensity"])
    if top_group == "Individual" and len(relative_parts) >= 3:
        group = relative_parts[1]
        raw_individual[mol_name].append((df, group, str(file)))
    elif top_group == "Mixtures":
        group = relative_parts[1] if len(relative_parts) >= 2 else "Ungrouped"
        raw_mixtures[mol_name].append((df, group, str(file)))

# Compute common binning
all_mz   = pd.concat([df["mz"]
                      for entries in raw_individual.values()
                      for df, *_ in entries]
                    + [df["mz"]
                       for entries in raw_mixtures.values()
                       for df, *_ in entries])
min_mz   = np.floor(all_mz.min())
max_mz   = np.ceil (all_mz.max())
bin_edges   = np.arange(min_mz - 0.5, max_mz + 0.5 + 1e-6, bin_width)
bin_centers = np.arange(min_mz, max_mz + 1)

def passes_peak_filter(df):
    binned, _ = np.histogram(df["mz"], bins=bin_edges,
                             weights=df["intensity"])
    if binned.max() == 0:
        return True
    norm = binned / binned.max()
    for mz in (371, 372):
        if mz < bin_centers[0] or mz > bin_centers[-1]:
            continue
        idx = int(mz - bin_centers[0])
        if norm[idx] > 0.5:
            return False
    return True

# Now build final spectra dicts and metadata, with unique keys
spectra_individual = {}
metadata_individual = []
for mol, entries in raw_individual.items():
    for i, (df, grp, path) in enumerate(entries, start=1):
        key = f"{mol}_{i}"
        if not passes_peak_filter(df):
            continue
        spectra_individual[key] = df
        metadata_individual.append({
            "molecule": key,
            "group":    grp,
            "file":     path,
            "short_molecule": mol
        })

spectra_mixtures = {}
metadata_mixtures = []
for mol, entries in raw_mixtures.items():
    for i, (df, grp, path) in enumerate(entries, start=1):
        key = f"{mol}_{i}"
        if not passes_peak_filter(df):
            continue
        spectra_mixtures[key] = df
        metadata_mixtures.append({
            "molecule": key,
            "group":    grp,
            "file":     path
        })

def build_matrix(spectra_dict):
    matrix = pd.DataFrame(index=bin_centers)
    for name, df in spectra_dict.items():
        binned, _ = np.histogram(
            df["mz"], bins=bin_edges,
            weights=df["intensity"]
        )
        matrix[name] = binned.astype(int)
    return matrix

matrix_individual = build_matrix(spectra_individual)
matrix_mixtures   = build_matrix(spectra_mixtures)

def zero_out_bins(matrix, mz_list=(371, 372)):
    for mz in mz_list:
        if mz in matrix.index:
            matrix.loc[mz, :] = 0

zero_out_bins(matrix_individual)
zero_out_bins(matrix_mixtures)

# Save out
matrix_individual.to_csv("mass_spectra_individual.csv")
pd.DataFrame(metadata_individual).to_csv(
    "mass_spectra_metadata_individual.csv", index=False)

matrix_mixtures.to_csv("mass_spectra_mixtures.csv")
pd.DataFrame(metadata_mixtures).to_csv(
    "mass_spectra_metadata_mixtures.csv", index=False)

# Quick preview
if not matrix_individual.empty:
    cols = matrix_individual.columns[:10]
    mid  = len(matrix_individual) // 2
    win  = 10
    print("Preview of Individual Matrix:")
    print(matrix_individual.loc[mid - win: mid + win, cols])
    print(f"✅ Loaded {len(spectra_individual)} individual spectra")
else:
    print("⚠️ No Individual matrix data found.")
