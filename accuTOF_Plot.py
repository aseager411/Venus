# Author: Alex Seager
# Last Version: 7/8/25
#
# Description: Visualizing mass spec data from accuTOF

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for servers
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import colormaps
import umap


# import data from csv
def LoadRealMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
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
# accepts simple list of molecule names or dict of names and concentrations
def GetSample(molecules, spectral_df):
    spectra = np.zeros(spectral_df.shape[0])

    # Convert if it's a plain list of names
    if isinstance(molecules, list) and all(isinstance(m, str) for m in molecules):
        # Treat all concentrations as 1.0
        molecules = {name: 1.0 for name in molecules}
    elif isinstance(molecules, list):
        # Convert list of (name, conc) pairs to dict
        molecules = dict(molecules)
    elif not isinstance(molecules, dict):
        raise ValueError("Input must be a list of names, list of (name, conc) pairs, or a dict.")

    for name, conc in molecules.items():
        if name not in spectral_df.columns:
            raise ValueError(f"Molecule '{name}' not found in spectral data.")
        spectra += conc * spectral_df[name].values

    return spectra, list(molecules.keys())


# Plot given spectra
# 
# Arguments: float vector spectra -> a vector representing generated spectra
# Returns: None
def PlotSingleSpectra(spectra, bin_width=1.0, mz_min=50.0, title="Sample Spectrum"):
    spectra = np.array(spectra, dtype=float)

    # Normalize spectrum
    max_val = spectra.max()
    if max_val > 0:
        spectra /= max_val

    num_bins = len(spectra)
    x = np.arange(num_bins) * bin_width + mz_min
    mz_max = x[-1]

    plt.figure(figsize=(10, 4))  # Match height to PlotMultipleSpectra
    plt.plot(x, spectra, label="Spectrum", linewidth=1.0, alpha=0.7)

    plt.xlabel('m/z')
    plt.ylabel('Normalized Intensity')
    plt.title(title)
    plt.legend(fontsize=8, loc='upper right')

    # Tick spacing logic
    range_mz = mz_max - mz_min
    if range_mz <= 100:
        tick_spacing = 5
    elif range_mz <= 400:
        tick_spacing = 25
    else:
        tick_spacing = 50

    xticks = np.arange(mz_min, mz_max + 1, tick_spacing)
    plt.xticks(xticks, fontsize=6)

    plt.tight_layout()
    filename = f"{title.replace(' ', '_').replace(':','')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# plot overlayed spectra of multiple samples
# Plot multiple spectra overlaid in different colors
def PlotMultipleSpectra(spectra_list, labels, bin_width=1.0, mz_min=50.0, title="Overlaid Spectra"):
    if len(spectra_list) != len(labels):
        raise ValueError("Number of spectra and number of labels must match.")

    num_bins = len(spectra_list[0])
    x = np.arange(num_bins) * bin_width + mz_min
    mz_max = x[-1]

    plt.figure(figsize=(10, 4))

    for spec, label in zip(spectra_list, labels):
        spec = np.array(spec, dtype=float)
        max_val = spec.max()
        if max_val > 0:
            spec = spec / max_val
        plt.plot(x, spec, label=label, linewidth=1.0, alpha=0.7)

    plt.xlabel('m/z')
    plt.ylabel('Normalized Intensity')
    plt.title(title)
    plt.legend(fontsize=8, loc='upper right')

    range_mz = mz_max - mz_min
    if range_mz <= 100:
        tick_spacing = 5
    elif range_mz <= 400:
        tick_spacing = 25
    else:
        tick_spacing = 50

    xticks = np.arange(mz_min, mz_max + 1, tick_spacing)
    plt.xticks(xticks, fontsize=6)

    plt.tight_layout()
    filename = f"{title.replace(' ', '_').replace(':','')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# visualize the variation of spectra for multiple runs of the same sample
def PlotMeanAndErrorSpectrum(df, short_name, title_prefix="Mean Spectrum", bin_width=1.0, mz_min=50.0, min_separation=2):
    matched_cols = [col for col in df.columns if col.startswith(short_name)]
    if not matched_cols:
        print(f"⚠️ No spectra found for '{short_name}'")
        return

    subset = df[matched_cols].astype(float)
    if subset.empty:
        print(f"⚠️ Data subset for '{short_name}' is empty.")
        return

    normalized = subset.div(subset.max(axis=0).replace(0, 1), axis=1)
    mean_spectrum = normalized.mean(axis=1)
    std_spectrum = normalized.std(axis=1)

    x = np.arange(len(mean_spectrum)) * bin_width + mz_min

    plt.figure(figsize=(10, 4))
    plt.plot(x, mean_spectrum, label=f"Mean of {short_name}", linewidth=1.2, color='black')

    # --- Filter major peaks ---
    major_mask = mean_spectrum >= 0.05
    major_indices = mean_spectrum[major_mask].sort_values(ascending=False).index.tolist()

    # Suppress nearby peaks
    # Convert Series index to integer positions (row numbers)
    all_indices = mean_spectrum.index.to_list()
    selected_positions = []
    for idx in major_indices:
        i = all_indices.index(idx)
        if all(abs(i - sel) >= min_separation for sel in selected_positions):
            selected_positions.append(i)

    major_x = x[selected_positions]
    major_y = mean_spectrum.iloc[selected_positions]
    major_err = std_spectrum.iloc[selected_positions]


    # Plot red error bars with caps
    plt.errorbar(major_x, major_y, yerr=major_err, fmt='o', color='red',
                 ecolor='red', elinewidth=1, capsize=3, markeredgewidth=0.5,
                 markersize=3, label='±1σ (major peaks)')

    plt.xlabel('m/z')
    plt.ylabel('Normalized Intensity')
    plt.title(f"{title_prefix}: {short_name}")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    filename = f"{title_prefix.replace(' ', '_')}_{short_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()



###maybe delete these pretty useless
#visualize similarity of molecular groups
def plot_pca_by_group(A, group_labels, molecule_labels=None, n_components=2):
    """
    A: numpy array of shape (features, samples)
    group_labels: list of class labels for each sample (same order as columns of A)
    molecule_labels: optional, list of molecule names
    n_components: 2 or 3 for PCA
    """

    # Transpose to shape (samples, features)
    A_T = A.T

    # Standardize features (not molecule-level scaling)
    A_scaled = StandardScaler().fit_transform(A_T)

    # PCA
    pca = PCA(n_components=n_components)
    A_pca = pca.fit_transform(A_scaled)

    unique_groups = sorted(set(group_labels))
    colors = plt.cm.get_cmap("tab10", len(unique_groups))

    plt.figure(figsize=(10, 7))
    for i, group in enumerate(unique_groups):
        idx = [j for j, g in enumerate(group_labels) if g == group]
        plt.scatter(A_pca[idx, 0], A_pca[idx, 1], label=group, alpha=0.7, s=80, edgecolors='k', c=[colors(i)])

    if molecule_labels:
        for i, name in enumerate(molecule_labels):
            plt.text(A_pca[i, 0], A_pca[i, 1], name, fontsize=6, alpha=0.6)

    colors = colormaps.get_cmap("tab10")
    plt.figure(figsize=(10, 7))
    for i, group in enumerate(unique_groups):
        idx = [j for j, g in enumerate(group_labels) if g == group]
        color = colors(i / max(1, len(unique_groups) - 1))
        plt.scatter(A_pca[idx, 0], A_pca[idx, 1], label=group, alpha=0.7, s=80, edgecolors='k', c=[color])


    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of Molecule Spectra Colored by Group")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA.png", dpi=300)


#more visualization
def plot_umap_by_group(A, group_labels, molecule_labels=None, n_neighbors=5, min_dist=0.1):
    """
    A: numpy array of shape (features, samples)
    group_labels: list of class labels for each sample (same order as columns of A)
    molecule_labels: optional, list of molecule names
    """
    A_T = A.T
    A_scaled = StandardScaler().fit_transform(A_T)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    A_umap = reducer.fit_transform(A_scaled)

    unique_groups = sorted(set(group_labels))
    colors = plt.colormaps.get_cmap("tab10")

    plt.figure(figsize=(10, 7))
    for i, group in enumerate(unique_groups):
        idx = [j for j, g in enumerate(group_labels) if g == group]
        color = colors(i / max(1, len(unique_groups) - 1))
        plt.scatter(A_umap[idx, 0], A_umap[idx, 1], label=group, alpha=0.7, s=80, edgecolors='k', c=[color])

    if molecule_labels:
        for i, name in enumerate(molecule_labels):
            plt.text(A_umap[i, 0], A_umap[i, 1], name, fontsize=6, alpha=0.6)

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of Molecule Spectra Colored by Group")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("UMAP.png", dpi=300)
    plt.close()
    
###################
#  MAIN FUNCTION  #
###################

def main():
    individual = "mass_spectra_individual.csv"
    spectralMatrix, df1 = LoadRealMatrix(individual)
    individual_names = df1.columns.tolist()

    mixtures = "mass_spectra_mixtures.csv"
    samples, df2 = LoadRealMatrix(mixtures)
    mixture_names = df2.columns.tolist()

    # PlotMeanAndErrorSpectrum(df1, short_name="Sulfur", title_prefix="Mean Spectrum - Individual")
    # PlotMeanAndErrorSpectrum(df2, short_name="B6M2", title_prefix="Mean Spectrum - Mixture")
    
    
    # meta = pd.read_csv("mass_spectra_metadata_individual.csv")  # Assumes this file has 'molecule' and 'group' columns

    # # Ensure matching order between df1.columns and metadata
    # col_to_group = dict(zip(meta["molecule"], meta["group"]))
    # group_labels = [col_to_group.get(name, "Unknown") for name in individual_names]

    # # PCA plot
    # #plot_pca_by_group(spectralMatrix, group_labels, individual_names)   
    # plot_umap_by_group(spectralMatrix, group_labels, individual_names)




    ### truth vs model guess overlay pro ser thr

    # spectra1, _ = GetSample(["Pro", "Ser", "Thr"], df1)
    # spectra2, _ = GetSample(["ProSerThr"], df2)
    # #ABESS guess
    # spectra3, _ = GetSample(['N-methylpyrrole', '246-Trimethylpyridine', 'Nile red', 'Methylcyclopentane'], df1)
    # #LASSO guess
    # spectra4, _ = GetSample(["Pro", "Thr"], df1)
    # PlotMultipleSpectra(
    #     [spectra2, spectra4],
    #     labels=["Pro, Ser, Thr", "guessed combo"],
    #     title="Overlay of true and closest match Spectra"
    # )

    ### truth vs model guess overlay Ala, Arg, Glu, Gly

    # spectra1, _ = GetSample(["Ala", "Arg", "Glu", "Gly"], df1)
    # spectra2, _ = GetSample(["AlaArgGluGly"], df2)
    # #ABESS guess
    # spectra3, _ = GetSample({'N-methylpyrrole': 4, '246-Trimethylpyridine': 9, 'Nile red': 21, 'Methylcyclopentane': 22, 'Cyclopentane': 30}, df1)
    # #LASSO guess
    # spectra4, _ = GetSample(['Cyclopentane', 'Ala', 'Gly', 'Pro', 'P2'], df1)
    # PlotMultipleSpectra(
    #     [spectra2, spectra4],
    #     labels=["Pro, Ser, Thr", "guessed combo"],
    #     title="Overlay of true and closest match Spectra"
    # )


    ###truth vs model guess overlay 1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide

    # spectra2, _ = GetSample(["1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide"], df2)
   
    # #LASSO guess
    # spectra4, _ = GetSample([('Benzenesulfonic acid', 0.01466872009225554), ('1-Chloro-3-methoxybenzene', 0.8038279654858083), ('Dodecanoic acid', 0.07637775898494034)], df1)
    # PlotMultipleSpectra(
    #     [spectra2, spectra4],
    #     labels=["sample", "guessed combo"],
    #     title="Overlay of true and closest match Spectra"
    # )



    # ### Truth vs model guess overlay 1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide
    # #incorrect guesses:
    # spectra1, _ = GetSample([('Ala', 0.011404128270954328), ('Phenanthrene', 0.2217878523929781)], df1)
    # #not identified by model:
    # spectra2, _ = GetSample(["Benzenesulfonic acid"], df1)

    # #true sample
    # spectra3, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
   
    # #LASSO guess
    # spectra4, _ = GetSample([('N-methylpyrrole', 1.3282611345263897), ('Ala', 0.011404128270954328), ('Pyrene', 0.28486107747102996), ('Phenanthrene', 0.2217878523929781), ('16-diphenyl-135-hexatriene', 0.11069791839345598)], df1)
    # PlotMultipleSpectra(
    #     [spectra3, spectra4],
    #     labels=["sample (Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene)", "guessed combo: N-methylpyrrole, Ala, Pyrene, Phenanthrene, 16-diphenyl-135-hexatriene"],
    #     title="Overlay of true and closest match Spectra"
    # )
    # #what the model missed vs added
    # PlotMultipleSpectra(
    #     [spectra1, spectra2],
    #     labels=["false positives: Ala, Phenanthrene", "false negatives: Benzenesulfonic acid"],
    #     title="Overlay of false positives and negatives"
    # )
    
    # spectra, names = GetSample(["dodeca", "mcyclopen", "Gly", "Ala", "P2", "Benzene", "d-gluc", ], df1)
    # PlotSingleSpectra(spectra, title=f"Environmental Sample")

    # spectra, names = GetSample(['N-methylpyrrole', '246-Trimethylpyridine', 'Nile red', 'Methylcyclopentane'], df1)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    spectra, names = GetSample(["d-rib"], df1)
    PlotSingleSpectra(spectra, title=f"Sample: {' + '.join(names)}")
    
    # spectra1, _ = GetSample(["Dodeca"], df1)
    # spectra2, _ = GetSample(["undeca"], df1)
    # spectra3, _ = GetSample(["tridodeca"], df1)
    
    # # spectra1, _ = GetSample([('Ala', 0.011404128270954328), ('Phenanthrene', 0.2217878523929781)], df1)
    # # #not identified by model:
    # # spectra2, _ = GetSample(["Benzenesulfonic acid"], df1)

    # # #true sample
    # spectra3, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
   
    # # #LASSO guess
    # # spectra4, _ = GetSample([('N-methylpyrrole', 1.3282611345263897), ('Ala', 0.011404128270954328), ('Pyrene', 0.28486107747102996), ('Phenanthrene', 0.2217878523929781), ('16-diphenyl-135-hexatriene', 0.11069791839345598)], df1)
    # PlotMultipleSpectra(
    #     [spectra1, spectra2, spectra 3],
    #     labels=["sample (Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene)", "guessed combo: N-methylpyrrole, Ala, Pyrene, Phenanthrene, 16-diphenyl-135-hexatriene"],
    #     title="Overlay of true and closest match Spectra"
    # )
    # # spectra, names = GetSample(["1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide"], df2)
    # # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    # spectra, names = GetSample(["Benzene + Dodecyltrimethylammonium bromide"], df2)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    # spectra, names = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    # spectra, names = GetSample(["AlaArg"], df2)
    # PlotSingleSpectra(spectra, title=f"Real Sample: {' + '.join(names)}")

    #PlotSpectra(spectralMatrix[:, 0])
    #PlotSpectra(samples[:, 0])

if __name__ == "__main__":
    main()