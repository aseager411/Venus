# Author: Alex Seager
# Last Version: 7/8/25
#
# Description: Visualizing mass spec data from accuTOF

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for servers
import matplotlib.pyplot as plt
import seaborn as sns


# import data from csv
# averaging duplicate molecule columns
def LoadAvgMatrix(csv_path, numMolecules=None, numWavelengths=None, normalize=True):
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

# doesn't group by name, used for seeing variation across runs
def LoadRawMatrix(csv_path, numWavelengths=None):
    df = pd.read_csv(csv_path, index_col=0).loc[51:450]
    if numWavelengths is not None:
        df = df.iloc[:numWavelengths, :]
    return df


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


def PlotSpectralHeatmap(
    A,
    molecule_names,
    bin_width=1.0,
    mz_min=50.0,
    mz_max=450.0,
    normalize=True,
    title="Spectral Heatmap",
    max_fig_width=20,   # inches
    max_fig_height=15,  # inches
):
    """
    A: 2D numpy array (num_bins, num_molecules)
    molecule_names: list of molecule names corresponding to columns in A
    bin_width: width of each m/z bin
    mz_min: lower m/z value to start the axis at (corresponds to bin 0)
    mz_max: maximum m/z value to display (inclusive); if None, shows all bins
    """
    if normalize:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1
        A = A / norms

    # Apply m/z window
    if mz_max is not None:
        if mz_max < mz_min:
            raise ValueError("mz_max must be >= mz_min")
        max_bins = int((mz_max - mz_min) / bin_width) + 1
        max_bins = min(max_bins, A.shape[0])
        A = A[:max_bins, :]

    A_T = A.T  # shape: (num_molecules, displayed_bins)
    num_molecules, num_bins = A_T.shape

    # Base size per molecule and per bin (tunable)
    height_per_molecule = 0.25  # inches
    width_per_bin = 0.15        # inches

    fig_height = max(4, num_molecules * height_per_molecule)
    fig_width = max(6, num_bins * width_per_bin)

    # Cap to maximums to avoid runaway size
    fig_height = min(fig_height, max_fig_height)
    fig_width = min(fig_width, max_fig_width)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        A_T,
        cmap="Greys",
        cbar_kws={"label": "Normalized Intensity"},
        yticklabels=molecule_names,
        xticklabels=False,  # set manually below
        square=False,
        linewidths=0,  # optional: remove grid lines
    )

    # X-axis ticks (sampled to avoid overcrowding)
    if num_bins > 0:
        max_labels = 20
        step = max(1, num_bins // max_labels)
        x_ticks = np.arange(0, num_bins, step)
        x_labels = [f"{mz_min + i * bin_width:.0f}" for i in x_ticks]
        ax.set_xticks(x_ticks + 0.5)  # center labels if needed depending on heatmap alignment
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("m/z")
    ax.set_ylabel("Molecule")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def PlotPeakFrequency(
    spectra,              # 2D array: (num_bins, num_spectra) or (num_spectra, num_bins)
    threshold=0.05,       # intensity threshold to call a "peak"
    bin_width=1.0,
    mz_min=50.0,
    mz_max=420.0,
    normalize=True,      # if True, normalize each spectrum before thresholding
    title="Peak Frequency Above Threshold",
    xlabel="m/z",
    ylabel="Peak Count",
    figsize=(12, 3),
    save_name="peak_frequency_hist.png",
):
    """
    Counts how many spectra have intensity > threshold at each m/z bin and plots a histogram-like step plot.
    """
    spectra = np.array(spectra)
    if spectra.ndim != 2:
        raise ValueError("spectra must be 2D array")
    # Heuristic: if shape looks like (num_spectra, num_bins), transpose to (num_bins, num_spectra)
    if spectra.shape[0] < spectra.shape[1]:
        spectra = spectra.T

    if normalize:
        norms = np.linalg.norm(spectra, axis=0)
        norms[norms == 0] = 1
        spectra = spectra / norms

    num_bins, num_spectra = spectra.shape

    # Apply m/z window
    if mz_max is not None:
        if mz_max < mz_min:
            raise ValueError("mz_max must be >= mz_min")
        max_bin = int((mz_max - mz_min) / bin_width) + 1
        max_bin = min(max_bin, num_bins)
        spectra = spectra[:max_bin, :]
        num_bins = spectra.shape[0]

    # Count peaks per bin
    counts = np.sum(spectra > threshold, axis=1)

    mz_values = mz_min + np.arange(num_bins) * bin_width

    
    plt.figure(figsize=figsize)
    # Bar histogram with black outline and black fill
    plt.bar(
        mz_values,
        counts,
        width=bin_width * 0.9,
        align="center",
        edgecolor="black",
        color="black",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(mz_min - bin_width * 0.5, mz_values[-1] + bin_width * 0.5)
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

###################
#  MAIN FUNCTION  #
###################

def main():
    individual = "mass_spectra_individual.csv"
    spectralMatrix, df1 = LoadAvgMatrix(individual)
    individual_names = df1.columns.tolist()

    mixtures = "mass_spectra_mixtures.csv"
    samples, df2 = LoadAvgMatrix(mixtures)
    mixture_names = df2.columns.tolist()

    #PlotSpectralHeatmap(spectralMatrix, individual_names, title="Heatmap of Normalized Spectra")

    PlotPeakFrequency(spectralMatrix)
    df_raw1 = LoadRawMatrix("mass_spectra_individual.csv")
    PlotMeanAndErrorSpectrum(df_raw1, short_name="1-naphthalenesulfonic acid",
                             title_prefix="Mean Spectrum - Individual")
    
    # df_raw2 = LoadRawMatrix("mass_spectra_mixtures.csv")
    # PlotMeanAndErrorSpectrum(df_raw2, short_name="B6M3",
    #                          title_prefix="Mean Spectrum - Mixture")
    

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

    # spectra, names = GetSample(["Sulfur"], df1)
    # PlotSingleSpectra(spectra, title=f"Sample: {' + '.join(names)}")
    
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