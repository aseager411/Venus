# Author: Alex Seager
# Last Version: 7/8/25
#
# Description: Visualizing mass spec data from accuTOF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    plt.show()

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
        # normalize
        max_val = spec.max()
        if max_val > 0:
            spec = spec / max_val
        plt.plot(x, spec, label=label, linewidth=1.0, alpha=0.7)

    plt.xlabel('m/z')
    plt.ylabel('Normalized Intensity')
    plt.title(title)
    plt.legend(fontsize=8, loc='upper right')

    # Determine tick spacing
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
    plt.show()


###################
#  MAIN FUNCTION  #
###################

def main():
    individual = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_individual.csv"
    spectralMatrix, df1 = LoadRealMatrix(individual)
    individual_names = df1.columns.tolist()

    mixtures = "/Users/alexseager/Desktop/Summer Work 2025/Code/mass_spectra_mixtures.csv"
    samples, df2 = LoadRealMatrix(mixtures)
    mixture_names = df2.columns.tolist()


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



    ### Truth vs model guess overlay 1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide
    #incorrect guesses:
    spectra1, _ = GetSample([('Ala', 0.011404128270954328), ('Phenanthrene', 0.2217878523929781)], df1)
    #not identified by model:
    spectra2, _ = GetSample(["Benzenesulfonic acid"], df1)

    #true sample
    spectra3, _ = GetSample(["Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene"], df2)
   
    #LASSO guess
    spectra4, _ = GetSample([('N-methylpyrrole', 1.3282611345263897), ('Ala', 0.011404128270954328), ('Pyrene', 0.28486107747102996), ('Phenanthrene', 0.2217878523929781), ('16-diphenyl-135-hexatriene', 0.11069791839345598)], df1)
    PlotMultipleSpectra(
        [spectra3, spectra4],
        labels=["sample (Benzenesulfonic acid + DPH(1,6-Diphenyl-1,3,5-hexatriene) + N-methypyrrole + Pyrene)", "guessed combo: N-methylpyrrole, Ala, Pyrene, Phenanthrene, 16-diphenyl-135-hexatriene"],
        title="Overlay of true and closest match Spectra"
    )
    #what the model missed vs added
    PlotMultipleSpectra(
        [spectra1, spectra2],
        labels=["false positives: Ala, Phenanthrene", "false negatives: Benzenesulfonic acid"],
        title="Overlay of false positives and negatives"
    )
    
    # spectra, names = GetSample(["Pro", "Ser", "Thr"], df1)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    # spectra, names = GetSample(['N-methylpyrrole', '246-Trimethylpyridine', 'Nile red', 'Methylcyclopentane'], df1)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    spectra, names = GetSample(["Cyclopentane"], df1)
    PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")
    
    # spectra, names = GetSample(["Phenanthrene"], df1)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

    # spectra, names = GetSample(["1-chloro-3-methoxybenzene + Benzenesulfonic acid + Dodecyltrimethylammonium bromide"], df2)
    # PlotSingleSpectra(spectra, title=f"Artificial Sample: {' + '.join(names)}")

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