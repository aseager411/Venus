# Author: Alex Seager
# Last Version: 6/11/25
#
# This file can be used to generate simple fake spectra and contains useful functions to add
# noise and create samples which are passed to more advanced files
#
# Description: I am attempting to model and simulate MS data. I generate a matrix of molecules 
# and their artificial spectra with a random value at each 'Wavelength.' This creates the 
# spectral matrix where we can pull out columns (spectra) and combine them using mixing ratio 
# and other distortions (broadening?). Parameters of teh spectral matrix include sparsity, 
# similarity, and the number of molecules and wavelengths. one can also specify the resolution 
# of the data and parameters of the mixed signal

import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(42)  # Seed the random number generator for repropducability 

# Defining global parameters
NUMMOLECULES = 100
NUMWAVELENGTHS = 100
SPARSITY = 0.98
MAXSIMILARITY = 0.5

# Sampling parameters
SAMPLECOMPLEXITY = 3 #np.random.randint(1, NUMMOLECULES+1) # randomize this
SNR = 8

# Generate the random spectra of one molecule given mean sparsity 
#
# Arguments: float sparsity -> what proportion of entries in the matrix are zero
#            int numWavelengths -> how many different wavelengths do we want for each spectrum?
# Returns: a vector -> the spectra of the molecule
def GenerateSpectra(sparsity, numWavelengths):
    # Generate random floats
    values = np.random.rand(numWavelengths)
    # Draw a bernoulli mask -> 0 with prob=sparsity, 1 with prob=1–sparsity 
    mask = np.random.rand(numWavelengths) >= sparsity
    # If this is this zero vector ie the spectra is empty add one peak
    if not mask.any():  
        idx = np.random.randint(numWavelengths)
        mask[idx] = True
    # Zero out according to mask
    v = values * mask

    # Calculate the norm of this vector 
    norm = np.linalg.norm(v)
    return v, norm

# Generate the random spectra of a group of molecules as a matrix given a maximum similarity, 
# number of molecules, and number of wavelengths
#
# Arguments: int numMolecules -> how many molecules do we want to simulate spectra for?
#            int numWavelengths -> how many different wavelengths do we want for each spectrum?
#            float maxSimilarity -> the maximum similarity between any two columns we will allow for our matrix
# Returns: a vector -> matrix representing the spectra of the molecules
def GenerateMatrix(numMolecules, numWavelengths, maxSimilarity):
    
    # Store normalized columns for similarity checks
    A_norm = np.zeros((numWavelengths, 0))
    # List of accepted raw spectra
    columns = []

    # Keep going until we have enough columns
    while len(columns) < numMolecules:
        # Draw one sparse spectrum + its norm
        v, norm_v = GenerateSpectra(SPARSITY, NUMWAVELENGTHS)
        # Skip spectra which are empty (not using for now)
        if norm_v == 0:
            v_norm = v #it's the zero vector already
        else:
            # Normalize to check cosine similarity
            v_norm = v / norm_v

        # If it’s the first column, accept immediately
        if A_norm.shape[1] == 0:    # do we have zero columns?
            columns.append(v)
            A_norm = np.column_stack((A_norm, v_norm))
            continue

        # Compute max cosine similarity vs. what’s already in A_norm
        if (A_norm.T @ v_norm).max() <= maxSimilarity:
            columns.append(v)
            A_norm = np.column_stack((A_norm, v_norm))
        # Otherwise just loop back and redraw

    # Stack all accepted raw vectors into the final matrix
    A = np.column_stack(columns)
    return A


##############################
#for testing
spectralMatrix = GenerateMatrix(NUMMOLECULES, NUMWAVELENGTHS, MAXSIMILARITY)
##############################


# Get a combined spectra of multiple (randomly selected) molecules 
#
# Arguments: int sampleComplexity -> how many different molecules are we mixing
#            matrix spectralMatrix -> a matrix of spectral data
# Returns: a column vector -> the combined spectra of the sample
def GetSampleSpectrum(sampleComplexity, spectralMatrix):
    spectra = np.zeros(spectralMatrix.shape[0])
    # randomly decide which molecules to add to the sample without replacement
    molecules = np.random.choice(spectralMatrix.shape[1], size = sampleComplexity, replace=False)


    #print("test01: molecules are " , molecules)

    # Create an associated concentration for each molecule which will sum to 1
    concentrations = np.random.dirichlet(alpha=np.ones(sampleComplexity)) #Black box
    # Create counter to iterate concentrations
    j = 0

    #loop through chosen molecules assigning concentrations
    for i in molecules: # the actual integer value not the place in the list


        #concentration = concentrations[j]

        concentration = 1   #TEMPORARY FOR TESTING

        j += 1
       # print("test02: concentration of ", i, "is " , concentration)
        spectra += spectralMatrix[:, i] * concentration
    return spectra, molecules 

# Add multiplicative gaussian noise to a sample spectra
#
# Arguments: int SNR -> desired signal to noise ratio
#            vector spectra -> a spectral sample
# Returns:  vector -> the given spectra with noise added
def AddNoise(snr, spectra):
    spectra = spectra.copy()
    max_peak = np.max(spectra)
    if max_peak == 0:
        return spectra  # nothing to add noise to

    # multiplicative noise standard deviation relative to max peak
    sigma = 1.0 / snr  # σ of multiplicative factor
    factors = np.random.normal(loc=1.0, scale=sigma, size=spectra.shape)

    mask = (spectra != 0)
    spectra[mask] = spectra[mask] * factors[mask]
    return spectra


# Plot given spectra
# 
# Arguments: float vector spectra -> a vector representing generated spectra
# Returns: None
def PlotSpectra(spectra, bin_width=1.0, mz_min=0.0):
    spectra = np.array(spectra, dtype=float)

    # Normalize spectrum
    max_val = spectra.max()
    if max_val > 0:
        spectra /= max_val

    num_bins = len(spectra)
    x = np.arange(num_bins) * bin_width + mz_min
    mz_max = x[-1]

    plt.figure(figsize=(10, 3))
    plt.bar(x, spectra, width=bin_width * 0.9)
    plt.xlabel('m/z')
    plt.ylabel('Normalized Intensity')
    plt.title('Sample Spectrum')

    # Determine tick spacing based on range
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


def main():
    #spectralMatrix = GenerateMatrix(NUMMOLECULES, NUMWAVELENGTHS, MAXSIMILARITY)
    #print(spectralMatrix)
    #u = GenerateSpectra(SPARSITY, NUMWAVELENGTHS)
    #print(u)
    s, selectedMolecules = GetSampleSpectrum(SAMPLECOMPLEXITY, spectralMatrix)
    #print("spectrum: ", s)
    noisySpectra = AddNoise(SNR, s)
    #print("noisy spectra: ", noisySpectra)
    PlotSpectra(noisySpectra)
if __name__ == "__main__":
    main()

