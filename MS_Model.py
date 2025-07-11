# Author: Alex Seager
# Last Version: 6/6/25
#
# Description: I am attempting to model and simulate MS data. I generate a matrix of molecules 
# and their artificial spectra with a random value at each 'Wavelength.' This creates the 
# spectral matrix where we can pull out columns (spectra) and combine them using mixing ratio 
# and other distortions (broadening?). Parameters of teh spectral matrix include sparsity, 
# similarity, and the number of molecules and wavelengths. one can also specify the resolution 
# of the data and parameters of the mixed signal


#Tasks
# sparse spectra (k sparsity) -> generate whole matrix and if entry is less than k make it zero
# design matrix where maximum correlation between every pair is between a certain value - correlation calculatued by dot product between two c\olumns between zero and one 
# make a matrix from real spectra can be hand selected important features or can be data from something else entirely (Hi-Tran)


### Questions
# exact vs mean k sparsity?
# if we check every entry for >k we end up with only values >k
## How do the parameters converge ie when is it impossible to have a matrix of given parameters
# reccomended algorthm for similarity of vectors? worth it to store norms?
#
# When to save verisons / start a new file
# am i overdocumenting/should I just be using the global parameters

import numpy as np
#np.random.seed(420)  # Seed the random number generator for repropducability 

# Defining global parameters
NUMMOLECULES = 5
WAVELENGTHS = 10
SIGNALRESOLUTION = 1
SAMPLECOMPLEXITY = 2
MIXINGRESOLUTION = 1
SPARSITY = 0.8
MAXSIMILARITY = 0.5

# Creating a matrix of spectra
#
# Arguments: int numMolecules -> how many molecules do we want to simulate spectra for?
#            int wavelengths -> how many different wavelengths do we want for each spectrum?
#            float SignalResolution -> what is the resolution of the intensity of our spectra?
#            This will be between [0, 1) where 1 means there is a binary signal at each 
#            wavelength
# Returns: A spectral matrix with randomly generated spectra based on given parameters
def SimpleSpectralMatrix(numMolecules, wavelengths, signalResolution):
    N = int(1.0 / signalResolution)  # Determine how many signal strengths are possible
    # Create the matrix (wavelngth x molecules) using numpy with random floats 
    A = np.random.randint(0, N+1, size=(wavelengths, numMolecules)) * signalResolution
    return A


# Creating a matrix of mean k sparsity 
#
# Arguments: int numMolecules -> how many molecules do we want to simulate spectra for?
#            int wavelengths -> how many different wavelengths do we want for each spectrum?
#            This will be between [0, 1) where 1 means there is a binary signal at each 
#            wavelength
#            float sparsity -> what proportion of entries in the matrix are zero       
# Returns: A spectral matrix (numpy matrix object) based on given parameters
def MeanSparseSpectralMatrix(numMolecules, wavelengths, sparsity):
    # Generate a boolean matrix representing which positions will be zero and non-zero (MEAN sparsity K)
    B = (np.random.rand(wavelengths, numMolecules) > sparsity) 
    # Generate actual values for the matrix which are uniform
    U = np.random.rand(wavelengths, numMolecules)
    # Create the sparse matrix by multiplying U element wise with B the boolean matrix
    A = U * B
    print("Test03: true sparsity is ", np.mean(A == 0))
    return A


# Creating a matrix of exactly K sparsity
#
# Arguments: int numMolecules -> how many molecules do we want to simulate spectra for?
#            int wavelengths -> how many different wavelengths do we want for each spectrum?
#            This will be between [0, 1) where 1 means there is a binary signal at each 
#            wavelength
#            float sparsity -> what proportion of entries in the matrix are zero      
# Returns: A spectral matrix (numpy matrix object) based on given parameters
def ExactSparseSpectralMatrix(numMolecules, wavelengths, sparsity):
    total = wavelengths * numMolecules
    num_nonzero = int((1.0 - sparsity) * total)  # how many entries should be nonzero
    # Generate zero matrix
    A = np.zeros((wavelengths, numMolecules), dtype=float)
    # Choose num_nonzero distinct indices so that exactly k fraction remain zero
    chosen_nonzero = np.random.choice(total, size=num_nonzero, replace=False)
    # Generate that many random floats in [0,1)
    random_vals = np.random.rand(num_nonzero)
    # Assign them into A at the chosen indices
    A.flat[chosen_nonzero] = random_vals
    #print("Test03: true sparsity is ", np.mean(A == 0))
    return A


# Calculate the cosine similarity of two spectra vectors
#
# Arguments: vectors a and b representing columns of a spectral matrix     
# Returns: a float (0, 1) where 0 means the vectors are orthogonal and 1 means they are the same vector
def SpectraSimilarity(a, b):
    # numpy dot product doesn't like the zero vector so deal with this case separately
    if ((a == 0).all() or (b == 0).all()):
        return 0
    # Compute dot product
    dot = np.dot(a, b)
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)
    


# Check the maximum cosine similarity of any two vectors in the matrix (using brute force)
#
# Arguments: matrix A -> molecular spectra
#            float maxSimilarity -> the maximum similarity we will allow for our matrix
# Returns: a float (0, 1) where 0 means the vectors are orthogonal and 1 means they are the same vector
def MatrixSimilarityChecker(A, maxSimilarity):
    n_cols = A.shape[1]
    for col1 in range(n_cols - 1): # Iterate first column which is never the last one
        for col2 in range(col1 + 1, n_cols): # Iterate second column which is never the first one
            if SpectraSimilarity(A[:, col1], A[:, col2]) > maxSimilarity:
                print(A)
                print("columns ", col1, "and ", col2, "have similarity ", SpectraSimilarity(A[:, col1], A[:, col2])
                      , " > ", maxSimilarity)
                return False
    return True


# Get the spectra of one molecule
#
# Arguments: int molecule -> the key of the selected molecule 
#            Matrix spectralMatrix -> a matrix of spectral data
# Returns: a column vector(array?) -> the spectra of that molecule
def GetMoleculeSpectra(molecule, spectralMatrix):
    return spectralMatrix[:, molecule]

# Get a combined spectra of multiple (randomly selected) molecules 
#
# Arguments: int sampleComplexity -> how many different molecules are we mixing
#            float mixingResolution -> what is the resolution of the concentration of each
#            molecule? This will be between [0, 1) where 1 means there is no variance in 
#            concentration of each molecule ie every molecule has equal concentration
#            int numMolecules -> how many molecules do we want to simulate spectra for?
#            matrix spectralMatrix -> a matrix of spectral data
#            int wavelengths -> how many different wavelengths do we want for each spectrum?
# Returns: a column vector -> the combined spectra of the sample
def GetSampleSpectrum(sampleComplexity, mixingResolution, spectralMatrix, numMolecules, wavelengths):
    spectra = np.zeros(wavelengths)
    # randomly decide which molecules to add to the sample without replacement
    molecules = np.random.choice(numMolecules, size = sampleComplexity, replace=False)
    print("test01: molecules are " , molecules)
    M = int(1.0 / mixingResolution)
    for i in molecules: # the actual integer value not the place in the list
        # randomly decide concentration of each molecule with replacement at specified resolution
        concentration = np.random.randint(1, M+1) * mixingResolution
        print("test02: concentration of ", i, "is " , concentration)
        spectra += spectralMatrix[:, i] * concentration
    return spectra 



# Generate a matrix with given parameters(molecules, number of wavelengths, similarity of spectra, and sparsity)
#
# Arguments: int numMolecules -> how many molecules do we want to simulate spectra for?
#            int wavelengths -> how many different wavelengths do we want for each spectrum?
#            float maxSimilarity -> the maximum similarity we will allow for our matrix
#            float sparsity -> what proportion of entries in the matrix are zero
# Returns: a column vector -> the combined spectra of the sample
def GenerateMatrix(numMolecules, wavelengths, maxSimilarity, sparsity):
    tries = 0
    hold = True
    while hold:
        tries += 1
        B = ExactSparseSpectralMatrix(numMolecules, wavelengths, sparsity)
        if MatrixSimilarityChecker(B, MAXSIMILARITY):
            hold = False
    print("it took ", tries, " tries to make a matrix with these parameters!")
    return B

def main():
    A = GenerateMatrix(NUMMOLECULES, WAVELENGTHS, MAXSIMILARITY, SPARSITY)
    print(A)
    #b = MatrixSimilarityChecker(A, MAXSIMILARITY)
    #print(b)
    #c = SpectraSimilarity(A[:, 0], A[:, 1])
    #print("similarity of columns 0 and 1: ", c)
    #x = GetSampleSpectrum(SAMPLECOMPLEXITY, MIXINGRESOLUTION, A, NUMMOLECULES, WAVELENGTHS)
    #print (x)
main()