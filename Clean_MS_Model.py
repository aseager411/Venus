# Author: Alex Seager
# Last Version: 6/9/25
#
# Description: I am attempting to model and simulate MS data. I generate a matrix of molecules 
# and their artificial spectra with a random value at each 'Wavelength.' This creates the 
# spectral matrix where we can pull out columns (spectra) and combine them using mixing ratio 
# and other distortions (broadening?). Parameters of teh spectral matrix include sparsity, 
# similarity, and the number of molecules and wavelengths. one can also specify the resolution 
# of the data and parameters of the mixed signal

import numpy as np
#np.random.seed(420)  # Seed the random number generator for repropducability 