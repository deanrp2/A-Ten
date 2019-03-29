#Main file for namsless should splinter later in development

class Atten:
    def __init():
        #the input parser should go here
        #material order: numpy vector with atomic number of each material in order
        #material density: numpy vetor
        
    def bin_calibration(self, cs137spec_file):
        """
        Function to set energy values to each bin. Must be based on cs137 spectrum.
        Parameters:
            cs137spec_file - file name  of a *.Spe file containing unattenuated spectrum from a Cs-137 source 
        Returns:
            bin_energies - numpy array containing the estimated energies associated with each bin
        """
        cs137_peak = 661.6 #keV
        
        #Smooth the spectrum and extract energy local maxima
        cs137_spec = smooth(pull_spectrum(cs137spec_file),30)
        peaks, empty_dict = scipy.signal.find_peaks(cs137_spec)
        
        #calculate the prominence associated with each maxima and find the bin with the largest prominence
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(cs137_spec, peaks)
        maxEpeak = prominences.argmax()
        
        #Find bin of maximum energy
        maxEbin = peaks[maxEpeak]
        delE = cs137_peak/maxEbin
        
        #Create bin energy array
        bin_energies = np.arange(0, cs137_spec.size * delE, delE)
        
        return bin_energies
    
    
    
    
    
