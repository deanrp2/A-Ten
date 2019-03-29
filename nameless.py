#Main file for namsless should splinter later in development

def pull_spectrum(spec_file):
    """
    Function to extract the spectrum from a *.Spe file
    Parameters:
        spec_file - file name of *.Spe file
    Returns:
        spectrum - numpy array of the number of counts per bin 
    """
    with open(spec_file) as data:
        spectrum = np.array([float(i.strip()) for i in data.readlines()[12:2060]])
    return spectrum

def smooth(y, box_pts):
    """
    Simple 1-D convolution function
    Parameters:
        y - 1-D numpy array to be smoothed
        box_pts - width of kernel
    Returns:
        y_smooth - smoothed version of y
    """
    #Credit to https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



class Atten:
    def __init():
        p = 3
        #the input parser should go here
        #material order: numpy vector with atomic number of each material in order
        #material density: numpy vetor
        #self.cs137spec_filepath
        
    def bin_calibration(self):
        """
        Function to set energy values to each bin. Must be based on cs137 spectrum.
        Parameters:
            cs137spec_file - file name  of a *.Spe file containing unattenuated spectrum from a Cs-137 source 
        Returns:
            bin_energies - numpy array containing the estimated energies associated with each bin
        """
        cs137_peak = 661.6 #keV
        
        #Smooth the spectrum and extract energy local maxima
        cs137_spec = smooth(pull_spectrum(self.cs137spec_filepath),30)
        peaks, empty_dict = scipy.signal.find_peaks(cs137_spec)
        
        #calculate the prominence associated with each maxima and find the bin with the largest prominence
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(cs137_spec, peaks)
        maxEpeak = prominences.argmax()
        
        #Find bin of maximum energy
        maxEbin = peaks[maxEpeak]
        delE = cs137_peak/maxEbin
        
        #Create bin energy array
        bin_energies = np.arange(0, cs137_spec.size * delE, delE)
        
        self.energy_spec
        
        return bin_energies
    
    
    
    
    
