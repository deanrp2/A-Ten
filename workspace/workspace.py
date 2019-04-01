#THIS IS INTENDED AS A WORKSPACE FOR WORKER FUNCTIONS AND SHOULD NOT BE INCLUDED IN THE FINAL COMPILATION OF THE CODE
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import scipy.optimize
plt.close()

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
    #Credit to https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def bin_calibration(cs137spec_file):
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
    
def id_groups(bin_energies, source_spec, plot_peaks = False, width_threshold = [20, 120]):
    """
    
    """
    #Define gaussian checker for later in function    
    def is_gaussian(n, x, y):
        def gaus(x,mu,sigma):
            return (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-mu)**2/(2*sigma**2))
        
        y = y/np.trapz(y,x)
        mean0 = sum(x*y)/n
        sigma0 = sum(y*(x-mean0)**2)/n
        
        #plt.figure()
        popt,pcov = scipy.optimize.curve_fit(gaus,x,y,p0=[mean0,sigma0])
        #perr = np.sqrt(np.diag(pcov))
        #plt.plot(x,y,'b+:',label='data')
        #plt.plot(x,gaus(x,*popt),'ro:',label='fit')
        if np.sqrt(np.diag(pcov)).sum() > 1.6:
            return False
        else:
            return True

    #Smooth signal to and easily identify spectrum peaks
    smoothed_spec = smooth(source_spec,30)
    peaks, empty_dict = scipy.signal.find_peaks(smoothed_spec)
    
    #Apply criteria that prominences must be greater than the average promenance
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(smoothed_spec, peaks)
    peaks = np.compress(prominences > prominences.mean(), peaks)

    #Apply criteria that widths must be greater than width_threshold
    widths, width_heights, leftips, rightips = scipy.signal.peak_widths(x = smoothed_spec, peaks = peaks, rel_height = 0.9)
    peaks = np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), peaks)
    leftips = np.floor(np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), leftips)).astype(np.int32)
    rightips = np.ceil(np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), rightips)).astype(np.int32)    
    widths = np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), widths)
    
    #Apply criteria that shape must be appropriately gaussian
    remove_i = np.array([]).astype(int)
    for i in range(peaks.size):
        n = rightips[i]-leftips[i]
        x = np.linspace(leftips[i],rightips[i],n)
        y = source_spec[leftips[i]:rightips[i]]
        if not is_gaussian(n,x,y):
            remove_i = np.append(remove_i, i)
        
    peaks = np.delete(peaks,remove_i)
    leftips = np.delete(leftips,remove_i)
    rightips = np.delete(rightips,remove_i)
    
    #integrate area under each peak
    group_counts = np.array([])
    
    for i in range(peaks.size):
        group_counts = np.append(group_counts, np.sum(source_spec[leftips[i]:rightips[i]+1]))
        
    print(group_counts)
    print(bin_energies[peaks])
    
    
    if plot_peaks:
        plt.figure()
        plt.plot(bin_energies, source_spec)
        
        for i in range(leftips.size):
            plt.axvline(bin_energies[leftips[i]], c = "crimson")
            plt.axvline(bin_energies[rightips[i]], c = "crimson")
            
        plt.ylabel("Counts [#/s]")
        plt.xlabel("Energy [keV]")

    group_counts = group_counts
    group_energies = bin_energies[peaks]

    

    
espec = pull_spectrum("na22_spectrum.Spe")
ebins = bin_calibration("cs137_spectrum.Spe")

id_groups(ebins, espec, plot_peaks = True)
#plt.plot(ebins,espec)