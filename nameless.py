#Main file for namsless should splinter later in development

#UNITS
#time:s
#weight:g
#length:cm
#density: g/cm^3

import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
from collections import OrderedDict

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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



class Aten:
    def __init__(self,input_filepath):
        #Reading contecnts of input file
        with open(input_filepath) as input_file:
            input_data = [line.strip() for line in input_file.readlines() if not (line[0] == "#" or line == "\n")]
        
        #Read raw input text into cards
        self.material_card = input_data[input_data.index('READ MATERIAL')+1:input_data.index('END MATERIAL')]
        self.geometry_card = input_data[input_data.index('READ GEOMETRY')+1:input_data.index('END GEOMETRY')]
        self.parameters_card = input_data[input_data.index('READ PARAMETERS')+1:input_data.index('END PARAMETERS')]
        self.paths_card = input_data[input_data.index('READ PATHS')+1:input_data.index('END PATHS')]
        
        #initializing material card
        self.material_process()
        
        #parsing parameter related inputs
        self.parameters = {x[0] : x[1] for x in [parameter.replace(" ", "").split("=") for parameter in self.parameters_card]}
        
        #parsing path related inputs
        self.paths = {x[0] : x[1] for x in [parameter.replace(" ", "").split("=") for parameter in self.paths_card]}
        self.bin_calibration(self.paths["cs137spec_filepath"])
        
        self.source_spec = pull_spectrum(self.paths["source_filepath"])/float(self.parameters["source_time"])
        
        if "background_filepath" in self.paths:
            self.source_spec -= pull_spectrum(self.paths["background_filepath"])/float(self.parameters["background_time"])
        
        #material information
        #layer_material
        #layer_thickness
        #bin energies: vector with energies of each bin using the cs137spec_filepath
        #source energy distribution called using simple pull_spectrum from source_filepath
        
        
    def bin_calibration(self,cs137spec_filepath):
        """
        Function to set energy values to each bin. Must be based on cs137 spectrum.
        Parameters:
            self - does not pull any attributes from self
            cs137spec_filepath - file name  of a *.Spe file containing unattenuated spectrum from a Cs-137 source 
        Returns:
            self.bin_energies - numpy array containing the estimated energies associated with each bin
        """
        cs137_peak = 661.6 #keV
        
        #Smooth the spectrum and extract energy local maxima
        cs137_spec = pull_spectrum(cs137spec_filepath)  
        cs137_spec = smooth(cs137_spec,30)
        peaks, empty_dict = scipy.signal.find_peaks(cs137_spec)
        
        #calculate the prominence associated with each maxima and find the bin with the largest prominence
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(cs137_spec, peaks)
        maxEpeak = prominences.argmax()
        
        #Find bin of maximum energy
        maxEbin = peaks[maxEpeak]
        delE = cs137_peak/maxEbin
        
        #Create bin energy array
        self.bin_energies = np.arange(0, cs137_spec.size * delE, delE)
                
        return 1
    
    def id_groups(self, plot_spec_peaks = False, plot_norm_peaks = False, width_threshold = [20, 120], width_rel_height = 0.7):
        """
        Function to identify energy group and count rate of each energy group.
        Parameters:
            self - both bin_energies and source_spec must be defined
            plot_spec_peaks - boolean to plot full source spectrum with e group identified
            plot_norm_peaks = boolean to plot norm fits as well. Should only be true if plot_spec_peaks is true
            width_threshold - tuning parameter for acceptable peak width
        Returns:
            self.group_counts - count rate associated with each group
            self.group_energies - energy value associated with each group
        """
        #Define gaussian checker for later in function    
        def is_gaussian(n, x, y, width_rel_height,be,plot_norm_peaks):
            def gaus(x,a0,mu,sigma):
                return a0*(sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-mu)**2/(2*sigma**2)) 

            a0 = np.trapz(y) #4253
            mean0 =  x[y.argmax()] #sum(x*y)/n #377.977
            sigma0 =  sum(y*(x-mean0)**2)/n # 114.967
            

            popt,pcov = scipy.optimize.curve_fit(gaus,x,y,p0=[a0,mean0,sigma0],maxfev=100000)
            perr = np.sqrt(np.diag(pcov)).sum()
            

            if perr > 80:
                return (False, 0)
            else:
                if plot_norm_peaks:
                    plt.plot(be,gaus(x,*popt),'o', c = "mediumseagreen", MarkerSize=5, label = "Gaussian Approximation")
                return (True, popt[0])
        
        #initializing figure if necessary
        if plot_spec_peaks or plot_norm_peaks:
            plt.figure(figsize=[15,8])
        
        #Smooth signal to and easily identify spectrum peaks
        smoothed_spec = smooth(self.source_spec,30)
        peaks, empty_dict = scipy.signal.find_peaks(smoothed_spec)
        
        
        #Apply criteria that prominences must be greater than the average promenance
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(smoothed_spec, peaks)
        peaks = np.compress(prominences > prominences.mean(), peaks)
        
        #Apply criteria that widths must be greater than width_threshold
        widths, width_heights, leftips, rightips = scipy.signal.peak_widths(x = smoothed_spec, peaks = peaks, rel_height = width_rel_height)
        peaks = np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), peaks)
        leftips = np.floor(np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), leftips)).astype(np.int32)
        rightips = np.ceil(np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), rightips)).astype(np.int32)    
        widths = np.compress((widths > width_threshold[0]) * (widths < width_threshold[1]), widths)
        
        
        #Apply criteria that shape must be appropriately gaussian
        #Hiding in here is getting the count rate under each peak as a return from the is_gaussian
        remove_i = np.array([]).astype(int)
        group_counts = np.array([])
        for i in range(peaks.size):
            n = rightips[i]-leftips[i]
            x = np.linspace(leftips[i],rightips[i],n)
            y = self.source_spec[leftips[i]:rightips[i]]
            gaussianality, group_count = is_gaussian(n,x,y, width_rel_height,self.bin_energies[leftips[i]:rightips[i]],plot_norm_peaks = plot_norm_peaks)
            if not gaussianality:
                remove_i = np.append(remove_i, i)
            else:
                group_counts = np.append(group_counts, group_count)
            
        peaks = np.delete(peaks,remove_i)
        leftips = np.delete(leftips,remove_i)
        rightips = np.delete(rightips,remove_i)
        
        #integrate area under each peak, %%%%%replaced with gaussian approximation
        #group_counts = np.array([])
        
        #for i in range(peaks.size):
            #group_counts = np.append(group_counts, np.sum(self.source_spec[leftips[i]:rightips[i]+1]))
            
        #Add optional peak plotted to make sure peaks were done correctly
        if plot_spec_peaks:
            plt.plot(self.bin_energies,self.source_spec, c = "royalblue",label = "Source Spectrum")
            
            for i in range(leftips.size):
                plt.axvline(self.bin_energies[peaks[i]], c = "crimson", label = "Group Energy Value")

                
            plt.ylabel("Counts [#/s]")
            plt.xlabel("Energy [keV]")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

    
        self.group_counts = group_counts
        self.group_energies = self.bin_energies[peaks]
    
    def material_process(self):
        self.materials = {}
        for entry in self.material_card:
            entry = entry.split()
            material = entry.pop(0)
            density = float(entry.pop(0))
            composition = [[int(entry[2*i]), float(entry[2*i-1])] for i in range(int(len(entry)/2))]
            self.materials[material] = [density, composition]
    
test = Aten("workspace/test_inp.at")    
test.id_groups(plot_spec_peaks = True, plot_norm_peaks = False)
print(test.group_counts)
#plt.figure()
#plt.plot(test.bin_energies,test.source_spec)
#print(test.group_energies)
