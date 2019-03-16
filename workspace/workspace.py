#THIS IS INTENDED AS A WORKSPACE FOR WORKER FUNCTIONS AND SHOULD NOT BE INCLUDED IN THE FINAL COMPILATION OF THE CODE
import numpy as np
import matplotlib.pyplot as plt


def pull_spectrum(spec_file):
    with open(spec_file) as data:
        spectrum = np.array([float(i.strip()) for i in data.readlines()[12:2060]])
    return spectrum



plt.plot(pull_spectrum("cs137_spectrum.Spe"))