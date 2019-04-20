import sys
import matplotlib.pyplot as plt

from ATen import ATen, pull_spectrum

class ATen_expr(ATen):
    def __init__(self, cs137_spec_path, background_path, background_time, outfile_name,  measurement_time, exprfiles):
        self.bin_calibration(cs137_spec_path)
        
        output = open(outfile_name, "w")
        for experiment in exprfiles:
            self.source_spec = pull_spectrum(experiment)/measurement_time
            self.source_spec -= pull_spectrum(background_path)/background_time
            self.id_groups(plot_spec_peaks = True, plot_norm_peaks = True)
            
            output.write(experiment + "\n")
            
            output.write("Group Energy [keV]                Group Count Rate [#/s]\n")            
            for i in range(len(self.group_counts)):
                grp = "%.5f"%(self.group_energies[i])
                output.write(grp + (34 - len(grp))*" " + "%.5f\n"%(self.group_counts[i]))



ms_input = sys.argv[2:]

if not ms_input:
    input_structure = """
The structure of the run command is as follows:
<path to ATen_expr.py> -i <cs137_spec_path> <background_path> <background_time> <outfile_name> <experimental measurement time> <list of experiment files>
"""
    print(input_structure)

else:
    main = ATen_expr(ms_input[0], ms_input[1], float(ms_input[2]), ms_input[3], float(ms_input[4]), ms_input[5:])
    
plt.show()