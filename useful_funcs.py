def pull_aten(path_to_output):
    with open(path_to_output) as out:
        file = out.readlines()
    
    startflag = "Grp Energy [keV]   Count Rate [#/s]   Depth [cm]   Atten Coeff [cm^2]   Material"
    startline = [i for i in range(len(file)) if  startflag in file[i]][0]
    endflag = "Attenuation Coefficient Values"
    endline =   [i for i in range(len(file)) if  endflag in file[i]][0] -2

    file = file[startline+1:endline]
    holder = []
    series_dict = {}
    for line in file:
        if "-  -  -  -  - " in line:
            full_data = np.array(holder).astype(np.float64)
            holder = []
            series_dict[full_data[0,0]] = full_data[:,1:]
        else:
            holder.append(line.split()[0:4])
    
    return series_dict
    
def pull_expr(path_to_expr):
    with open(path_to_expr) as out:
        file = out.readlines()
    
    data = {}
    working = False
    for i in range(len(file)):
        if ".Spe" in file[i]:
            working = file[i]
            data[working] = []
        elif "Group" in file[i]:
            pass
        else:
            data[working].append(file[i].split())
          
    for key in data:
        data[key] = np.array(data[key]).astype(np.float64)
    return data

