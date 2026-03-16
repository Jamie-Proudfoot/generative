import os
import sys
import csv
import pickle

os.chdir("X:/MultiObj_AL/Michael_Addition/Data/Processing")
#sys.path.insert(0, "X:/MultiObj_AL/Michael_Addition/Data/Processing") # location of saved common atom and TS bond index finding functions
import common_atom_connectivity_functions 
import find_ts_bond_index_functions
#sys.path.remove("X:/MultiObj_AL/Michael_Addition/Data/Processing")

#################################################################################################
# Here we will extract indices for the 1st atom of each substituent for the Nitro Michael Addition (NMA) ground-state dataset, the NMA transition state dataset,
# the malonate MA Ts dataset and the Azo MA TS dataset.
# We will also extract the indices of atoms in a TS bond from the NMA and malonate TS datasets. 
# The atomic indices for the position of the TS bond should be consistent within each dataset but we perform this extraction as an additional check.

# specify location of files (path must be absolute, not relative (i.e. '../..' cannot be used))
data_folder_dict = {"nma_gs": 'C:/Users/Niamh/Documents/PhD_MultiObj_AL/Data/Nitro/structures/structures/AM1/gs',"nma_ts":'C:/Users/Niamh/Documents/PhD_MultiObj_AL/Data/Nitro/structures/structures/AM1/ts',"mal_ts":'C:/Users/Niamh/Documents/PhD_MultiObj_AL/Data/Malonate/data_archive/data_archive/malonate_michael_addition/optimisations/am1_ts',"azo_ts":'C:/Users/Niamh/Documents/PhD_MultiObj_AL/Data/azo/DFT/transitionstates'}
#data_folder_dict = {"nma_gs": 'X:/MultiObj_AL/Michael_Addition/Data/Nitro/structures/structures/AM1/gs',"nma_ts":'X:/MultiObj_AL/Michael_Addition/Data/Nitro/structures/structures/AM1/ts',"mal_ts":'X:/MultiObj_AL/Michael_Addition/Data/Malonate/data_archive/data_archive/malonate_michael_addition/optimisations/am1_ts',"azo_ts":'X:/MultiObj_AL/Michael_Addition/Data/azo/DFT/transitionstates'}
#data_folder_dict = {"nma_gs": 'X:/MultiObj_AL/Michael_Addition/Data/Test/nitro/AM1/gs',"nma_ts":'X:/MultiObj_AL/Michael_Addition/Data/Test/nitro/AM1/ts',"mal_ts":'X:/MultiObj_AL/Michael_Addition/Data/Test/malonate/AM1/ts',"azo_ts":'X:/MultiObj_AL/Michael_Addition/Data/Test/azo/DFT/ts'}

# Define common atom list for structures
# We define R1 to be the substituent on the beta C on the opposite side of the alpha-beta C=C as the carbonyl C. R2 is the other substituent of the beta C. R3 is the alpha C substituent and R4 is the substituent on the carbonyl C.
# Here we only provide the core electrophile and nucleophile atoms e.g. not the first atom of R1. We attempt to distinguish between R1 and R2 by comparing relative distances to the carbonyl C. R2 should be closer.
#common_atoms_dict = {"nma_gs":[1,2,3,4,5],"nma_ts":[1,2,3,4,5,6,7,8,9,10,11],"mal_ts":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],"azo_ts":[1,2,3,4,5,6,7,8,9,10,11,12]}
common_atoms_dict = {"nma_gs":[1,2,3,4],"nma_ts":[1,2,3,4,5,6,7,8,9,10],"mal_ts":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"azo_ts":[1,2,3,4,5,6,7,8,9,10,11]}

# Create dictionaries giving the core atoms bearing substituents R2, R3 and R4
core_sub_atoms_nma = {"R2": 1, "R3": 2,"R4": 3}
core_sub_atoms_mal = {"R2": 1, "R3": 2,"R4": 3}
core_sub_atoms_azo = {"R2": 4, "R3": 1,"R4": 2}
subst_atoms_dict = {"nma_gs":core_sub_atoms_nma,"nma_ts":core_sub_atoms_nma,"mal_ts":core_sub_atoms_mal,"azo_ts":core_sub_atoms_azo}

# Outputs for substituent indices
subst_output_dict = {"nma_gs":"subst_indices_nma_gs.csv","nma_ts":"subst_indices_nma_ts.csv","mal_ts":"subst_indices_mal_ts.csv","azo_ts":"subst_indices_azo_ts.csv"}
# TS bonds
#ts_output_dict = {"nma_ts":"ts_bond_indices_nma_ts.pkl","mal_ts":"ts_bond_indices_mal_ts.pkl"}

# Functions to trim file paths
def _trim_path1(path):
    s = path.split("/")[-1].split(".")[0]
    t = s.split("_")
    trimmed_path = "-".join(t[0:2])
    return trimmed_path
def _trim_path2(path):
    s = path.split("/")[-1].split(".")[0]
    t = s.split("-")
    trimmed_path = "-".join(t[0:2])
    return trimmed_path
short_path_func_dict = {"nma_gs":_trim_path2,"nma_ts":_trim_path2,"mal_ts":_trim_path1,"azo_ts":_trim_path2}

###########################################################
# Run the substituent index extractions
structs_to_check_dict = {}
for k in ["nma_gs","nma_ts"]:
#for k in data_folder_dict.keys():
    structs_to_check_list = [] # list to store structures from each dataset that should be checked by eye to ensure R1 and R2 are assigned consistently
    
    folder = data_folder_dict[k]    
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".out") or filename.endswith(".log"):
            if not filename.endswith("SPE.out") and not filename.endswith("SPE.log"): # ignore any *_SPE files
                file = os.path.join(folder,filename)
                file_formatted = file.replace("\\","/")
                files.append(file_formatted)
    print(f"{k}: {len(files)} files to analyse for substituent atom indices.")            
    common_atoms = common_atoms_dict[k]
    core_subst_atoms = subst_atoms_dict[k]
    carb_C = core_subst_atoms["R4"] # atom index of carbonyl C to use to check we have R1 and R2 the right way around
    short_path_func = short_path_func_dict[k]
    output_file = subst_output_dict[k]
    
    atom_index_dict = {}
    for p in files:
        short_file_name = short_path_func(p)
        temp = {}
        for name, atom in core_subst_atoms.items():
            substituent_atoms = common_atom_connectivity_functions.find_substituent_atoms(p,atom,common_atoms)
            if name == "R2":
                if len(substituent_atoms) == 2: # 2 substituents bonded to beta C
                    # Compare distances between the 2 substituents and the carbonyl C. Assign R1 to the larger distance as defined as opposite side of C=C to carbonyl C.
                    idx1 = substituent_atoms[0]
                    idx2 = substituent_atoms[1]
                    dist1 = common_atom_connectivity_functions.find_distance(p,atom1=carb_C,atom2=idx1)
                    dist2 = common_atom_connectivity_functions.find_distance(p,atom1=carb_C,atom2=idx2)
                    if dist1 < dist2:
                        temp.update({"R1":substituent_atoms[1],"R2":substituent_atoms[0]})
                    else:
                        temp.update({"R1":substituent_atoms[0],"R2":substituent_atoms[1]})
                else:
                    print(f"Warning: {p} has more or less than 2 non-common atom connected to the given core atom. Structure needs manual checking.")
                    structs_to_check_list.append(p)    
            else:
                if len(substituent_atoms) == 1: # 1 substituent on alpha and carbonyl C
                    temp.update({name:substituent_atoms[0]})
                else:
                    print(f"Warning: {p} has more or less than 1 non-common atom connected to the given core atom. Structure needs manual checking.")
                    structs_to_check_list.append(p)                    
        atom_index_dict.update({short_file_name:temp})
    
    structs_to_check_dict.update({k:structs_to_check_list})
    
    # Output atom index dictionary for each dataset
    fields = ["structure","R1","R2","R3","R4"]
    with open(output_file,"w",newline='') as f:
        w = csv.DictWriter(f,fields)
        header = {}
        for i in fields:
            header[i] = i
        w.writerow(header) # header row
        for key, val in sorted(atom_index_dict.items()):
            row = {'structure': key}
            row.update(val)
            w.writerow(row)

with open("manual_check_structs_nma.pkl","wb") as f:
    pickle.dump(structs_to_check_dict,f)

#########################################################################
# Extract TS bond indices 
# for k in ["nma_ts","mal_ts"]:
#     folder = data_folder_dict[k]    
#     files = []
#     for filename in os.listdir(folder):
#         if filename.endswith(".out") or filename.endswith(".log"):
#             if not filename.endswith("SPE.out") and not filename.endswith("SPE.log"): # ignore any *_SPE files
#                 file = os.path.join(folder,filename)
#                 file_formatted = file.replace("\\","/")
#                 files.append(file_formatted)
#     print(f"{k}: {len(files)} files to analyse for TS bond indices.")            

#     short_path_func = short_path_func_dict[k]
#     ts_output_file = ts_output_dict[k]
#     # Find TS bonds. Set mode to 2 as only interested in identifying the bond that's forming (not changes in bond order in the TS for existing bonds).
#     ts_dict = find_ts_bond_index_functions.extract_ts_indices(files,short_path_func,num_ts_bonds="= 1",mode=2)   
#     # Output
#     with open(ts_output_file,"wb") as f:
#         pickle.dump(ts_dict,f)
