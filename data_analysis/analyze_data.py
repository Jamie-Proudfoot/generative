
#%%

import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#%%

# Main dataframe
df = pd.read_csv("aza_michael_barriers_pm6.csv")
df.head()

#%%

# Dataframe for mapping molecule substituent indices
df_idx = pd.read_csv("Substituent_Indices/subst_indices_nma_gs.csv")
df_idx["nitro_rxn"] = df_idx["structure"].apply(lambda x: int(x.split("-")[-1]))
df_idx.head()

# Atom mapping for core Michael acceptor atoms
# (nitro-Michael to match source of features)
# CB = C1, CA = C2, C = C3, O = O4
core_nitro_michael ={
    "O4": 4, "C3": 3, "C2": 2, "C1": 1
}

#%%

# Source of molecular/atomic features 
file = "nitro_ma_gs/nitro_ma_gs_pm6.hdf5"
# file = "NMA_GS.hdf5"
with h5py.File(file, "r") as f: 
    keys = f.keys()
    keylist = list(keys)
    for i in range(len(keylist)):
        print(keylist[i])
        print(list(f[keylist[i]]))

#%%

# Use the "files" HSF5 key to extract molecule-level indices
indices = []
with h5py.File(file, "r") as f: 
    subkey = list(f["files"])[0]
    data = f["files"][subkey]
    for d in data:
        i = int(d.decode("utf-8").split("/")[-1].split("-")[1])
        indices.append(i)

#%%

# keys: all features available in HDF5 file
# atomic_scalars: scalar values for each atom (e.g. Mulliken charges)
# atomic_arrays: array values for each atom (notably, sterimol parameters)
# scalars: molecule-level scalars (e.g. num_atoms)
# not_implemented: missing from hdf5
# other: file specification (bytes-like strings)

keys =  ['atomic_MRs', 'atomic_acharges', 'atomic_acharges_sum', 
         'atomic_aromatics', 'atomic_d4_coord_nums_multi', 'atomic_d4_coord_nums_single', 
         'atomic_estates', 'atomic_fcharges', 'atomic_heavies', 'atomic_hybridizations', 
         'atomic_hydrogens', 'atomic_logPs', 'atomic_mcharges', 'atomic_mcharges_sum', 
         'atomic_nums', 'atomic_pbvs', 'atomic_pcharges', 'atomic_pints', 'atomic_rings', 
         'atomic_sasas', 'atomic_sterimol_B1s', 'atomic_sterimol_B5s', 'atomic_sterimol_Ls', 
         'atomic_valences', 'atomic_vsas', 'cartesians', 'connectivity', 'energies', 'files', 
         'homo_energies', 'lumo_energies', 'num_atoms', 'num_bonds', 'one_hots']

atomic_scalars = ['atomic_MRs','atomic_aromatics','atomic_d4_coord_nums_multi',
                'atomic_estates','atomic_fcharges','atomic_heavies',
                'atomic_hybridizations','atomic_hydrogens','atomic_logPs',
                'atomic_mcharges','atomic_mcharges_sum',
                'atomic_pbvs','atomic_pcharges','atomic_pints','atomic_rings',
                'atomic_sasas','atomic_valences','atomic_vsas',
                'atomic_nums'
                ]

atomic_arrays = ['atomic_sterimol_B1s','atomic_sterimol_B5s','atomic_sterimol_Ls','cartesians','connectivity','one_hots']

scalars = ['energies','homo_energies','lumo_energies','num_atoms','num_bonds']

not_implemented = ['atomic_acharges','atomic_acharges_sum']

other = ['files']

#%%

# Convert recognisable names into hdf5 keys
name_conversion = {
    "Mulliken": "atomic_mcharges",
    "PEOE": "atomic_vsas",
    "Pint": "atomic_pints",
    "PBV": "atomic_pbvs",
    "B1": "atomic_sterimol_B1s",
    "B5": "atomic_sterimol_B5s",
    "Energy": 'energies',
    "HOMO": 'homo_energies',
    "LUMO": 'lumo_energies',
    "NAtoms": 'num_atoms',
    "NBonds": 'num_bonds'
}

#%%

# Select a feature (and an atom type for atom-level features)
# create a new column in the dataframe corresponding to this feature
# O4: carbonyl-O
# C3: carbonyl-C 
# C2: alpha-C
# C1: beta-C
# R4: first atom of substitent bonded to carbonyl
# R3: first atom of substituent bonded to alpha-C
# R2: first atom of substituent bonded to beta-C (cis to carbonyl)
# R1: first atom of substituent bonded to beta-C (trans to carbonyl)

feat = "Mulliken"
atom = "O4" # (ignored for non-atomic features)

if feat in name_conversion.keys(): fname = name_conversion[feat]
else: fname = feat

dlist = []
base = [i for i in range(1,1001)] # convert unix to base-1 ordering
unix = sorted(base, key = lambda x: str(x))
with h5py.File(file, "r") as f:

    for index, row in df.iterrows():

        # i = row["nitro_rxn"] # 1-indexed
        # i = unix.index(i)+1 # convert to unix ordering
        try: i =  indices.index(row["nitro_rxn"])+1
        except: pass
        print(f"Index: {row['nitro_rxn']} ({i})")

        if fname in atomic_scalars + atomic_arrays:
            if atom in core_nitro_michael.keys(): idx = core_nitro_michael[atom]
            elif atom in df_idx.columns[1:5]: idx = df_idx[df_idx["nitro_rxn"]==i][atom].values[0]
            else: raise ValueError(f"{atom} is an invalid atom specifier")
        
        print(list(f[fname]))
        if "PM6" in list(f[fname]): subkey = "PM6"
        elif "DFT" in list(f[fname]): subkey = "DFT"
        else: subkey = list(f[fname])[0]

        data = (f[fname][subkey])
        try:
            if data.shape != (0,):
                if hasattr(data[i-1], "shape"):
                    if data[i-1].shape:
                        if fname in atomic_scalars:
                            print(f"Data[{idx-1}]: \n{data[i-1][idx-1]}")
                            dlist.append(data[i-1][idx-1])
                        elif fname in atomic_arrays:
                            if fname == "atomic_sterimol_B1s":
                                array = data[i-1][idx-1].flatten()
                                array = array[np.nonzero(array)]
                                b1 = min(array) if array.any() else 0
                                print(f"min(Data[{idx-1}]): \n{b1}")
                                dlist.append(b1)
                            elif fname == "atomic_sterimol_B5s":
                                array = data[i-1][idx-1].flatten()
                                array = array[np.nonzero(array)]
                                b5 = max(array) if array.any() else 0
                                print(f"max(Data[{idx-1}]): \n{b5}")
                                dlist.append(b5)
                            else: raise ValueError(f"{fname} not implemented yet")
                        elif fname in scalars: 
                            print(f"Data: {data[i-1]}")
                            dlist.append(data[i-1])
                        else: raise ValueError(f"{fname} not implemented yet")
                    else: 
                        print(f"Data: {[data[i-1]]}")
                        dlist.append([data[i-1]])
                else: 
                    print(f"Data: {data[i-1].decode('utf-8')}")
                    dlist.append([data[i-1].decode('utf-8')])
            else: raise ValueError("Empty data, shape: (0,)")
        except Exception as e:
            print(e)
            dlist.append([None])

if fname in atomic_scalars + atomic_arrays: colname = "_".join((feat,atom))
else: colname = feat
df[colname] = np.array(dlist).flatten()


# Plot scatter plot of the feature of interest and print Pearson's R^2
target = "low_barrier" # target for correlation ("low_barrier" or "barrier")
R2=np.round(pearsonr(df[df[colname].notnull()][colname].astype(float),df[df[colname].notnull()][target].astype(float)).statistic**2,4)
rng = np.random.default_rng(seed=int(str(int.from_bytes(colname.encode(),'little'))[:9]))
plt.scatter(df[df[colname].notnull()][colname],df[df[colname].notnull()][target], c=(rng.random(),rng.random(),rng.random()), alpha=0.5)
plt.xlabel(colname)
plt.ylabel(target)
plt.title(f"{colname} vs {target} ($R^2$: {R2})")
# Display column in DataFrame
df.head()


#%%