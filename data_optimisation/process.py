#%%

import os
import sys
import re
import subprocess
import time

from rdkit import Chem
from rdkit.Chem import AllChem
from morfeus import read_xyz, Sterimol, BuriedVolume

import cclib

import numpy as np
import pandas as pd
from tqdm import tqdm

from subprocess import Popen, DEVNULL
import glob

from pandarallel import pandarallel


#%%


def getLastMulliken(logfile):
    charges = []
    buffer_size = 8192
    found = False
    leftover = ""

    with open(logfile, 'rb') as f:
        f.seek(0, 2)  # eof
        pos = f.tell()
        accumulated = ""

        while pos > 0 and not found:
            read_size = min(buffer_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size).decode(errors='ignore')
            accumulated = chunk + accumulated

            lines = accumulated.splitlines()
            for i in reversed(range(len(lines))):
                if lines[i].strip().startswith("Mulliken charges:"):
                    found = True
                    block_lines = []
                    j = i + 2  # skip extra line after header
                    while j < len(lines):
                        l_strip = lines[j].strip()
                        if not l_strip or not l_strip[0].isdigit():
                            break
                        parts = l_strip.split()
                        try:
                            block_lines.append(float(parts[-1]))
                        except ValueError:
                            pass
                        j += 1
                    charges = block_lines
                    break

    if not charges: raise ValueError("No Mulliken charges found in the file.")

    return charges


def get_Mulliken(outfile, i):
    charges = getLastMulliken(outfile)
    mulliken = charges[i]
    return round(mulliken,6)

def get_PBV(xyzfile, i):
    elements, coordinates = read_xyz(xyzfile)
    pbv_C1 = BuriedVolume(elements, coordinates, i+1, excluded_atoms=None).fraction_buried_volume
    return round(pbv_C1,6)

def get_LUMO(outfile):
    # LUMO energy
    data = cclib.io.ccread(outfile)
    mo_energies = data.moenergies[0] # eV
    homo = data.homos[0] # HOMO idx
    lumo = homo + 1
    lumo_energy = mo_energies[lumo]
    return round(lumo_energy,6)

def get_vibfreq(outfile):
    data = cclib.io.ccread(outfile)
    vibfreq = data.vibfreqs[0] # cm-1
    return round(vibfreq,6)    

def get_descriptors(id, folder):
    # Read outfile and test if terminated correctly
    outfile = os.path.join(folder,f"{id}.out")
    with open(outfile,"r") as f: data = f.read()
    if "Error termination" in data: 
        # print("Error termination")
        return

    # Create temporary SDF and XYZ files for processing
    p1 = Popen(["obabel", outfile, "-O", f"{outfile.split('.')[0]}.sdf"], stdout=DEVNULL, stderr=DEVNULL)
    p2 = Popen(["obabel", outfile, "-O", f"{outfile.split('.')[0]}.xyz"], stdout=DEVNULL, stderr=DEVNULL)
    p1.wait(), p2.wait()

    # RDKit molecule object
    mol = Chem.SDMolSupplier(f"{outfile.split('.')[0]}.sdf", removeHs=False, sanitize=False)[0]
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY) # add aromaticity
    # Collect important atom indices
    michael = Chem.MolFromSmarts("O=C([*])/C([*])=C([*])/[*]")
    matching = True
    try: matches = mol.GetSubstructMatches(michael)
    except Exception as e: matching = False
    if not matching or not matches: 
        # print("No matching Michael acceptor")
        return
    
    # If matches > 1, select match with highest charge (~ most reactive)
    atomsets, charges = [], []
    for match in matches:
        atoms = dict(zip(["O4","C3","R4","C2","R3","C1","R1","R2"],match))
        atomsets.append(atoms)
        charges.append(get_Mulliken(outfile, atoms["O4"]))
    i = np.argmax(charges)
    atoms = atomsets[i]

    # Mulliken O4
    mulliken_O4 = charges[i]

    # PBV C1
    pbv_C1 = get_PBV(f"{outfile.split('.')[0]}.xyz", atoms["C1"])

    data = cclib.io.ccread(outfile)

    # LUMO energy
    mo_energies = data.moenergies[0] # eV
    homo = data.homos[0] # HOMO idx
    lumo = homo + 1
    lumo_energy = round(mo_energies[lumo],6)

    # Lowest vibrational frequency
    # (required for checking if ground-state)
    vibfreq = round(data.vibfreqs[0],6)   

    # Remove temp files
    p1 = Popen(["rm", f"{outfile.split('.')[0]}.sdf"], stdout=DEVNULL, stderr=DEVNULL)
    p2 = Popen(["rm", f"{outfile.split('.')[0]}.xyz"], stdout=DEVNULL, stderr=DEVNULL)

    descriptors = mulliken_O4, pbv_C1, lumo_energy, vibfreq
    return descriptors

#%%

c = 1
nproc = 12

pandarallel.initialize(progress_bar=False, nb_workers=nproc)

# (Smaller) reopts
chunk = f"{c}_err"
ids = []
for outfile in tqdm(glob.glob(f"{chunk}/*.out")[:]):
    id = os.path.basename(outfile).split(".")[0]
    if not id.startswith(f"ZINC_{chunk}"): ids.append(id)
df1 = pd.DataFrame({"id": ids})
df1["data"] = df1.parallel_apply(lambda row: get_descriptors(row["id"], chunk), axis='columns')

# (Larger) main 
chunk = str(c)
ids = []
for outfile in tqdm(glob.glob(f"{chunk}/*.out")[:]):
    id = os.path.basename(outfile).split(".")[0]
    if not id.startswith(f"ZINC_{chunk}"): ids.append(id)
df2 = pd.DataFrame({"id": ids})
df2["data"] = df2.parallel_apply(lambda row: get_descriptors(row["id"], chunk), axis='columns')

# Combine dataframes
df = pd.concat([df1, df2], ignore_index=True)
# Expand data column to new columns
cols = ["mulliken_O4", "pbv_C1", "lumo", "vibfreq"]
split = pd.DataFrame(df["data"].tolist(), columns=cols)
df = pd.concat([df, split], axis=1)
df = df.dropna(how="any", axis=0)
df = df.drop("data", axis=1)
# Filter non-stationary points
df = df[df["vibfreq"]>=0]

#%%

df.to_csv(f"ZINC_{c}.csv",index=False)

#%%
