#%%

import os
import sys
import subprocess
import time

from rdkit import Chem
from rdkit.Chem import AllChem
from morfeus import Sterimol, read_xyz

import cclib

import pandas as pd
from tqdm import tqdm

#%%

def mol_to_gaussian_input(mol, filename="input.gjf", route_card="# pm6 opt=(calcfc,maxstep=5,maxcycles=100) freq=(noraman)", nproc=6, mem="12GB"):
    """
    Converts an RDKit Mol object with 3D coords to a Gaussian .com file.
    """
    # Add hydrogens
    mol_h = Chem.AddHs(mol)
    chg = Chem.GetFormalCharge(mol_h)
    # Embed 1 conformer in RDKit
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    AllChem.EmbedMolecule(mol_h, params)
    if mol_h.GetNumConformers() == 0: raise ValueError("ETKDGv3 embedding failed")
    # Optimise with MMFF 
    AllChem.MMFFOptimizeMolecule(mol_h)
    if mol_h.GetNumConformers() == 0: raise ValueError("MMFF optimization failed")

    chk_name = filename.replace(".com", ".chk")
    f = open(filename, "w")
    # f.write(f"%chk={chk_name}\n")
    f.write(f"%mem={mem}\n")
    f.write(f"%nprocshared={nproc}\n")
    f.write(f"%nosave\n\n") # do not save scratch rwf

    # Route card and comment line
    f.write(f"{route_card}\n\n")
    f.write("RDKit Generated Molecule\n\n")

    # Charge + multiplicity (assume singlet)
    f.write(f"{chg} 1\n")

    # Write atom coordinates
    for atom in mol_h.GetAtoms():
        pos = mol_h.GetConformer(0).GetAtomPosition(atom.GetIdx())
        atom_symbol = atom.GetSymbol()
        # Gaussian format for GJF file
        f.write(f"{atom_symbol:<2} {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}\n")
    f.write("\n")

    f.close()
    # print(f"Gaussian input file '{filename}' created successfully.")

#%%

# test_1000.smi - 1000 mols
# test_10000.smi - 10000 mols
# data = pd.read_csv("test_10000.smi", sep=" ")
data = pd.read_csv("ZINC_Michael_250k_500_100.csv.gz")
run = 10
runs = 10
size = int(len(data)/runs)
if not os.path.exists(str(run)): os.mkdir(str(run))

#%%

errs = []
print(f"{int((run-1)*size)}:{int(run*size)}")
for index, row in tqdm(data.iloc[int((run-1)*size):int(run*size)].iterrows(),total=int(size)):
# for index, row in data.iloc[int((run-1)*size):int(run*size)+1].iterrows():

    smi = row.smiles
    name = row.id
    
    # Create molecule object
    try: mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        errs.append(f"{name} {e}")
        pass

    # Create Gaussian input file
    try: mol_to_gaussian_input(mol, filename=f"{run}/{name}.gjf")
    except Exception as e:
        errs.append(f"{name} {e}")
        pass

with open(f"{run}/{run}.err","w+") as f:
    for line in errs: f.write(f"{line}\n")

#%%