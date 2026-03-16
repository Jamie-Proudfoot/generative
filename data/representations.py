#%%

import re
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

from tqdm import tqdm
tqdm.pandas()

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
df.head()

#%%

def to_canonical(smi, remove_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if remove_stereo: Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)

def to_ordered(smi, ma_smarts="O=C([*])/C([*])=C([*])/[*]", remove_stereo=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    n = mol.GetNumAtoms()
    michael = Chem.MolFromSmarts(ma_smarts) # "[*]C(=O)/C([*])=C([*])/[*]"
    matches = mol.GetSubstructMatches(michael)
    if not matches: raise ValueError("No matching Michael acceptor")
    # If matches > 1, select first match
    vals = matches[0]
    new_indices = list(vals)+[i for i in range(n) if i not in vals]
    mol = Chem.RenumberAtoms(mol, new_indices)
    if remove_stereo: Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=False)

def to_fragmented(smi, remove_stereo=False, warnings=False, removeHs=False, corder=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    n = mol.GetNumAtoms()
    ma_smarts = "O=C([*])/C([*])=C([*])/[*]"
    atoms = ["O4","C3","R4","C2","R3","C1","R1","R2"]
    michael = Chem.MolFromSmarts(ma_smarts) # "[*]C(=O)/C([*])=C([*])/[*]"
    matches = mol.GetSubstructMatches(michael)
    if not matches: return smi
    # If matches > 1, select first match
    vals = list(matches[0])
    # Correct RDKit behaviour that forces cis double bonds to be trans
    tofix = False
    bd = mol.GetBondBetweenAtoms(vals[atoms.index("C2")],vals[atoms.index("C1")])
    bondstereo = bd.GetStereo()
    R2_atm = mol.GetAtomWithIdx(vals[atoms.index("R2")]).GetAtomicNum()
    R3_atm = mol.GetAtomWithIdx(vals[atoms.index("R3")]).GetAtomicNum()
    R3_aro = mol.GetAtomWithIdx(vals[atoms.index("R3")]).GetIsAromatic()
    if bondstereo == Chem.BondStereo.STEREOZ and \
        ((R2_atm == 1) or (R2_atm == R3_atm)) and not (R3_atm > 6 or R3_aro): tofix = True
    elif bondstereo == Chem.BondStereo.STEREOE and \
        ((R3_atm == 1 and R2_atm != 1) or (R3_atm > 6 or R3_aro)): tofix = True
    if tofix:
        if warnings: print("WARNING: APPLYING MANUAL FIX")
        vals[atoms.index("R1")], vals[atoms.index("R2")] = vals[atoms.index("R2")], vals[atoms.index("R1")]
    new_indices = vals+[i for i in range(n) if i not in vals]
    mol = Chem.RenumberAtoms(mol, new_indices)
    b4 = mol.GetBondBetweenAtoms(atoms.index("C3"),atoms.index("R4")).GetIdx()
    b3 = mol.GetBondBetweenAtoms(atoms.index("C2"),atoms.index("R3")).GetIdx()
    b2 = mol.GetBondBetweenAtoms(atoms.index("C1"),atoms.index("R2")).GetIdx()
    b1 = mol.GetBondBetweenAtoms(atoms.index("C1"),atoms.index("R1")).GetIdx()
    mol = Chem.FragmentOnBonds(mol, addDummies=True, bondIndices=[b4,b3,b2,b1], dummyLabels=[(4,4),(3,3),(2,2),(1,1)])
    if remove_stereo: Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    frags = Chem.MolToSmiles(mol, canonical=True).split(".")
    frags = [f for f in frags if f != '[1*]C([2*])=C([3*])C([4*])=O']
    frags = sorted(frags, key=lambda x: int(re.search(r'\[(\d+)\*\]', x).group(1)))
    if removeHs: frags = [f for f in frags if not re.match(r'\[(\d+)\*\]\[H\]', f)]
    if corder: return ".".join(frags[2:][::-1]+frags[:2]) # order is R4, R3, R1 (trans), R2 (cis)
    else: return ".".join(frags) # order is R1 (trans), R2 (cis), R3 (alpha), R4 (carbonyl)

def reconstruct(frag_smi, remove_stereo=False):
    for i in range(1,5): # add hydrogens if missing
        if f"[{i}*]" not in frag_smi: frag_smi += f".[{i}*][H]"
    ma_smarts = "O=C([4*])/C([3*])=C([2*])/[1*]"
    ma_smarts = re.sub(r'\[(\d+)\*\]', r'[*:\1]', ma_smarts)
    frag_smi = re.sub(r'\[(\d+)\*\]', r'[*:\1]', frag_smi)
    frag_smi = ".".join([ma_smarts,frag_smi])
    mol = Chem.molzip(Chem.MolFromSmiles(frag_smi))
    if remove_stereo: Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)

#%%

i = np.random.choice(len(df))
smiles = df.smiles[i]
print(smiles)
canon_smi = to_canonical(smiles, remove_stereo=True)
print(canon_smi)
ordered_smi = to_ordered(smiles, remove_stereo=True)
print(ordered_smi)
frag_smi = to_fragmented(smiles, remove_stereo=True, warnings=True, removeHs=True)
print(frag_smi)
recon_smi = reconstruct(frag_smi, remove_stereo=False)
# print(recon_smi)
Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(frag_smi), Chem.MolFromSmiles(recon_smi)])

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
canon_smi = df["smiles"].progress_apply(lambda x: to_canonical(x, remove_stereo=True))
df["smiles"] = canon_smi
df.to_csv("ZINC_Michael_canon.csv.gz",index=False,compression="gzip")

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
canon_smi = df["smiles"].progress_apply(lambda x: to_canonical(x, remove_stereo=False))
df["smiles"] = canon_smi
df.to_csv("ZINC_Michael_canon_stereo.csv.gz",index=False,compression="gzip")

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
ordered_smi = df["smiles"].progress_apply(lambda x: to_ordered(x, remove_stereo=True))
df["smiles"] = ordered_smi
df.to_csv("ZINC_Michael_ordered.csv.gz",index=False,compression="gzip")

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
frag_smi = df["smiles"].progress_apply(lambda x: to_fragmented(x, remove_stereo=True))
df["smiles"] = frag_smi
df.to_csv("ZINC_Michael_fragments.csv.gz",index=False,compression="gzip")

#%%

df = pd.read_csv("ZINC_Michael_data.csv.gz")
frag_smi = df["smiles"].progress_apply(lambda x: to_fragmented(x, remove_stereo=True, removeHs=True))
df["smiles"] = frag_smi
df.to_csv("ZINC_Michael_fragments_noHs.csv.gz",index=False,compression="gzip")

#%%