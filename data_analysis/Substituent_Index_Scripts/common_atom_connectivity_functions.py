import os
import re
import numpy as np
import pandas as pd
import itertools
import pickle
import sys
import warnings
import openbabel
from openbabel import pybel
from openbabel.pybel import Outputfile
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import Descriptors, Descriptors3D
import rdkit.ML.Descriptors.MoleculeDescriptors
from rdkit import DataStructs
import freesasa
# import molml.utils
from molml.constants import TYPE_ORDER
from scipy.spatial.distance import cdist

#################################################################################
# Functions taken from code compiled by E. Farrar

# set folder location of scripts
sys.path.insert(0, "X:/ML_and_Feature_Codes/structure_reading_scripts")
import distance, angle, dihedral
sys.path.remove("X:/ML_and_Feature_Codes/structure_reading_scripts")

# set location of local copy of cclib
import cclib

sys.path.insert(0, "C:/Users/Niamh/OneDrive/PhD/Git/mmltoolkit") # insert path to mmltoolkit repo here
from mmltoolkit.featurizations import coulombmat_and_eigenvalues_as_vec
from mmltoolkit.featurizations import sum_over_bonds
sys.path.remove("C:/Users/Niamh/OneDrive/PhD/Git/mmltoolkit") # insert path to mmltoolkit repo here


# define BOND_LENGTHS with approximate covalent radii of Br
BOND_LENGTHS = {'C': {'3': 0.62, '2': 0.69, 'Ar': 0.72, '1': 0.85},
 'Cl': {'1': 1.045},
 'F': {'1': 1.23},
 'H': {'1': 0.6},
 'N': {'3': 0.565, '2': 0.63, 'Ar': 0.655, '1': 0.74},
 'O': {'3': 0.53, '2': 0.59, 'Ar': 0.62, '1': 0.695},
 'P': {'2': 0.945, 'Ar': 0.985, '1': 1.11},
 'S': {'2': 0.905, 'Ar': 0.945, '1': 1.07},
 'Br': {'1': 1.2}}

# manually copied from molml
def get_connections(elements1, coords1, elements2=None, coords2=None):
    disjoint = True
    if elements2 is None or coords2 is None:
        disjoint = False
        elements2 = elements1
        coords2 = coords1
    dist_mat = cdist(coords1, coords2)
    connections = {i: {} for i in range(len(elements1))}
    for i, element1 in enumerate(elements1):
        for j, element2 in enumerate(elements2):
            if not disjoint and i >= j:
                continue
            dist = dist_mat[i, j]
            bond_type = get_bond_type(element1, element2, dist)
            if not bond_type:
                continue
            connections[i][j] = bond_type
            if not disjoint:
                connections[j][i] = bond_type
    return connections

# manually copied from molml
def get_bond_type(element1, element2, dist):
    bad_eles = [x for x in (element1, element2) if x not in BOND_LENGTHS]
    if len(bad_eles):
        msg = "The following elements are not in BOND_LENGTHS: %s" % bad_eles
        warnings.warn(msg)
        return

    for key in TYPE_ORDER[::-1]:
        try:
            cutoff = BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]
            if dist < cutoff:
                return key
        except KeyError:
            continue


# function to convert atom numbers to atom symbols
def GetAtomSymbol(AtomNum):
    Lookup = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
              'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
              'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
              'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
              'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
              'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
              'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

    if AtomNum > 0 and AtomNum < len(Lookup):
        return Lookup[AtomNum-1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0

# find the atoms connected to the given atom in the given structure
def find_connected_atoms(path,atom):
    # get atom numbers and coordinations from cclib
    nums = cclib.io.ccread(path).atomnos
    coords = cclib.io.ccread(path).atomcoords
    # convert arrays to lists
    nums = nums.tolist()
    coords = coords[-1].tolist()
    # convert numbers to symbols
    symbs = []
    for num in nums:
        symbs.append(GetAtomSymbol(num))
    # calculate connectivity matrix
    connectivity = get_connections(symbs,coords)
    # find which atoms atom of interest is connected to
    our_atom = connectivity.get(atom-1)
    connected_atoms = []
    for i in our_atom.keys():
        connected_atoms.append(i+1)
    return connected_atoms

# find the atom on the substituent that is connected to the given atom in the given structure
def find_substituent_atoms(path,atom,common_atoms):
    connected_atoms = find_connected_atoms(path,atom)
    to_remove = []
    for i in connected_atoms:
        if i in common_atoms:
            to_remove.append(i)
    for i in to_remove:
        connected_atoms.remove(i)
    return connected_atoms

# Find distance between 2 indexed atoms in a structure
def find_distance(path,atom1,atom2):
    # get coordinations from cclib
    coords = cclib.io.ccread(path).atomcoords
    #anos = cclib.io.ccread(path).atomnos
    # convert arrays to lists
    coords = coords[-1].tolist()
    # Extract the coordinates for the atom indexes of interest
    coords1 = np.array(coords[atom1-1]).reshape((1,-1))
    coords2 = np.array(coords[atom2-1]).reshape((1,-1))
    # Calculate distance
    distance = cdist(coords1,coords2)
    return distance
