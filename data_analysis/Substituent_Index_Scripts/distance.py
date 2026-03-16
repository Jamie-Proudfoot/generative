#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017 Kristaps Ermanis for GetAtomSymbol and ReadGeometry
All else Elliot Farrar (ehef20) 2019

Script that measures bond distance between two atoms
"""

import argparse
import os
import math
import numpy as np


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


def ReadGeometry(GOutpFile):
    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()

    index = 0
    atoms = []
    coords = []
    gindexes = []
    chindex = None

    #Find the geometry section and charge section
    for index in range(len(GOutp)):
        if 'Charge =' in GOutp[index]:
            chindex = index
        if ('Input orientation:' in GOutp[index]) or ("Standard orientation:" in GOutp[index]):
            gindexes.append(index + 5)

    #Read geometries
    for line in GOutp[gindexes[-1]:]:
        if '--------------' in line:
            break
        else:
            data = list(filter(None, line[:-1].split(' ')))
            atoms.append(GetAtomSymbol(int(data[1])))
            coords.append(data[3:])

    if chindex != None:
        line = GOutp[chindex].split('Charge = ')
        line = line[1].split(' Multiplicity = ')
        charge = int(line[0])
    else:
        charge = -1000

    gausfile.close()

    for i, x in enumerate(coords):
    	x.insert(0,i+1)
    return atoms, coords, charge


def main(filenames,DistAtoms):
    conformers = []
    distances = []
    outputs = []
    for filename in filenames:
        atoms, coords, charge = ReadGeometry(filename)
        AtomSymbols = []
        conformers.append(coords)

        a = DistAtoms[0] - 1
        b = DistAtoms[1] - 1

        x1_coords = float(coords[a][1])
        y1_coords = float(coords[a][2])
        z1_coords = float(coords[a][3])
        x2_coords = float(coords[b][1])
        y2_coords = float(coords[b][2])
        z2_coords = float(coords[b][3])

        p1 = np.array([x1_coords, y1_coords, z1_coords])
        p2 = np.array([x2_coords, y2_coords, z2_coords])

        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        distances.append(dist)

        for an in DistAtoms:
            AtomSymbols.append(atoms[an-1])

        output = str(filename) + " (" + str(AtomSymbols[0]) + "-" + str(AtomSymbols[1]) + ")" + " : " + str(dist)

        #print(output)

        outputs.append(output)

    #print("\n")

    return outputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bond measurement script')
    parser.add_argument('OutputFiles', help="Space seperated list of output files", nargs='+')
    parser.add_argument('AtomsList', help="Comma seperated list of 2 atom numbers to measure distance between." +\
                        "Numbering starts with 1. Distances in angstroms")
    args = parser.parse_args()
        	    
    #print('\n' + 'Filename : ' + 'Distance (Atoms ' + args.AtomsList + ')' + '\n')

    SchrodEnv = os.getenv('SCHRODINGER')
    if SchrodEnv != None:
        settings.SCHRODINGER = SchrodEnv

    DistAtoms = [int(x) for x in args.AtomsList.split(',')]

    main(args.OutputFiles, DistAtoms)