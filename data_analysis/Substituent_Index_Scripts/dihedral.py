#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017 Kristaps Ermanis for GetAtomSymbol and ReadGeometry
All else Elliot Farrar (ehef20) 2019-2021

Script that measures dihedral angle between four atoms
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


def main(filenames,DihedralAtoms):
    conformers = []
    dihedrals = []
    outputs = []
    for filename in filenames:
        atoms, coords, charge = ReadGeometry(filename)
        AtomSymbols = []
        conformers.append(coords)
        a = DihedralAtoms[0] - 1
        b = DihedralAtoms[1] - 1
        c = DihedralAtoms[2] - 1
        d = DihedralAtoms[3] - 1

        x1_coords = float(coords[a][1])
        y1_coords = float(coords[a][2])
        z1_coords = float(coords[a][3])
        x2_coords = float(coords[b][1])
        y2_coords = float(coords[b][2])
        z2_coords = float(coords[b][3])
        x3_coords = float(coords[c][1])
        y3_coords = float(coords[c][2])
        z3_coords = float(coords[c][3])
        x4_coords = float(coords[d][1])
        y4_coords = float(coords[d][2])
        z4_coords = float(coords[d][3])

        p1 = np.array([x1_coords, y1_coords, z1_coords])
        p2 = np.array([x2_coords, y2_coords, z2_coords])
        p3 = np.array([x3_coords, y3_coords, z3_coords])
        p4 = np.array([x4_coords, y4_coords, z4_coords])

        q1 = np.subtract(p2,p1) # b - a
        q2 = np.subtract(p3,p2) # c - b
        q3 = np.subtract(p4,p3) # d - c

        q1_x_q2 = np.cross(q1,q2)
        q2_x_q3 = np.cross(q2,q3)

        n1 = q1_x_q2/np.sqrt(np.dot(q1_x_q2,q1_x_q2))
        n2 = q2_x_q3/np.sqrt(np.dot(q2_x_q3,q2_x_q3))

        u1 = n2
        u3 = q2/(np.sqrt(np.dot(q2,q2)))
        u2 = np.cross(u3,u1)

        cos_theta = np.dot(n1,u1)
        sin_theta = np.dot(n1,u2)

        theta = -math.atan2(sin_theta,cos_theta)
        theta_deg = np.degrees(theta)
        if theta_deg <= 0:
                theta_deg += 360

        dihedrals.append(theta_deg)

        for an in DihedralAtoms:
            AtomSymbols.append(atoms[an-1])

        output = str(filename) + " (" + str(AtomSymbols[0]) + "-" + str(AtomSymbols[1]) + "-" + str(AtomSymbols[2]) + "-" + str(AtomSymbols[3])  + ")" + " : " + str(theta_deg)

        #print(output)

        outputs.append(output)

    #print("\n")

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bond measurement script')
    parser.add_argument('OutputFiles', help="Space seperated list of output files", nargs='+')
    parser.add_argument('AtomsList', help="Comma seperated list of 4 atom numbers to measure dihedral between." +\
                        "Numbering starts with 1. Dihedrals in degrees")
    args = parser.parse_args()
        	    
    #print('\n' + 'Filename : ' + 'Dihedral (Atoms ' + args.AtomsList + ')' + '\n')

    SchrodEnv = os.getenv('SCHRODINGER')
    if SchrodEnv != None:
        settings.SCHRODINGER = SchrodEnv

    DihedralAtoms = [int(x) for x in args.AtomsList.split(',')]

    main(args.OutputFiles, DihedralAtoms)
