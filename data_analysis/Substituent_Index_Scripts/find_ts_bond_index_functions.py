import operator
import cclib
import numpy as np 
import sys
import warnings
from scipy.spatial.distance import cdist

# sys.path.insert(0, "X:/ML_and_Feature_Codes/structure_reading_scripts")
# import distance, angle, dihedral
# sys.path.remove("X:/ML_and_Feature_Codes/structure_reading_scripts")

# Functions for finding the atom indices joined by a TS bond in a structure
# Functions taken from E. Farrar with minor adaptations

##################################################################
def element_mapping():
    """
    Convert atom numbers to atom symbols (or vice versa).
    """
    mapping = {
        "C": 6,
        "H": 1,
        "B": 5,
        "Br": 35,
        "Cl": 17,
        "D": 0,
        "F": 9,
        "I": 53,
        "N": 7,
        "O": 8,
        "P": 15,
        "S": 16,
        "Se": 34,
        "Si": 14,
    }
    reverse_mapping = dict([reversed(pair) for pair in mapping.items()])
    mapping.update(reverse_mapping)
    return mapping

class MolML():
    """
    Functions and dictionaries borrowed or inspired by the molml package.
    Extras added from: https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    """
    # Specifies the order in which to check for bond types.
    TYPE_ORDER = ['1', 'Ar', '2', '3', 'H']
    # Specifies the covalent radii to check for bond types.
    COV_RADII = {'C': {'3': 0.62, '2': 0.69, 'Ar': 0.72, '1': 0.85},
                'Cl': {'1': 1.045},
                'F': {'1': 1.23},
                'H': {'1': 0.6},
                'N': {'3': 0.565, '2': 0.63, 'Ar': 0.655, '1': 0.74},
                'O': {'3': 0.53, '2': 0.59, 'Ar': 0.62, '1': 0.695},
                'P': {'2': 0.945, 'Ar': 0.985, '1': 1.11},
                'S': {'2': 0.905, 'Ar': 0.945, '1': 1.07},
                'B': {'1': 0.843},
                'Br': {'1': 1.14},
                'I': {'1': 1.33},
                }
    # Specifies the VDW radii to check for hydrogen bonding.
    VDW_RADII = {#'C': {'H': 1.7},
                #'Cl': {'H': 1.75},
                #'F': {'H': 1.47},
                'H': {'H': 1.2},
                'N': {'H': 1.55},
                'O': {'H': 1.52},
                #'P': {'H': 1.8},
                #'S': {'H': 1.8},
                #'Br': {'H': 1.85},
                #'I': {'H': 198},
                }

    def get_bond_type(element1, element2, dist, radii="covalent"):
        """
        For a given pair of elememts with some distance between them, determines the 
        bond type between those elements based off covalent radii in COV_RADII or the
        Van der Waals radii in VDW_RADII (for hydrogen bonding).
        """
        if radii == "covalent":
            radii = MolML.COV_RADII
            bad_eles = [x for x in (element1, element2) if x not in radii]
            if len(bad_eles):
                msg = "The following elements are not in the list of radii: %s" % bad_eles
                warnings.warn(msg)
                return
        elif radii == "vdw":
            radii = MolML.VDW_RADII
        else:
            msg = "No radii specified for bond identification."
            warnings.warn(msg)
            return
        
        for key in MolML.TYPE_ORDER[::-1]:
            try:
                cutoff = radii[element1][key] + radii[element2][key]
                if dist < cutoff:
                    # convert aryl bonds to weight
                    if key == "Ar":
                        return 1.5
                    # convert hydrogen bonds to weight
                    elif key == "H":
                        # hydrogen bonds must include one hydrogen
                        if (element1 == 'H' and element2 != 'H') or (element2 == 'H' and element1 != 'H'):
                            return 0.35
                        else:
                            return 0
                    else:
                        return key
            except KeyError:
                continue
    
    def get_connections(elements1, coords1, radii="covalent"):
        """
        For a given set of coordinates and a given atom calculates all bonds to that atom.
        By default uses covalent radii to detect bonding, but can be changed to VDW radii
        to detect hydrogen bonding.
        """
        dist_mat = cdist(coords1, coords1)
        connections = {i: {} for i in range(len(elements1))}
        for i, element1 in enumerate(elements1):
            for j, element2 in enumerate(elements1):
                if i >= j:
                    continue
                dist = dist_mat[i, j]
                bond_type = MolML.get_bond_type(element1, element2, dist, radii=radii)
                if not bond_type:
                    continue
                connections[i][j] = bond_type
                connections[j][i] = bond_type

        return connections

class Connectivity():
    """
    Class for generating connectivity matrices.
    """

    def gen_connectivity_matrix(nums, coords, radii="covalent"):
        """
        Generates a connectivity matrix for a given set of atomic numbers and coordinates.
        
        :param nums (array): array of atomic numbers.
        :param coords (array): array of atomic coordinates.
        :returns bonds (array): an array of bond connections (keys) and weights (values).
        """
        symbs = [element_mapping()[num] for num in nums]
        cm = MolML.get_connections(symbs, coords, radii=radii)
        
        # convert to numpy array without repeats
        done, bonds = [], []
        for key, values in cm.items():
            for atom, weight in values.items():
                if (key, atom) not in done and (atom, key) not in done:
                    bonds.append([key, atom, float(weight)])
                done.append((key, atom))

        return np.array(bonds, dtype='float32')

def _convert_cm(cm):
    """
    # convert connectivity matrix back to dictionary
    # we use this later for quick connectivities
    """
    dic_cm = {}
    for atom in np.unique(np.concatenate([np.unique(cm[:, 0]), np.unique(cm[:, 1])])):
        atom_dic = {}
        for bond in cm:
            if bond[0] == atom:
                atom_dic.update({int(bond[1]) : bond[2]})
            if bond[1] == atom:
                atom_dic.update({int(bond[0]) : bond[2]})
        dic_cm.update({int(atom) : atom_dic})
    return dic_cm

def _add_ts_bonds_with_cm(cm, data, factor=1, mode=3):
    """
    Identify TS bonds and return the corresponding atomic indices
    
    :param cm (array): a connectivity matrix (without ts bonds)
    :param data (cclib object): a cclib data object.
    :factor (float): factor by which to scale the displacement matrix
    :mode (bool): what kind of TS bonds to look for.
        - mode 1: looks for changes in bond order only.
        - mode 2: looks for single bond forming or breaking only.
        - mode 3: looks for both kinds of TS bonds.
    :returns cm (array): connectivity matrix with added ts bonds 
    :returns n (int): the number of ts bonds.
    :returns ts_list (list): list of atomic indices for TS bonds
    """
    ts_list = []
    
    # in this method we displace the coordinates towards the reactant 
    # and product and compare the connectivity matrices for each
    # there are two possible situations:
    # 1. if a bond changes order from R to P we put the average bond
    #    order in the TS CM, e.g., double to triple would be 2.5
    # 2. if a bond is broken or created form R to P we assume the
    #    bond is a single bond and set the TS bond order to 0.5
    #    We do this because the bond order changes depending on
    #    the frequency, e.g., for the MA dataset sometimes the 
    #    bond-forming interaction is 0-1, 0-1.5, 0-2, or 0-3.
    #    We tried normalizing the frequencies by multiplying
    #    the displacement vector by a factor depending on the
    #    average negative frequency, however this did not help.
    # One possibility is to provide a list of weights in the CM
    # rather than just one weight where each weight corresponds 
    # to reactant, TS, product, e.g., [R, TS, P]. For bonds that
    # don't change these would all be the same value, but for TSs
    # this would help encode changes in bond order. However, the
    # direction of the frequency is not always consistent, e.g., 
    # for the MA dataset, sometimes the positive displacement is
    # the product and negative reactant and other times vice versa.
    # We could provide every structure twice in training, once 
    # encoded each way, however this is probably not worth it.
    # Note, some structures won't register a TS bond via this 
    # method, usually because the magnitude of the frequencies
    # are too low and thus no bond is registered in the displaced
    # R and P. These structures can be avoided by setting an
    # appropriate frequency threshold to remove structures.
    nums = data.atomnos
    coords = data.atomcoords[-1]

    # identify displacement vector corresponding to negative freq
    for vib in data.vibfreqs:
        if vib < 0:
            position = data.vibfreqs.tolist().index(vib)
    disp_vect = data.vibdisps[position] * factor # optional scaling factor

    # displace coordinates
    c1 = coords + disp_vect # R or P
    c2 = coords - disp_vect # R or P

    # calculate connectivities for displaced coordinates
    cm1 = Connectivity.gen_connectivity_matrix(nums, c1)
    cm2 = Connectivity.gen_connectivity_matrix(nums, c2)

    # generate one-key dictionaries
    dic_cm1 = _convert_cm(cm1)
    dic_cm2 = _convert_cm(cm2)

    # convert to two-key dictionaries
    cm0 = {f"{int(i[0])}_{int(i[1])}":i[-1] for i in cm}
    cm1 = {f"{int(i[0])}_{int(i[1])}":i[-1] for i in cm1}
    cm2 = {f"{int(i[0])}_{int(i[1])}":i[-1] for i in cm2}

    # compare connectivities of displaced coordinates
    banned = [17, 35] # list of atoms to ignore for TS bonds
    combine_keys = set(list(cm1.keys()) + list(cm2.keys()))
    for key in combine_keys:
        s = key.split("_")
        atom1, atom2 = int(s[0]), int(s[1]) # atom numbers
        sym1, sym2 = nums[atom1], nums[atom2] # atom symbols

        # 1. if atom pair in both coordinates (bond order changing)
        if mode == 1 or mode == 3:
            if key in cm1.keys() and key in cm2.keys():
                # check weights are different (bond is changing)
                if cm1[key] == cm2[key]:
                    continue
                # remove TS interactions between hydrogens
                # H-H interactions can only occur with H-H reduction and formation
                # this situation will be picked up by situation 2 (single bond forming/breaking)
                if sym1 == sym2 == 1:
                    continue
                # remove TS interactions involving Cl or Br
                # remove this if needed, e.g., for SN2 with Cl or Br as nucleophiles
                if sym1 in banned or sym2 in banned:
                    continue
                # set bond order to average of R/P bond orders
                w = (int(cm1[key]) + int(cm2[key]))/2
                cm0.update({key:w})
                ts_list.append(key)

        # 2. if atom pair unique to one set of coordinates (bond forming or breaking)
        if mode == 2 or mode == 3:
            if (key in cm1.keys() and key not in cm2.keys()) or (key in cm2.keys() and key not in cm1.keys()):
                # only count TS bonds for H-H where the atoms are only connected to one another
                # this includes H-H reduction and formation, but excludes erroneous H-H TS interactions
                # H-H bonds can only be single bonds, so we only need to perform this check for situation 2
                if sym1 == sym2 == 1:
                    # define the one-key cm for the displacement with the bond
                    if key in cm1.keys():
                        dic_cm = dic_cm1
                    if key in cm2.keys():
                        dic_cm = dic_cm2
                    # find directly connected atoms
                    connected1, connected2 = [], []
                    if atom1 in dic_cm:
                        connected1 = list(dic_cm[atom1].keys())
                    if atom2 in dic_cm:
                        connected2 = list(dic_cm[atom2].keys())
                    # if hydrogens are connected only to each other a H-H TS bond correct
                    if not (atom1 == connected2[0] and atom2 == connected1[0]):
                        continue
                # remove Cl/Br edge cases (as above)
                if (sym1 in banned or sym2 in banned):
                    continue
                # set bond order to 0.5 (single bond forming/breaking)
                cm0.update({key:0.5}) 
                ts_list.append(key)
                
    # convert back to array
    bonds = []
    for k, v in cm0.items():
        s = [int(float(i)) for i in k.split("_")]
        s.extend([v])
        bonds.append(s)
    cm = np.array(bonds)

    # get the number of TS bonds (check for 0.5 in last column)
    n = len([i for i in cm[:, 2] if i == 0.5])
    # For the TS atomic index list, need to add 1 to each index so the values correspond to atom indices in each Gaussian (starting at 1), rather than the connectivity matrix indices (starting at 0).
    ts_list_converted = []
    for i in ts_list:
        idx_a, idx_b = int(i.split("_")[0])+1, int(i.split("_")[1])+1
        new_key = "-".join([str(idx_a),str(idx_b)])
        ts_list_converted.append(new_key)
    ts_list = ts_list_converted
    return cm, n, ts_list
    
def extract_ts_indices(paths,file_strip_func=None,num_ts_bonds="= 1",mode=3):
    """
    Extract indices of atoms in ts bonds for structures in a list of paths
    
    :param paths (list): a list of file paths containing TS structures.
    :param strip_file_func (callable): a function to use to shorten the file paths to create the keys to use in the output dictionary. If None, the full file path will be used.
    :param num_ts_bonds (str): number of TS bonds to look for in each structure. Default "= 1".
    :param mode (int): the type of TS bond to look for. 1 = changes in bond order. 2= single bond forming/breaking. 3 = both.
    :returns ts_index_dict (dict): a dictionary giving the atomic indices of atoms joined by a TS bond for each structure. Dictionary keys are the shortened file paths produced by calling the supplied function strip_file_func.
    """ 
    # Possible operators to use when checking if number of TS bonds founds matches required condition specified by num_ts_bonds input.
    ops = { ">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "=": operator.eq}
    
    ts_index_dict = {}
    for p in paths:
        data = cclib.io.ccread(p)
        nums = data.atomnos
        coords = data.atomcoords[-1]

        # calculate connectivity matrix with covalent radii
        cm = Connectivity.gen_connectivity_matrix(nums, coords, radii="covalent")
        # weights values are based on the bond type: 
        # none (0), single (1), aromatic (1.5), double (2), triple (3)
        # other weights are defined later for TS bonds and hydrogen bonds

        # Calculate bond breakings/formings.
        # there are several possible methods for this but we will use the 
        # connectivity matrix method. Furthermore, if no TS bond-forming
        # or bond-breaking is detected from the first pass, further attempts
        # will be made by scaling up the displacement vectors until a TS
        # bond is identified. Some structures will fail and be removed.
        if data.vibfreqs[0] < 0:
            
            # find changes in bond order using a scaling factor of 1
            # this finds more subtle changes in bond order, e.g., double to single
            # this is less sensitive, less error prone, and less important overall
            # thus we perform this with no displacement scaling (factor = 1)
            if mode == 1 or mode == 3:
                cm, n, ts_list = _add_ts_bonds_with_cm(cm, data, factor=1, mode=1)
            else:
                ts_list = []   
            # find single bond forming/breakings using a variable scaling factor
            # these are the most important and most difficult to identify
            # thus we perform a series of checks with varying scaling factors
            # these checks are based on the criteria in self.ts_bonds
            op, th = num_ts_bonds.split(" ")
            # we change equals to equals to or greater than
            # this accounts for times when we find too many bonds
            if op == "=":
                new_op = ">="
            else:
                new_op = op
            
            # we start by reducing the displacement vector to prevent
            # detection of false positives, such as C-Cl bonds in NMA,
            # and make a first pass through the bond finder function
            f, step = 0.45, 0.2
            if mode == 2 or mode == 3:
                cm, n, ts_list_new = _add_ts_bonds_with_cm(cm, data, factor=f, mode=2)
                if mode == 3: #combine lists
                    ts_list = list(set(ts_list).union(set(ts_list_new)))
                elif mode == 2:
                    ts_list = ts_list_new
                
                # if the number of TS bonds doesn't fulfil our criteria from 
                # num_ts_bonds, then we gradually increase f until we find more
                if not ops[new_op](int(n), int(th)):
                    # we set a maximum displacement factor of 3 because if scaled 
                    # enough every bond will eventually be identified as a TS bond
                    for i in np.arange(f+step, 3, step):
                        if not ops[new_op](int(n), int(th)):
                            cm, n, ts_list_new = _add_ts_bonds_with_cm(cm, data, factor=i, mode=2)
                            ts_list = list(set(ts_list).union(set(ts_list_new))) # combine ts index lists
                        # if we meet the TS bond criteria, we can stop finding bonds
                        else:
                            break

            # if we still don't fulfil the criteria we remove that structure
            if not ops[new_op](int(n), int(th)):
                print(f"WARNING: insufficient TS interactions were identified for: {p}. Corresponding TS will be omitted from the output.")
            else:
                path_key = file_strip_func(p)
                ts_index_dict.update({path_key:ts_list})
        
    return ts_index_dict
