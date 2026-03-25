# generative

Repository for generative molecular modelling.

The current focus of this repository is on Michael acceptors from the ZINC-20 dataset.  

This repository contains a dataset of 242,455 Michael acceptor (MA) compounds taken from the ZINC-20 dataset. 

These compounds have been subsequently optimized to local minima in potential energy at the PM6 level of theory using Gaussian 16, and have been labelled with three properties:  

1. `mulliken_O4` (the Mulliken charge of the MA carbonyl oxygen)
2. `pbv_C1` (the Morfeus percent buried volume of the MA beta carbon)
3. `lumo` (the overall LUMO energy of the molecule, units: eV)
4. `vibfreq` (the lowest vibrational frequency of the molecule, units: cm⁻¹)

## LSTM
- small LSTM from from [dlchem101](https://github.com/rociomer/dl-chem-101/tree/main/03_gen_SMILES_LSTM)  

## VAE
- vanilla VAE from [Akshay Subramania](https://github.com/aksub99/molecular-vae/blob/master/Molecular_VAE.ipynb)  

##
See also the [chemical_vae](https://github.com/the-grayson-group/chemical_vae) repository on property-dependent (conditional) generation using a two-head VAE.  
