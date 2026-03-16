#%%

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
import io
import requests
from huggingface_hub import hf_hub_url
from pandarallel import pandarallel

#%%

def get_ZINC(chunk="00"):
    # Choose the file to download
    repo_id="zpn/zinc20"
    filename = f"zinc_processed/smiles_all_{chunk}_clean.jsonl.gz"
    # Get the direct download URL from Hugging Face
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(url)
    # Stream the file directly into memory
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_json(io.BytesIO(response.content), compression="gzip", lines=True)
    return df

#%%

# Choose a subset of 1B ZINC20 molecules to load
# each subset is 1/100th, so approx. 10M molecules

chunk = "00"
ncpu = 10
# Takes ~ 1 minute to download, unzip and load (approx. 10M molecules)
df = get_ZINC(chunk="00")

#%%

smarts = "O=CC=C"
patt = Chem.MolFromSmarts(smarts)

# Takes ~ 3 minutes for full dataset
# (approx. 3 minutes per 10_000_000 compounds)
pandarallel.initialize(nb_workers=ncpu, progress_bar=True)
df["HasMatch"] = df["smiles"].parallel_apply(lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(patt))

#%%

Michael_df = df[df["HasMatch"]==True]
del df # clear large df from memory
print(Michael_df.shape)
Michael_df.head()

#%%

filter_smarts = ["O=C([OH])C=C","O=C([O-])C=C",
                 "O=C(F)","O=C(Cl)","O=C(Br)","O=C(I)",
                 "O=CC([OH])=C","O=CC=C[OH]",
                 "O=CC([NH2])=C","O=CC=C[NH2]",
                 "O=CC=CC=O"]
filter_mols = [Chem.MolFromSmarts(fs) for fs in filter_smarts]
# Can be relatively slow, ~ 1 minute
PandasTools.AddMoleculeColumnToFrame(Michael_df, smilesCol='smiles')
tqdm.pandas()
Michael_df["Filter"] = Michael_df["ROMol"].progress_apply(lambda x: any(x.HasSubstructMatch(patt) for patt in filter_mols))
Draw.MolsToGridImage(mols=filter_mols, legends=filter_smarts, molsPerRow=4)

#%%

Michael_df = Michael_df[Michael_df["Filter"]==False]
print(Michael_df.shape)
Michael_df.drop(["HasMatch","ROMol","Filter"], axis=1, inplace=True)
Michael_df.to_csv(f"ZINC_{chunk}_Michael.csv.gz",index=False,compression='gzip')
Michael_df.head()

#%%

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
Michael_df = pd.read_csv(f"ZINC_{chunk}_Michael.csv.gz")

m = 20
mols = [Chem.MolFromSmiles(smi) for smi in Michael_df["smiles"].values[:m]]
ids = Michael_df["id"].values[:m]
Draw.MolsToGridImage(mols=mols, legends=ids.tolist(), molsPerRow=4, useSVG=True)

#%%

# Train-validation-test split

import numpy as np
np.random.seed(42)

n_train = len(Michael_df) - 50_000
n_valid, n_test = 25_000, 25_000
smi_id = Michael_df[["smiles","id"]].values
np.random.shuffle(smi_id)
train = smi_id[:n_train].astype(str)
valid = smi_id[n_train:n_train+n_valid].astype(str)
test = smi_id[n_train+n_valid:].astype(str)

header = np.array([["SMILES","Name"]])
np.savetxt("train.smi", np.concatenate((header, train)), 
    delimiter=" ", fmt="%s")
np.savetxt("valid.smi", np.concatenate((header, valid)), 
           delimiter=" ", fmt="%s")
np.savetxt("test.smi", np.concatenate((header, test)), 
           delimiter=" ", fmt="%s")

#%%