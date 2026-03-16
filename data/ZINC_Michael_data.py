import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
import io, os
import requests
from huggingface_hub import hf_hub_url
from pandarallel import pandarallel

def get_ZINC(chunk="00"):
    # Download, unzip and load ZINC data from zpn/zinc20 (approx. 10M molecules)
    # Choose the file to download
    repo_id="zpn/zinc20"
    filename = f"zinc_processed/smiles_all_{chunk}_clean.jsonl.gz"
    # Get the direct download URL from Hugging Face
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    # print(url)
    # Stream the file directly into memory
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_json(io.BytesIO(response.content), compression="gzip", lines=True)
    return df

smarts = "O=CC=C"
patt = Chem.MolFromSmarts(smarts)

filter_smarts = ["O=C([OH])C=C","O=C([O-])C=C",
                 "O=C(F)","O=C(Cl)","O=C(Br)","O=C(I)",
                 "O=CC([OH])=C","O=CC=C[OH]",
                 "O=CC([NH2])=C","O=CC=C[NH2]",
                 "O=CC=CC=O"]
filters = [Chem.MolFromSmarts(fs) for fs in filter_smarts]

def has_match(smiles, smarts="O=CC=C"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        patt = Chem.MolFromSmarts(smarts)
        return mol.HasSubstructMatch(patt)
    except Exception:
        return False

def MatchMichael(df, patt=patt):
    # (Parallel) match Michael acceptor substructure (~ 2 mins)
    df.loc[:, "HasMatch"] = df["smiles"].parallel_apply(has_match)
    filtered_df = df[df["HasMatch"]==True]
    filtered_df = filtered_df.drop("HasMatch", axis=1)
    return filtered_df

def FilterSubstruct(df, filters=filters):
    # Filter unwanted (reactive) substructures (~ 1 min)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
    df.loc[:, "Filter"] = df["ROMol"].apply(lambda x: any(x.HasSubstructMatch(patt) for patt in filters))
    filtered_df = df[df["Filter"]==False]
    filtered_df = filtered_df.drop(["ROMol","Filter"], axis=1)
    return filtered_df

# Choose a subset of the 1Bn ZINC20 molecules to load
# each subset is 1/100th, so approx. 10Mn molecules

ncpu = 10
chunks = [f'{n:02}' for n in range(100)]
pandarallel.initialize(nb_workers=ncpu, progress_bar=False, verbose=0)
for chunk in tqdm(chunks[:]): # if this crashes, continue using list slices, e.g. chunks[50:]
    df = MatchMichael(get_ZINC(chunk=chunk))
    fdf = FilterSubstruct(df)
    outfile = os.path.join("ZINC","tranches",f"ZINC_{chunk}_Michael.csv.gz")
    fdf.to_csv(outfile,index=False,compression='gzip')
