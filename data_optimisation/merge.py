#%%


import numpy as np
import pandas as pd

#%%

df = pd.read_csv("ZINC_Michael_250k_500_100.csv.gz")

files = [os.path.join("tranches",f"ZINC_{i+1}.csv") for i in range(10)]

data = pd.concat([pd.read_csv(file) for file in files])

#%%

df = df.merge(data, on="id")

#%%

df = df.dropna(how="any",axis=0)
df.to_csv("ZINC_Michael_data.csv.gz", compression="gzip")

#%%
