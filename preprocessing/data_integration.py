import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

df1 = pd.read_csv("data/train_1952.csv")
df2 = pd.read_csv("data/bindingdb_data.csv")

df2 = df2.rename(columns={'Ligand SMILES': 'Smiles', 'IC50 (nM)': 'IC50_nM'})

merged_df = pd.merge(df1, df2, on='Smiles', how='outer', suffixes=('', '_df2'))
merged_df['IC50_nM'] = merged_df['IC50_nM'].combine_first(merged_df['IC50_nM_df2'])

merged_df = merged_df.drop(columns=['IC50_nM_df2'])

merged_df.to_csv("data/train.csv", index=False)