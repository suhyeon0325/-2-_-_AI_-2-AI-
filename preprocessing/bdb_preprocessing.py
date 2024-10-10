import pandas as pd

columns_to_load = [
    'Ligand SMILES', 'Ligand InChI', 'Target Name', 'IC50 (nM)', 'pH', 'Temp (C)',
    'Curation/DataSource', 'Patent Number', 'Authors', 'Institution'
]
data = pd.read_csv('/data/BindingDB_All.tsv', sep='\t', usecols=columns_to_load)

filtered_data = (
    data.dropna(subset=['IC50 (nM)', 'Ligand InChI'])
    .query("`Target Name` == 'Interleukin-1 receptor-associated kinase 4' and `Curation/DataSource` != 'ChEMBL'")
    .loc[~data['IC50 (nM)'].str.contains('[<>]', regex=True)]
)

filtered_data['IC50 (nM)'] = pd.to_numeric(filtered_data['IC50 (nM)'], errors='coerce').round(2)

filtered_data = filtered_data.drop_duplicates(subset=['Ligand SMILES', 'IC50 (nM)'])
duplicated_data = filtered_data[filtered_data['Ligand SMILES'].duplicated(keep=False)]
non_duplicated_data = filtered_data[~filtered_data['Ligand SMILES'].isin(duplicated_data['Ligand SMILES'])]

sorted_duplicated_data = duplicated_data.sort_values(by='Ligand SMILES')
deduplicated_data = sorted_duplicated_data.drop_duplicates(
    subset=['Ligand SMILES', 'pH', 'Temp (C)', 'Curation/DataSource', 'Patent Number', 'Authors', 'Institution']
)
ligand_inchi_counts = deduplicated_data['Ligand SMILES'].value_counts()

dup_ligand_data = deduplicated_data[deduplicated_data['Ligand SMILES'].isin(ligand_inchi_counts[ligand_inchi_counts > 1].index)]
unique_ligand_data = deduplicated_data[deduplicated_data['Ligand SMILES'].isin(ligand_inchi_counts[ligand_inchi_counts == 1].index)]

def process_group(group):
    ic50_diff = group['IC50 (nM)'].max() - group['IC50 (nM)'].min()
    if ic50_diff <= 1.0:
        mean_ic50 = group['IC50 (nM)'].mean()
        most_common_values = group.mode().iloc[0]
        result_row = most_common_values.copy()
        result_row['IC50 (nM)'] = mean_ic50
        return result_row.to_frame().T
    else:
        return None

processed_data = dup_ligand_data.groupby('Ligand SMILES').apply(process_group).dropna(how='all').reset_index(drop=True)

combined_data = pd.concat([non_duplicated_data, unique_ligand_data, processed_data], ignore_index=True)

final_data = combined_data[['Ligand SMILES', 'IC50 (nM)']]
final_data = final_data[final_data['IC50 (nM)'] != 1000000.0].sort_values(by='IC50 (nM)')

final_data.to_csv("data/bindingdb_data.csv", index=False)