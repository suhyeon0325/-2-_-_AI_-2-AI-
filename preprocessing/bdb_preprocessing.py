import pandas as pd

# 사용할 열만 로드
columns_to_load = [
    'Ligand SMILES', 'Ligand InChI', 'Target Name', 'IC50 (nM)', 'pH', 'Temp (C)',
    'Curation/DataSource', 'Patent Number', 'Authors', 'Institution'
]
data = pd.read_csv('/data/BindingDB_All.tsv', sep='\t', usecols=columns_to_load)

# IC50와 Ligand InChI가 없는 데이터를 제거
# IRAK4이고 출처가 ChEMBL아닌 데이터만 필터링
# IC50값에 부호가 포함되면 제거
filtered_data = data.dropna(subset=['IC50 (nM)', 'Ligand InChI']).query(
    "`Target Name` == 'Interleukin-1 receptor-associated kinase 4' and `Curation/DataSource` != 'ChEMBL'"
)
filtered_data = filtered_data[~filtered_data['IC50 (nM)'].str.contains('[<>]', regex=True)]

# 중복된 SMILES 제거
filtered_data['IC50 (nM)'] = pd.to_numeric(filtered_data['IC50 (nM)'], errors='coerce').round(2)
filtered_data = filtered_data.drop_duplicates(subset=['Ligand SMILES', 'IC50 (nM)'])

# 중복된 Ligand는 따로 분리
ligand_counts = filtered_data['Ligand SMILES'].value_counts()
duplicated_data = filtered_data[filtered_data['Ligand SMILES'].isin(ligand_counts[ligand_counts > 1].index)]
non_duplicated_data = filtered_data[filtered_data['Ligand SMILES'].isin(ligand_counts[ligand_counts == 1].index)]

# IC50값이 1보다 작으면 평균으로 통일
def process_group(group):
    ic50_range = group['IC50 (nM)'].max() - group['IC50 (nM)'].min()
    if ic50_range <= 1.0:
        mean_ic50 = group['IC50 (nM)'].mean()
        result_row = group.iloc[0].copy() 
        result_row['IC50 (nM)'] = mean_ic50 
        return result_row
    return None

# 중복 데이터 처리
processed_data = duplicated_data.groupby('Ligand SMILES').apply(process_group).dropna().reset_index(drop=True)

# 중복되지 않았던 데이터 + IC50 중복값 처리한 데이터
final_data = pd.concat([non_duplicated_data, processed_data], ignore_index=True)

# IC50 값과 Ligand SMILES만 남김
# IC50이 1000000인건 제거(이상치)
final_data = final_data[['Ligand SMILES', 'IC50 (nM)']]
final_data = final_data[final_data['IC50 (nM)'] != 1000000.0].sort_values(by='IC50 (nM)')

# 데이터 저장
final_data.to_csv("data/bindingdb_data.csv", index=False)