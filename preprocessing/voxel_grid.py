import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

VOXEL_SIZE = 1.0  
GRID_DIM = 32 

def smiles_to_voxel(smiles, grid_dim=GRID_DIM, voxel_size=VOXEL_SIZE):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)  
        AllChem.EmbedMolecule(mol)  
        AllChem.MMFFOptimizeMolecule(mol)  
        
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        
        voxel_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        grid_center = grid_dim // 2
        
        for coord in coords:
            x, y, z = coord / voxel_size + grid_center
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            
            if 0 <= x < grid_dim and 0 <= y < grid_dim and 0 <= z < grid_dim:
                voxel_grid[x, y, z] = 1.0  
        
        return voxel_grid
    
    except Exception as e:
        return None
    
def add_voxel_data_to_dataframe(df):
    df['Voxel_Data'] = df['Smiles'].apply(smiles_to_voxel)
    df = df.dropna(subset=['Voxel_Data']).reset_index(drop=True)
    
    return df

train_voxel = add_voxel_data_to_dataframe(train)
test_voxel = add_voxel_data_to_dataframe(test)

np.save('data/train_voxel.npy', np.array(train_voxel['Voxel_Data'].tolist()))
np.save('data/test_voxel.npy', np.array(test_voxel['Voxel_Data'].tolist()))