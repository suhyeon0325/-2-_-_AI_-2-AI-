import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 데이터 로드
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 복셀 크기, 그리드 차원 설정
VOXEL_SIZE = 1.0  
GRID_DIM = 32 

# SMILES를 3차원 복셀 그리드로 변환하는 함수
def smiles_to_voxel(smiles, grid_dim=GRID_DIM, voxel_size=VOXEL_SIZE):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 수소 원자 추가하고 3D 좌표 생성한 뒤 에너지 최적화
        mol = Chem.AddHs(mol) 
        AllChem.EmbedMolecule(mol)  
        AllChem.MMFFOptimizeMolecule(mol)  
        
        # 최적화된 분자의 원자 좌표 추출
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        
        # 복셀 그리드 생성
        voxel_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        grid_center = grid_dim // 2
        
        # 각 원자 좌표를 복셀 그리드에 매핑함
        for coord in coords:
            x, y, z = coord / voxel_size + grid_center
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            
            if 0 <= x < grid_dim and 0 <= y < grid_dim and 0 <= z < grid_dim:
                voxel_grid[x, y, z] = 1.0  
        
        return voxel_grid
    
    except Exception as e:
        return None
    
# train에 복셀 추가
train['Voxel_Data'] = train['Smiles'].apply(smiles_to_voxel)
train = train.dropna(subset=['Voxel_Data']).reset_index(drop=True)

# test에 복셀 추가
test['Voxel_Data'] = test['Smiles'].apply(smiles_to_voxel)
test = test.dropna(subset=['Voxel_Data']).reset_index(drop=True)

# npy 파일로 저장
np.save('data/train_voxel.npy', np.array(train['Voxel_Data'].tolist()))
np.save('data/test_voxel.npy', np.array(test['Voxel_Data'].tolist()))