import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.nn import Sequential, Linear
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

# 랜덤 시드 설정
seed_value = 325
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 값이 해당 집합에 포함되어 있는지를 인코딩 하는 함수
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# 원자 특성 추출 함수
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding_unk(atom.GetDegree(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))) +
                    [atom.GetIsAromatic()])

# SMILES를 그래프로 변환하는 함수
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    features = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()], dtype=torch.long).t().contiguous()
    return Data(x=features, edge_index=edge_index)

# 데이터셋 생성
def create_dataset(smiles_list, labels=None):
    dataset = []
    for i, smile in enumerate(smiles_list):
        graph = smile_to_graph(smile)
        if labels is not None:
            graph.y = torch.tensor([labels[i]], dtype=torch.float)
        dataset.append(graph)
    return dataset

# 데이터 로드
train = pd.read_csv('data/train.csv')  
test = pd.read_csv('data/test.csv')
train['pIC50'] = -np.log10(train['IC50_nM']) + 9

# 데이터 분할
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)

# 데이터셋
train_dataset = create_dataset(train_data['Smiles'].values, train_data['pIC50'].values)
val_dataset = create_dataset(val_data['Smiles'].values, val_data['pIC50'].values)
test_dataset = create_dataset(test['Smiles'].values)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mish 활성화 함수 정의
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# GIN 모델 정의
class GIN_model(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, dim=96, output_dim=128, dropout=0.2):
        super(GIN_model, self).__init__()
        
        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

        nn1 = Sequential(Linear(num_features_xd, dim), Mish(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)
        
        nn2 = Sequential(Linear(dim, dim), Mish(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)
        self.fc1 = Linear(output_dim, 512)
        self.fc2 = Linear(512, 128)
        self.out = Linear(128, n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.mish(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.mish(self.conv2(x, edge_index))
        x = self.bn2(x)
        
        x = global_add_pool(x, batch)
        
        x = self.mish(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.mish(self.fc2(x))
        x = self.dropout(x)
        
        return self.out(x)
    
# 학습 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, verbose=True)
criterion = nn.MSELoss()

# 파라미터 설정
num_epochs = 1000
early_stopping_patience = 30
best_val_loss = float('inf')
no_improvement_count = 0
best_model_state = None

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    scheduler.step(avg_loss)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.y.view(-1, 1).float())
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_count = 0
        best_model_state = model.state_dict()
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stopping_patience:
        break

# 최적의 모델 저장 및 로드
torch.save(best_model_state, 'best_model.pth')
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 테스트 데이터 예측
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch)
        test_preds.extend(output.cpu().numpy())

# 예측 결과 저장
ic50_predictions = 10 ** (9 - test_preds.flatten())
submission = pd.DataFrame({'ID': test['ID'], 'IC50_nM': ic50_predictions})
submission = test[['ID', 'IC50_nM']]
submission.to_csv('submission_gin.csv', index=False)