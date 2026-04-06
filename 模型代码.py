# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 23:41:20 2025

@author: CD
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
import torch_geometric
import random
from aohe import Group_feature
from sklearn.preprocessing import StandardScaler

# ============================
# 1. 3D坐标版分子图预处理
# ============================
class MolecularGraphDataset(Dataset):
    def __init__(self, smiles_list, labels):
        # 双重过滤：有效SMILES + 可生成3D坐标
        self.valid_smiles = []
        self.labels = []
        # self.indices = []  # 存储每个数据点的原始索引
        for smile, label in zip(smiles_list, labels):
            mol = self._get_3d_molecule(str(smile).strip())
            if mol is not None:
                self.valid_smiles.append(smile)
                self.labels.append(label)
        # self.indices.append(index)  # 添加索引
        # 原子特征编码
        self.atom_features = {
            'C': [1,0,0,0,0], 'H': [0,1,0,0,0], 
            'O': [0,0,1,0,0], 'N': [0,0,0,1,0],
            'S': [0,0,0,0,1]
        }
        
        # 化学键特征编码
        self.bond_features = {
            Chem.rdchem.BondType.SINGLE: [1,0,0],
            Chem.rdchem.BondType.DOUBLE: [0,1,0],
            Chem.rdchem.BondType.AROMATIC: [0,0,1]
        }

    def _get_3d_molecule(self, smile):
        """生成带3D坐标的分子对象，失败返回None"""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:  # 返回0表示成功
            return None
        
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        # 获取已验证的3D分子
        mol = self._get_3d_molecule(self.valid_smiles[idx])
        
        # 原子特征（3D坐标）
        node_feats = []
        for atom in mol.GetAtoms():
            # 原子类型特征（5维）
            feat = self.atom_features.get(atom.GetSymbol(), [0]*5)[:5]
            # 化学属性（4维）
            feat += [
                atom.GetFormalCharge()/2.0,    # 归一化电荷
                atom.GetDegree()/4.0,          # 归一化连接度
                int(atom.GetIsAromatic()),     # 芳香性
                atom.GetHybridization().real/3.0  # 杂化状态
            ]
            # 3D坐标（3维）
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            feat += [pos.x, pos.y, pos.z]
            node_feats.append(feat)  # 总维度：5+4+3=12
        
        # 化学键处理
        edge_indices = []
        edge_feats = []
        num_atoms = mol.GetNumAtoms()
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_features.get(bond.GetBondType(), [0,0,0])
            # 无向图双向连接
            edge_indices.extend([[i,j], [j,i]])
            edge_feats.extend([bond_type, bond_type])
            
        self_loops = [[i, i] for i in range(num_atoms)]
        edge_indices.extend(self_loops)
        edge_feats.extend([[0, 0, 0]] * num_atoms)
        
        return Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
            y=torch.tensor([self.labels[idx]], dtype=torch.float32)
            # index=torch.tensor(self.indices, dtype=torch.long) # 将原始索引传递到 Data 对象
        )

# ========================
# 2. GAT模型
# ========================
class ChemGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, heads=4, fea_num=454):
        super().__init__()
        # 边特征编码
        self.edge_encoder = torch.nn.Linear(edge_dim, node_dim)
        
        # self.s = StandardScaler()
        # GAT层堆叠
        self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, edge_dim=node_dim)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1, edge_dim=node_dim)
        
        # 回归头
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + fea_num, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
    
        
    def forward(self, data, data_fea_index, pep_features_all):
        
        features_wh = pep_features_all[data_fea_index].float()
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        edge_emb = self.edge_encoder(edge_attr)
        
        x = self.conv1(x, edge_index, edge_emb)
        
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_emb)
    
        x = global_mean_pool(x, data.batch)
        
        x = torch.cat((x,features_wh),dim=1)
        
        # x = self.s.fit_transform(x)
        
        return F.sigmoid(self.regressor(x))

# ========================
# 3. 训练与测试
# ========================

def train_test_data(labels, pep_all, data_input):
    smiles = []
    data = pep_all[data_input]
    label = labels[data_input]
    for index_k, pep in enumerate(data):
        smile = Chem.MolToSmiles(Chem.MolFromFASTA(pep[0]))
        smiles.append(smile)
    return smiles, label

def cal_acc(pred, labels):
    pred = pred.view(-1,)
    preds = (pred > 0.5).float()
    correct = (preds == labels).sum().item()
    return correct

if __name__ == "__main__":
    epochs = 100
    
    # 数据加载
    data_positive = np.array(pd.read_csv(r'C:\Users\CD\Desktop\实验数据\实验数据\实验数据\螯合肽\positive.csv'))[:,0]
    data_negative = np.array(pd.read_csv(r'C:\Users\CD\Desktop\实验数据\实验数据\实验数据\螯合肽\250.csv'))[:,1]

    #训练数据库
    order_index = torch.arange(0, 500, 1).reshape(-1,1)
    labels = torch.cat((torch.ones(250,), torch.zeros(250,)))
    pep_all = np.concatenate((data_positive, data_negative))
    X_train, X_test, y_train, y_test = train_test_split(order_index, labels, test_size=0.2, random_state=42)
    d_l = DataLoader(X_train, batch_size=1, shuffle=False, drop_last=True)
    dataset = []
    num_data_index = []
    for i in d_l:
        num_data_index.append(i)
        a = train_test_data(labels, pep_all, i)
        data_mo = MolecularGraphDataset(a[0], a[1])
        dataset.append(data_mo[0])
    # print(len(dataset))
    
    ###螯合肽物化特征池
    g_f_ne = Group_feature(data=data_negative).Group_feature()
    g_f_po = Group_feature(data=data_positive).Group_feature()
    pep_all_features = torch.cat((g_f_po, g_f_ne),dim=0)[num_data_index]
    
    ##数据加载数据池
    data_index = torch.arange(0, pep_all_features.shape[0], 1)
    dataloader = DataLoader(data_index, batch_size=16, shuffle=True, drop_last=True)

    # train_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=16, shuffle=True)
    
    #测试数据池
    
    t_l = DataLoader(X_test, batch_size=1, shuffle=False, drop_last=True)
    val_dataset = []
    val_num_data_index = []
    for i in t_l:
        val_num_data_index.append(i)
        a = train_test_data(labels, pep_all, i)
        data_mo = MolecularGraphDataset(a[0], a[1])
        val_dataset.append(data_mo[0])
    val_pep_all_features = torch.cat((g_f_po, g_f_ne),dim=0)[val_num_data_index]
    
    val_data_index = torch.arange(0, val_pep_all_features.shape[0], 1)
    # val_dataloader = DataLoader(val_data_index,bat)
    # val_data = train_test_data(labels, pep_all, X_test)
    
    # 模型定义
    model = ChemGAT(node_dim=len(dataset[0].x[0]), edge_dim=len(dataset[0].edge_attr[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss(reduction='sum')
    
    ####epoch_file
    epoch_loss = r'D:\aohe\epoch_loss.csv'
    epoch_acc = r'D:\aohe\epoch_acc.csv'
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_num_train = 0
        for data_num in dataloader:
            a_dataset = [dataset[i] for i in data_num]
            train_loader = torch_geometric.loader.DataLoader(a_dataset, batch_size=16, shuffle=False)
            
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data, data_num, pep_all_features).view(-1,)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_num_train += data.y.shape[0]
        mean_loss_train = total_loss / total_num_train
        # 评估模型
        model.eval()
        # total_correct = 0
        # total_num = 0
        with torch.no_grad():
            test_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            for data_eval in test_dataloader:
                out = model(data_eval,data_index,pep_all_features)
                correct_num = cal_acc(out, data_eval.y)
            train_acc = correct_num / data_eval.y.shape[0] * 100
            # eval_dataloader = torch_geometric.loader.DataLoader
            
            ###val
            val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
            for val_data in val_dataloader:
                out = model(val_data,val_data_index,val_pep_all_features)
                correct_num = cal_acc(out, val_data.y)
            val_acc = correct_num / val_data.y.shape[0] * 100
            
        print(f'EPOCH: {epoch + 1} | loss: {mean_loss_train:.4f} | train_accuracy: {train_acc:.4f}% | test_accuracy: {val_acc:.4f}%')
        
        new_loss_data = pd.DataFrame({'Epoch': [epoch + 1], 'Loss': [mean_loss_train]})
        if (epoch + 1) == 1:
            new_loss_data.to_csv(epoch_loss, mode='w',  header=True, index=False)
        else:
            new_loss_data.to_csv(epoch_loss, mode='a', header=False, index=False)
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            model_save_path = rf'D:\aohe\model\model{epoch+1}.pth'
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch_loss':mean_loss_train
                },model_save_path)
            
        new_acc_data = pd.DataFrame({'Epoch': [epoch + 1], 'Train_acc': [f'{train_acc:.2f}%'], 'Test_acc': [f'{val_acc:.2f}%']})
        if epoch == 0:
            new_acc_data.to_csv(epoch_acc, mode='w',  header=True, index=False)
        elif (epoch + 1) % 100 == 0:
            new_acc_data.to_csv(epoch_acc, mode='a', header=False, index=False)


            
