import itertools
from collections import defaultdict
from operator import neg
import random
import math
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# # 读取数据
# df_ddi_train = pd.read_csv('/root/DSN-DDI/drugbank_test/inductive_data/fold3/train.csv')
# df_ddi_s1 = pd.read_csv('/root/DSN-DDI/drugbank_test/inductive_data/fold3/s1.csv')
# df_ddi_s2 = pd.read_csv('/root/DSN-DDI/drugbank_test/inductive_data/fold3/s2.csv')
# # 计算并打印每个数据集中的药物种类数量
# for df, prefix in zip([df_ddi_train, df_ddi_s1, df_ddi_s2], ['train', 's1', 's2']):
#     unique_drugs = pd.concat([df['d1'], df['d2']]).unique()
#     print(f'The {prefix} dataset has {len(unique_drugs)} unique drugs.')

# # 对每个数据集绘制柱状图并保存
# for df, prefix in zip([df_ddi_train, df_ddi_s1, df_ddi_s2], ['train', 's1', 's2']):
#     # 统计"type"的数量
#     type_counts = df['type'].value_counts()

#     # 创建柱状图
#     plt.figure(figsize=(10,6))
#     type_counts.plot(kind='bar')

#     # 设置图的标题和坐标轴的标签
#     plt.title(f'Distribution of Type for {prefix}')
#     plt.xlabel('Type')
#     plt.ylabel('Count')

#     # 获取最多和最少的类别及其数量
#     max_type = type_counts.idxmax()
#     max_count = type_counts.max()
#     min_type = type_counts.idxmin()
#     min_count = type_counts.min()

#     # 在图上添加标注
#     plt.text(max_type, max_count, f'max: {max_count}', ha='center', va='bottom')
#     plt.text(min_type, min_count, f'min: {min_count}', ha='center', va='top')

#     # 保存图像
#     plt.savefig(f'{prefix}_Type.png')
#     plt.show()
###########################################################################

df_drugs_smiles = pd.read_csv('drugbank_test/drugbank/drug_smiles.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

# print(f"DRUG_TO_INDX_DICT has {len(DRUG_TO_INDX_DICT)} entries.")
# print("First 5 entries in DRUG_TO_INDX_DICT:")
# for i, (key, value) in enumerate(DRUG_TO_INDX_DICT.items()):
#     print(f"{key}: {value}")
#     if i == 4:
#         break

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

# print(f"drug_id_mol_graph_tup has {len(drug_id_mol_graph_tup)} entries.")
# print("First 5 entries in drug_id_mol_graph_tup:")
# for i, tup in enumerate(drug_id_mol_graph_tup):
#     print(tup)
#     if i == 4:
#         break

drug_to_mol_graph = {id:Chem.MolFromSmiles(smiles.strip()) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])}

# print(f"drug_to_mol_graph has {len(drug_to_mol_graph)} entries.")
# print("First 5 entries in drug_to_mol_graph:")
# for i, (key, value) in enumerate(drug_to_mol_graph.items()):
#     print(f"{key}: {value}")
#     if i == 4:
#         break

# Gettings information and features of atoms 
# 提取原子的特征，如原子符号、度数（连接的原子数量）、氢的数量、隐性价态等 | # 获得分子中原子的一些统计信息

# 表示一个分子中最大的原子数量。这对于创建神经网络输入特别重要，因为我们需要将所有的分子表示成相同的维度。
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup]) 
# print(f"ATOM_MAX_NUM: {ATOM_MAX_NUM}")

# 是一个列表，其中包含了所有可能出现的原子的符号，比如 'H', 'C', 'O', 'N'等等。
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)}) # 获取可用的原子符号
# print(f"AVAILABLE_ATOM_SYMBOLS has {len(AVAILABLE_ATOM_SYMBOLS)} types: {AVAILABLE_ATOM_SYMBOLS}")

# 是一个列表，其中包含了所有可能出现的原子的度（与该原子相连的原子数）。例如，一个度为2的原子连接了2个其他原子
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)}) # 获取可用的原子度数（连接的原子数量）
# print(f"AVAILABLE_ATOM_DEGREES has {len(AVAILABLE_ATOM_DEGREES)} types: {AVAILABLE_ATOM_DEGREES}")

# 是一个列表，其中包含了所有可能出现的原子的氢原子数。例如，一个氢原子数为2的原子有2个氢原子。
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)}) # 获取可用的原子氢数
# print(f"AVAILABLE_ATOM_TOTAL_HS has {len(AVAILABLE_ATOM_TOTAL_HS)} types: {AVAILABLE_ATOM_TOTAL_HS}")

# 是所有原子中最大的隐式价态（Implicit Valence），这是一个原子可以形成的化学键的数量。
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)) 
max_valence = max(max_valence, 9)
# print(f"max_valence: {max_valence}")

# 是一个列表，包含了从0到max_valence的所有值。
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)
# print(f"AVAILABLE_ATOM_VALENCE has {len(AVAILABLE_ATOM_VALENCE)} types: {AVAILABLE_ATOM_VALENCE}")

# 是所有原子中最大的正式电荷（Formal Charge）。如果MAX_ATOM_FC为0，那么保持为0；否则，使用上一步得到的MAX_ATOM_FC。
MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
# print(f"MAX_ATOM_FC: {MAX_ATOM_FC}")

# 是所有原子中最大的自由电子数（Radical Electrons）。原子的自由电子数是指原子上的未配对电子的数量。
# 如果MAX_RADICAL_ELC为0，那么保持为0；否则，使用上一步得到的MAX_RADICAL_ELC。
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0
# print(f"MAX_RADICAL_ELC: {MAX_RADICAL_ELC}")

# 定义一个one-hot编码函数，如果x不在allowable_set中，就将x设为allowable_set中的最后一个元素
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
# print(f"One-hot encoding example: {one_of_k_encoding_unk('C', ['H', 'C', 'N'])}")

# 定义原子特征提取函数，该函数会返回一个包含多个原子特征的列表
def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)
# mol = Chem.MolFromSmiles('CCO')
# atom = mol.GetAtomWithIdx(0)  # 取得第一个原子，一个碳原子
# print(f"Atom features example: {atom_features(atom)}")

# # 获取原子特征
# def get_atom_features(atom, mode='one_hot'):

#     if mode == 'one_hot':
#         atom_feature = torch.cat([
#             one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
#             one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
#             one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
#             one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
#             torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
#         ])
#     else:
#         atom_feature = torch.cat([
#             one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
#             torch.tensor([atom.GetDegree()]).float(),
#             torch.tensor([atom.GetTotalNumHs()]).float(),
#             torch.tensor([atom.GetImplicitValence()]).float(),
#             torch.tensor([atom.GetIsAromatic()]).float()
#         ])

#     return atom_feature

# 获取分子边列表和特征
def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list 
    return undirected_edge_list.T, n_features

# mol = Chem.MolFromSmiles('CCO')
# print(f"Molecule edge list and feature matrix example: {get_mol_edge_list_and_feat_mtx(mol)}")

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) 
                                for drug_id, mol in drug_id_mol_graph_tup}
# 这段代码的目的是过滤掉MOL_EDGE_LIST_FEAT_MTX字典中值为None的项。换句话说，这段代码会去掉那些对应分子数据不存在的药物。
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

# 打印 key-value 对的数量
print(f"Number of key-value pairs in MOL_EDGE_LIST_FEAT_MTX: {len(MOL_EDGE_LIST_FEAT_MTX)}")

# 打印前五个样例数据
for i, (drug_id, (edge_list, feature_matrix)) in enumerate(MOL_EDGE_LIST_FEAT_MTX.items()):
    if i == 2:
        break
    print(f"Drug id: {drug_id}")
    print(f"Edge list shape: {edge_list.shape}, Edge list: {edge_list}")
    print(f"Feature matrix shape: {feature_matrix.shape}, Feature matrix: {feature_matrix}")

# 这行代码获取了特征矩阵的维度，也就是原子特征的总数。
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])

# # 打印过滤后的 key-value 对的数量
# print(f"Number of key-value pairs in MOL_EDGE_LIST_FEAT_MTX after filtering: {len(MOL_EDGE_LIST_FEAT_MTX)}")
# 打印总原子特性数
print(f"Total number of atom features: {TOTAL_ATOM_FEATS}")
#########################################################################

import pickle

# Save the dictionary to a pickle file
with open('mol_edge_list_feat_mtx.pkl', 'wb') as f:
    pickle.dump(MOL_EDGE_LIST_FEAT_MTX, f)

def get_drug_feature(edge_list, feature_matrix):
    return torch.sum(feature_matrix, axis=0)

NEW_MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_drug_feature(edge_list, feature_matrix) 
                               for drug_id, (edge_list, feature_matrix) in MOL_EDGE_LIST_FEAT_MTX.items()}

for i, (drug_id, drug_feature) in enumerate(NEW_MOL_EDGE_LIST_FEAT_MTX.items()):
    if i == 2:  # 只打印前两个样例
        break
    print(f"Drug id: {drug_id}")
    print(f"Drug feature shape: {drug_feature.shape}, Drug feature: {drug_feature}\n")

# Save the dictionary to a pickle file
with open('new_mol_edge_list_feat_mtx.pkl', 'wb') as f:
    pickle.dump(NEW_MOL_EDGE_LIST_FEAT_MTX, f)