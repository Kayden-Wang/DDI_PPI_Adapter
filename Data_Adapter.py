import pandas as pd
from tqdm import tqdm
import copy
import pickle
import torch
from torch_geometric.data import Data
import numpy as np
import gc
import json

class DrugInteractionNetwork:
    # data reading 
    def __init__(self, ddi_path, max_len=2000, drug1_index='d1', drug2_index='d2', type_index='type', graph_undirection=True):
        '''
        ddi_path: 
            the path of drug-drug interaction (DDI) file 
        max_len: 
            the max length of acc number per drug
            default: 2000
        drug1_index: 
            the column name of drug 1 in DDI file
            default: 'd1'
        drug2_index: 
            the column name of drug 2 in DDI file
            default: 'd2'
        type_index: 
            the column name of interaction type in DDI file
            default: 'type'
        graph_undirection: 
            whether to make the graph undirection
            default: True
        '''
        self.ddi_list = [] 
        self.ddi_dict = {}
        self.ddi_label_list = []
        self.drug_dict = {} 
        self.drug_name = {}
        self.ddi_path = ddi_path
  
        name = 0
        ddi_name = 0

        df_ddi_train = pd.read_csv(ddi_path)
        train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train[drug1_index], df_ddi_train[drug2_index], df_ddi_train[type_index])]

        for tup in tqdm(train_tup):

            # get node and node name
            # name: the number of node
            if tup[0] not in self.drug_name.keys():
                self.drug_name[tup[0]] = name
                name += 1
            
            if tup[1] not in self.drug_name.keys():
                self.drug_name[tup[1]] = name
                name += 1

            # get edge and its label
            # ddi_name: the number of edge
            temp_data = ""
            if tup[0] < tup[1]:
                temp_data = tup[0] + "__" + tup[1]
            else:
                temp_data = tup[1] + "__" + tup[0]
            
            if temp_data not in self.ddi_dict.keys():
                self.ddi_dict[temp_data] = ddi_name
                self.ddi_label_list.append(tup[2])
                ddi_name += 1

        # This Loop is to get the ddi_list from ddi_dict
        # "drug1__drug2" -> [drug1, drug2]
        for ddi in tqdm(self.ddi_dict.keys()):
            name = self.ddi_dict[ddi]
            temp = ddi.strip().split('__')
            self.ddi_list.append(temp)

        # Convert the drug name in ddi_list to IDs (name -> ID)
        ddi_num = len(self.ddi_list)
        self.origin_ddi_list = copy.deepcopy(self.ddi_list)
        assert len(self.ddi_list) == len(self.ddi_label_list)
        for i in tqdm(range(ddi_num)):
            seq1_name = self.ddi_list[i][0]
            seq2_name = self.ddi_list[i][1]
            self.ddi_list[i][0] = self.drug_name[seq1_name]
            self.ddi_list[i][1] = self.drug_name[seq2_name]
        
        # Add the reverse edge (if undirected graph)
        if graph_undirection:
            for i in tqdm(range(ddi_num)):
                temp_ddi = self.ddi_list[i][::-1]
                temp_ddi_label = self.ddi_label_list[i]
                self.ddi_list.append(temp_ddi)
                self.ddi_label_list.append(temp_ddi_label)

        self.node_num = len(self.drug_name)
        self.edge_num = len(self.ddi_list)
    
    def get_feature_origin(self, mol_feature_path="/root/DSN-DDI/new_mol_edge_list_feat_mtx.pkl"):
        # Load drug feature from pickle file
        with open(mol_feature_path, 'rb') as f:
            self.mol_feature = pickle.load(f)

        self.drug_dict = {}
        for name in tqdm(self.drug_name.keys()):
            if name in self.mol_feature:
                self.drug_dict[name] = self.mol_feature[name]
            else:
                print(f"Drug {name} not found in mol_feature dict.")

    # generate pyg data 
    def generate_data(self):
        ddi_list = np.array(self.ddi_list)
        ddi_label_list = np.array(self.ddi_label_list)

        self.edge_index = torch.tensor(ddi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ddi_label_list, dtype=torch.long)
        def drug_generator():
            i = 0
            for name in self.drug_name:
                assert self.drug_name[name] == i
                i += 1
                yield self.drug_dict[name]

        self.x = torch.stack(list(drug_generator()))
        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr=self.edge_attr)
        del self.x
        gc.collect()
    
    def split_dataset(self,train_valid_index_path, train_path, val_1_path, val_2_path, random_new=False):
        if random_new:
            df_ddi_train = pd.read_csv(train_path)
            df_ddi_s1 = pd.read_csv(val_1_path)
            df_ddi_s2 = pd.read_csv(val_2_path)

            train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
            s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
            s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]
            
            self.ppi_split_dict = {}
            self.ppi_split_dict['train_index'] = [self.ddi_dict[f"{min(a,b)}__{max(a,b)}"] for a, b, _ in train_tup if f"{min(a,b)}__{max(a,b)}" in self.ddi_dict]
            self.ppi_split_dict['valid1_index'] = [self.ddi_dict[f"{min(a,b)}__{max(a,b)}"] for a, b, _ in s1_tup if f"{min(a,b)}__{max(a,b)}" in self.ddi_dict]
            self.ppi_split_dict['valid2_index'] = [self.ddi_dict[f"{min(a,b)}__{max(a,b)}"] for a, b, _ in s2_tup if f"{min(a,b)}__{max(a,b)}" in self.ddi_dict]
            
            jsobj = json.dumps(self.ppi_split_dict)
            with open(train_valid_index_path, 'w') as f:
                f.write(jsobj)
                f.close()
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()

def main():
    # path to your csv file
    ddi_path = '/root/DSN-DDI/drugbank_test/drugbank/ddis.csv'
    
    ddi_network = DrugInteractionNetwork(ddi_path)
    
    # print number of nodes and edges in the network
    print(f"Number of nodes: {ddi_network.node_num}")
    print(f"Number of edges: {ddi_network.edge_num}")

    # print the first 5 drugs and their IDs
    print("First 5 drugs and their IDs:")
    for i, (drug, id) in enumerate(ddi_network.drug_name.items()):
        if i >= 5:
            break
        print(f"{drug}: {id}")
    
    # print the first 5 drug-drug interactions
    print("First 5 drug-drug interactions:")
    for i in range(min(5, len(ddi_network.ddi_list))):
        print(f"{ddi_network.ddi_list[i]}: {ddi_network.ddi_label_list[i]}")

    # print shape of the interactions list and labels list
    print(f"Shape of interactions list: {len(ddi_network.ddi_list)}")
    print(f"Shape of labels list: {len(ddi_network.ddi_label_list)}")
        # Calling the new function to load drug features
    
    ddi_network.get_feature_origin()

    # Print the features of the first 5 drugs
    print("Features of the first 5 drugs:")
    for i, (drug, feature) in enumerate(ddi_network.drug_dict.items()):
        if i >= 5:
            break
        print(f"{drug}: {feature.shape}")
    
    # Generate PyG data
    ddi_network.generate_data()
    
    # Print the shapes of PyG data
    print(f"Edge index shape: {ddi_network.data.edge_index.shape}")
    print(f"Edge attribute shape: {ddi_network.data.edge_attr.shape}")
    print(f"Node attribute shape: {ddi_network.data.x.shape}")

    # Specify file paths
    train_valid_index_path = "train_valid_index.json"
    train_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/train.csv"
    val_1_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/s1.csv"
    val_2_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/s2.csv"
    random_new = True  # Set to True to generate new split, False to load from train_valid_index_path

    # Split dataset
    ddi_network.split_dataset(train_valid_index_path, train_path, val_1_path, val_2_path, random_new=random_new)
    
    # Print the results
    print("Split results:")
    print(f"Number of training examples: {len(ddi_network.ppi_split_dict['train_index'])}")
    print(f"Number of validation examples (set 1): {len(ddi_network.ppi_split_dict['valid1_index'])}")
    print(f"Number of validation examples (set 2): {len(ddi_network.ppi_split_dict['valid2_index'])}")

        
if __name__ == "__main__":
    main()
