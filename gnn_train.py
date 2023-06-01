import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from Data_Adapter import DrugInteractionNetwork
from gnn_model import GIN_Net2
from utils import Metrictor_PPI, print_file, timeSince
import datetime
# from tensorboardX import SummaryWriter

import wandb
os.environ["WANDB_API_KEY"] = "..."
wandb.login()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class CFG:
    description = "Train Script"
    project_name = "DDI-PPI-Adapt"
    ddi_path = '/root/DSN-DDI/drugbank_test/drugbank/ddis.csv'
    mol_feature_path = "/root/DSN-DDI/new_mol_edge_list_feat_mtx.pkl"
    train_valid_index_path = "/root/DSN-DDI/train_valid_index.json"
    split_new = False
    use_lr_scheduler = True
    save_path = "/root/DSN-DDI/drugbank_test/Adapte_to_GNN-PPI/save_model"
    graph_only_train = False
    batch_size = 512
    epochs = 50

def class_to_dict(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")}



def train(model, graph, ddi_list, loss_fn, optimizer, device,
        result_file_path, save_path,
        batch_size=512, epochs=1000, scheduler=None, 
        got=False):
    
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    truth_edge_num = graph.edge_index.shape[1] // 2
    start = time.time()
    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)
        '''
            函数 train 中，当 got=True 时，将使用 train_mask_got 作为训练集的样本 ID,从而训练一个 GO-terms 模型。
            这个模型的目标是根据蛋白质之间的相似性，预测它们在哪些 GO-terms 中具有共同的功能。
            GO-terms 是 Gene Ontology (GO) 词汇集合中的一个子集，用于描述生物学过程、分子功能和细胞成分。
        '''
        for step in range(steps):  
            if step == steps-1:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size:]
                else:
                    train_edge_id = graph.train_mask[step*batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size : step*batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step*batch_size : step*batch_size + batch_size]
            
            if got:
                output = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr[train_edge_id]
            
            # label = label.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            # 转换为独热编码
            num_classes = output.shape[1]  # 假设输出的形状是[N, num_classes]
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=num_classes).float()
            loss = loss_fn(output, label_one_hot)
            
            # loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            # metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)
            metrics = Metrictor_PPI(pre_result.cpu().data, label_one_hot.cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            # summary_writer.add_scalar('train/loss', loss.item(), global_step)
            # summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            # summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            # summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                        .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                    os.path.join(save_path, 'gnn_model_train.ckpt'))
        
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps-1:
                    valid_edge_id = graph.val_mask[step*batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step*batch_size : step*batch_size + batch_size]
                
                output = model(graph.x, graph.edge_index, valid_edge_id)
                label = graph.edge_attr[valid_edge_id]

                label = label.type(torch.LongTensor).to(device)
                # 转换为独热编码
                num_classes = output.shape[1]  # 假设输出的形状是[N, num_classes]
                label_one_hot = torch.nn.functional.one_hot(label, num_classes=num_classes).float()
                loss = loss_fn(output, label_one_hot)
                
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label_one_hot.cpu().data)
        
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']), save_file_path=result_file_path)
        
        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict()},
                        os.path.join(save_path, 'gnn_model_best.ckpt'))
        
        # summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        # summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        # summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        # summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        
        print_file("epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                    .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1, global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)
        print_file('Elapsed {remain:s} '.format(remain = timeSince(start, float(epoch+1)/epochs))) 
        # use wandb format print these information
        wandb.log({'epoch':epoch, 
                    'train_loss': loss, 
                    'train_recall': recall, 
                    'train_precision': precision, 
                    'train_F1': f1, 
                    'valid_loss': valid_loss, 
                    'valid_recall': metrics.Recall, 
                    'valid_precision': metrics.Precision, 
                    'valid_F1': metrics.F1, 
                    'best_valid_F1': global_best_valid_f1, 
                    'best_valid_F1_epoch': global_best_valid_f1_epoch})

    wandb.finish()
def main():
    ddi_data = DrugInteractionNetwork(ddi_path=CFG.ddi_path)
    print("use_get_feature_origin")
    ddi_data.get_feature_origin(mol_feature_path=CFG.mol_feature_path)
    ddi_data.generate_data()
    
    print("----------------------- start split train and valid index -------------------")
    train_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/train.csv"
    val_1_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/s1.csv"
    val_2_path = "/root/DSN-DDI/drugbank_test/inductive_data/fold0/s2.csv"
    ddi_data.split_dataset(CFG.train_valid_index_path, train_path, val_1_path, val_2_path, random_new=CFG.split_new)
    print("----------------------- Done split train and valid index -------------------")

    graph = ddi_data.data
    print(graph.x.shape)
    ddi_list = ddi_data.ddi_list

    # graph.train_mask 是一个布尔型的一维张量，用于指示哪些节点在训练时应该被包含
    graph.train_mask = ddi_data.ppi_split_dict['train_index']
    graph.val_mask = ddi_data.ppi_split_dict['valid1_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    
    # 将 graph 的原始边索引 edge_index 中训练集所包含的部分赋值给 edge_index_got 属性，并且将其中的起点和终点顺序反转，再与原有的边索引拼接在一起。
    graph.edge_index_got = torch.cat((graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    # 将 graph 的边属性 edge_attr 中训练集所包含的部分赋值给 edge_attr_got 属性，并将其复制一份，再拼接在一起。
    graph.edge_attr_got = torch.cat((graph.edge_attr[graph.train_mask], graph.edge_attr[graph.train_mask]), dim=0)
    # 是一个列表，包含了一个从 0 开始的连续整数序列，长度与 graph.train_mask 相同, 这个列表可以被用来指定在训练过程中要使用的样本 ID
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    graph.to(device)

    # Ajust_model for esm_2_large
    model = GIN_Net2(in_len=1, in_feature=55, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1).to(device)

    # Add wandb to script
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=CFG.project_name, config = class_to_dict(CFG), name = nowtime, save_code=True)
    model.run_id = wandb.run.id

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = None
    if CFG.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(CFG.save_path):
        os.mkdir(CFG.save_path)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(CFG.save_path, "gnn_{}_{}".format(CFG.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        CFG_dict = CFG.__dict__
        for key in CFG_dict:
            f.write("{} = {}".format(key, CFG_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    
    # # log the dataset
    # GNN_ddI_dataset = wandb.Artifact('PPI', type='dataset')
    # GNN_ddI_dataset.add_file(f'{CFG.ppi_path}')
    # GNN_ddI_dataset.add_file(f'{CFG.pseq_path}')
    # GNN_ddI_dataset.add_file(f'{CFG.vec_path}')
    # GNN_ddI_dataset.add_file(f'{CFG.train_valid_index_path}')
    # wandb.log_artifact(GNN_ddI_dataset)
    
    # # log the code
    # GNN_PPI_code = wandb.Artifact('Python_code', type='code')
    # GNN_PPI_code.add_file("./gnn_train.py")
    # GNN_PPI_code.add_file("./gnn_data.py")
    # GNN_PPI_code.add_file("./gnn_model.py")
    # GNN_PPI_code.add_file("./run.py")
    # wandb.log_artifact(GNN_PPI_code)

    train(model, graph, ddi_list, loss_fn, optimizer, device,
        result_file_path, save_path,
        batch_size=CFG.batch_size, epochs=CFG.epochs, scheduler=scheduler, 
        got=CFG.graph_only_train)
    
    wandb.finish()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    seed_everything(seed=42)
    main()