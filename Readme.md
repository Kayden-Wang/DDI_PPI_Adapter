# Readme

以下代码用以将DSN-DDI中的DrugBank数据训练集验证集划分, 以及其数据的预处理方式后的数据适配到GNN-PPI框架中进行训练. 

`data_preprocessing_GNN.py` 文件用以提取Drugbank 药物分子的特征, 并将其进行特征的压缩, 成为单一的药物分子表示. 

`Data_Adapter.py` 文件用以进行药物交互关系网络的构建, 然后将DDI数据包装成GNN的输入数据格式, 并且进行训练集验证集的划分.

`gnn_model.py` 是封装着适配过的GNN模型

`gnn_train.py` 是GNN的模型训练代码

`utils.py` 中封装这可能用到的工具库

下列文件再DSN-DDI的路径如下
<img src=".\image-20230601153710838.png" alt="image-20230601153710838" style="zoom:33%;" /> 