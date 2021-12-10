import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics

from utils import load_data, build_graph, weight_reset
from model4 import GRAND


def Train(directory, epochs, n_classes, in_size, out_dim, dropout, slope, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)


    context = torch.device('cpu')

    g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())

    g.to(context)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):     # 返回训练集和测试集的索引train：test 4:1
        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1 # 多加一列，将训练集记为1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)        # 正向反向加边，更新边上的数据
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train'])       # 过滤出被记为train的边
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)       # 从异构图中创建子图，train集的子图
        # g_train.copy_from_parent()
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()          # 训练集的边
        # 筛选出label_train
        # label_train0 = g_train.edata['dm'].unsqueeze(1)
        # label_train0 = label_train0.numpy().tolist()
        # label_train1 = g_train0.edata['md'].unsqueeze(1)
        # label_train1 = label_train1.numpy().tolist()
        # count = 0
        # for x1 in range(len(train_eid0)):
        #     if train_eid0[x1]<10860:
        #         count = count + 1
        # label_train = label_train0[0:count] + label_train0[0:count]
        # label_train = torch.tensor(label_train)
        # z1 = 0
        # # label_train_1 = label_train.numpy().tolist()
        # for z2 in range(len(label_train0)):
        #     if label_train0[z2] == [1.]:
        #          z1 = z1+1
        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)   # 原图中选出标记为0的记为测试集
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)       # 测试集的边
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))
