import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics

from utils import load_data, build_graph, weight_reset
from model import GATMDA, GATMDA_only_attn, GATMDA_without_attn


def Train(directory, epochs, attn_size, attn_heads, out_dim, dropout, slope, lr, wd, random_seed, cuda, model_type):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)

    # if cuda:
    #     context = torch.device('cuda:0')
    # else:
    #     context = torch.device('cpu')
    context = torch.device('cpu')
    # g为miRNA和disease图，g0为miRNA、disease和lncRNA图
    g, g0, disease_vertices, mirna_vertices, ID, IM, IL, samples, ml_associations, ld_associations = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())
    print('## lncrna nodes: ', torch.sum(g0.ndata['type'] == 2).numpy())
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

    for train_idx, test_idx in kf.split(samples[:, 2]):
        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

        g0.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g0.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        # g相关代码
        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train0 = g0.edge_subgraph(train_eid, preserve_nodes=True)
        # g_train.copy_from_parent()

        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)
        # g0相关代码
        # label_train = label_train.numpy().tolist()
        # x0 = 0
        # for x1 in range(len(label_train)):
        #     if label_train[x1] == [1.0]:
        #         x0 = x0 + 1
        #
        # label_test = label_test.numpy().tolist()
        # y0 =0
        # for y1 in range(len(label_test)):
        #     if label_test[y1] == [1.0]:
        #         y0 = y0 + 1

        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        if model_type == 'GATMDA':
            model = GATMDA(G=g_train0,
                           meta_paths_list=['md', 'dm', 'ml', 'dl'],
                           feature_attn_size=attn_size,
                           num_heads=attn_heads,
                           num_diseases=ID.shape[0],
                           num_mirnas=IM.shape[0],
                           num_lncrnas=IL.shape[0],
                           d_sim_dim=ID.shape[1],
                           m_sim_dim=IM.shape[1],
                           l_sim_dim=IL.shape[1],
                           out_dim=out_dim,
                           dropout=dropout,
                           slope=slope,
                           )

        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.BCELoss()



    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result