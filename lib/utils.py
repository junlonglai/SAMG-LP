import numpy as np
import scipy as sp
import torch


def get_gnn_sup(adj):
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj  # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]

    return sup

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def WGMA(adj_tnr_seq, moving_step, decay_theta, device):
    # 衰减系数
    decay_tnr_list = [torch.FloatTensor([(1 - decay_theta) ** (moving_step - k - 1)]).to(device)
                      for k in range(moving_step)]
    # 计算加权平均值并存储在新的列表中
    avg_tnr_seq = []
    for i in range(len(adj_tnr_seq) - moving_step + 1):
        adj_tnr_list = adj_tnr_seq[i:i + moving_step]
        result_tnr = torch.zeros_like(adj_tnr_seq[0])  # 创建与列表元素相同形状的零张量
        for j, adj_tnr in enumerate(adj_tnr_list):   # 加权求和
            result_tnr += decay_tnr_list[j] * adj_tnr
        result_tnr = result_tnr/sum(decay_tnr_list)
        avg_tnr_seq.append(result_tnr)

    return avg_tnr_seq

