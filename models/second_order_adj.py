import numpy as np


def second_order(first_order_adj):
    in_adj = np.zeros((first_order_adj.shape[0], first_order_adj.shape[1]))
    out_adj = np.zeros((first_order_adj.shape[0], first_order_adj.shape[1]))

    row_sum = np.sum(first_order_adj, axis=0)
    cal_sum = np.sum(first_order_adj, axis=1)

    for i in range(first_order_adj.shape[0]):
        for j in range(i+1, first_order_adj.shape[1]):
            for k in range(first_order_adj.shape[0]):
                in_adj[i][j] += first_order_adj[k][i]*first_order_adj[k][j]/row_sum[k]
                out_adj[i][j] += first_order_adj[i][k]*first_order_adj[j][k]/cal_sum[k]

            in_adj[j][i] = in_adj[i][j]
            out_adj[j][i] = out_adj[i][j]

    return in_adj, out_adj


def gen_A0(adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums

    return _adj