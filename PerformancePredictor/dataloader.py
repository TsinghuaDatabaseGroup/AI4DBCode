import os
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch


# ## Load Data

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dataset, path):

    print('Loading {} dataset...'.format(dataset))
#    print("now dataset: {}\n".format(os.path.join(path, dataset)))
    vmatrix = np.genfromtxt("{}.content".format(os.path.join(path, dataset)),
                                        dtype=np.dtype(str))

    ematrix = np.genfromtxt("{}.cites".format(os.path.join(path, dataset)),
                                    dtype=np.float32)

    return load_data_from_matrix(vmatrix, ematrix)


def load_data_from_matrix(vmatrix, ematrix):
    from performance_graphembedding_checkpoint import node_dim
    idx_features_labels = vmatrix

    # encode vertices
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # encode labels
    # labels = encode_onehot(idx_features_labels[:, -2])
    labels = idx_features_labels[:, -1].astype(float)

    # encode edges
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)


    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = ematrix[:, :-1]

    # print(list(map(idx_map.get, edges_unordered.flatten())))
    # print(edges_unordered.flatten())

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # edges (weights are computed in gcn)

    # modified begin.
    edges_value = ematrix[:, -1:]
    # modified end.

    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(node_dim, node_dim),dtype=np.float32)
    # print("old_adj = ", adj)
    adj = sp.coo_matrix((edges_value[:,0], (edges[:, 0], edges[:, 1])),shape=(node_dim, node_dim),dtype=np.float32)
    # print("new_adj = ", adj)
    # print(adj.shape)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    operator_num = adj.shape[0]
    idx_train = range(int(0.8 * operator_num))
    # print("idx_train", idx_train)
    idx_val = range(int(0.8 * operator_num), int(0.9 * operator_num))
    idx_test = range(int(0.9 * operator_num), int(operator_num))

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # padding to the same size
    # print(features.shape)
    # print(node_dim - features.shape[0])
    dim=(0, 0, 0,  node_dim - features.shape[0])
    features=F.pad(features, dim, "constant", value=0)

    labels = labels.astype(np.float32)
    labels = torch.from_numpy(labels)
    # print(labels[idx_train].dtype)
    labels.unsqueeze(1)
    labels = labels * 10
    labels=F.pad(labels, [0, node_dim - labels.shape[0]], "constant", value=0)

    # print("features", features.shape)
    return adj, features, labels, idx_train, idx_val, idx_test