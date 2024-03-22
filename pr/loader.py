import pandas as pd
import numpy as np
import os
import random

from utils.data_processing import Data


def load_data(dataset_name, data_dir="./data", is_split_val_test=True, time_scale=1):
    # Load data and train val test split
    graph_df = pd.read_csv(os.path.join(data_dir, dataset_name, "dgl/edges.csv"))
    node_features = np.load(os.path.join(data_dir, dataset_name, "dgl/nodes_feat.npy"))

    n_edges = len(graph_df)
    # 边向量用0向量代替
    edge_feature_dim = 16
    edges_feature = np.zeros((n_edges + 1, edge_feature_dim))

    # 读取数据
    sources = graph_df.src_id.values
    destinations = graph_df.dst_id.values
    edge_idxs = graph_df.idx.values
    labels = np.zeros(n_edges)
    timestamps = graph_df.ts.values
    timestamps = timestamps * time_scale  # 缩放时间戳
    train_mask = graph_df.train_mask.values
    test_val_mask = graph_df.test_mask.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

    if not is_split_val_test:
        test_data = Data(sources[test_val_mask], destinations[test_val_mask], timestamps[test_val_mask],
                     edge_idxs[test_val_mask], labels[test_val_mask])
        print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                     full_data.n_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes))
        print("The dim of node features: {}".format(node_features.shape[1]))
        print("The dim of edge features: {}".format(edges_feature.shape[1]))
        return node_features, edges_feature, full_data, train_data, test_data

    random.seed(2023)
    # 随机划分test/val集合
    test_val_index = [i for i, mask in enumerate(test_val_mask) if mask]
    val_radio = 0.5
    val_size = int(val_radio * len(test_val_index))
    random.shuffle(test_val_index)

    val_indices = test_val_index[:val_size]
    test_indices = test_val_index[val_size:]

    val_mask = np.zeros(n_edges, dtype=bool)
    test_mask = np.zeros(n_edges, dtype=bool)
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # build test/val data
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The dim of node features: {}".format(node_features.shape[1]))
    print("The dim of edge features: {}".format(edges_feature.shape[1]))

    return node_features, edges_feature, full_data, train_data, val_data, test_data


def load_nodes_meta(dataset_name, data_dir="./data"):
    # Load node meta data
    nodes_df = pd.read_csv(os.path.join(data_dir, dataset_name, "dgl/nodes.csv"))

    # meta data
    node_ids = nodes_df.node_id.values
    # pids = nodes_df.pid.values
    years = nodes_df.year.values

    # insert 0
    node_ids = np.insert(node_ids, 0, 0)
    # pids = np.insert(pids, 0, "None")
    years = np.insert(years, 0, 0)

    return node_ids, years



