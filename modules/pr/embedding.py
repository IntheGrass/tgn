# 对embedding_module.py中部分模块的重新实现，并且，新模块不使用边特征向量，
import torch
from torch import nn
import numpy as np

from model.temporal_attention import TemporalAttentionLayer
from model.time_encoding import TimeEncode


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, neighbor_finder, time_encoder: TimeEncode,
                 n_layers, embedding_dimension, device, dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.node_feature_dim = node_features.shape[1]
        self.time_feature_dim = time_encoder.dimension
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        return NotImplemented


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, neighbor_finder, time_encoder, n_layers, embedding_dimension,
                 device, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, neighbor_finder, time_encoder, n_layers,
                                             embedding_dimension, device, dropout)
        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None):
        """Recursive implementation of curr_layers temporal graph attention layers.

        source_nodes [batch_size]: papers input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))  # (batch_size, 1, dim)

        source_node_features = self.node_features[source_nodes_torch, :]  # 初始节点特征

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features  # 记忆模块，memory仅仅使用用于确定第0层的节点向量

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_embedding(memory,
                                                                 source_nodes,
                                                                 timestamps,
                                                                 n_layers=n_layers - 1,
                                                                 n_neighbors=n_neighbors)

            neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(
                self.device)  # (batch_size, n_neighbors), 邻居不足的部分设为0

            edge_deltas = timestamps[:, np.newaxis] - edge_times  # 计算邻居边timestamp与当前的差值，必>=0

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()  # (batch_size * n_neighbors, )
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors,
                                                           -1)  # (batch_size, n_neighbors, dim)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)  # (batch_size, n_neighbors, dim)

            mask = neighbors_torch == 0  # 不全的邻居设mask=true

            source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              mask)  # (batch_size, dim)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, neighbor_finder, time_encoder, n_layers, embedding_dimension,
                 device, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                embedding_dimension=embedding_dimension,
                                                device=device, dropout=dropout,
                                                use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + self.time_feature_dim,
                                                             embedding_dimension)
                                             for _ in range(n_layers)])  # 邻居节点的线性变换层
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + self.node_feature_dim + self.time_feature_dim,
                             embedding_dimension) for _ in range(n_layers)])  # 聚合源节点与邻居节点的线性变换层

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, neighbor_finder, time_encoder, n_layers, embedding_dimension,
                 device, n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, neighbor_finder, time_encoder,
                                                      n_layers, embedding_dimension, device, dropout,
                                                      use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=self.node_feature_dim,
            n_neighbors_features=self.node_feature_dim,
            n_edge_features=0,
            time_dim=self.time_feature_dim,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=self.embedding_dimension)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              # 使用空的边向量
                                              torch.zeros(source_node_features.shape[0], 0).to(self.device),
                                              mask)

        return source_embedding


class SageEmbedding(GraphEmbedding):
    def __init__(self, node_features, neighbor_finder, time_encoder, n_layers, embedding_dimension,
                 device, dropout=0.1, use_memory=True, bias=True, activation=True):
        super(SageEmbedding, self).__init__(node_features, neighbor_finder, time_encoder,
                                            n_layers, embedding_dimension, device, dropout,
                                            use_memory)
        self.activation = activation
        self.fc_neigh = torch.nn.ModuleList([torch.nn.Linear(
            self.node_feature_dim + self.time_feature_dim,
            self.node_feature_dim if i < n_layers-1 else embedding_dimension, bias=False) for i in range(n_layers)])

        self.fc_self = torch.nn.ModuleList([torch.nn.Linear(
            self.node_feature_dim + self.time_feature_dim,
            self.node_feature_dim if i < n_layers-1 else embedding_dimension, bias=False) for i in range(n_layers)])
        if bias:
            self.bias = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.zeros(embedding_dimension))
                                                for _ in range(n_layers)])
        self.feat_drop = nn.Dropout(self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for i in range(self.n_layers):
            torch.nn.init.xavier_normal_(self.fc_neigh[i].weight, gain=gain)
            torch.nn.init.xavier_normal_(self.fc_self[i].weight, gain=gain)

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, mask):
        neighbor_embeddings = self.feat_drop(neighbor_embeddings)
        source_node_features = self.feat_drop(source_node_features)
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings],
                                       dim=2)
        neighbor_embeddings = self.fc_neigh[n_layer - 1](neighbors_features)
        neighbors_mean = torch.nn.functional.relu(torch.mean(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = self.fc_self[n_layer - 1](source_features)

        target = source_embedding + neighbors_mean

        if self.bias is not None:
            target = target + self.bias[n_layer - 1]

        if self.activation:
            target = torch.nn.functional.relu(target)  # 激活函数
        return target


def get_embedding_module(module_type, node_features, neighbor_finder,
                         time_encoder, n_layers, embedding_dimension, device,
                         n_heads=2, dropout=0.1, use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 embedding_dimension=embedding_dimension,
                                 device=device, dropout=dropout, use_memory=use_memory)

    elif module_type == "sage":
        return SageEmbedding(node_features=node_features,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             embedding_dimension=embedding_dimension,
                             device=device, dropout=dropout, use_memory=use_memory)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
