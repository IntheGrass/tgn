from collections import defaultdict

import numpy as np
import torch
from torch import nn

from modules.memory import Memory
from modules.memory_updater import get_memory_updater
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.pr.embedding import get_embedding_module
from utils.utils import MergeLayer


class MyTgn(nn.Module):
    # 图卷积模型，根据图节点生成其嵌入向量
    def __init__(self, neighbor_finder, node_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False, n_neighbors=10,
                 message_dimension=100, memory_dimension=500, embedding_module_type="sage",
                 message_function="mlp", aggregator_type="last", memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False):
        super(MyTgn, self).__init__()
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)

        # general config
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.use_memory = use_memory
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message

        # model parameter
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors  # 采样的邻居数
        self.node_feature_dim = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.embedding_dim = self.node_feature_dim  # output dim. TODO: 设置为可自定义？、

        # modules
        if self.use_memory:
            self.memory_dimension = memory_dimension
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension

            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module = get_embedding_module(self.embedding_module_type,
                                                     self.node_raw_features,
                                                     self.neighbor_finder,
                                                     None,
                                                     self.n_layers,
                                                     self.node_feature_dim,
                                                     self.device,
                                                     dropout=dropout,
                                                     use_memory=self.use_memory)

    def forward(self, src_nodes, des, cut_time_l, n_neighbors=20, is_update_memory=False):
        pass

    def compute_pair_edge_embedding(self, src_nodes, dst_nodes, cut_time_l, n_neighbors=20):
        # 计算成对节点（即边的两个节点）的嵌入，会更行memory和message
        n_samples = len(src_nodes)
        nodes = np.concatenate([src_nodes, dst_nodes])

        memory = None
        if self.use_memory:
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=cut_time_l,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors)

        src_embedding = node_embedding[:n_samples]
        dst_embedding = node_embedding[n_samples:]

        if self.use_memory:
            unique_nodes, node_id_to_messages = self.get_raw_messages(src_nodes,
                                                                      src_embedding,
                                                                      dst_nodes,
                                                                      dst_embedding,
                                                                      cut_time_l)

            self.update_memory(unique_nodes, node_id_to_messages)
            self.memory.clear_messages(nodes)

        return src_embedding, dst_embedding

    def embedding_nodes(self, nodes, cut_time_l, n_neighbors=20):
        # 计算独立节点的嵌入，不更新memory
        memory = None
        if self.use_memory:
            memory_features = self.memory.get_memory(list(range(self.n_nodes)))

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=memory_features,
                                                                 source_nodes=nodes,
                                                                 timestamps=cut_time_l,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors)
        return node_embedding

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)  # 聚合nodes中的所有消息，默认聚合方式为保留最新消息，返回的unique_nodes包含了实际聚合节点（即消息不为空的节点）

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)  # 聚合同一节点的消息，默认取最新值

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times):  # 该方法用于生成消息
        # 默认情况下，原始消息仅包含：源节点与目标的memory + 边特征 + 当前时间与上次更新时间的差值
        edge_times = torch.from_numpy(edge_times).float().to(self.device)

        source_memory = self.memory.get_memory(source_nodes) if not \
          self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
          not self.use_destination_embedding_in_message else destination_node_embedding  # 使用原始特征或memory中特征用于生成消息

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
          source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, source_time_delta_encoding],
                                   dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder








