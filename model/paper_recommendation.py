from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from model.my_tgn import MyTgn
from model.time_encoding import TimeEncode
from utils.utils import MergeLayer


class IPaperRecommendation(ABC):
    @abstractmethod
    def compute_pair_probabilities(self, src_nodes, dst_nodes, edge_times, n_neighbors=20):
        pass


class PrModel(nn.Module, IPaperRecommendation):
    def __init__(self, neighbor_finder, node_features, node_timestamps, device,
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=True, n_neighbors=10,
                 time_dimension=100,
                 message_dimension=100, memory_dimension=500, embedding_module_type="sage",
                 message_function="identity", aggregator_type="last", memory_updater_type="gru", ):
        self.neighbor_finder = neighbor_finder
        self.node_features = torch.from_numpy(node_features).float().to(device)
        self.node_timestamps = torch.from_numpy(node_timestamps).float().to(self.device)

        self.node_feature_dim = self.node_raw_features.shape[1]
        self.text_feature_dim = self.node_feature_dim  # 文本嵌入输出的向量维度
        self.graph_feature_dim = self.node_feature_dim  # 图嵌入输出的向量维度

        # modules
        self.time_encoder = TimeEncode(dimension=self.n_node_features)

        self.graph_embedding_layer = MyTgn(neighbor_finder, node_features, device, n_layers,
                                           n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                           n_neighbors=n_neighbors,
                                           message_dimension=message_dimension, memory_dimension=memory_dimension,
                                           embedding_module_type=embedding_module_type,
                                           message_function=message_function,
                                           aggregator_type=aggregator_type, memory_updater_type=memory_updater_type)
        self.fc = nn.Linear(self.text_feature_dim + self.graph_feature_dim, self.node_feature_dim)  # 结合文本向量与图向量

        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    def forward(self, src_nodes, dst_nodes, neg_nodes, edge_times, n_neighbors=20, is_sigmoid=True):
        self.compute_edge_probabilities(src_nodes, dst_nodes, neg_nodes, edge_times,
                                        n_neighbors=n_neighbors, is_sigmoid=is_sigmoid)

    def compute_edge_probabilities(self, src_nodes, dst_nodes, neg_nodes, edge_times,
                                   edge_idxs=None, n_neighbors=20, is_sigmoid=True):
        batch_size = len(src_nodes)
        assert batch_size == len(dst_nodes) == len(neg_nodes), "batch size is not matched"

        src_graph_embedding, dst_graph_embedding = self.graph_embedding_layer.compute_pair_edge_embedding(
            src_nodes, dst_nodes, edge_times, n_neighbors=n_neighbors)
        neg_graph_embedding = self.graph_embedding_layer.embedding_nodes(neg_nodes, edge_times, n_neighbors=n_neighbors)

        nodes = torch.cat([src_nodes, dst_nodes, neg_nodes], dim=0)
        text_embedding = self.compute_text_embedding(nodes)
        graph_embedding = torch.cat([src_graph_embedding, dst_graph_embedding, neg_graph_embedding], dim=0)

        node_embedding = self.compose_text_graph_embedding(text_embedding, graph_embedding)  # (3*batch_size, feat_dim)

        src_node_embedding = node_embedding[:batch_size]
        dst_node_embedding = node_embedding[batch_size:2 * batch_size]
        neg_node_embedding = node_embedding[2 * batch_size:]

        score = self.affinity_score(torch.cat([src_node_embedding, src_node_embedding], dim=0),
                                    torch.cat([dst_node_embedding, neg_node_embedding])).squeeze(dim=0)  # 合并计算评分

        pos_score = score[:batch_size]
        neg_score = score[batch_size:]

        if not is_sigmoid:
            return pos_score, neg_score

        return pos_score.sigmoid(), neg_score.sigmoid()

    def compute_pair_probabilities(self, src_nodes, dst_nodes, edge_times, n_neighbors=20):
        assert len(src_nodes) == len(dst_nodes), "The size of src_nodes don't match dst_nodes"
        batch_size = len(src_nodes)
        nodes = torch.cat([src_nodes, dst_nodes], dim=0)

        text_embedding = self.compute_text_embedding(nodes)
        cut_time_l = np.concatenate([edge_times, edge_times])
        graph_embedding = self.graph_embedding_layer.embedding_nodes(nodes, cut_time_l, n_neighbors=n_neighbors)

        node_embedding = self.compose_text_graph_embedding(text_embedding, graph_embedding)  # (2*batch_size, feat_dim)
        src_node_embedding = node_embedding[:batch_size]
        dst_node_embedding = node_embedding[batch_size:]

        score = self.affinity_score(src_node_embedding, dst_node_embedding).squeeze(dim=0)

        return score

    def compute_text_embedding(self, nodes, is_with_time=True):
        # 计算文本与时间结合的嵌入
        text_embedding = self.node_features[nodes]
        if is_with_time:
            # 若要求向量包含时间编码，则按照add的方式加入
            # TODO 添加concat的方式
            cut_time = self.node_timestamps[nodes].unsqueeze(dim=1)
            time_features = self.time_encoder(cut_time).squeeze(dim=1)
            text_embedding = text_embedding + time_features
        return text_embedding

    def compose_text_graph_embedding(self, text_embedding, graph_embedding):
        """
        text_embedding: (batch_size, text_feature_dim)
        graph_embedding: (batch_size, graph_feature_dim)
        """
        return self.fc(torch.cat(text_embedding, graph_embedding, dim=1))
