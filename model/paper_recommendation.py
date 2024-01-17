from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from model.my_tgn import MyTgn
from model.time_encoding import TimeEncode
from utils.utils import MergeLayer


class IPaperRecommendation(ABC):
    @abstractmethod
    def predict_edge_probabilities(self, src_nodes, dst_nodes, edge_times, n_neighbors=20):
        pass


class PrModel(nn.Module, IPaperRecommendation):
    def __init__(self, neighbor_finder, node_features, node_timestamps, device,
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=True, n_neighbors=10,
                 message_dimension=100, memory_dimension=500,
                 embedding_module_type="sage", message_function="identity", aggregator_type="last",
                 memory_updater_type="gru", use_text=True):
        super(PrModel, self).__init__()
        IPaperRecommendation.__init__(self)
        self.use_text = use_text  # 是否结合文本向量
        self.neighbor_finder = neighbor_finder
        self.node_features = torch.from_numpy(node_features).float().to(device)
        self.node_timestamps = torch.from_numpy(node_timestamps).float().to(device)

        self.node_feature_dim = self.node_features.shape[1]
        self.text_feature_dim = self.node_feature_dim  # 文本嵌入输出的向量维度
        self.graph_feature_dim = self.node_feature_dim  # 图嵌入输出的向量维度

        # modules
        self.time_encoder = TimeEncode(dimension=self.node_feature_dim)

        self.graph_embedding_layer = MyTgn(neighbor_finder, node_features, device, self.time_encoder, n_layers,
                                           n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                           n_neighbors=n_neighbors,
                                           message_dimension=message_dimension, memory_dimension=memory_dimension,
                                           embedding_module_type=embedding_module_type,
                                           message_function=message_function,
                                           aggregator_type=aggregator_type, memory_updater_type=memory_updater_type)
        if self.use_text:
            self.fc = nn.Linear(self.text_feature_dim + self.graph_feature_dim, self.node_feature_dim)  # 结合文本向量与图向量

        self.affinity_score = MergeLayer(self.node_feature_dim, self.node_feature_dim,
                                         self.node_feature_dim,
                                         1)

    def set_neighbor_finder(self, ngh_finder):
        self.graph_embedding_layer.set_neighbor_finder(ngh_finder)

    def forward(self, src_nodes, dst_nodes, neg_nodes, edge_times, n_neighbors=20, is_sigmoid=True):
        self.compute_edge_probabilities(src_nodes, dst_nodes, neg_nodes, edge_times,
                                        n_neighbors=n_neighbors, is_sigmoid=is_sigmoid)

    def compute_edge_probabilities(self, src_nodes, dst_nodes, neg_nodes, edge_times,
                                   edge_idxs=None, n_neighbors=20, is_sigmoid=True):
        # 训练时调用，会修改memory
        batch_size = len(src_nodes)
        assert batch_size == len(dst_nodes) == len(neg_nodes), "batch size is not matched"

        # Note: 必须首先计算不影响memory的负节点嵌入，否则，会导致计算负节点嵌入使用正结点所计算的memory
        neg_graph_embedding = self.graph_embedding_layer.embedding_nodes(neg_nodes, edge_times, n_neighbors=n_neighbors)
        src_graph_embedding, dst_graph_embedding = self.graph_embedding_layer.compute_pair_edge_embedding(
            src_nodes, dst_nodes, edge_times, n_neighbors=n_neighbors)
        graph_embedding = torch.cat([src_graph_embedding, dst_graph_embedding, neg_graph_embedding], dim=0)

        if self.use_text:
            # 结合文本嵌入与图嵌入构建节点嵌入
            nodes = np.concatenate([src_nodes, dst_nodes, neg_nodes])
            text_embedding = self.compute_text_embedding(nodes)

            node_embedding = self.compose_text_graph_embedding(text_embedding, graph_embedding)  # (3*batch_size, feat_dim)
        else:
            node_embedding = graph_embedding

        src_node_embedding = node_embedding[:batch_size]
        dst_node_embedding = node_embedding[batch_size:2 * batch_size]
        neg_node_embedding = node_embedding[2 * batch_size:]
        # predict score
        score = self.affinity_score(torch.cat([src_node_embedding, src_node_embedding], dim=0),
                                    torch.cat([dst_node_embedding, neg_node_embedding])).squeeze(dim=0)  # 合并计算评分

        pos_score = score[:batch_size]
        neg_score = score[batch_size:]

        if not is_sigmoid:
            return pos_score, neg_score

        return pos_score.sigmoid(), neg_score.sigmoid()

    def predict_edge_probabilities(self, src_nodes, dst_nodes, edge_times, n_neighbors=20):
        # 预测时调用，不会影响memory
        assert len(src_nodes) == len(dst_nodes), "The size of src_nodes don't match dst_nodes"
        batch_size = len(src_nodes)
        nodes = np.concatenate([src_nodes, dst_nodes])
        cut_time_l = np.concatenate([edge_times, edge_times])


        graph_embedding = self.graph_embedding_layer.embedding_nodes(nodes, cut_time_l, n_neighbors=n_neighbors)

        # 得到最终的节点嵌入
        if self.use_text:
            # 结合文本与图嵌入构建节点嵌入
            text_embedding = self.compute_text_embedding(nodes)
            node_embedding = self.compose_text_graph_embedding(text_embedding,
                                                               graph_embedding)  # (2*batch_size, feat_dim)
        else:
            node_embedding = graph_embedding

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
        raw_embedding = torch.cat([text_embedding, graph_embedding], dim=1)
        return self.fc(raw_embedding)
