import torch
import numpy as np

from model.time_encoding import TimeEncode
from utils.utils import MergeLayer, DistanceLayer


class MlpPredictor(torch.nn.Module):
    def __init__(self, feat_dim, node_features, device, layer_type="merge"):
        super(MlpPredictor, self).__init__()
        assert feat_dim == node_features.shape[1], "feat_dim don't match raw node feature dim"
        self.node_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.node_feature_dim = feat_dim
        self.device = device
        self.affinity_score = get_score_layer(self.node_feature_dim, self.node_feature_dim, layer_type)

    def forward(self, source_nodes, destination_nodes):
        # 返回source到dst边成立的可能性分数
        source_node_embeddings = self.node_features[source_nodes]
        destination_node_embeddings = self.node_features[destination_nodes]
        score = self.affinity_score(source_node_embeddings, destination_node_embeddings)
        return score

    def compute_pos_neg_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes):
        assert len(source_nodes) == len(destination_nodes) == len(negative_nodes), "The number of nodes is not matched"

        n_sample = len(source_nodes)
        all_src_nodes = np.tile(source_nodes, 2)
        all_dst_nodes = np.concatenate((destination_nodes, negative_nodes))

        score = self(all_src_nodes, all_dst_nodes)
        pos_score, neg_score = score[:n_sample], score[n_sample:]

        return pos_score, neg_score


class MlpTimePredictor(MlpPredictor):
    def __init__(self, feat_dim, node_features, timestamps, device, layer_type="merge", time_type="add", time_dim=768):
        super(MlpTimePredictor, self).__init__(feat_dim, node_features, device)
        self.timestamps = timestamps
        self.time_feature_dim = time_dim
        self.time_type = time_type  # 决定时序向量与节点向量的组合形式，可选值: add, concat
        self.time_encoder = TimeEncode(dimension=self.time_feature_dim)
        if self.time_type == "concat":
            input_dim = self.node_feature_dim + self.time_feature_dim
        else:
            assert self.node_feature_dim == self.time_feature_dim, "time feature dim don't match node feature dim"
            input_dim = self.node_feature_dim
        self.affinity_score = get_score_layer(input_dim, self.node_feature_dim, layer_type)

    def forward(self, source_nodes, destination_nodes):
        src_time_features = self._calculate_timestamp_features(self.timestamps[source_nodes])
        dst_time_features = self._calculate_timestamp_features(self.timestamps[destination_nodes])

        if self.time_type == "add":
            source_node_embeddings = self.node_features[source_nodes] + src_time_features
            destination_node_embeddings = self.node_features[destination_nodes] + dst_time_features
        else:
            source_node_embeddings = torch.concat(
                [self.node_features[source_nodes], src_time_features], dim=1)
            destination_node_embeddings = torch.concat(
                [self.node_features[destination_nodes], dst_time_features], dim=1)
        score = self.affinity_score(source_node_embeddings, destination_node_embeddings)
        return score

    def _calculate_timestamp_features(self, timestamps):
        # input timestamps shape: (batch_size, )
        timestamps = torch.from_numpy(timestamps).float().to(self.device)  # to gpu/cpu
        timestamps = torch.unsqueeze(timestamps, dim=1)  # (batch_size, ) -> (batch_size, 1)

        time_features = self.time_encoder(timestamps)  # (batch_size, 1, time_feature_dim)
        time_features = torch.squeeze(time_features, dim=1)  # (batch_size, time_feature_dim)
        return time_features


def get_score_layer(input_dim, feat_dim, layer_type="merge"):
    layer_type = layer_type.lower()
    if layer_type == "distance":
        return DistanceLayer(input_dim, feat_dim)
    elif layer_type == "merge":
        return MergeLayer(input_dim, input_dim, feat_dim, 1)
    else:
        raise Exception(f"unsupported layer type: {layer_type}")
