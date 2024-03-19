import torch
import numpy as np

from model.time_encoding import TimeEncode
from utils.utils import MergeLayer, DistanceLayer, AttnScoreLayer


class MlpPredictor(torch.nn.Module):
    def __init__(self, feat_dim, node_features, device, layer_type="mlp"):
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
    def __init__(self, feat_dim, node_features, timestamps, device, layer_type="mlp", time_type="add", time_dim=768):
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


class WeightedMlpAttnPredictor(MlpTimePredictor):
    def __init__(self, feat_dim, node_features, timestamps, device):
        super(WeightedMlpAttnPredictor, self).__init__(feat_dim, node_features, timestamps, device)
        self.mlp_layer = get_score_layer(self.node_feature_dim, self.node_feature_dim, "mlp")
        self.time_attn_layer = get_score_layer(self.time_feature_dim, self.time_feature_dim, "attn")

        self.affinity_score = torch.nn.Linear(2, 1)

    def forward(self, source_nodes, destination_nodes):
        # 获取输入张量
        src_features = self.node_features[source_nodes]
        dst_features = self.node_features[destination_nodes]
        src_time_features = self._calculate_timestamp_features(self.timestamps[source_nodes])
        dst_time_features = self._calculate_timestamp_features(self.timestamps[destination_nodes])

        # 计算预测评分
        mlp_score = self.mlp_layer(src_features, dst_features)  # (n,1)
        time_attn_score = self.time_attn_layer(src_time_features, dst_time_features)  # (n,1)
        tmp_input = torch.cat((mlp_score, time_attn_score), dim=1)  # (n, 2)
        score = self.affinity_score(tmp_input)  # (n, 1)

        return score


def get_score_layer(input_dim, hidden_dim, layer_type="mlp"):
    layer_type = layer_type.lower()
    if layer_type == "distance":
        return DistanceLayer(input_dim, hidden_dim)
    elif layer_type == "mlp":
        return MergeLayer(input_dim, input_dim, hidden_dim, 1)
    elif layer_type == "attn":
        return AttnScoreLayer(input_dim)
    else:
        raise Exception(f"unsupported layer type: {layer_type}")
