import torch
import numpy as np

from utils.utils import MergeLayer


class MlpPredictor(torch.nn.Module):
    def __init__(self, feat_dim, node_features, device):
        super(MlpPredictor, self).__init__()
        assert feat_dim == node_features.shape[1], "feat_dim don't match raw node feature dim"
        self.node_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.n_node_features = feat_dim
        self.device = device
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    def forward(self, source_nodes, destination_nodes):
        # 返回source到dst边成立的可能性分数
        source_node_embeddings = self.node_features[source_nodes]
        destination_node_embeddings = self.node_features[destination_nodes]
        score = self.affinity_score(source_node_embeddings, destination_node_embeddings)
        return score

    def compute_pos_neg_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes):
        assert len(source_nodes) == len(destination_nodes) == len(negative_nodes), "The number of nodes is not matched"

        n_sample = len(source_nodes)
        source_node_embeddings = self.node_features[source_nodes]
        destination_node_embeddings = self.node_features[destination_nodes]
        negative_node_embeddings = self.node_features[negative_nodes]
        score = self.affinity_score(torch.cat([source_node_embeddings, source_node_embeddings], dim=0),
                                    torch.cat([destination_node_embeddings, negative_node_embeddings], dim=0))
        pos_score, neg_score = score[:n_sample], score[n_sample:]

        return pos_score, neg_score

