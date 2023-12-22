import math

import torch
import pickle
import numpy as np
import argparse

from model.tgn import TGN
from pr.loader import load_data
from pr.metrics import Metrics
from utils.data_processing import Data, compute_time_statistics
from utils.utils import get_neighbor_finder

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser('TGN self-supervised pr model evaluation')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit or aan or dblp)',
                    default='aan')
parser.add_argument('--model-path', type=str, default='./saved_models/whole-pr-tgn-attn-ann.pth',
                    help='path to the trained model')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')

args = parser.parse_args()

# set global params
GPU = args.gpu
DATA = args.data
NUM_NEIGHBORS = args.n_degree

WHOLE_MODEL_SAVE_PATH = args.model_path

def build_test_dict(test_data: Data):
    test_set = {}

    for (source, dst, timestamp) in zip(test_data.sources, test_data.destinations, test_data.timestamps):
        if source not in test_set:
            test_set[source] = {}

        test_set[source]["timestamp"] = timestamp
        if "positive" not in test_set[source]:
            test_set[source]["positive"] = []
        test_set[source]["positive"].append(dst)

    return test_set


def eval_pr(model: TGN, train_data: Data, test_data: Data):

    train_nodes = np.arange(1, max(train_data.unique_nodes)+1)
    test_set = build_test_dict(test_data)
    # timestamp =

    batch_size = min(500, len(train_nodes))
    batch_num = int(len(train_nodes) / batch_size)
    metrics = Metrics()
    for test_node in test_set:
        scores = []
        timestamp = test_set[test_node]["timestamp"]
        for k in range(batch_num):
            start = k * batch_size
            end = min(start+batch_size, len(train_nodes))
            destination_nodes = train_nodes[start: end]
            source_nodes = np.repeat(test_node, len(destination_nodes))
            edge_indices = np.zeros(len(destination_nodes))
            timestamps = np.repeat(timestamp, len(destination_nodes))

            if model.use_memory:
                backup_memory = model.memory.backup_memory()
            # TODO 由于代码实现的限制，目前的negative_nodes长度必须与source_nodes匹配，占用计算资源，后续想办法优化
            negative_nodes = np.zeros(len(destination_nodes), dtype=int)
            pred_score, _ = model.compute_edge_probabilities(source_nodes, destination_nodes, negative_nodes,
                                                             edge_indices, timestamps, n_neighbors=NUM_NEIGHBORS)
            if model.use_memory:
                model.memory.restore_memory(backup_memory)  # 这一步是为了防止模型记忆测试数据

            scores.extend(pred_score.view(-1).tolist())  # 展开为一维再添加

        sorted_indices = np.argsort(scores)[::-1]  # 评分
        ranked_recommendation = train_nodes[sorted_indices]
        metrics.add(ranked_recommendation, test_set[test_node]["positive"])
    metrics.printf()


def main():
    node_features, edge_features, full_data, train_data, test_data = load_data(DATA, is_split_val_test=False)

    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    tgn = torch.load(WHOLE_MODEL_SAVE_PATH, map_location=torch.device(device_string))
    # tgn = TGN(neighbor_finder=full_ngh_finder, node_features=node_features,
    #           edge_features=edge_features, device=device)

    tgn.set_neighbor_finder(full_ngh_finder)
    tgn.device = device
    tgn.embedding_module.device = device
    tgn.to(device)

    eval_pr(tgn, train_data, test_data)


if __name__ == '__main__':
    main()