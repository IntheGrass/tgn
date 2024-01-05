import math
import os.path

import torch
from tqdm import tqdm
import numpy as np
import argparse

from model.tgn import TGN
from pr.loader import load_data
from pr.metrics import Metrics
from utils.data_processing import Data, compute_time_statistics, build_test_dict
from utils.utils import get_neighbor_finder

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser('TGN self-supervised pr model evaluation')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit or aan or dblp)',
                    default='aan')
parser.add_argument('--prefix', type=str, default='pr-tgn-attn',
                    help='path to the trained model')
parser.add_argument('--test-size', type=int, default=0,
                    help='Test set size for evaluation. If less than or equal to 0, use all test sets')
parser.add_argument('--n_epoch', type=int, default=0, help='The epochs of checkpoint')
parser.add_argument('--bs', type=int, default=200, help='Batch_size for test')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--time-scale', type=int, default=1, help='Timestamp scaling factor')

args = parser.parse_args()

# set global params
GPU = args.gpu
BATCH_SIZE = args.bs
DATA = args.data
NUM_NEIGHBORS = args.n_degree
TEST_SIZE = args.test_size

MODEL_NAME = f"{args.prefix}-{args.data}"
WHOLE_MODEL_SAVE_PATH = os.path.join("./saved_models", f"whole-{MODEL_NAME}.pth")


def get_model(device):
    model = torch.load(WHOLE_MODEL_SAVE_PATH, map_location=device)
    checkpoint_epoch = args.n_epoch - 1  # checkpoint的epoch从0开始计数
    if checkpoint_epoch >= 0:
        # 加载指定epoch下的模型
        checkpoint_path = os.path.join("./saved_checkpoints", f"{MODEL_NAME}-{str(checkpoint_epoch)}.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"load checkpoint: {checkpoint_path}")
    return model


def eval_pr(model: TGN, train_data: Data, test_data: Data):

    train_nodes = np.arange(1, max(train_data.unique_nodes)+1)
    test_set = build_test_dict(test_data)

    batch_size = min(BATCH_SIZE, len(train_nodes))
    batch_num = math.ceil(len(train_nodes) / batch_size)

    metrics = Metrics()

    with torch.no_grad():
        model = model.eval()
        total = TEST_SIZE if TEST_SIZE > 0 else len(test_set)
        for i, test_node in tqdm(enumerate(test_set), total=total, desc="eval model"):
            if i >= total:
                break
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
            metrics.add(ranked_recommendation, test_set[test_node]["positive"])  # 计算指标

    metrics.printf()  # 打印结果


def main():
    node_features, edge_features, full_data, train_data, test_data = load_data(DATA, is_split_val_test=False,
                                                                               time_scale=args.time_scale)

    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    tgn = get_model(device)

    tgn.set_neighbor_finder(full_ngh_finder)
    tgn.device = device
    tgn.embedding_module.device = device
    tgn.to(device)

    eval_pr(tgn, train_data, test_data)


if __name__ == '__main__':
    main()
