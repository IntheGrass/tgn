import math

import torch
from tqdm import tqdm
import numpy as np
import argparse

from model.mlp_predictor import MlpPredictor
from pr.loader import load_data
from pr.metrics import Metrics
from utils.data_processing import Data, build_test_dict

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser('TGN self-supervised pr model evaluation')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit or aan or dblp)',
                    default='aan')
parser.add_argument('--model-path', type=str, default='./saved_models/mlp-aan.pth',
                    help='path to the trained model')
parser.add_argument('--test-size', type=int, default=0,
                    help='Test set size for evaluation. If less than or equal to 0, use all test sets')
parser.add_argument('--bs', type=int, default=200, help='Batch_size for test')

args = parser.parse_args()

# set global params
GPU = 0
BATCH_SIZE = args.bs
DATA = args.data
TEST_SIZE = args.test_size

MODEL_SAVE_PATH = args.model_path


def eval_pr(model: MlpPredictor, train_data: Data, test_data: Data):
    train_nodes = np.arange(1, max(train_data.unique_nodes) + 1)
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
                timestamps = np.repeat(timestamp, len(destination_nodes))

                pred_score = model(source_nodes, destination_nodes)

                scores.extend(pred_score.view(-1).tolist())  # 展开为一维再添加

            sorted_indices = np.argsort(scores)[::-1]  # 评分
            ranked_recommendation = train_nodes[sorted_indices]
            metrics.add(ranked_recommendation, test_set[test_node]["positive"])  # 计算指标

    metrics.printf()  # 打印结果


if __name__ == '__main__':
    node_features, edge_features, full_data, train_data, test_data = load_data(DATA, is_split_val_test=False)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    model = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device_string))
    model.device = device
    model.to(device)

    eval_pr(model, train_data, test_data)
