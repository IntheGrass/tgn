import argparse
import math
import time
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model.mlp_predictor import MlpPredictor, MlpTimePredictor, WeightedMlpAttnPredictor
from pr.loader import load_data, load_nodes_meta
from utils.logger import setup_logger
from utils.utils import get_neighbor_finder, RandEdgeSampler

torch.manual_seed(0)
np.random.seed(0)

# parse params
parser = argparse.ArgumentParser('self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit or aan or dblp)',
                    default='aan')
parser.add_argument('--bs', type=int, default=512, help='Batch_size')
parser.add_argument('--n-epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--prefix', type=str, default='mlp', help='Prefix to name the checkpoints')
parser.add_argument('--feat-dim', type=int, default=768, help='Dimensions of the node')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--layer-type', type=str,
                    help='The layer to calculate similarity. optional: merge, distance, attn', default='mlp')
parser.add_argument('--weighted-time', action='store_true', help='Whether to use weighted time attn score')
parser.add_argument('--use-time', action='store_true', help='Whether to use time features as input')
parser.add_argument('--use-time-type', type=str, help='how to use time features. optional: add, concat',
                    default='add')
args = parser.parse_args()

# set global params
BATCH_SIZE = args.bs
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
GPU = 0
DATA = args.data
FEAT_DIM = args.feat_dim
LEARNING_RATE = args.lr

# paths
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'


def eval_edge_prediction(model, negative_edge_sampler, data, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]

            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_pos_neg_edge_probabilities(sources_batch, destinations_batch,
                                                                        negatives_batch)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def main():
    # set up logger
    logger = setup_logger()

    node_features, edge_features, full_data, train_data, val_data, test_data = load_data(DATA)
    _, timestamps = load_nodes_meta(DATA)

    # Initialize negative samplers
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=0)
    test_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=1)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    if args.weighted_time:
        logger.info(f"use predictor with weighted time score")
        model = WeightedMlpAttnPredictor(FEAT_DIM, node_features, timestamps, device)
    elif args.use_time:
        logger.info(f"use predictor with time: {args.use_time_type}")
        model = MlpTimePredictor(FEAT_DIM, node_features, timestamps, device, time_type=args.use_time_type,
                                 layer_type=args.layer_type)
    else:
        model = MlpPredictor(FEAT_DIM, node_features, device, layer_type=args.layer_type)

    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    logger.info(f"current loss function: {str(criterion)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))

    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for batch_idx in range(0, num_batch):
            optimizer.zero_grad()

            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                train_data.destinations[start_idx:end_idx]
            # timestamps_batch = train_data.timestamps[start_idx:end_idx]

            size = len(sources_batch)

            _, negatives_batch = train_rand_sampler.sample(size)  # 纯随机地选取源节点与目标节点，不区分是否是真实存在的边

            model = model.train()
            pos_score, neg_score = model.compute_pos_neg_edge_probabilities(sources_batch, destinations_batch,
                                                                            negatives_batch)
            ys = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_score.squeeze(), neg_score.squeeze(), ys)
            # 后向传播
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

        # end one epoch
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        # 在验证集中评估训练到当前epoch的性能
        val_ap, val_auc = eval_edge_prediction(model, val_rand_sampler, val_data)
        val_aps.append(val_ap)

        train_losses.append(np.mean(m_loss))
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val ap: {}, val auc: {}'.format(val_ap, val_auc))

    # finish train
    test_ap, test_auc = eval_edge_prediction(model, test_rand_sampler, test_data)
    logger.info(
        'Test statistics: auc: {}, ap: {}'.format(test_auc, test_ap))
    logger.info('Saving TGN model')
    torch.save(model, MODEL_SAVE_PATH)
    logger.info('TGN model saved')


if __name__ == '__main__':
    main()

