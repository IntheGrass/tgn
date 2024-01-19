import argparse
import pickle

import torch
import numpy as np
import math
import time
from pathlib import Path

from model.paper_recommendation import PrModel
from model.tgn import TGN
from evaluation.evaluation import eval_pr_edge_prediction
from utils.logger import setup_logger
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_diff_neighbor_finder
from utils.data_processing import compute_time_statistics
from pr.loader import load_data, load_nodes_meta

torch.manual_seed(0)
np.random.seed(0)

# parse params
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit or aan or dblp)',
                    default='aan')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='pr', help='Prefix to name the checkpoints')
parser.add_argument('--neigh_finder', type=str, default="all", choices=["all", "in", "out"],
                    help='The type of neighbor finder')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--loss', type=str, default="bce", choices=[
    "bce", "margin"], help='Type of loss function')
parser.add_argument('--time-scale', type=int, default=1, help='Timestamp scaling factor')

parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_text', action='store_true',
                    help='Whether to use text embedding')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time", "sage"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=768, help='Dimensions of the memory for '
                                                                'each user')  # TODO 降低dim
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--resume_epoch', type=int, default=0, help='指定断点续传时所用的模型epoch'),
parser.add_argument('--interval', type=int, default=1, help='The interval to save checkpoint')

args = parser.parse_args()

# set global params
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

# paths
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
WHOLE_MODEL_SAVE_PATH = f'./saved_models/whole-{args.prefix}-{args.data}.pth'


def get_checkpoint_path(epoch):
    return f'./saved_checkpoints/{args.prefix}-{args.data}-{str(epoch)}.pth'


def get_loss_function(loss_type):
    if loss_type == "margin":
        return torch.nn.MarginRankingLoss(margin=1.0)
    elif loss_type == "bce":
        return torch.nn.BCELoss()
    return torch.nn.BCELoss()  # 默认交叉熵损失


def main():
    # set up logger
    logger = setup_logger()

    node_features, edge_features, full_data, train_data, val_data, test_data = load_data(DATA,
                                                                                         time_scale=args.time_scale)
    _, _, node_timestamps = load_nodes_meta(DATA)

    # 初始化邻居采样器
    train_ngh_finder = get_diff_neighbor_finder(train_data, args.uniform, finder_type=args.neigh_finder)
    full_ngh_finder = get_diff_neighbor_finder(full_data, args.uniform, finder_type=args.neigh_finder)

    # Initialize negative samplers
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=0)
    test_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=1)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    model = PrModel(train_ngh_finder, node_features, node_timestamps, device,
                    n_layers=NUM_LAYER, n_heads=NUM_HEADS, dropout=DROP_OUT,
                    use_memory=USE_MEMORY, n_neighbors=NUM_NEIGHBORS,
                    message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                    embedding_module_type=args.embedding_module, message_function=args.message_function,
                    aggregator_type=args.aggregator, memory_updater_type=args.memory_updater,
                    use_text=args.use_text)

    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(get_checkpoint_path(args.resume_epoch-1)))


    criterion = get_loss_function(args.loss)
    logger.info(f"current loss: {str(criterion)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []
    Path("results/").mkdir(parents=True, exist_ok=True)
    results_path = f"results/{args.prefix}-{DATA}.pkl"

    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()

        if USE_MEMORY:
            model.graph_embedding_layer.memory.__init_memory__()

        # Train using only training graph
        model.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for batch_idx in range(0, num_batch):
            optimizer.zero_grad()

            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                train_data.destinations[start_idx:end_idx]
            edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]

            size = len(sources_batch)

            _, negatives_batch = train_rand_sampler.sample(size)  # 纯随机地选取源节点与目标节点，不区分是否是真实存在的边

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
                neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)

            model = model.train()

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                  timestamps_batch, n_neighbors=NUM_NEIGHBORS,
                                                                  is_sigmoid=False)

            # 损失后向传播
            if args.loss == "margin":
                loss = criterion(pos_prob.squeeze(), neg_prob.squeeze(), pos_label)
            else:
                pos_prob, neg_prob = pos_prob.sigmoid(), neg_prob.sigmoid()
                loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            if USE_MEMORY:
                model.graph_embedding_layer.memory.detach_memory()  # 我猜测是由于memory和message更新，所以每一轮都要重新detach

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        # 在验证集中评估训练到当前batch的性能
        # Validation uses the full graph
        model.set_neighbor_finder(full_ngh_finder)  # 因为该邻居检索器只会获取timestamp小于当前值的邻居，因此使用全数据也能保证邻居不会包含理应未知的测试边

        val_ap, val_auc = eval_pr_edge_prediction(model=model,
                                                  negative_edge_sampler=val_rand_sampler,
                                                  data=val_data,
                                                  n_neighbors=NUM_NEIGHBORS)

        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val ap: {}, val auc: {}'.format(val_ap, val_auc))

        if (epoch + 1) % args.interval == 0:
            torch.save(model.state_dict(), get_checkpoint_path(epoch))  # save checkpoint

    # 评估测试集结果
    model.set_neighbor_finder(full_ngh_finder)
    test_ap, test_auc = eval_pr_edge_prediction(model=model,
                                                negative_edge_sampler=test_rand_sampler,
                                                data=test_data,
                                                n_neighbors=NUM_NEIGHBORS)

    logger.info(
        'Test statistics: auc: {}, ap: {}'.format(test_auc, test_ap))
    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "test_ap": test_ap,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))
    logger.info('Saving TGN model')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.save(model, WHOLE_MODEL_SAVE_PATH)
    logger.info('TGN model saved')


if __name__ == '__main__':
    main()
