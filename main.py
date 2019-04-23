import os
import argparse
from util import Utils
from data_loader import DataLoader
from trainer import Trainer
from evaluator import Evaluator
from timeit import default_timer as timer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for Fake News Detection')

    # Data Related
    parser.add_argument("--train", dest="train", type=str, default='data/fulltrain.csv')
    parser.add_argument("--dev", dest="dev", type=str, default='data/balancedtest.csv')
    parser.add_argument("--test", dest="test", type=str, default='data/test.xlsx', help='Out of domain test set')
    parser.add_argument("--pte", dest="pte", type=str, default='', help='Pre-trained embeds')

    # Hyper-parameters
    parser.add_argument("--freq_cutoff", dest="freq_cutoff", type=int, default=2)
    parser.add_argument("--emb_dim", dest="emb_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=100)
    parser.add_argument("--node_emb_dim", dest="node_emb_dim", type=int, default=32)
    parser.add_argument("--filters", dest="filters", type=int, default=100)
    parser.add_argument("--kernel_size", dest="kernel_size", type=int, default=3)
    parser.add_argument("--max_epochs", dest="max_epochs", type=int, default=20)
    parser.add_argument("--max_sent_len", dest="max_sent_len", type=int, default=500)
    parser.add_argument("--max_sents_in_a_doc", dest="max_sents_in_a_doc", type=int, default=10000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.2)
    parser.add_argument("--ntags", dest="ntags", type=int, default=2)
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--encoder", dest="encoder", type=int, default=2, help='0: LSTM encoder for text,'
                                                                               '1: CNN encoder for text'
                                                                               '2: GCN encoder for text'
                                                                               '3: GCN + attention'
                                                                               '4: GAT'
                                                                               '5: GAT with 2 attn heads')
    parser.add_argument("--config", dest="config", type=str, default='lstm_no_pte', help='Name for saving plots')
    parser.add_argument("--model_file", dest="model_file", type=str, default='model_gat_adj_latest.t7', help='For '
                                                                                'evaluating a saved model')
    parser.add_argument("--plot", dest="plot", type=int, default=0, help='set to plot attn')
    parser.add_argument("--use_ss", dest="use_ss", type=int, default=0, help='use ss model')
    parser.add_argument("--mode", dest="mode", type=int, default=0, help='0: train, 1:test')
    if not os.path.exists("models/"):
        os.makedirs("models/")
    return parser.parse_args()


def main():
    params = parse_arguments()
    s_t = timer()
    dl = DataLoader(params)
    u = Utils(params, dl)

    if params.mode == 0:
        # Start training
        trainer = Trainer(params, u)
        trainer.log_time['data_loading'] = timer() - s_t
        trainer.train()
        print(trainer.log_time)
        print("Total time taken (in seconds): {}".format(timer() - s_t))

    elif params.mode == 1:
        # Evaluate on the test set
        evaluator = Evaluator(params, u, dl)
        evaluator.evaluate()

    else:
        # Nothing implemented yet
        pass


if __name__ == '__main__':
    main()
