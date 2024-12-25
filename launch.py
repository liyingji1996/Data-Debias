from __future__ import absolute_import, division, print_function

import logging
import os
import random
import utils.utils as my_utils
import utils.model_loader as model_loader
import argparse

import numpy as np
import torch
from transformers import BertForSequenceClassification, AutoModelForMaskedLM

from hiex import SamplingAndOcclusionExplain
from poe import MiD
from int_gradient import ING, write_features

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))


logger = logging.getLogger(__name__)


def get_hardware_setting(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return device, n_gpu


def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main_ing(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device, n_gpu = get_hardware_setting(args)
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    assert args.gradient_accumulation_steps >= 1, 'Invalid gradient_accumulation_steps parameter, should be >= 1'
    assert args.do_train ^ args.do_eval, 'Activate either do_train or do_eval at a time'
    set_random_seed(args)

    tokenizer, processor = model_loader.get_processors(args)
    label_list = processor.get_labels()
    ing = ING(args, logger=logger, tokenizer=tokenizer, processor=processor, num_labels=len(label_list))
    if args.model_name_or_path == 'roberta-base':
        f_train = os.path.join("runs/{}/train_roberta".format(args.task_name), "train_data.pkl")
        f_eval = os.path.join("runs/{}/train_roberta".format(args.task_name), "eval_data.pkl")
    elif args.model_name_or_path == 'gpt2':
        f_train = os.path.join("runs/{}/train_gpt2".format(args.task_name), "train_data.pkl")
        f_eval = os.path.join("runs/{}/train_gpt2".format(args.task_name), "eval_data.pkl")
    else:
        f_train = os.path.join("runs/{}/train_tiny-bert".format(args.task_name), "train_data.pkl")
        f_eval = os.path.join("runs/{}/train_tiny-bert".format(args.task_name), "eval_data.pkl")

    if not (os.path.exists(f_train) and os.path.exists(f_eval)):
        train_features, train_examples = my_utils.load_text_as_feature(args, processor, tokenizer, 'train')
        eval_features, eval_examples = my_utils.load_text_as_feature(args, processor, tokenizer, 'eval')
        write_features(train_features, f_train)
        write_features(eval_features, f_eval)
    ing.load_data(f_train, f_eval)
    logger.info('\n')
    logger.info('='*my_utils.MAX_LINE_WIDTH)
    ing.train()
    if args.find_words:
        if not args.add_jsd:
            target_dir = args.output_dir + '_best-f1'
            args.load_path = os.path.join(target_dir, 'ck.pth')
            ing.load_model(args.load_path)
        ing.find_words()

def config_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--ing", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="vanilla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha_1", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.3)

    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--add_jsd", type=bool, default=False)
    parser.add_argument("--debias", type=bool, default=False)
    parser.add_argument("--find_words", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=5.0e-05)
    parser.add_argument("--decay_rate", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--fw_batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=20)

    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--early_stop",  type=int, default=5)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # directories
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="runs/sst2-train_tiny-bert")
    parser.add_argument("--data_dir", type=str, default="data/sst2/")
    parser.add_argument("--model_name_or_path", type=str, help="choose from prajjwal1/bert-tiny, gpt2, distilbert-base-uncased, bert-base-uncased, roberta-base")
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--do_lower_case", type=bool, default=True)
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--neg_words_file", type=str, default="")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--bias_direction", type=str, default="")
    parser.add_argument("--projection_matrix", type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    _args = config_args()
    if _args.ing:
        output_dir = _args.output_dir
        # 原始训练或去偏训练
        if _args.debias or not _args.add_jsd:
            logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d %H:%M:%S',
                                level=logging.INFO if _args.local_rank in [-1, 0] else logging.WARN)
            logger.info('=' * my_utils.MAX_LINE_WIDTH)
            logger.info('{}'.format(_args))

            main_ing(_args)

        # 增强bias训练
        else:
            for i in range(_args.iterations):
                logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s -   %(message)s',
                                    datefmt='%m/%d %H:%M:%S',
                                    level=logging.INFO if _args.local_rank in [-1, 0] else logging.WARN)
                logger.info('=' * my_utils.MAX_LINE_WIDTH)
                logger.info('{}'.format(_args))
                if _args.task_name == 'bios':
                    bias_acc = main_ing(_args)
                else:
                    _args.output_dir = output_dir + '_iterations_{}'.format(i)
                    bias_acc = main_ing(_args)
                    _args.bias_acc = bias_acc
                    _args.neg_words_file = os.path.join(_args.output_dir, "token_and_score.csv")
