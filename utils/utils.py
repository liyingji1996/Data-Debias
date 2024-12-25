from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loader import convert_examples_to_features
from collections import Counter
import re
import os
import torch
import json
import yaml
import numpy as np
import string
import pandas as pd
import traceback

CONFIG_NAME = "config.json"     # TODO: do multiple config to separate model from framework
WEIGHTS_NAME = "pytorch_model.bin"
PHASE_NAMES = ['normal', 'correcting', 'debiasing']
MAX_LINE_WIDTH = 150


def load_config(pth):
    with open(pth, 'r') as f:
        return yaml.full_load(f)

def load_suppress_words_new(args, pth, tokenizer):
    if pth == '':
        return dict(), dict()
    frame = pd.read_csv(pth)
    word = list(frame['word'].values)
    score = list(frame['score'].values)
    words = dict()
    words_ids = dict()
    num = int(len(word) * args.eta)
    for i in range(num):
        try:
            w = word[i].lower()
            words[w] = score[i]
            # words_ids[tokenizer.vocab[w]] = score[i]
            words_ids[tokenizer.convert_tokens_to_ids(w)] = score[i]
        except AttributeError:
            traceback.print_exc()
            continue
    return words, words_ids

def load_suppress_words(pth, tokenizer, suppress_weighted=False):
    if pth == '':
        return dict(), dict()
    with open(pth) as f:
        words = dict()
        words_ids = dict()
        for line in f.readlines():
            segs = line.strip().split('\t')
            word = segs[0]
            val = float(segs[1]) if suppress_weighted else 1.  # false
            canonical = tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            words[word] = val
            words_ids[tokenizer.vocab[word]] = val
        assert words
    return words, words_ids


def words_count(corpus_a, stop_words_ids):
    words_ids = []
    for t in corpus_a:
        words_ids += [w for w in t if w not in stop_words_ids]
    words_ids_count = Counter(words_ids).most_common()  #
    return words_ids_count, len(words_ids)


def load_text_as_feature(args, processor, tokenizer, dataset, output_mode='classification'):
    valid_choices = ['train', 'test', 'eval']
    assert dataset in valid_choices, 'Invalid dataset is given: [{}], valid choices {}'.format(dataset, valid_choices)
    if dataset == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif dataset == 'eval':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples, args.max_seq_length, tokenizer, output_mode)
    return features, examples


def load_word_from_file(pth):
    with open(pth) as f:
        words = []
        for line in f.readlines():
            segs = line.strip().split('\t')
            word = segs[0]
            words.append(word)
        assert words
    return words


def record_attr_change(soc, input_pack, attr_obj, key='test'):
    target = attr_obj.idx_test_dict if key == 'test' else attr_obj.idx_train_dict
    all_ids_a, all_ids_b, all_mask_a, all_mask_b, all_segments_a, all_segments_b = input_pack  # data feature
    j = 0
    inp_enb_a = []
    inp_enb_mask_a = []
    inp_enb_b = []
    inp_enb_mask_b = []
    inp_ex_a = []
    inp_ex_mask_a = []
    inp_ex_b = []
    inp_ex_mask_b = []
    segment_ids_a = []
    segment_ids_b = []
    for instance_id in target.keys():
        for pos in target[instance_id]:
            x_region = (pos, pos)
            inp_ids = soc.algo.do_attribution(all_ids_a[instance_id], all_ids_b[instance_id], all_mask_a[instance_id],
                                              all_mask_b[instance_id], all_segments_a[instance_id], all_segments_b[instance_id], x_region, additional_mask=target[instance_id])
            inp_enb_a.append(inp_ids[0])
            inp_enb_mask_a.append(inp_ids[1])
            inp_enb_b.append(inp_ids[2])
            inp_enb_mask_b.append(inp_ids[3])
            inp_ex_a.append(inp_ids[4])
            inp_ex_mask_a.append(inp_ids[5])
            inp_ex_b.append(inp_ids[6])
            inp_ex_mask_b.append(inp_ids[7])
            segment_ids_a.append(inp_ids[8])
            segment_ids_b.append(inp_ids[9])
            j += 1
            break
    return [inp_enb_a, inp_enb_mask_a, inp_enb_b, inp_enb_mask_b, inp_ex_a, inp_ex_mask_a, inp_ex_b, inp_ex_mask_b, segment_ids_a, segment_ids_b, j]


def compute_metrics(preds, labels, pred_probs):
    assert len(preds) == len(labels), \
        'Unmatched length between predictions [{}] and ground truth [{}]'.format(len(preds), len(labels))
    return acc_and_f1(preds, labels, pred_probs)


def acc_and_f1(preds, labels, pred_probs):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    p, r = precision_score(y_true=labels, y_pred=preds, average='macro'), recall_score(y_true=labels, y_pred=preds, average='macro')
    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs, average='macro', multi_class='ovr')
    except ValueError:
        roc = 0.
    return {
        "acc": acc, "f1": f1, "precision": p, "recall": r, "auc_roc": roc
    }


class AttrRecord:
    def __init__(self, w, w_id):
        self.w = w
        self.id = w_id

        self.checked_test = False
        self.checked_train = False
        self.idx_test_dict = dict()  # test new words
        self.idx_train_dict = dict()
        self.attr_changes = OrderedDict()   # values are list of attributions for instances
        self.fpr_changes = OrderedDict()    # values are tuples of #FP #TP #FPR

    def record_attr_change(self, steps, attr):
        self.attr_changes[steps] = attr
        print("self.attr_changes[steps]:", self.attr_changes[steps])

    def record_fpr_change(self, steps, attr):
        self.fpr_changes[steps] = attr

    def update_test_dict(self, idx, pos):
        self.checked_test = True
        if idx in self.idx_test_dict:
            self.idx_test_dict[idx].append(pos)
        else:
            self.idx_test_dict[idx] = [pos]

    def update_train_dict(self, idx, pos):
        self.checked_train = True
        if self.idx_train_dict:
            self.idx_train_dict[idx].append(pos)
        else:
            self.idx_train_dict[idx] = [pos]

    def disable_test_dict(self):
        self.idx_test_dict = None

    def disable_train_dict(self):
        self.idx_train_dict = None

    def get_test_idxs(self):
        return self.idx_test_li

    def get_train_idxs(self):
        return self.idx_train_li

    def get_change(self):
        epoch_li = [epoch for epoch in self.attr_changes.keys()]
        attr_li = [self.attr_changes[epoch] for epoch in epoch_li]
        return epoch_li, attr_li


def save_model_new(args, model, epoch, postfix):
    target_dir = args.output_dir + postfix
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    torch.save(model.state_dict(), os.path.join(target_dir, 'ck.pth'))
    model.config.to_json_file(os.path.join(target_dir, 'epoch_{}_config.json'.format(epoch)))
    if args.do_train:
        f = open(os.path.join(target_dir, 'args.json'), 'w')
        json.dump(args.__dict__, f, indent=4)
        f.close()


def seconds2hms(s, get_int=True):
    h = s//3600
    m = (s % 3600) // 60
    s = s % 60
    if get_int:
        return int(h), int(m), int(s)
    return h, m, s


def get_args_diff(d_arg_yaml, d_arg):
    diff = dict()
    for k in d_arg:
        if d_arg[k] != d_arg_yaml[k]:
            diff[k] = d_arg[k]
    return diff


def heading(msg):
    remains = MAX_LINE_WIDTH - len(msg) - 2
    return '|' + ' '*(remains // 2) + msg + ' '*(remains // 2 + remains % 2) + '|'


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass
