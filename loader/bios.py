from .common import *
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
import random
import pandas as pd
from collections import defaultdict
import itertools
import re

words4 = [["woman", "man"], ["women", "men"], ["girl", "boy"], ["girls", "boys"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["mary", "john"]]


class BiosProcessor(DataProcessor):
    """
    Data processor using DataProcessor class provided by BERT
    """
    def __init__(self, configs, tokenizer=None):
        super().__init__()
        self.data_dir = configs.data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = configs.max_seq_length
        self.configs = configs

    def _create_examples(self, data_dir, split):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        frame = pd.read_csv(os.path.join(data_dir, '%s.csv' % split), encoding='utf-8')
        bio_text_pronouns = list(frame['bio_text_pronoun'].values)
        bio_text = list(frame['bio_text'].values)
        names = []
        s_a = dict()
        s_b = dict()
        for bio in bio_text:
            name = self.get_name_coms(bio)
            names.append(name)
        labels = list(frame['job'].values)
        with open(os.path.join(data_dir, "prof2ind.json")) as json_file:
            mapping = json.load(json_file)
        examples = []
        all_pairs = defaultdict(lambda: defaultdict(list))
        for i in range(len(bio_text_pronouns)):
            bio = bio_text_pronouns[i]
            all_data = self.template(bio, all_pairs, names[i])
        for group_id in all_data:
            def_group = all_data[group_id]
            f_sents = def_group['f']
            m_sents = def_group['m']
            for sent_id, (f_sent, m_sent) in enumerate(zip(f_sents, m_sents)):
                if sent_id in s_a.keys():
                    s_a[sent_id].append(f_sent)
                    s_b[sent_id].append(m_sent)
                else:
                    s_a[sent_id] = [f_sent]
                    s_b[sent_id] = [m_sent]
        for key in s_a.keys():
            example = InputExample(guid=key, text=bio_text_pronouns[key].lower(), text_a=s_a[key], text_b=s_b[key], label=mapping[labels[key]])
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, 'test')

    def get_labels(self):
        with open(os.path.join(self.data_dir, "prof2ind.json")) as json_file:
            mapping = json.load(json_file)
        return mapping

    def template(self, bio, all_pairs, names):
        for i, (female, male) in enumerate(words4):
            sent_f = None
            for name in names:
                sent_f = bio.replace(name, female)
            if sent_f:
                sent_list_f = sent_f.lower().split(' ')
            else:
                sent_list_f = bio.lower().split(' ')
            for j in range(len(sent_list_f)):
                if sent_list_f[j] == 'she' or sent_list_f[j] == 'he' or sent_list_f[j] == 'her' or sent_list_f[j] == 'his':
                    sent_list_f[j] = female
            sent_f = ' '.join(sent_list_f)

            sent_m = None
            for name in names:
                sent_m = bio.replace(name, male)
            if sent_m:
                sent_list_m = sent_m.lower().split(' ')
            else:
                sent_list_m = bio.lower().split(' ')
            for j in range(len(sent_list_m)):
                if sent_list_m[j] == 'she' or sent_list_m[j] == 'he' or sent_list_m[j] == 'her' or sent_list_m[j] == 'his':
                    sent_list_m[j] = male
            sent_m = ' '.join(sent_list_m)
            all_pairs[i]['f'].append(sent_f)
            all_pairs[i]['m'].append(sent_m)
        return all_pairs

    def get_name_coms(self, bio):
        names = []
        cand_names = bio.split()[:4]
        cand_names = [each for each in cand_names if each.istitle()]
        cand_names1 = itertools.combinations(cand_names, 2)
        cand_names2 = itertools.combinations(cand_names, 3)
        for each in cand_names1:
            if ' '.join(each) not in names:
                names.append(' '.join(each))
        for each in cand_names2:
            if ' '.join(each) not in names:
                names.append(' '.join(each))
        names.extend(cand_names)
        names = sorted(names, key=lambda a:len(a))
        return names


class BiosDataset(Dataset):
    """
    torch.utils.Dataset instance for building torch.utils.DataLoader, for training the language model.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)
