from .common import *
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import defaultdict
import itertools
import re

words2 = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["gal", "guy"], ["female", "male"]]
words4 = [["woman", "man"], ["women", "men"], ["girl", "boy"], ["girls", "boys"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]
words3 = ["woman", "man", "women", "men", "girl", "boy", "girls", "boys", "she", "he", "mother", "father", "daughter", "son", "gal", "guy", "female", "male", "her", "his", "herself", "himself", "Mary", "John"]

class SST2Processor(DataProcessor):
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
        frame = pd.read_csv(os.path.join(data_dir, '%s.tsv' % split), sep="\t", header=None)
        text = list(frame.iloc[:, 0])
        labels = list(frame.iloc[:, 1])
        examples = []
        s_a = dict()
        s_b = dict()
        text_new = []
        labels_new = []
        all_pairs = defaultdict(lambda: defaultdict(list))
        for i in range(len(text)):
            t = text[i]
            match, all_data = self.template(t, all_pairs)
            if match:
                text_new.append(t)
                labels_new.append(labels[i])
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
            example = InputExample(guid=key, text=text_new[key], text_a=s_a[key], text_b=s_b[key], label=labels_new[key])
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, 'test')

    def get_labels(self):
        return [0, 1]

    def template(self, text, all_pairs):
        for i, (female, male) in enumerate(words4):
            sent_list = text.lower().split(' ')
            sent_list_f = []
            sent_list_m = []
            match = 0
            for j in range(len(sent_list)):
                if sent_list[j] in words3:
                    match = 1
                    sent_list_f.append(female)
                    sent_list_m.append(male)
                else:
                    sent_list_f.append(sent_list[j])
                    sent_list_m.append(sent_list[j])
            if match:
                sent_f = ' '.join(sent_list_f)
                sent_m = ' '.join(sent_list_m)
                all_pairs[i]['f'].append(sent_f)
                all_pairs[i]['m'].append(sent_m)
        return match, all_pairs

    def match(self, text):
        sent_list = text.lower().split(' ')
        for j in range(len(sent_list)):
            if sent_list[j] in words3:
                sent = ' '.join(sent_list)
                return sent


class SST2Dataset(Dataset):
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
