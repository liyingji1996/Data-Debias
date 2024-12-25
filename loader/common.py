import json, os, csv, sys, logging
import torch
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)

class DotDict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

def dotdict_collate(batch):
    return DotDict(**default_collate(batch))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid  # unused
        self.text = text
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DualInputFeatures(object):
    """A Dual set of features of data."""

    def __init__(self, guid, input_ids, input_ids_a, input_ids_b, input_mask, input_mask_a, input_mask_b, segment_ids, segment_ids_a, segment_ids_b, label_id, ref_input_ids_a, ref_input_ids_b, mask_labels):
        self.guid = guid
        self.input_ids = input_ids
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask = input_mask
        self.input_mask_a = input_mask_a
        self.input_mask_b = input_mask_b
        self.segment_ids = segment_ids
        self.segment_ids_a = segment_ids_a
        self.segment_ids_b = segment_ids_b
        self.label_id = label_id
        self.ref_input_ids_a = ref_input_ids_a
        self.ref_input_ids_b = ref_input_ids_b
        self.mask_labels = mask_labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def convert_examples_to_features(examples, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        mask_labels = input_ids + [-100] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        input_ids_a_ = []
        input_mask_a_ = []
        segment_ids_a_ = []
        ref_input_ids_a_ = []
        input_ids_b_ = []
        input_mask_b_ = []
        segment_ids_b_ = []
        ref_input_ids_b_ = []
        for i in range(len(example.text_a)):
            tokens_a = tokenizer.tokenize(example.text_a[i])
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            # The input_id of all zeros serves as the baseline for the integrated gradient
            ref_tokens_a = [tokenizer.cls_token] + [tokenizer.pad_token] * len(tokens_a) + [tokenizer.sep_token]
            ref_input_ids_a = tokenizer.convert_tokens_to_ids(ref_tokens_a)
            tokens_a = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
            input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
            segment_ids_a = [0] * len(input_ids_a)
            input_mask_a = [1] * len(input_ids_a)
            padding_a = [0] * (max_seq_length - len(input_ids_a))
            input_ids_a += padding_a
            input_mask_a += padding_a
            segment_ids_a += padding_a
            ref_input_ids_a += padding_a

            tokens_b = tokenizer.tokenize(example.text_b[i])
            if len(tokens_b) > max_seq_length - 2:
                tokens_b = tokens_b[:(max_seq_length - 2)]
            # The input_id of all zeros serves as the baseline for the integrated gradient
            ref_tokens_b = [tokenizer.cls_token] + [tokenizer.pad_token] * len(tokens_b) + [tokenizer.sep_token]
            ref_input_ids_b = tokenizer.convert_tokens_to_ids(ref_tokens_b)
            tokens_b = [tokenizer.cls_token] + tokens_b + [tokenizer.sep_token]
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
            segment_ids_b = [0] * len(input_ids_b)
            input_mask_b = [1] * len(input_ids_b)
            padding_b = [0] * (max_seq_length - len(input_ids_b))
            input_ids_b += padding_b
            input_mask_b += padding_b
            segment_ids_b += padding_b
            ref_input_ids_b += padding_b
            assert len(input_ids) == len(input_ids_a) == len(input_ids_b) == max_seq_length
            assert len(input_mask) == len(input_mask_a) == len(input_mask_b) == max_seq_length
            assert len(segment_ids) == len(segment_ids_a) == len(segment_ids_b) == max_seq_length

            input_ids_a_.append(input_ids_a)
            input_mask_a_.append(input_mask_a)
            segment_ids_a_.append(segment_ids_a)
            ref_input_ids_a_.append(ref_input_ids_a)
            input_ids_b_.append(input_ids_b)
            input_mask_b_.append(input_mask_b)
            segment_ids_b_.append(segment_ids_b)
            ref_input_ids_b_.append(ref_input_ids_b)

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            DualInputFeatures(guid=example.guid,
                              input_ids=input_ids,
                              input_ids_a=input_ids_a_,
                              input_ids_b=input_ids_b_,
                              input_mask=input_mask,
                              input_mask_a=input_mask_a_,
                              input_mask_b=input_mask_b_,
                              segment_ids=segment_ids,
                              segment_ids_a=segment_ids_a_,
                              segment_ids_b=segment_ids_b_,
                              label_id=label_id,
                              ref_input_ids_a=ref_input_ids_a_,
                              ref_input_ids_b=ref_input_ids_b_,
                              mask_labels=mask_labels))

    return features

