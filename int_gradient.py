from __future__ import absolute_import, division, print_function

import os
import utils.utils as my_utils
from tqdm import tqdm
import time
import pickle
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import functional as F
import pandas as pd
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification


class JSD(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=-1)
        net_2_probs = F.softmax(net_2_logits, dim=-1)
        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = (F.kl_div(F.log_softmax(net_1_logits, dim=-1), total_m, reduction=self.reduction) + F.kl_div(F.log_softmax(net_2_logits, dim=-1), total_m, reduction=self.reduction)) * 0.5
        return loss


class FH(nn.Module):
    def __init__(self, model):
        super(FH, self).__init__()
        self.model = model
        self.device_ids = model.device_ids

    def forward(self, input_ids, token_type_ids, attention_mask):
        logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=None, return_dict=True).logits
        return logits


def write_features(fs, file):
    print("--------------write features------------")
    guid = np.array([f.guid for f in fs])
    input_ids = np.array([f.input_ids for f in fs])
    input_ids_a = np.array([f.input_ids_a for f in fs])
    input_ids_b = np.array([f.input_ids_b for f in fs])
    input_mask = np.array([f.input_mask for f in fs])
    input_mask_a = np.array([f.input_mask_a for f in fs])
    input_mask_b = np.array([f.input_mask_b for f in fs])
    segment_ids = np.array([f.segment_ids for f in fs])
    segment_ids_a = np.array([f.segment_ids_a for f in fs])
    segment_ids_b = np.array([f.segment_ids_b for f in fs])
    ref_input_ids_a = np.array([f.ref_input_ids_a for f in fs])
    ref_input_ids_b = np.array([f.ref_input_ids_b for f in fs])
    label_ids = np.array([f.label_id for f in fs])
    mask_labels = np.array([f.mask_labels for f in fs])
    with open(file, 'wb') as wf:
        pickle.dump(guid, wf)
        pickle.dump(input_ids, wf)
        pickle.dump(input_ids_a, wf)
        pickle.dump(input_ids_b, wf)
        pickle.dump(input_mask, wf)
        pickle.dump(input_mask_a, wf)
        pickle.dump(input_mask_b, wf)
        pickle.dump(segment_ids, wf)
        pickle.dump(segment_ids_a, wf)
        pickle.dump(segment_ids_b, wf)
        pickle.dump(ref_input_ids_a, wf)
        pickle.dump(ref_input_ids_b, wf)
        pickle.dump(label_ids, wf)
        pickle.dump(mask_labels, wf)


def unpack_features(file, output_mode='classification'):
    print("------------------read features---------------")
    with open(file, 'rb') as rf:
        guid = pickle.load(rf)
        input_ids = pickle.load(rf)
        input_ids_a = pickle.load(rf)
        input_ids_b = pickle.load(rf)
        input_mask = pickle.load(rf)
        input_mask_a = pickle.load(rf)
        input_mask_b = pickle.load(rf)
        segment_ids = pickle.load(rf)
        segment_ids_a = pickle.load(rf)
        segment_ids_b = pickle.load(rf)
        ref_input_ids_a = pickle.load(rf)
        ref_input_ids_b = pickle.load(rf)
        label_ids = pickle.load(rf)
        mask_labels = pickle.load(rf)

    guid = torch.tensor(guid, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    mask_labels = torch.tensor(mask_labels, dtype=torch.long)
    input_ids_a = torch.tensor(input_ids_a, dtype=torch.long)
    input_ids_b = torch.tensor(input_ids_b, dtype=torch.long)
    input_mask_a = torch.tensor(input_mask_a, dtype=torch.long)
    input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)
    segment_ids_a =torch.tensor(segment_ids_a, dtype=torch.long)
    segment_ids_b = torch.tensor(segment_ids_b, dtype=torch.long)
    ref_input_ids_a = torch.tensor(ref_input_ids_a, dtype=torch.long)
    ref_input_ids_b = torch.tensor(ref_input_ids_b, dtype=torch.long)

    if output_mode == 'regression':
        label_ids = torch.tensor(label_ids, dtype=torch.float)
    else:
        label_ids = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, input_ids_a, input_ids_b, input_mask, input_mask_a, input_mask_b, segment_ids, segment_ids_a, segment_ids_b, ref_input_ids_a, ref_input_ids_b, label_ids, mask_labels, guid


def get_dataloader(ds, args, size):
    data = TensorDataset(*ds)
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    dl = DataLoader(data, sampler=sampler, batch_size=size)
    return dl


def find_incorrect(model, batch):
    input_ids_a = batch[0].cuda()
    input_ids_b = batch[1].cuda()
    input_mask_a = batch[2].cuda()
    input_mask_b = batch[3].cuda()
    segment_ids_a = batch[4].cuda()  # Not used
    segment_ids_b = batch[5].cuda()  # Not used

    with torch.no_grad():
        logits_a = model(input_ids=input_ids_a, token_type_ids=segment_ids_a, attention_mask=input_mask_a, return_dict=True).logits
        logits_b = model(input_ids=input_ids_b, token_type_ids=segment_ids_b, attention_mask=input_mask_b, return_dict=True).logits
        pred_a = logits_a.max(1).indices
        pred_b = logits_b.max(1).indices
    matched = pred_a == pred_b
    matched = matched.to('cpu').numpy()
    return np.where(matched == 0)[0]


def feature_fpp(count_dict, total_dict):
    dict_ratio = dict()
    for w in count_dict.keys():
        dict_ratio[w] = float(count_dict[w]) / total_dict[w]
    return dict_ratio


class Model(nn.Module):
    def __init__(self, args, num_labels):
        super(Model, self).__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        self.config = self.mlm.bert.config
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, output_hidden_states, return_dict):
        output = self.mlm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if output_hidden_states:
            cls = output.hidden_states[-1][:, 0, :]
            logits = self.classifier(cls)
            return logits
        else:
            return output.loss


class ING:
    def __init__(self, args, logger, tokenizer, processor, num_labels):
        self.args = args
        ''' For training and recording '''
        self.logger, self.tokenizer, self.processor = logger, tokenizer, processor
        self.global_step, self.step_in_phase, self.num_labels = 0, 0, len(self.processor.get_labels())
        self.losses, self.reg_losses = [], []
        self.suppress_records = [[], []]
        self.attr_change_dict, self.manual_change_dict = dict(), dict()  # dict of AttrRecord

        # self.model = Model(self.args, num_labels)
        # my model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.load_path or self.args.model_name_or_path, num_labels=num_labels)
        if self.args.load_path:
            dict_load = torch.load(self.args.load_path)
            self.model.load_state_dict(dict_load)
            self.logger.info('---------------Load model finished!----------------')
        device_ids = []
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.cuda()
        params = [{'params': self.model.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        self.ds_train, self.ds_eval = None, None  # ids, mask, segment_ids, label
        self.dl_train, self.dl_eval = None, None
        self.fw_train = None

        self.loss_fct = nn.CrossEntropyLoss()
        self.desc, self.pbar = my_utils.DescStr(), None  # For nested progress bar

        self.neg_suppress_words, self.neg_suppress_words_ids = my_utils.load_suppress_words_new(args, self.args.neg_words_file, self.tokenizer)

        self.stop_words, self.stop_words_ids = self.get_stop_words()
        self.word_count_dict, self.word_appear_records = dict(), dict()
        self.jsd = JSD()
        self.token_and_score = dict()
        self.alpha_1 = self.args.alpha_1
        self.bias_degree = dict()
        self.bias_acc = self.args.bias_acc
        self.f1 = self.args.f1

    def load_model(self, load_path):
        dict_load = torch.load(load_path)
        self.model.module.load_state_dict(dict_load)

    def load_data(self, f_train, f_eval):
        self.ds_train = unpack_features(f_train)
        self.ds_eval = unpack_features(f_eval)

    def init_trainer(self):
        """
        Init the trainer just before the training, such that no information will be omitted
        """
        self.dl_train = get_dataloader(self.ds_train, self.args, self.args.train_batch_size)
        self.dl_eval = get_dataloader(self.ds_eval, self.args, self.args.eval_batch_size)

    def train(self):
        self.init_trainer()     # TODO: 6. handling the case of resuming training
        start_at = time.time()
        val_best_acc, val_worst_acc = -1, 1
        iterations = 1
        output_dir = self.args.output_dir
        if self.args.add_jsd and self.neg_suppress_words:
            # if not self.neg_suppress_words and not self.args.debias:
            #     self.find_words()
            # self.neg_suppress_words, self.neg_suppress_words_ids = my_utils.load_suppress_words_new(
            #     self.args.neg_words_file, self.tokenizer)
            self.degree()  # Calculate the biased word score for each sample

        with tqdm(total=self.args.epoch, ncols=my_utils.MAX_LINE_WIDTH, desc='#Iter') as pbar:
            self.pbar = pbar
            for epoch_idx in range(self.args.epoch):
                total_jsd_loss = 0
                for step, batch in enumerate(tqdm(self.dl_train)):
                    self.model.train()
                    input_ids = batch[0].cuda()
                    token_type_ids = batch[6].cuda()
                    attention_mask = batch[3].cuda()
                    labels = batch[11].cuda()
                    if "distilbert" in self.args.model_name_or_path:
                        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True).loss
                    else:
                        loss = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, return_dict=True).loss
                    loss = torch.sum(loss, dim=-1)

                    self.logger.info('loss:%f', loss)
                    # JSD loss
                    if self.args.add_jsd and self.neg_suppress_words:
                        if self.args.debias:
                            jsd_loss = self.jsd_loss(batch)
                        else:
                            jsd_loss = max(1 - self.jsd_loss(batch), 0.5)
                        total_jsd_loss += jsd_loss
                        self.logger.info('jsd_loss:%f', jsd_loss)
                        loss = loss + self.alpha_1 * jsd_loss
                    loss.backward()

                    if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:  # Batch normalization
                        # FIXME: support FP16 if needed
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.global_step += 1
                self.logger.info('total_jsd_loss:%f', total_jsd_loss)
                self.pbar.update(1)

                self.logger.info('******************** Validation *******************\n')
                val_res, val_res_1 = self.validate(epoch_idx)
                val_acc = val_res_1['acc']

                if self.args.add_jsd and not self.args.debias:
                    val_1 = val_res_1
                    val = val_res
                    iteration = epoch_idx
                    iterations += 1
                    self.args.output_dir = output_dir + '_iterations_{}'.format(iterations)
                    if not os.path.exists(self.args.output_dir):
                        os.makedirs(self.args.output_dir)
                    my_utils.save_model_new(self.args, self.model.module, epoch_idx, postfix='_bias')
                    self.find_words()
                else:
                    if val_acc > val_best_acc:
                        val_best_acc = val_acc
                        best_fairness_1 = val_res_1
                        best_fairness = val_res
                        best_fairness_iteration = epoch_idx
                        my_utils.save_model_new(self.args, self.model.module, epoch_idx, postfix='_best-fairness')
        # make sure that model from new phase will be recorded
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.learning_rate

        self.logger.info('--------------------> Training complete')
        time_cost = my_utils.seconds2hms(time.time() - start_at)
        if self.args.add_jsd and not self.args.debias:
            self.logger.info('Bias model on epoch [{}]'.format(int(iteration + 1)))
            for key in sorted(val.keys()):
                self.logger.info('\t{} = {:.4f}'.format(key, val[key]))
            for key in sorted(val_1.keys()):
                self.logger.info('\t{} = {:.4f}'.format(key, val_1[key]))
        else:
            self.logger.info('Best fairness model on epoch [{}]'.format(int(best_fairness_iteration + 1)))
            for key in sorted(best_fairness.keys()):
                self.logger.info('\t{} = {:.4f}'.format(key, best_fairness[key]))
            for key in sorted(best_fairness_1.keys()):
                self.logger.info('\t{} = {:.4f}'.format(key, best_fairness_1[key]))
        self.logger.info('--> Time cost: {:>2}:{:>2}:{:>2}\n'.format(*time_cost))

    def jsd_change(self):
        self.logger.info('------------------------Calculate JSD Change Start------------------------')
        tokens2score = {}
        for step, batch in enumerate(tqdm(self.dl_train)):
            for b in range(batch[1].size(0)):  # batch size
                current_batch_inp_mask_a = batch[1][b, :, :]  # 11*128
                current_batch_inp_mask_b = batch[2][b, :, :]
                replaced_tokens = []
                for j in range(current_batch_inp_mask_a.size(1)):
                    if current_batch_inp_mask_a[0,j].item() == current_batch_inp_mask_b[0, j].item() and current_batch_inp_mask_a[0, j].item() in self.neg_suppress_words_ids.keys():
                        replaced_tokens.append(current_batch_inp_mask_a[0, j].item())
                for replaced_token in replaced_tokens:
                    replaced_a = torch.where(current_batch_inp_mask_a == replaced_token, self.tokenizer.mask_token_id, current_batch_inp_mask_a)
                    replaced_b = torch.where(current_batch_inp_mask_b == replaced_token, self.tokenizer.mask_token_id, current_batch_inp_mask_b)
                    logits_a = self.model(input_ids=current_batch_inp_mask_a.cuda(),
                                                    attention_mask=batch[4][b, :, :].cuda(),
                                                    token_type_ids=batch[7][b, :, :].cuda(), labels=None,
                                                    return_dict=True).logits  # M(x)
                    logits_b = self.model(input_ids=current_batch_inp_mask_b.cuda(),
                                                    attention_mask=batch[5][b, :, :].cuda(),
                                                    token_type_ids=batch[8][b, :, :].cuda(), labels=None,
                                                    return_dict=True).logits
                    replaced_logits_mask_a = self.model(input_ids=replaced_a.cuda(),
                                                           attention_mask=batch[4][b, :, :].cuda(),
                                                           token_type_ids=batch[7][b, :, :].cuda(), labels=None,
                                                           return_dict=True).logits  # M(x)
                    replaced_logits_mask_b = self.model(input_ids=replaced_b.cuda(),
                                               attention_mask=batch[5][b, :, :].cuda(),
                                               token_type_ids=batch[8][b, :, :].cuda(), labels=None,
                                               return_dict=True).logits
                    jsd_score_for_each_example_tokens = []
                    mask_jsd_score_for_each_example_tokens = []
                    for zzz in range(replaced_logits_mask_a.size(0)):
                        jsd_score = self.jsd(logits_a[zzz], logits_b[zzz])
                        mask_jsd_score = self.jsd(replaced_logits_mask_a[zzz], replaced_logits_mask_b[zzz])
                        jsd_score_for_each_example_tokens.append(jsd_score)
                        mask_jsd_score_for_each_example_tokens.append(mask_jsd_score)
                    exmaple_jsd_score_for_token = sum(jsd_score_for_each_example_tokens)/len(jsd_score_for_each_example_tokens)
                    exmaple_mask_jsd_score_for_token = sum(mask_jsd_score_for_each_example_tokens)/len(mask_jsd_score_for_each_example_tokens)
                    difference_value = exmaple_jsd_score_for_token - exmaple_mask_jsd_score_for_token
                    if self.tokenizer.convert_ids_to_tokens(replaced_token) not in tokens2score.keys():
                        tokens2score[self.tokenizer.convert_ids_to_tokens(replaced_token)] = [difference_value.item()]
                    else:
                        tokens2score[self.tokenizer.convert_ids_to_tokens(replaced_token)].append(difference_value.item())
        results_path = os.path.join(self.args.output_dir, "mask_token_difference.csv")
        difference = []
        mask_word = []
        for key in tokens2score.keys():
            mask_word.append(key)
            difference.append(sum(tokens2score[key]) / len(tokens2score[key]))
        dataframe = pd.DataFrame({'mask_word': mask_word, 'difference': difference})
        dataframe.to_csv(results_path, index=False, sep=',')
        self.logger.info('------------------------Calculate JSD Change Finish------------------------')

    def degree(self):
        self.logger.info('------------------------Calculate Bias Degree Start------------------------')
        for step, batch in enumerate(tqdm(self.dl_train)):
            for i in range(batch[0].size(0)):
                score = 0
                for j in range(batch[0].size(1)):
                    if batch[0][i][j].item() in self.neg_suppress_words_ids.keys():
                        score += self.neg_suppress_words_ids[batch[0][i][j].item()]
                self.bias_degree[batch[-1][i].item()] = score
        min_degree = min(self.bias_degree.values())
        difference = max(self.bias_degree.values()) - min(self.bias_degree.values())
        for key in self.bias_degree.keys():
            self.bias_degree[key] = (self.bias_degree[key] - min_degree) / difference
        self.logger.info('------------------------Calculate Bias Degree Finish------------------------')

    def jsd_loss(self, batch):
        sum_score = 0
        guid = batch[-1]
        j = 0
        for i in range(batch[1].size(1)):
            if "distilbert" in self.args.model_name_or_path:
                logits_enb_a = self.model(input_ids=batch[1][:, i, :].cuda(), attention_mask=batch[4][:, i, :].cuda(),
                                          labels=None, return_dict=True).logits  # M(x)
                logits_enb_b = self.model(input_ids=batch[2][:, i, :].cuda(), attention_mask=batch[5][:, i, :].cuda(),
                                          labels=None, return_dict=True).logits
            else:
                logits_enb_a = self.model(input_ids=batch[1][:, i, :].cuda(), attention_mask=batch[4][:, i, :].cuda(),
                                          token_type_ids=batch[7][:, i, :].cuda(), labels=None, return_dict=True).logits  # M(x)
                logits_enb_b = self.model(input_ids=batch[2][:, i, :].cuda(), attention_mask=batch[5][:, i, :].cuda(),
                                          token_type_ids=batch[8][:, i, :].cuda(), labels=None, return_dict=True).logits
            for k in range(logits_enb_a.size(0)):
                if self.bias_degree[guid[k].item()] != 0:
                    j += 1
                    sum_score += self.bias_degree[guid[k].item()] * self.jsd(logits_enb_a[k], logits_enb_b[k])
        if j:
            sum_score = sum_score / j
        return sum_score

    def find_words(self):
        self.model.eval()
        self.fw_train = get_dataloader(self.ds_train, self.args, self.args.fw_batch_size)
        for step, batch in enumerate(tqdm(self.fw_train)):
            self.regular(batch)
        # you should average a token if it occurs multi-times in the sentence
        top_tokens = sorted(self.token_and_score.items(), key=lambda a: sum(a[1]) / len(a[1]), reverse=True)
        word = []
        score = []
        # self.args.neg_words_file = os.path.join(self.args.output_dir, "token_and_score.csv")
        results_path = os.path.join(self.args.output_dir, "token_and_score.csv")
        for each in top_tokens:
            word.append(self.tokenizer.convert_ids_to_tokens(each[0]))
            score.append(sum(each[1]) / len(each[1]))
        dataframe = pd.DataFrame({'word': word, 'score': score})
        dataframe.to_csv(results_path, index=False, sep=',')
        self.logger.info('Writing Biasd Words finished!')

    def regular(self, batch):
        fh = FH(self.model)
        # ig = LayerIntegratedGradients(fh, self.model.module.mlm.bert.embeddings)
        ig = LayerIntegratedGradients(fh, self.model.module.bert.embeddings)
        self.logger.info('---------------------------------------Regularization------------------------------------')
        input_ids_a = batch[1]
        input_ids_b = batch[2]
        base_ids_a = batch[9]
        base_ids_b = batch[10]
        token_type_ids_a = batch[7]
        token_type_ids_b = batch[8]
        attention_mask_a = batch[4]
        attention_mask_b = batch[5]
        label = batch[11].cuda()
        for k in range(input_ids_a.size(1)):
            input_a = input_ids_a[:, k, :]
            input_b = input_ids_b[:, k, :]
            attributions_a, approximation_error_a = ig.attribute(inputs=input_ids_a[:, k, :].cuda(), baselines=base_ids_a[:, k, :].cuda(),
                                                                 additional_forward_args=(
                                                                 token_type_ids_a[:, k, :].cuda(), attention_mask_a[:, k, :].cuda()), target=label,
                                                                 return_convergence_delta=True)
            attributions_a = abs(attributions_a).sum(dim=-1).squeeze(0)

            attributions_a = attributions_a / torch.linalg.norm(attributions_a)

            attributions_b, approximation_error_b = ig.attribute(inputs=input_ids_b[:, k, :].cuda(), baselines=base_ids_b[:, k, :].cuda(),
                                                                 additional_forward_args=(
                                                                 token_type_ids_b[:, k, :].cuda(), attention_mask_b[:, k, :].cuda()), target=label,
                                                                 return_convergence_delta=True)
            attributions_b = abs(attributions_b).sum(dim=-1).squeeze(0)
            attributions_b = attributions_b / torch.linalg.norm(attributions_b)
            attributions = abs(attributions_a - attributions_b)

            for i in range(attributions.size(0)):  # batch_size
                for j in range(attributions.size(-1)):  # max_len
                    token_a = input_a[i][j].item()
                    token_b = input_b[i][j].item()
                    if token_a == token_b and self.tokenizer.convert_ids_to_tokens(token_a).isalpha() and len(self.tokenizer.convert_ids_to_tokens(token_a)) > 2:
                        score = attributions[i][j].item()
                        if token_a not in self.token_and_score.keys():
                            self.token_and_score[token_a] = [score]
                        else:
                            self.token_and_score[token_a].append(score)

    def validate(self, epoch):
        self.logger.info('\t--> Running evaluation')
        self.logger.info('\t\tNum examples = %d', len(self.dl_eval.dataset))
        self.model.train(False)
        preds, preds_a, preds_b, ys, ys_1 = [], [], [], [], []
        for step, batch in enumerate(tqdm(self.dl_eval, file=self.desc, desc='#Eval')):
            labels = batch[11].cuda()
            with torch.no_grad():
                if "distilbert" in self.args.model_name_or_path:
                    logits = self.model(input_ids=batch[0].cuda(), attention_mask=batch[3].cuda(), labels=None,
                                      return_dict=True).logits
                else:
                    logits = self.model(input_ids=batch[0].cuda(), token_type_ids=batch[6].cuda(), attention_mask=batch[3].cuda(), labels=None, return_dict=True).logits  # be careful with the order
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    ys.append(labels.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    ys[0] = np.append(ys[0], labels.detach().cpu().numpy(), axis=0)
                for i in range(batch[1].size(1)):
                    if "distilbert" in self.args.model_name_or_path:
                        logits_a = self.model(input_ids=batch[1][:, i, :].cuda(),
                                              attention_mask=batch[4][:, i, :].cuda(), labels=None,
                                              return_dict=True).logits
                        logits_b = self.model(input_ids=batch[2][:, i, :].cuda(),
                                              attention_mask=batch[5][:, i, :].cuda(), labels=None,
                                              return_dict=True).logits
                    else:
                        logits_a = self.model(input_ids=batch[1][:, i, :].cuda(),
                                              token_type_ids=batch[7][:, i, :].cuda(),
                                              attention_mask=batch[4][:, i, :].cuda(), labels=None,
                                              return_dict=True).logits  # be careful with the order
                        logits_b = self.model(input_ids=batch[2][:, i, :].cuda(),
                                              token_type_ids=batch[8][:, i, :].cuda(),
                                              attention_mask=batch[5][:, i, :].cuda(), labels=None,
                                              return_dict=True).logits
                    if len(preds_a) == 0:
                        preds_a.append(logits_a.detach().cpu().numpy())
                        preds_b.append(logits_b.detach().cpu().numpy())
                        ys_1.append(labels.detach().cpu().numpy())
                    else:
                        preds_a[0] = np.append(preds_a[0], logits_a.detach().cpu().numpy(), axis=0)
                        preds_b[0] = np.append(preds_b[0], logits_b.detach().cpu().numpy(), axis=0)
                        ys_1[0] = np.append(ys_1[0], labels.detach().cpu().numpy(), axis=0)

            self.pbar.set_description(self.desc.read())

        preds = preds[0]
        preds_a = preds_a[0]
        preds_b = preds_b[0]
        ys = ys[0]
        ys_1 = ys_1[0]
        pred_labels = np.argmax(preds, axis=1)  # FIXME: Currently, only support classification
        pred_labels_a = np.argmax(preds_a, axis=1)  # FIXME: Currently, only support classification
        pred_labels_b = np.argmax(preds_b, axis=1)  # FIXME: Currently, only support classification
        pred_prob = nn.functional.softmax(torch.from_numpy(preds).float(), -1).numpy()
        pred_prob_a = nn.functional.softmax(torch.from_numpy(preds_a).float(), -1).numpy()
        pred_prob_b = nn.functional.softmax(torch.from_numpy(preds_b).float(), -1).numpy()
        result = my_utils.compute_metrics(pred_labels, ys, pred_prob)  # return {"acc": acc, "f1": f1, "precision": p, "recall": r, "auc_roc": roc}
        result_a = my_utils.compute_metrics(pred_labels_a, pred_labels_b, pred_prob_a)
        j, m, n, k = 0, 0, 0, 0
        for i in range(len(pred_labels_a)):
            if pred_labels_a[i] == pred_labels_b[i]:
                j += 1
                if pred_labels_a[i] == ys_1[i]:
                    m += 1
                else:
                    n += 1
            else:
                k += 1
        assert m + n == j
        assert k + j == len(pred_labels_a)
        fr_tr = k / j
        tpr = m / j

        result_a1 = my_utils.compute_metrics(pred_labels_a, ys_1, pred_prob_a)
        result_b1 = my_utils.compute_metrics(pred_labels_b, ys_1, pred_prob_b)
        result_a['FR/TR'] = fr_tr
        result_a['TPR'] = tpr

        split = 'dev'
        output_eval_file = os.path.join(self.args.output_dir, "eval_results_epoch-%d_%s_%s.txt" % (epoch, split, self.args.task_name))
        with open(output_eval_file, "a") as writer:
            self.logger.info('\n')
            self.logger.info('********** Eval results **********\n')
            self.logger.info('Epoch {}\n'.format(int(epoch+1)))
            writer.write('********** Eval results **********\n')
            writer.write('{} = {}\n'.format('Epoch', epoch))
            for key in sorted(result.keys()):
                self.logger.info('\t\t{} = {:.4f}'.format(key, result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

            self.logger.info('**************************')
            writer.write('********** Eval results_a **********\n')
            for key in sorted(result_a.keys()):
                self.logger.info('\t\t{} = {:.4f}'.format(key, result_a[key]))
                writer.write("%s = %s\n" % (key, str(result_a[key])))

            self.logger.info('**************************')
            writer.write('********** Eval results_a1 **********\n')
            for key in sorted(result_a1.keys()):
                self.logger.info('\t\t{} = {:.4f}'.format(key, result_a1[key]))
                writer.write("%s = %s\n" % (key, str(result_a1[key])))

            self.logger.info('**************************')
            writer.write('********** Eval results_b1 **********\n')
            for key in sorted(result_b1.keys()):
                self.logger.info('\t\t{} = {:.4f}'.format(key, result_b1[key]))
                writer.write("%s = %s\n" % (key, str(result_b1[key])))

        self.model.train(True)
        return result, result_a

    """
    ================================================================
    |                  Suspicious detector                         |
    ================================================================
    """

    def get_ratios(self, group_li_a):
        count, _ = my_utils.words_count(group_li_a, self.stop_words_ids)
        res = feature_fpp(dict(count), self.word_count_dict)
        return res

    def update_suppress_words(self, wrong_li):
        f_a = []
        for idx in range(len(wrong_li[0])):
            f_a.append(wrong_li[0][idx].numpy().tolist())

        f_word_ratio = self.get_ratios(f_a)
        sorted_fpp = sorted(f_word_ratio.items(), key=lambda item: item[1])[::-1]

        word = []
        ratio = []
        for p in sorted_fpp:
            word.append(self.tokenizer.convert_ids_to_tokens[p[0]])
            ratio.append(p[1])
        dataframe = pd.DataFrame({'word': word, 'ratio': ratio})
        dataframe.to_csv(os.path.join(self.args.output_dir, "words_ratio.csv"), index=False, sep=',')

        for p in sorted_fpp:
            if p[1] <= self.args.eta:  #
                break
            if self.tokenizer.convert_ids_to_tokens[p[0]].isalpha() and len(self.tokenizer.convert_ids_to_tokens[p[0]]) > 2:
                w = self.tokenizer.convert_ids_to_tokens[p[0]]
                self.neg_suppress_words_ids[p[0]] = p[1]
                self.neg_suppress_words[w] = p[1]
        return f_word_ratio

    def filter_suppress_words(self, ws):
        for w in ws:
            w_id = self.tokenizer.vocab[w]
            del self.neg_suppress_words_ids[w_id]
            del self.neg_suppress_words[w]

    def _get_word_counts(self):
        word_count_dict = dict()
        for w_ids in self.word_appear_records.keys():
            word_count_dict[w_ids] = sum(self.word_appear_records[w_ids])
        return word_count_dict

    def get_stop_words(self):
        stop_words = {'[CLS]', '[PAD]', '[SEP]', 'she', 'he', 'at', 'in', 'on', 'to', 'above', 'over', 'below', 'under',
                      'beside', 'behind', 'between', 'after', 'from', 'since', 'for', 'across', 'through', 'past',
                      'towards', 'onto', 'into', 'up', 'down', 'about', 'by', 'with'}
        for pc in string.punctuation:
            stop_words.add(pc)

        stop_words_ids = set()
        for s in stop_words:
            try:
                stop_words_ids.add(self.tokenizer.vocab[s])
            except KeyError:
                self.logger.warning('=' * my_utils.MAX_LINE_WIDTH)
                self.logger.warning('WARNING: Cannot find target word in vocab:\t{}'.format(s))
                self.logger.warning('=' * my_utils.MAX_LINE_WIDTH + '\n\n')
        return stop_words, stop_words_ids
