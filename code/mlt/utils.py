# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 8:05 下午
# @Author  : zhengjiawei
# @FileName: utils.py
# @Software: PyCharm
import gc
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from .nezha.config import NeZhaConfig
from .nezha.model import NeZhaPreTrainedModel, NeZhaModel, NeZhaForSequenceClassification
from .nezha.nezha_utils import Lookahead, WarmupLinearSchedule
from sklearn import metrics
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertPreTrainedModel, BertConfig, BertModel, DebertaPreTrainedModel, \
    DebertaConfig, DebertaModel, AdamW, BertForSequenceClassification


def load_data(config, tokenizer):
    with open(config['data_cache_path'], 'rb') as f:
        data = pickle.load(f)
    train_dev_data, test_data = data['train'], data['test']
    test_dataset = TCDataset(test_data)
    eval_train_dataset = TCDataset(train_dev_data)
    collate_fn = Data_Collator(config['max_seq_len'], tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=4, collate_fn=collate_fn)
    eval_train_dataloader = DataLoader(dataset=eval_train_dataset, batch_size=config['batch_size'],
                                       shuffle=False, num_workers=4, collate_fn=collate_fn)
    return collate_fn, test_dataloader, train_dev_data, eval_train_dataloader


def seed_everyone(seed_):
    torch.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    np.random.seed(seed_)
    random.seed(seed_)
    return seed_


class TCDataset(Dataset):

    def __init__(self, data_dict: dict):
        super(TCDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index],
                self.data_dict['attention_mask'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['labels'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Data_Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def truncate_and_pad(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, labels_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] + [self.tokenizer.sep_token_id],
                                                      dtype=torch.long)
            token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels = self.truncate_and_pad(input_ids_list, token_type_ids_list,
                                                                                  attention_mask_list, labels_list,
                                                                                  max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor):
        q = self.fc(hidden_state).squeeze(dim=-1)
        q = q.masked_fill(mask, -np.inf)
        w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        h = w @ hidden_state
        return h.squeeze(dim=1)


class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        mask = input_ids == 0
        h = self.attn(hidden_states, mask)
        out = self.fc(h)
        return out


class DeepAttentionClassifier(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int):
        super(DeepAttentionClassifier, self).__init__()
        self.interact_attn = Attention(hidden_size=hidden_size)
        self.a_self_attn = Attention(hidden_size=hidden_size)
        self.b_self_attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(5 * hidden_size, 3 * hidden_size)
        self.out = nn.Linear(3 * hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        pad_mask, a_mask, b_mask = get_mask(input_ids)
        interact_h = self.interact_attn(hidden_states, pad_mask)
        a_h = self.a_self_attn(hidden_states, a_mask)
        b_h = self.b_self_attn(hidden_states, b_mask)
        dot_h = a_h * b_h
        sub_h = a_h - b_h
        final_h = torch.cat([interact_h, a_h, b_h, dot_h, sub_h], dim=1)
        final_h = self.dropout(final_h)
        logits = self.out(torch.relu(self.fc(final_h)))
        return logits


class FusionClassifier(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int):
        super(FusionClassifier, self).__init__()
        self.fc = nn.Linear(5 * hidden_size, 2 * hidden_size)
        self.out = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        pad_mask, a_mask, b_mask = get_mask(input_ids)
        hidden_states = hidden_states * pad_mask.view(*pad_mask.size(), 1)
        last_hidden = hidden_states.max(dim=1)[0]
        a_hidden = hidden_states * a_mask.view(*a_mask.size(), 1)
        b_hidden = hidden_states * b_mask.view(*a_mask.size(), 1)
        a_hidden = a_hidden.max(dim=1)[0]
        b_hidden = b_hidden.max(dim=1)[0]
        a_sub_b = (a_hidden - b_hidden).abs()
        a_mul_b = a_hidden * b_hidden
        fusion_hidden = torch.cat([
            last_hidden, a_hidden, b_hidden,
            a_sub_b, a_mul_b
        ], dim=-1)
        logits = self.out(torch.relu(self.fc(self.dropout(fusion_hidden))))
        return logits


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BertForTc(BertPreTrainedModel):

    def __init__(self, config: BertConfig, model_path: str):
        super(BertForTc, self).__init__(config)

        self.bert = BertModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

    def forward(self,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = self.dropout(outputs[0])

        last_hidden = torch.cat(
            [last_hidden.max(dim=1)[0],
             last_hidden.mean(dim=1)],
            dim=-1
        )
        logits = self.classifier(last_hidden)
        outputs = (logits,)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        outputs = (loss,) + outputs

        return outputs


class DeBertForOppo(DebertaPreTrainedModel):

    def __init__(self, config: DebertaConfig, model_path: str):
        super(DeBertForOppo, self).__init__(config)
        self.bert = DebertaModel.from_pretrained(model_path, config=config)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        last_hidden = outputs[0]
        last_hidden = self.dropout(last_hidden)
        last_hidden = torch.cat([
            last_hidden.max(dim=1)[0],
            last_hidden.mean(dim=1)
        ], dim=-1)

        logits = self.classifier(last_hidden)

        outputs = (logits,)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        outputs = (loss,) + outputs

        return outputs


class NeZhaForTc(NeZhaPreTrainedModel):
    def __init__(self, config: NeZhaConfig, model_path: str):
        super(NeZhaForTc, self).__init__(config)

        self.bert = NeZhaModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

    def forward(self,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = self.dropout(outputs[0])

        last_hidden = torch.cat(
            [last_hidden.max(dim=1)[0],
             last_hidden.mean(dim=1)],
            dim=-1
        )
        logits = self.classifier(last_hidden)
        outputs = (logits,)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        outputs = (loss,) + outputs

        return outputs


def roc_score(labels, preds,
              ):
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    roc_auc = metrics.roc_auc_score(labels, preds)
    pr_auc = metrics.average_precision_score(labels, preds)
    preds = np.where(preds >= 0.5, 1, 0)
    acc = metrics.accuracy_score(y_true=labels, y_pred=preds)
    return roc_auc, pr_auc, acc


def read_data(config: dict, tokenizer, debug=False) -> str:
    train_file_path = os.path.join('../tcdata/nlp_round2_data/gaiic_track3_round2_train_20210407.tsv')
    test_file_path = os.path.join('../tcdata/nlp_round1_data/gaiic_track3_round1_testB_20210317.tsv')
    train_df = pd.read_csv(train_file_path, header=None, sep='\t')
    test_df = pd.read_csv(test_file_path, header=None, sep='\t')
    if debug:
        train_df = train_df.head(1000)
        test_df = test_df.head(1000)
    data_df = {'train': train_df, 'test': test_df}
    processed_data = {}
    for data_type, df in data_df.items():
        inputs = defaultdict(list)
        if data_type == 'train':
            inverse_inputs = defaultdict(list)
        for i, row in tqdm(df.iterrows(), desc=f'Preprocessing {data_type} data', total=len(df)):
            label = 0 if data_type == 'test' else row[2]
            sentence_a, sentence_b = row[0], row[1]
            build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer)
            if data_type.startswith('test'):
                build_bert_inputs(inputs, label, sentence_b, sentence_a, tokenizer)
            if data_type == 'train':
                build_bert_inputs(inverse_inputs, label, sentence_b, sentence_a, tokenizer)

        processed_data[data_type] = inputs
        if data_type == 'train':
            processed_data[f'inverse_{data_type}'] = inverse_inputs

    data_cache_path = config['data_cache_path']
    if not os.path.exists(os.path.dirname(data_cache_path)):
        os.makedirs(os.path.dirname(data_cache_path))
    with open(data_cache_path, 'wb') as f:
        pickle.dump(processed_data, f)

    return data_cache_path


def build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])
    inputs['labels'].append(label)


def load_cv_data(collate_fn, config, dev_idxs, train_dev_data,
                 train_idxs, inverse_train, test_pseudo_data):
    train_data = defaultdict(list)
    dev_data = defaultdict(list)
    for key, values in train_dev_data.items():
        train_data[key] = [values[idx] for idx in train_idxs]
        dev_data[key] = [values[idx] for idx in dev_idxs]
    train_dataset = TCDataset(train_data)
    dev_dataset = TCDataset(dev_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=4, collate_fn=collate_fn)
    return dev_dataloader, train_dataloader


def get_mask(input_ids: torch.Tensor):
    pad_mask = input_ids == 0
    cls_mask = input_ids == 1
    sep_mask = input_ids == 2
    content_mask = ~ (pad_mask | cls_mask | sep_mask)
    tmp_mask = torch.cumsum(sep_mask, dim=1).bool()
    b_mask = content_mask & tmp_mask
    a_mask = content_mask ^ b_mask
    return pad_mask.long(), a_mask, b_mask


def evaluation(config, model, val_dataloader):
    model.eval()
    preds = []
    labels = []
    val_loss = 0.
    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            labels.append(batch['labels'])
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss, logits = model(**batch_cuda)[:2]
            probs = torch.softmax(logits, dim=-1)

            # if config['n_gpus'] > 1:
            #     loss = loss.mean()

            val_loss += loss.item()
            preds.append(probs[:, 1].detach().cpu())

    avg_val_loss = val_loss / len(val_dataloader)
    roc_auc_score, pr_auc, acc = roc_score(labels, preds)
    return avg_val_loss, roc_auc_score, pr_auc, acc


def train(config, train_dataloader, dev_dataloader, fold):
    model = NeZhaForSequenceClassification.from_pretrained(config['model_path'])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'],
                      correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    total_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = WarmupLinearSchedule(optimizer,
                                        warmup_steps=int(config['warmup_ratio'] * total_steps),
                                        t_total=total_steps)
    model.to(config['device'])
    if config['adv'] == 'fgm':
        fgm = FGM(model)
    else:
        pgd = PGD(model)
        K = 3
    epoch_iterator = trange(config['num_epochs'])
    global_steps = 0
    train_loss = 0.
    logging_loss = 0.
    best_roc_auc = 0.
    best_model_path = ''

    # if config['n_gpus'] > 1:
    #     model = nn.DataParallel(model)

    for _ in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss = model(**batch_cuda)[0]
            # if config['n_gpus'] > 1:
            #     loss = loss.mean()
            loss.backward()

            if config['adv'] == 'fgm':
                fgm.attack(epsilon=config['eps'])
                loss_adv = model(**batch_cuda)[0]
                loss_adv.backward()
                fgm.restore()
            else:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(epsilon=config['eps'], alpha=config['alpha'], is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch_cuda)[0]
                    # if config['n_gpus'] > 1:
                    #     loss_adv = loss_adv.mean()
                    loss_adv.backward()
                pgd.restore()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config['ema_start']:
                ema.update()

            train_loss += loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            lr = lr_scheduler.get_last_lr()[0]
            if global_steps % config['logging_step'] == 0:
                if global_steps >= config['ema_start_step'] and not config['ema_start']:
                    print('\n>>> EMA starting ...')
                    config['ema_start'] = True
                    ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

                print_train_loss = (train_loss - logging_loss) / config['logging_step']
                logging_loss = train_loss

                if config['ema_start']:
                    ema.apply_shadow()
                val_loss, roc_auc_score, pr_auc_score, acc = evaluation(config, model, dev_dataloader)

                print_log = f'\n>>> training loss: {print_train_loss:.6f}, valid loss: {val_loss:.6f}, '

                if roc_auc_score > best_roc_auc:
                    model_save_path = os.path.join(config['output_path'],
                                                   f'fold-{fold}-checkpoint-{global_steps}-{roc_auc_score:.6f}')
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(model_save_path)
                    best_roc_auc = roc_auc_score
                    best_model_path = model_save_path

                print_log += f'valid roc-auc: {roc_auc_score:.6f}, valid pr-auc: {pr_auc_score:.6f},' \
                             f' valid acc: {acc:.6f}'

                print(print_log)
                model.train()
                if config['ema_start']:
                    ema.restore()

    if config['adv'] == 'fgm':
        del fgm
    else:
        del pgd
    gc.collect()
    torch.cuda.empty_cache()

    return best_model_path


def predict(config, model, test_dataloader, mode='test'):
    test_iterator = tqdm(test_dataloader, desc=f'Predicting-{mode}', total=len(test_dataloader))
    test_preds = []
    test_logits = []

    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            logits = model(**batch_cuda)[1]
            probs = torch.softmax(logits, dim=-1)
            test_preds.append(probs[:, 1].detach().cpu())
            test_logits.append(logits.detach().cpu())

    test_preds = torch.cat(test_preds)
    test_logits = torch.cat(test_logits)
    if mode == 'test':
        test_preds = torch.stack(test_preds.split(2), dim=0).mean(dim=1).numpy()
        test_logits = torch.stack(test_logits.split(2), dim=0).mean(dim=1).numpy()
    else:
        test_preds = test_preds.numpy()
        test_logits = test_logits.numpy()
    return test_preds, test_logits


def bert_train(config, train_dataloader, dev_dataloader, fold):
    model = BertForSequenceClassification.from_pretrained(config['model_path'])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'],
                      correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    total_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = WarmupLinearSchedule(optimizer,
                                        warmup_steps=int(config['warmup_ratio'] * total_steps),
                                        t_total=total_steps)
    model.to(config['device'])
    if config['adv'] == 'fgm':
        fgm = FGM(model)
    else:
        pgd = PGD(model)
        K = 3
    epoch_iterator = trange(config['num_epochs'])
    global_steps = 0
    train_loss = 0.
    logging_loss = 0.
    best_roc_auc = 0.
    best_model_path = ''

    # if config['n_gpus'] > 1:
    #     model = nn.DataParallel(model)

    for _ in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss = model(**batch_cuda)[0]
            # if config['n_gpus'] > 1:
            #     loss = loss.mean()
            loss.backward()

            if config['adv'] == 'fgm':
                fgm.attack(epsilon=config['eps'])
                loss_adv = model(**batch_cuda)[0]
                loss_adv.backward()
                fgm.restore()
            else:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(epsilon=config['eps'], alpha=config['alpha'], is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch_cuda)[0]
                    # if config['n_gpus'] > 1:
                    #     loss_adv = loss_adv.mean()
                    loss_adv.backward()
                pgd.restore()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config['ema_start']:
                ema.update()

            train_loss += loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            lr = lr_scheduler.get_last_lr()[0]
            if global_steps % config['logging_step'] == 0:
                if global_steps >= config['ema_start_step'] and not config['ema_start']:
                    print('\n>>> EMA starting ...')
                    config['ema_start'] = True
                    ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

                print_train_loss = (train_loss - logging_loss) / config['logging_step']
                logging_loss = train_loss

                if config['ema_start']:
                    ema.apply_shadow()
                val_loss, roc_auc_score, pr_auc_score, acc = evaluation(config, model, dev_dataloader)

                print_log = f'\n>>> training loss: {print_train_loss:.6f}, valid loss: {val_loss:.6f}, '

                if roc_auc_score > best_roc_auc:
                    model_save_path = os.path.join(config['output_path'],
                                                   f'fold-{fold}-checkpoint-{global_steps}-{roc_auc_score:.6f}')
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(model_save_path)
                    best_roc_auc = roc_auc_score
                    best_model_path = model_save_path

                print_log += f'valid roc-auc: {roc_auc_score:.6f}, valid pr-auc: {pr_auc_score:.6f},' \
                             f' valid acc: {acc:.6f}'

                print(print_log)
                model.train()
                if config['ema_start']:
                    ema.restore()

    if config['adv'] == 'fgm':
        del fgm
    else:
        del pgd
    gc.collect()
    torch.cuda.empty_cache()

    return best_model_path