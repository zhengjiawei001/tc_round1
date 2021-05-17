import os
import random
from collections import defaultdict
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, BertForSequenceClassification
# from transformers.modeling_bert import BertLayer, BertModel
from transformers import BertTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# wandb.login(key='8cefb8016177b89343b4f6c8eed0c154b55b006b')
# os.system('wandb online')
# os.environ['WANDB_PROJECT'] = 'bert_oppo_finetune'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def roc_score(labels: list, preds: list,
              ) -> Tuple[float, float, float]:
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    roc_auc = metrics.roc_auc_score(labels, preds)
    pr_auc = metrics.average_precision_score(labels, preds)
    preds = np.where(preds >= 0.5, 1, 0)
    acc = metrics.accuracy_score(y_true=labels, y_pred=preds)
    return roc_auc, pr_auc, acc


def read_data(config: dict, tokenizer: BertTokenizer,debug=False) -> str:
    train_file_path = os.path.join('../tcdata/nlp_round2_data/pretrain_data.tsv')
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
        for i, row in tqdm(df.iterrows(), desc=f'Preprocessing {data_type} data', total=len(df)):
            label = 0 if data_type == 'test' else row[2]
            sentence_a, sentence_b = row[0], row[1]
            inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                                return_token_type_ids=True, return_attention_mask=True)
            inputs['input_ids'].append(inputs_dict['input_ids'])
            inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
            inputs['attention_mask'].append(inputs_dict['attention_mask'])
            inputs['labels'].append(label)

        processed_data[data_type] = inputs

    return processed_data


def load_data(data, config, tokenizer):
    train_dev_data, test_data = data['train'], data['test']
    test_dataset = OppoDataset(test_data)
    collate_fn = Collator(config['max_seq_len'], tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=4, collate_fn=collate_fn)
    return collate_fn, test_dataloader, train_dev_data


def load_cv_data(collate_fn, config, dev_idxs, train_dev_data, train_idxs):
    train_data = defaultdict(list)
    dev_data = defaultdict(list)
    for key, values in train_dev_data.items():
        train_data[key] = [values[idx] for idx in train_idxs]
        dev_data[key] = [values[idx] for idx in dev_idxs]
    train_dataset = OppoDataset(train_data)
    dev_dataset = OppoDataset(dev_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=4, collate_fn=collate_fn)
    return dev_dataloader, train_dataloader


class OppoDataset(Dataset):

    def __init__(self, data_dict: dict):
        super(OppoDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index], self.data_dict['labels'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
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

        input_ids, token_type_ids, attention_mask, labels = self.pad_and_truncate(input_ids_list, token_type_ids_list,
                                                                                  attention_mask_list, labels_list,
                                                                                  max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
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
        # emb_name这个参数要换成你模型中embedding的参数名
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
        # emb_name这个参数要换成你模型中embedding的参数名
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

            if config['n_gpus'] > 1:
                loss = loss.mean()

            val_loss += loss.item()
            preds.append(probs[:, 1].detach().cpu())

    avg_val_loss = val_loss / len(val_dataloader)
    roc_auc_score, pr_auc, acc = roc_score(labels, preds)
    return avg_val_loss, roc_auc_score, pr_auc, acc


def train(config, train_dataloader):
    # bert_config = BertConfig.from_pretrained(config['model_path'])
    model = BertForSequenceClassification.from_pretrained(config['model_path'])

    #wandb.watch(model)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    # lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5,
    #                                  patience=2, verbose=True)
    model.to(config['device'])
    # fgm = FGM(model)
    pgd = PGD(model)
    K = 3
    epoch_iterator = trange(config['num_epochs'])
    global_steps = 0
    train_loss = 0.
    logging_loss = 0.
    best_roc_auc = 0.
    best_model_path = ''

    if config['n_gpus'] > 1:
        model = nn.DataParallel(model)

    for epoch in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss = model(**batch_cuda)[0]

            if config['n_gpus'] > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()

            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(**batch_cuda)[0]
                if config['n_gpus'] > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
            pgd.restore()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            #wandb.log({'running training loss': loss.item()}, step=global_steps)

            if global_steps % config['logging_step'] == 0:
                print_train_loss = (train_loss - logging_loss) / config['logging_step']
                logging_loss = train_loss

                #val_loss, roc_auc_score, pr_auc_score, acc = evaluation(config, model, dev_dataloader)

                print_log = f'>>> training loss: {print_train_loss:.4f}'
                print(print_log)
                # lr_scheduler.step(metrics=roc_auc_score, epoch=global_steps // config['logging_step'])

                # if roc_auc_score > best_roc_auc:
                #     model_save_path = os.path.join(config['output_path'],
                #                                    f'checkpoint-{global_steps}-{roc_auc_score:.3f}')
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     model_to_save.save_pretrained(model_save_path)
                #     best_roc_auc = roc_auc_score
                #     best_model_path = model_save_path
                #
                # print_log += f'valid roc-auc: {roc_auc_score:.3f}, valid pr-auc: {pr_auc_score}, valid acc: {acc:.3f}'
                # print(print_log)
                # log_wandb_metrics = {
                #     'training loss': print_train_loss,
                #     'valid loss': val_loss,
                #     'valid roc-auc': roc_auc_score,
                #     'valid pr-auc': pr_auc_score,
                #     'valid acc': acc
                # }
                # wandb.log(log_wandb_metrics, step=global_steps)
                model.train()

    return model, best_model_path


def predict(config, model, test_dataloader):
    test_iterator = tqdm(test_dataloader, desc='Predicting', total=len(test_dataloader))
    test_preds = []

    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            logits = model(**batch_cuda)[1]
            probs = torch.softmax(logits, dim=-1)
            test_preds.append(probs[:, 1].detach().cpu())

    submission_path = os.path.join(config['output_path'], 'submission.tsv')
    test_preds = torch.cat(test_preds).numpy()
    test_df = pd.DataFrame(data={'prediction': test_preds})
    test_df.to_csv(submission_path, index=False, header=False, encoding='utf8', sep='\t')
    with ZipFile(os.path.join(config['output_path'], 'submission.zip'), 'w') as myzip:
        myzip.write(submission_path, 'submission.tsv')


def main():
    config = {
        'overwrite': True,
        'data_path': '../tcdata/nlp_round2_data',
        'data_cache_path': '../user_data/tmp_data/finetune_output/whole_word_mask_processed/data.pkl',
        'output_path': '../user_data/tmp_data/finetune_output/ngram_results',
        'model_path': '../user_data/tmp_data/pretrain_output/whole_word_mask_bert_output/best_model_ckpt',
        'best_model_path': '',
        'batch_size': 128,  # 64
        'num_epochs': 4,  # 3
        'num_folds': 5,  # 5
        'cv': '',
        'max_seq_len': 42,
        'learning_rate': 2e-5,
        'eps': 0.1,
        'alpha': 0.3,
        'adv': 'fgm',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step':1000,  # 500
        'ema_start_step':1500,  # 1500
        'ema_start': False,
        'seed': 20200409
    }

    if not torch.cuda.is_available():
        config['device'] = 'cpu'
    else:
        config['n_gpus'] = torch.cuda.device_count()
        config['batch_size'] *= config['n_gpus']

    if not os.path.exists(config['output_path']):
        os.makedirs((config['output_path']))

    tokenizer = BertTokenizer.from_pretrained('../user_data/vocab_.txt')
    data = read_data(config, tokenizer,debug=False)

    collate_fn, test_dataloader, train_data = load_data(data, config, tokenizer)
    train_dataset = OppoDataset(train_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4, collate_fn=collate_fn)

    seed_everything(config['seed'])

    if not config['best_model_path']:
        model, best_model_path = train(config, train_dataloader)
    else:
        best_model_path = config['best_model_path']



if __name__=='__main__':
    main()
