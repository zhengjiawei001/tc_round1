import os
import pickle
import random
from collections import defaultdict
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from .nezha.model import NeZhaForSequenceClassification

multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_data(config: dict, tokenizer) -> str:
    test_file_path = os.path.join(config['data_path'], 'test.tsv')
    test_df = pd.read_csv(test_file_path, header=None, sep='\t')
    # test_df = test_df.head(1000)

    data_df = {'test': test_df}
    processed_data = {}

    for data_type, df in data_df.items():
        inputs = defaultdict(list)
        for i, row in tqdm(df.iterrows(), desc=f'Preprocessing {data_type} data', total=len(df)):
            label = 0
            sentence_a, sentence_b = row[0], row[1]
            build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer)
            if data_type.startswith('test'):
                build_bert_inputs(inputs, label, sentence_b, sentence_a, tokenizer)

        processed_data[data_type] = inputs

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


def load_data(config, tokenizer):
    with open(config['data_cache_path'], 'rb') as f:
        data = pickle.load(f)
    test_data = data['test']
    test_dataset = OppoDataset(test_data)
    collate_fn = Collator(config['max_seq_len'], tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=4, collate_fn=collate_fn)
    return collate_fn, test_dataloader


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


def main():
    config = {
        'overwrite': True,
        'data_path': '../tcdata/oppo_breeno_round1_data',
        'data_cache_path': '../user_data/nezha-cv10-infer-processed/data.pkl',
        'output_path': '../user_data/nezha-cv10-infer-results',
        'model_prefix': '../user_data/nezha-cv10-results',
        'vocab_path': '../user_data/green-nezha/vocab.txt',
        'batch_size': 128,
        'max_seq_len': 32,
        'device': 'cuda',
        'seed': 2021
    }

    if not torch.cuda.is_available():
        config['device'] = 'cpu'
    else:
        config['n_gpus'] = torch.cuda.device_count()
        config['batch_size'] *= config['n_gpus']

    if not os.path.exists(config['output_path']):
        os.makedirs((config['output_path']))

    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    if not os.path.exists(config['data_cache_path']) or config['overwrite']:
        read_data(config, tokenizer)

    collate_fn, test_dataloader = load_data(config, tokenizer)

    test_pred_df = pd.DataFrame(data={'id': range(len(test_dataloader.dataset) // 2),
                                      'fold1-probs': [0.0] * (len(test_dataloader.dataset) // 2),
                                      'fold1-logits0': [0.0] * (len(test_dataloader.dataset) // 2),
                                      'fold1-logits1': [0.0] * (len(test_dataloader.dataset) // 2),
                                      })
    # 将这里改成多折最好的模型路径
    best_model_paths = ['fold-1-checkpoint-4000-0.976395', 'fold-2-checkpoint-4000-0.973806',
                        'fold-3-checkpoint-4000-0.975206', 'fold-4-checkpoint-4000-0.976415',
                        'fold-5-checkpoint-4000-0.975606', 'fold-6-checkpoint-4000-0.975377',
                        'fold-7-checkpoint-4000-0.975410', 'fold-8-checkpoint-4000-0.975481',
                        'fold-9-checkpoint-4000-0.975342', 'fold-10-checkpoint-4000-0.974909']

    for fold, best_model_path in enumerate(best_model_paths, start=1):
        print(f'Fold {fold} starting ...')
        best_model_path = os.path.join(config['model_prefix'], best_model_path)
        model = NeZhaForSequenceClassification.from_pretrained(best_model_path)
        model.to(config['device'])

        test_pred_probs, test_pred_logits = predict(config, model, test_dataloader, mode='test')
        test_pred_df.loc[:, f'fold{fold}-probs'] = test_pred_probs
        test_pred_df.loc[:, f'fold{fold}-logits0'] = test_pred_logits[:, 0]
        test_pred_df.loc[:, f'fold{fold}-logits1'] = test_pred_logits[:, 1]

    print('\n>>> Saving train test predict probability ...')
    test_pred_df.to_csv(os.path.join(config['output_path'], 'test_predicted.tsv'),
                        sep='\t', index=False, encoding='utf8')
    print('\n>>> Saving submission file ...')
    submission_path = os.path.join(config['output_path'], 'submission.tsv')
    test_pred_df.loc[:, 'submission'] = test_pred_df.loc[
                                        :, [f'fold{i}-probs' for i in range(1, len(best_model_paths) + 1)]
                                        ].mean(axis=1)
    test_pred_df.loc[:, ['submission']].to_csv(submission_path, index=False, header=False, encoding='utf8', sep='\t')
    with ZipFile(os.path.join(config['output_path'], 'submission.zip'), 'w') as myzip:
        myzip.write(submission_path, 'submission.tsv')


if __name__ == '__main__':
    main()
