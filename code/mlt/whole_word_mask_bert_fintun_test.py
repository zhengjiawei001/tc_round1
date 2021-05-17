# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 7:56 下午
# @Author  : zhengjiawei
# @FileName: nezha_ngram_finetune_cv7.py
# @Software: PyCharm


import os

import torch
from sklearn.model_selection import StratifiedKFold
from torch import multiprocessing
from transformers import BertTokenizer

from .utils import load_data, seed_everyone, read_data, load_cv_data, bert_train

multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    config = {
        'overwrite': True,
        'data_path': '../tcdata/nlp_round2_data',
        'data_cache_path': '../user_data/tmp_data/finetune_output/whole_word_mask_processed/data.pkl',
        'output_path': '../user_data/tmp_data/finetune_output/whole_word_mask_results',
        'model_path': '../user_data/tmp_data/pretrain_output/whole_word_mask_bert_output/best_model_ckpt',
        'best_model_path': '',
        'batch_size': 128,  # 64
        'num_epochs': 3,  # 3
        'num_folds': 5,  # 5
        'cv': '',
        'max_seq_len': 32,
        'learning_rate': 2e-5,
        'eps': 0.1,
        'alpha': 0.3,
        'adv': 'fgm',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 1000,  # 500
        'ema_start_step': 1500,  # 1500
        'ema_start': False,
        'seed': 20200409
    }
    print(config['data_cache_path'])
    if not torch.cuda.is_available():
        config['device'] = 'cpu'
    else:
        config['n_gpus'] = torch.cuda.device_count()
        config['batch_size'] *= config['n_gpus']

    if not os.path.exists(config['output_path']):
        os.makedirs((config['output_path']))

    tokenizer = BertTokenizer.from_pretrained('../user_data/vocab_.txt')
    if not os.path.exists(config['data_cache_path']) or config['overwrite']:
        read_data(config, tokenizer, debug=False)

    collate_fn, test_dataloader, train_dev_data, eval_train_dataloader = load_data(config, tokenizer)

    fold = 0
    skf = StratifiedKFold(shuffle=True, n_splits=config['num_folds'], random_state=config['seed'])
    for train_idxs, dev_idxs in skf.split(X=train_dev_data['input_ids'], y=train_dev_data['labels']):
        fold += 1
        config['ema_start'] = True

        dev_dataloader, train_dataloader = load_cv_data(collate_fn, config, dev_idxs,
                                                        train_dev_data, train_idxs, None, None)
        seed_everyone(config['seed'])

        if not config['best_model_path']:
            best_model_path = bert_train(config, train_dataloader, dev_dataloader, fold)
        else:
            best_model_path = config['best_model_path']

        if not config['cv']:
            break


if __name__ == '__main__':
    main()
