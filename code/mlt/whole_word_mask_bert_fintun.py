# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 7:56 下午
# @Author  : zhengjiawei
# @FileName: nezha_ngram_finetune_cv7.py
# @Software: PyCharm


import gc
import os
from zipfile import ZipFile

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import multiprocessing
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer

from .utils import load_data, seed_everyone, read_data, load_cv_data, predict, bert_train

multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    config = {
        'overwrite': True,
        'data_path': '../tcdata/nlp_round1_data',
        'data_cache_path': '../user_data/tmp_data/finetune_output/whole_word_mask_cv5_processed/data.pkl',
        'output_path': '../user_data/tmp_data/finetune_output/whole_word_mask_cv5_results',
        'model_path': '../user_data/tmp_data/pretrain_output/whole_word_mask_bert_output/checkpoint-300000',
        'best_model_path': '',
        'batch_size': 64,  # 64
        'num_epochs': 3,  # 3
        'num_folds': 5,  # 5
        'cv': 'cv-',
        'max_seq_len': 32,
        'learning_rate': 2e-5,
        'eps': 0.1,
        'alpha': 0.3,
        'adv': 'fgm',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 500,  # 500
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

    test_pred_df = pd.DataFrame(data={'id': range(len(test_dataloader.dataset) // 2),
                                      'fold1-probs': [0.0] * (len(test_dataloader.dataset) // 2),
                                      'fold1-logits0': [0.0] * (len(test_dataloader.dataset) // 2),
                                      'fold1-logits1': [0.0] * (len(test_dataloader.dataset) // 2),
                                      })
    train_pred_df = pd.DataFrame(data={'id': range(len(train_dev_data['input_ids'])),
                                       'fold1-probs': [0.0] * len(train_dev_data['input_ids']),
                                       'fold1-logits0': [0.0] * len(train_dev_data['input_ids']),
                                       'fold1-logits1': [0.0] * len(train_dev_data['input_ids'])}
                                 )

    fold = 0
    skf = StratifiedKFold(shuffle=True, n_splits=config['num_folds'], random_state=config['seed'])
    for train_idxs, dev_idxs in skf.split(X=train_dev_data['input_ids'], y=train_dev_data['labels']):
        fold += 1
        config['ema_start'] = False

        dev_dataloader, train_dataloader = load_cv_data(collate_fn, config, dev_idxs,
                                                        train_dev_data, train_idxs, None, None)
        seed_everyone(config['seed'])

        if not config['best_model_path']:
            best_model_path = bert_train(config, train_dataloader, dev_dataloader, fold)
        else:
            best_model_path = config['best_model_path']

        if best_model_path:
            print('\n>>> Loading best model ...')
            bert_config = BertConfig.from_pretrained(config['model_path'])
            model = BertForSequenceClassification.from_pretrained(best_model_path, config=bert_config
                                                                  )

            model.to(config['device'])

            # if config['n_gpus'] > 1:
            #     model = nn.DataParallel(model)

        train_pred_probs, train_pred_logits = predict(config, model, eval_train_dataloader, mode='valid')
        train_pred_df.loc[:, f'fold{fold}-probs'] = train_pred_probs
        train_pred_df.loc[:, f'fold{fold}-logits0'] = train_pred_logits[:, 0]
        train_pred_df.loc[:, f'fold{fold}-logits1'] = train_pred_logits[:, 1]
        test_pred_probs, test_pred_logits = predict(config, model, test_dataloader, mode='test')
        test_pred_df.loc[:, f'fold{fold}-probs'] = test_pred_probs
        test_pred_df.loc[:, f'fold{fold}-logits0'] = test_pred_logits[:, 0]
        test_pred_df.loc[:, f'fold{fold}-logits1'] = test_pred_logits[:, 1]
        del model, train_dataloader, dev_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    print('\n>>> Saving train test predict probability ...')
    train_pred_df.to_csv(os.path.join(config['output_path'], 'train_predicted.tsv'),
                         sep='\t', index=False, encoding='utf8')
    test_pred_df.to_csv(os.path.join(config['output_path'], 'test_predicted.tsv'),
                        sep='\t', index=False, encoding='utf8')
    print('\n>>> Saving submission file ...')
    submission_path = os.path.join(config['output_path'], 'submission.tsv')
    test_pred_df.loc[:, 'submission'] = test_pred_df.loc[:,
                                        [f'fold{i}-probs' for i in range(1, config['num_folds'] + 1)]].mean(axis=1)
    test_pred_df.loc[:, ['submission']].to_csv(submission_path, index=False, header=False, encoding='utf8', sep='\t')
    with ZipFile(os.path.join(config['output_path'], 'submission.zip'), 'w') as myzip:
        myzip.write(submission_path, 'submission.tsv')


if __name__ == '__main__':
    main()
