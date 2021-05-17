# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 4:32 下午
# @Author  : zhengjiawei
# @FileName: nezha_ngram_pretrain.py
# @Software: PyCharm

import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from .nezha.config import NeZhaConfig
from .nezha.model import NeZhaForMaskedLM
from .utils import seed_everyone

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def read_data(train_file_path, tokenizer: BertTokenizer, debug=False) -> dict:
    train_data = pd.read_csv(train_file_path, header=None, sep='\t')
    if debug:
        train_data = train_data.head(1000)
    data_dict = defaultdict(list)
    for i, row in tqdm(train_data.iterrows(), desc=f'Preprocessing train data', total=train_data.shape[0]):
        text_a, text_b = row[0], row[1]
        inputs_dict = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        data_dict['input_ids'].append(inputs_dict['input_ids'])
        data_dict['token_type_ids'].append(inputs_dict['token_type_ids'])
        data_dict['attention_mask'].append(inputs_dict['attention_mask'])

    return data_dict


class Data_Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_p=0.15):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_p = mlm_p
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def truncate_and_pad(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):

        inputs_id = torch.zeros((len(input_ids_list), max_seq_len),
                                dtype=torch.long)
        attention_mask = torch.zeros_like(inputs_id)
        tokens_type_id = torch.zeros_like(inputs_id)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                inputs_id[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                inputs_id[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
            tokens_type_id[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
        return inputs_id, tokens_type_id, attention_mask

    def nezha_ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len, seed=i)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def nezha_mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _ngram_mask(self, input_ids, max_seq_len, seed):
        np.random.seed(seed)

        cand_indexes = []

        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])

        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_p)))

        ngrams = np.arange(1, 4, dtype=np.int64)
        pvals = 1. / np.arange(1, 4)
        pvals /= pvals.sum(keepdims=True)

        # favor_shorter_ngram:
        pvals = pvals[::-1]

        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)

        np.random.shuffle(ngram_indexes)

        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            # Repeatedly looking for a candidate that does not exceed the
            # maximum number of predictions by trying shorter ngrams.
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.truncate_and_pad(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        nezha_ngram_batch_mask = self.nezha_ngram_mask(input_ids_list, max_seq_len)

        input_ids, mlm_labels = self.nezha_mask_tokens(input_ids, nezha_ngram_batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


class TcDataset(Dataset):

    def __init__(self, data_dict):
        super(TcDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index):
        data = (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self):
        return len(self.data_dict['input_ids'])


def set_path():
    pair_path = '../user_data/duality_pair_pretrain_no_nsp_25w.tsv'
    model_path = '../user_data/model_data/nezha-cn-base'
    output_path = '../user_data/tmp_data/pretrain_output/nezha_ngram_output'
    return pair_path, model_path, output_path


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    transformers.logging.set_verbosity_info()
    seed_everyone(20210312)
    raw_data_path, model_path, output_path = set_path()
    nezha_ngram_config = NeZhaConfig.from_pretrained(model_path)
    nezha_ngram_tokenizer = BertTokenizer.from_pretrained(model_path)
    data = read_data(raw_data_path, nezha_ngram_tokenizer, debug=False)
    train_dataset = TcDataset(data)
    nehze_model = NeZhaForMaskedLM.from_pretrained(model_path, config=nezha_ngram_config)

    data_collator = Data_Collator(max_seq_len=42, tokenizer=nezha_ngram_tokenizer, mlm_p=0.15)

    model_save_path = os.path.join(output_path, 'nezha_ngram_best_model_ckpt')
    nezha_ngram_tokenizer_and_config = os.path.join(output_path, 'nezha_ngram_tokenizer_and_config')
    check_path(model_save_path)
    check_path(nezha_ngram_tokenizer_and_config)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=25,
        fp16_backend='amp',
        per_device_train_batch_size=128,
        save_steps=1000,
        logging_steps=1000,
        save_total_limit=10,
        prediction_loss_only=True,
        run_name='44',
        logging_first_step=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        seed=20210409
    )

    nezha_ngram_trainer = Trainer(
        model=nehze_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    nezha_ngram_trainer.train()

    nezha_ngram_trainer.save_model(model_save_path)
    nezha_ngram_config.save_pretrained(nezha_ngram_tokenizer_and_config)
    nezha_ngram_tokenizer.save_pretrained(nezha_ngram_tokenizer_and_config)


if __name__ == '__main__':
    main()
