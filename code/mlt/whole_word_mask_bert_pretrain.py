import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertOnlyMLMHead,
    MaskedLMOutput,
)

wandb.login(key='8cefb8016177b89343b4f6c8eed0c154b55b006b')
os.system('wandb online')
os.environ['WANDB_PROJECT'] = 'bert_oppo_pretrain'

from .utils import seed_everyone

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config=config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_file_path, tokenizer: BertTokenizer, debug=False) -> dict:
    train_data = pd.read_csv(train_file_path, header=None, sep='\t')
    if debug:
        train_data = train_data.head(1000)

    inputs = defaultdict(list)
    for i, row in tqdm(train_data.iterrows(), desc=f'Preprocessing train data', total=len(train_data)):
        sentence_a, sentence_b = row[0], row[1]
        inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class TcDataset(Dataset):

    def __init__(self, data_dict: dict):
        super(TcDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


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

    def bert_ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len, seed=i)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def bert_mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        ngram_batch_mask = self.bert_ngram_mask(input_ids_list, max_seq_len)

        input_ids, mlm_labels = self.bert_mask_tokens(input_ids, ngram_batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


# class Whole_Word_Mask_Collator:
#     def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
#         self.max_seq_len = max_seq_len
#         self.tokenizer = tokenizer
#         self.mlm_probability = mlm_probability
#
#     def truncate_and_pad(self, input_ids_list, token_type_ids_list,
#                          attention_mask_list, max_seq_len):
#         input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
#         token_type_ids = torch.zeros_like(input_ids)
#         attention_mask = torch.zeros_like(input_ids)
#         for i in range(len(input_ids_list)):
#             seq_len = min(len(input_ids_list[i]), max_seq_len)
#             if seq_len <= max_seq_len:
#                 input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
#             else:
#                 input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
#                                                       [self.tokenizer.sep_token_id], dtype=torch.long)
#             token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
#             attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
#         return input_ids, token_type_ids, attention_mask
#
#     def _whole_word_mask(self, input_ids_list: List[str], max_seq_len: int, max_predictions=512):
#         cand_indexes = []
#         for (i, token) in enumerate(input_ids_list):
#             if (token == str(self.tokenizer.cls_token_id)
#                     or token == str(self.tokenizer.sep_token_id)):
#                 continue
#
#             if len(cand_indexes) >= 1 and token.startswith("##"):
#                 cand_indexes[-1].append(i)
#             else:
#                 cand_indexes.append([i])
#
#         random.shuffle(cand_indexes)
#         num_to_predict = min(max_predictions, max(1, int(round(len(input_ids_list) * self.mlm_probability))))
#         masked_lms = []
#         covered_indexes = set()
#         for index_set in cand_indexes:
#             if len(masked_lms) >= num_to_predict:
#                 break
#             if len(masked_lms) + len(index_set) > num_to_predict:
#                 continue
#             is_any_index_covered = False
#             for index in index_set:
#                 if index in covered_indexes:
#                     is_any_index_covered = True
#                     break
#             if is_any_index_covered:
#                 continue
#             for index in index_set:
#                 covered_indexes.add(index)
#                 masked_lms.append(index)
#
#         assert len(covered_indexes) == len(masked_lms)
#         mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids_list))]
#         mask_labels += [0] * (max_seq_len - len(mask_labels))
#         return torch.tensor(mask_labels)
#
#     def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#
#         labels = inputs.clone()
#
#         probability_matrix = mask_labels
#
#         special_tokens_mask = [
#             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#         ]
#         probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
#         if self.tokenizer.pad_token is not None:
#             padding_mask = labels.eq(self.tokenizer.pad_token_id)
#             probability_matrix.masked_fill_(padding_mask, value=0.0)
#
#         masked_indices = probability_matrix.bool()
#         labels[~masked_indices] = -100
#
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
#
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]
#
#         return inputs, labels
#
#     def whole_word_mask(self, input_ids_list: List[list], max_seq_len: int) -> torch.Tensor:
#         mask_labels = []
#         for input_ids in input_ids_list:
#             wwm_id = random.choices(range(len(input_ids)), k=int(len(input_ids) * 0.2))
#             input_id_str = [f'##{id_}' if i in wwm_id else str(id_) for i, id_ in enumerate(input_ids)]
#             mask_label = self._whole_word_mask(input_id_str, max_seq_len)
#             mask_labels.append(mask_label)
#         return torch.stack(mask_labels, dim=0)
#
#     def __call__(self, examples: list) -> dict:
#         input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
#         cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
#         max_seq_len = min(cur_max_seq_len, self.max_seq_len)
#
#         input_ids, token_type_ids, attention_mask = self.truncate_and_pad(input_ids_list,
#                                                                           token_type_ids_list,
#                                                                           attention_mask_list,
#                                                                           max_seq_len)
#         batch_mask = self.whole_word_mask(input_ids_list, max_seq_len)
#         input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
#         data_dict = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'token_type_ids': token_type_ids,
#             'labels': mlm_labels
#         }
#
#         return data_dict


def main():
    seed_everyone(20210318)

    raw_data_path = '../user_data/duality_pair_pretrain_no_nsp.tsv'
    output_dir = '../user_data/tmp_data/pretrain_output/whole_word_mask_bert_output'

    tokenizer = BertTokenizer.from_pretrained('../user_data/vocab_.txt')
    data = read_data(raw_data_path, tokenizer, debug=False)

    train_dataset = TcDataset(data)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=100,
        type_vocab_size=2,
        pad_token_id=0,
    )
    model = BertForMaskedLM(config=config)
    wandb.init(project=f"bert_oppo_pretrain1", entity="zjw", dir=output_dir)
    data_collator = Data_Collator(max_seq_len=42, tokenizer=tokenizer, mlm_p=0.15)

    model_save_dir = os.path.join(output_dir, 'best_model_ckpt')
    tokenizer_and_config = os.path.join(output_dir, 'tokenizer_and_config')
    check_dir(model_save_dir)
    check_dir(tokenizer_and_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=100,
        fp16_backend='auto',
        per_device_train_batch_size=128,
        save_steps=500,
        logging_steps=500,
        save_total_limit=10,
        prediction_loss_only=True,
        run_name='0419',
        logging_first_step=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        seed=202104
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(model_save_dir)
    config.save_pretrained(tokenizer_and_config)
    tokenizer.save_pretrained(tokenizer_and_config)


if __name__ == '__main__':
    main()
