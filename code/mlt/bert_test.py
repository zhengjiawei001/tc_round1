# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 5:48 下午
# @Author  : zhengjiawei
# @FileName: bert_test.py
# @Software: PyCharm
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained('/zjw/tianchi_oppo/user_data/tmp_data/finetune_output/whole_word_mask_results/checkpoint-7000-0.992')

tokenizer = BertTokenizer.from_pretrained('/zjw/tianchi_oppo/user_data/vocab_.txt')

query_A = '73 74 75 76'
query_B = '77 43 78 76'

model.eval()
example_inputs = tokenizer([query_A, query_B], [query_B, query_A], return_tensors='pt')
logits = model(**example_inputs)[0]
probs = torch.softmax(logits, dim=-1)
outputs = probs[:, 1].mean().item()


print(outputs)

