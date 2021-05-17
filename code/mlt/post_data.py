# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 3:15 上午
# @Author  : zhengjiawei
# @FileName: post_data.py
# @Software: PyCharm
import os
from zipfile import ZipFile

import pandas as pd


def main():
    data1 = pd.read_csv('../user_data/tmp_data/finetune_output/nezha_ngram_cv7_results/submission.tsv', header=None)
    data2 = pd.read_csv('../user_data/tmp_data/finetune_output/nezha_base_cv7_results/submission.tsv', header=None)
    data3 = pd.read_csv('../user_data/tmp_data/finetune_output/whole_word_mask_cv5_results/submission.tsv', header=None)
    data = data1 * 0.4 + data2 * 0.3 + data3 * 0.3
    data.to_csv('../prediction_result/sub.tsv', header=False, index=False, sep='\t')
    with ZipFile(os.path.join('../../prediction_result', 'submission.zip'), 'w') as myzip:
        myzip.write('../prediction_result', 'sub.tsv')


if __name__ == '__main__':
    main()
