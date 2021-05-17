import pandas as pd


def main():
    train_data_1 = pd.read_csv('../tcdata/nlp_round1_data/gaiic_track3_round1_train_20210228.tsv', sep='\t',
                               header=None)
    train_data_2 = pd.read_csv('../tcdata/nlp_round2_data/gaiic_track3_round2_train_20210407.tsv', sep='\t',
                               header=None)

    test_data_a = pd.read_csv('../tcdata/nlp_round1_data/gaiic_track3_round1_testA_20210228.tsv', sep='\t', header=None)
    test_data_b = pd.read_csv('../tcdata/nlp_round1_data/gaiic_track3_round1_testB_20210317.tsv', sep='\t', header=None)

    data = pd.concat([train_data_1.iloc[:, 0:2], train_data_2.iloc[:, 0:2], test_data_a, test_data_b], axis=0)
    data1 = data.iloc[:, [1, 0]]
    data1.columns = [0, 1]
    data2 = pd.concat([data, data1], axis=0)
    data2.to_csv('../user_data/duality_pair_pretrain_no_nsp_25w.tsv', sep='\t', header=None, index=0)
    vocab = []
    for i in range(data.shape[0]):
        data_line_list_ = data.iloc[i, :]
        data_line_list = data_line_list_[0].split(' ') + data_line_list_[1].split(' ')
        vocab.extend(data_line_list)
    vocab = list(set(vocab))

    with open('../user_data/vocab_.txt', 'w') as f:
        vocab_ = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
        for each in vocab_:
            f.write(each)
            f.write('\n')
        for i in range(len(vocab)):
            if len(vocab[i]) > 0:
                f.write(vocab[i])
                f.write('\n')
            if i % 1000 == 0:
                print('进行中')


def main1():
    train_data_1 = pd.read_csv('../tcdata/nlp_round1_data/gaiic_track3_round1_train_20210228.tsv', sep='\t',
                               header=None)
    train_data_2 = pd.read_csv('../tcdata/nlp_round2_data/gaiic_track3_round2_train_20210407.tsv', sep='\t',
                               header=None)
    train_df = pd.concat([train_data_1, train_data_2], axis=0)
    train_df.to_csv('../tcdata/nlp_round2_data/pretrain_data.tsv', sep='\t', header=None, index=0)


if __name__ == '__main__':
    # main()
    main1()
