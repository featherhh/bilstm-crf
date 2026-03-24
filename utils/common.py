import os
from LSTM_CRF.config import *
conf = Config()

'''构造数据集'''


def build_data():
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD", 'UNK']
    for line in open(conf.train_path, 'r', encoding='utf-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        if char not in vocab_list:
            vocab_list.append(char)
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    word2id = {wd: index for index, wd in enumerate(vocab_list)}
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id


'''保存字典文件'''


def write_file(wordlist, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(wordlist))


if __name__ == '__main__':
    datas, word2id = build_data()
    print(len(datas))
    print(datas[:4])
    print(word2id)
    print(len(word2id))