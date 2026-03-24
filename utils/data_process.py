import json
import os
from collections import Counter
os.chdir('..')
cur = os.getcwd()
print('当前数据处理默认工作目录：', cur)

class TransferData():
    def __init__(self):
        self.label_dict = json.load(open(os.path.join(cur, './data/labels.json')))
        self.seq_tag_dict = json.load(open(os.path.join(cur,'./data/tag2id.json')))
        self.origin_path = os.path.join(cur, './data_origin')
        self.train_filepath = os.path.join(cur, './data/train.txt')

    def transfer(self):
        with open(self.train_filepath, 'w', encoding='utf-8') as fr:
            for root, dirs, files in os.walk(self.origin_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if 'original' not in filepath:
                        continue
                    label_filepath = filepath.replace('.txtoriginal','')
                    print(filepath, '\t\t', label_filepath)
                    res_dict = self.read_label_text(label_filepath)
                    with open(filepath, 'r', encoding='utf-8')as f:
                        content = f.read().strip()
                        for indx, char in enumerate(content):
                            char_label = res_dict.get(indx, 'O')
                            fr.write(char + '\t' + char_label + '\n')
                        # fr.write('\n') # 注意是否直接按行取出样本

    def read_label_text(self, label_filepath):
        res_dict = {}
        for line in open(label_filepath, 'r', encoding='utf-8'):
            # line--》[右髋部\t21\t23\t身体部位]
            res = line.strip().split('\t')
            # res-->['右髋部', '21', '23', '身体部位']
            start = int(res[1])
            end = int(res[2])
            label = res[3]
            label_tag = self.label_dict.get(label)
            for i in range(start, end + 1):
                if i == start:
                    tag = "B-" + label_tag
                else:
                    tag = "I-" + label_tag
                res_dict[i] = tag
        return res_dict


if __name__ == '__main__':
    handler = TransferData()
    handler.transfer()
    pass