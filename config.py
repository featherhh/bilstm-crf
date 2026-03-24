import os
import torch
import json


class Config(object):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        # self.device = 'mps'
        self.train_path = 'F:\\ai资料\第八阶段-nlp2\完整代码\MedicalKB\\Ner\LSTM_CRF\data\\train.txt'
        self.vocab_path = 'F:\\ai资料\第八阶段-nlp2\完整代码\MedicalKB\\Ner\LSTM_CRF\\vocab\\vocab.txt'
        self.embedding_dim = 300
        self.epochs = 5
        self.batch_size = 16
        self.hidden_dim = 256
        self.lr = 2e-3
        # self.crf_lr = 1e-3
        self.dropout = 0.2
        self.model = "BiLSTM"
        # self.model = "BiLSTM_CRF"
        self.tag2id = json.load(open('F:\\ai资料\第八阶段-nlp2\完整代码\MedicalKB\\Ner\LSTM_CRF\data\\tag2id.json'))

if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.tag2id)