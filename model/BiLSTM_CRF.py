import torch
import torch.nn as nn
from TorchCRF import CRF
import os
import sys

# ========== 核心排查代码 ==========
# 1. 获取当前文件绝对路径
current_file_path = os.path.abspath(__file__)
print(f"【当前文件完整路径】: {current_file_path}")

# 2. 获取model文件夹路径
model_dir = os.path.dirname(current_file_path)
print(f"【model文件夹路径】: {model_dir}")

# 3. 获取LSTM_CRF根目录（model的父目录）
root_dir = os.path.dirname(model_dir)
print(f"【LSTM_CRF根目录】: {root_dir}")

# 4. 把根目录加入sys.path，并打印sys.path确认
sys.path.append(root_dir)
print(f"\n【Python搜索路径清单】:\n{sys.path}")

# 5. 检查root_dir下是否存在utils文件夹
utils_path = os.path.join(root_dir, "utils")
print(f"\n【utils文件夹是否存在】: {os.path.exists(utils_path)}")
print(f"【utils文件夹路径】: {utils_path}")

# 6. 检查utils下是否存在data_loader.py
data_loader_path = os.path.join(utils_path, "data_loader.py")
print(f"【data_loader.py是否存在】: {os.path.exists(data_loader_path)}")
print(f"【data_loader.py路径】: {data_loader_path}")
# ========== 排查代码结束 ==========
# 2. 把根目录放到搜索清单最前面（最高优先级，避免同名干扰）
sys.path.insert(0, root_dir)
# 尝试导入
from utils.data_loader import *

class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        #lstm模型得到的结果
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs

    def log_likelihood(self, x, tags, mask):
        # lstm模型得到的结果
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        # 计算损失
        return - self.crf(outputs, tags, mask)

    def get_lstm2linear(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs
if __name__ == '__main__':

    train_dataloader, dev_dataloader = get_data()
    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    my_model = NERLSTM_CRF(embedding_dim, hidden_dim, dropout, word2id, tag2id)
    my_model.train()
    for input_ids_padded, labels_padded, attention_mask in train_dataloader:
        print(input_ids_padded.shape)
        print(labels_padded.shape)
        print(attention_mask.shape)
        attention_mask = attention_mask.to(torch.bool)
        # outputs = my_model(x=input_ids_padded, mask=attention_mask)
        outputs = my_model.log_likelihood(x=input_ids_padded,tags=labels_padded,mask=attention_mask)
        print(f'outputs--》{outputs}')
        break
