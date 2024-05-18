import torch
import collections
import re
import math
from torch import nn
from torch.nn import functional as F


# ***********************数据预处理************************************
time_machine = "/home/peihuanni/d2l/d2l-zh/pytorch/data/timemachine.txt"

def read_time_machine():
    """把timemachine中的每一行忽略标点符号和字母大写存到到一个列表里,列表的每一行都是一个字符串"""
    with open(time_machine, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='char'):
    """根据token类型把每一行拆成单个字母char或单个单词word"""
    if token == 'char':
        return list(list(line) for line in lines)
    if token == 'word':
        return list(line.split() for line in lines)

def count_corpus(tokens):
    """统计次元的频率"""
    if isinstance(tokens[0], list):
        tokens = list(token for line in tokens for token in line)
    return tokens, collections.Counter(tokens)

def get_second_element(item):
    return item[1]

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 返回一个统计tokens频率的字典，输出形如输出: Counter({'banana': 2, 'apple': 3, 'orange': 1})
        _, counter = count_corpus(tokens)
        # sorted函数返回一个按频率降序序排列的tokens列表
        self._token_freqs = sorted(counter.items(), key=get_second_element, reverse=True)
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.unk = 0
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            elif token not in self.token_to_idx:
                self.idx_to_token.append(token)
                # 列表，按频率存token，达成idx_to_token的效果，每append一次，len(idx_to_token)加一
                self.token_to_idx[token] = len(self.idx_to_token)-1
                # 字典，key为token，value为从零开始的idx

    def __len__(self):
        return len(self.idx_to_token)
    
    # 这个函数可以获得这个token的索引
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


lines = read_time_machine()
tokens_char = tokenize(lines, token='word')
tokens_1D, count = count_corpus(tokens_char)
vocab = Vocab(tokens_1D)

for i in [1,10]:
    print('text', tokens_char[i])
    print('index', vocab[tokens_char[i]])


# **************************RNN搭建*********************************
batch_size, num_steps = 32, 35
