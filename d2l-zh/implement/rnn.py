import torch
import collections
import re

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
        _, counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=get_second_element, reserve=True)



lines = read_time_machine()
tokens_char = tokenize(lines, token='word')
tokens_1D, count = count_corpus(tokens_char)
# 注意这里，虽然在这个函数中对tokens作了修改
# 但事实上，token_char（实参）是传入给了tokens（型参）
# 在函数里面对型参的操作，并不会影响实参
# 可以运行：
# print(tokens_char)
# print(tokens_1D)
#  二者结果截然不同
