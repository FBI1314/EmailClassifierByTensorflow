#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: data_helpers.py
@time: 2019/1/15 19:46
@desc:
'''
import re
import numpy as np
import pickle


def load_positive_negative_data_files(positive_data_file,negative_data_file):
    # 从文件加载数据
    positive_examples = read_and_clean_zh_file(positive_data_file)
    negative_examples = read_and_clean_zh_file(negative_data_file)
    #合并数据
    x_test=positive_examples+negative_examples

    #生成Label,one-hot形式
    positive_labels=[[0,1] for _ in positive_examples]
    negative_labels=[[1,0] for _ in negative_examples]

    y=np.concatenate([positive_labels,negative_labels],0)
    return [x_test,y]

#将输入数据转换成固定大小的embedding
def padding_sentences(input_sentences,padding_token,padding_sentence_length=None):
    sentences=[sentence.split(' ') for sentence in input_sentences]
    max_sentence_length=padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    new_sentences=[]
    for sentence in sentences:
        if len(sentence)>max_sentence_length:
            sentence=sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token]*(max_sentence_length-len(sentence)))
        new_sentences.append(sentence)
    return (new_sentences,max_sentence_length)

# 批量数据生成器
def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    print('data_size==',data_size)
    num_batches_per_epoch=int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices=np.random.shuffle(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data
        for batch_num in range(num_batches_per_epoch):
            start_idx=batch_num*batch_size
            end_idx=min((batch_num+1)*batch_size,data_size)
            print(start_idx,end_idx)
            yield shuffled_data[start_idx:end_idx]
#切词
def seperate_line(line):
    return ''.join([word +' ' for word in line])

#读取文件并清洗数据
def read_and_clean_zh_file(data_file,output_cleaned_file=None):
    lines=list(open(data_file,'rb').readlines())
    lines=[clean_str(seperate_line(line.decode('utf-8'))) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file,'w') as f:
            for line in lines:
                f.write((line+'\n').encode('utf-8'))
    return lines


def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict