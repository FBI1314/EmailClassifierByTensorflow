#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: word2vec_helpers.py
@time: 2019/1/15 20:06
@desc:
'''
from gensim.models import Word2Vec
import multiprocessing

#是用词向量表征输入样本
def embedding_sentences(sentences,embedding_size=128,window=5,min_count=5,file_to_load=None,file_to_save=None):
    if file_to_load is not None:
        w2vModel=Word2Vec.load(file_to_load)
    else:
        w2vModel=Word2Vec(sentences,size=embedding_size,window=window,min_count=min_count,workers=multiprocessing.cpu_count())
        if(file_to_save is not None):
            w2vModel.save(file_to_save)
    all_vectors=[]
    embeddingDim=w2vModel.vector_size
    embeddingUnknown=[0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector=[]
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    print('allvectors len is:',len(all_vectors))
    return all_vectors
