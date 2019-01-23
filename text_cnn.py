#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: text_cnn.py
@time: 2019/1/15 21:04
@desc:
'''
import tensorflow as tf
class TextCNN(object):
    def __init__(self,sequence_length, num_classes,
        embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        #Placeholders for input,output,droupout
        self.input_x=tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        #默认正则化为0.0
        l2_loss=tf.constant(0.0)

        #增加一维 #-1表示最后一维:类似图片的三通道信息，文本设置一个通道
        self.embedded_chars=self.input_x
        self.embedded_chars_expended=tf.expand_dims(self.embedded_chars,-1)

        pooled_outputs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%filter_size):
                #构造4维张量：卷积核，[卷积核的高度，词向量维度（卷积核的宽度），1（图像通道数），卷积核个数（输出通道数）]
                filter_shape=[filter_size,embedding_size,1,num_filters]
                # 初始化W，生成的值服从具有指定平均值和标准偏差的正态分布
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
                #卷积 输出feature map：shape是[batch, height, width, channels]这种形式
                conv=tf.nn.conv2d(self.embedded_chars_expended,
                                  W,
                                  # 图像各维步长,一维向量，长度为4，图像通常为[1, x, x, 1]
                                  strides=[1,1,1,1],
                                  # 卷积方式，'SAME'为等长卷积, 'VALID'为窄卷积
                                  padding='VALID',
                                  name='conv')
                #添加relu激活层
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                #添加池化层,返回值shape仍然是[batch, height, width, channels]这种形式
                pooled=tf.nn.max_pool(h,
                                      # 池化窗口大小，长度（大于）等于4的数组，与value的维度对应，一般为[1,height,width,1]，batch和channels上不池化
                                      ksize=[1,sequence_length-filter_size+1,1,1],
                                      # 与卷积步长类似
                                      strides=[1,1,1,1],
                                      padding="VALID",
                                      name="pool"
                                      )
                pooled_outputs.append(pooled)

        #合并池化结果
        num_filters_total=num_filters*len(filter_sizes)
        # 连接第3维为width，即对句子中的某个词，将不同核产生的计算结果（features）拼接起来。
        self.h_pool=tf.concat(pooled_outputs,3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

        #添加dropout
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        #最终分数和预测
        with tf.name_scope("output"):
            W=tf.get_variable("W",
                              shape=[num_filters_total,num_classes],
                              # xavier_initializer函数返回一个用于初始化权重的初始化程序 “Xavier”，这个初始化器是用来保持每一层的梯度大小都差不多相同。
                              initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_classes],name='b'))
            # W和b均为线性参数，因为加了两个参数所以增加了L2损失，都加到了l2_loss里
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)
            # xw_plus_b：matmul(x, weights) + biases.
            self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            #返回最大值所在下标
            self.predictions=tf.argmax(self.scores,1,name="predictions")

        #计算了模型预测值scores和真实值input_y之间的交叉熵损失
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            # 最终的损失值为平均交叉熵损失+L2正则损失，l2_reg_lambda是正则项系数
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

        #模型评估：精确度
        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")