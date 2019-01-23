#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: train.py
@time: 2019/1/15 19:18
@desc:
'''
import tensorflow as tf
import numpy as np
import os
import data_helpers
import time
import word2vec_helpers
from text_cnn import TextCNN
import datetime

# -----------数据加载参数---------------
# 训练集中用于交叉验证的比例
tf.flags.DEFINE_float("dev_sample_percentage",.1,"Percentage of the training data to use for validation")
# 正例样本路径
tf.flags.DEFINE_string("positive_data_file", "./data/ham_5000.utf8", "Data source for the positive data.")
# 负例样本路径
tf.flags.DEFINE_string("negative_data_file", "./data/spam_5000.utf8", "Data source for the negative data.")
#label个数
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")

# -----------模型参数---------------
#e词向量维度
tf.flags.DEFINE_integer("embedding_dim",128,"Dimensionality of character embedding(default: 128)")
#卷积核尺寸
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-spearated filter sizes (default: '3,4,5')")
#每个尺寸的卷积核数量
tf.flags.DEFINE_integer("num_filters",128,"Number of filters per filter size (default: 128)")
#droupout比例
tf.flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
#正则化大小
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

#-----------训练参数 - --------------
#批处理大小
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#总训练次数
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
#每训练多少步后验证一次
tf.flags.DEFINE_integer("evaluate_every", 100, "Evalue model on dev set after this many steps (default: 100)")
#每训练多少步后保存一下模型
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (defult: 100)")
#保存模型次数
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc parameters
# 加上一个布尔类型的参数，要不要自动分配
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# 加上一个布尔类型的参数，要不要打印日志
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#打印参数
FLAGS=tf.flags.FLAGS
#生成字典形式
FLAGS.flag_values_dict()
print("\nParameters:")
for k in FLAGS:
    v = FLAGS[k].value
    print("{}={}".format(k.upper(),v))
print("")

# 为模型和计算结果生成存放目录
# =======================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#加载数据
x_test,y=data_helpers.load_positive_negative_data_files(FLAGS.positive_data_file,FLAGS.negative_data_file)
print(len(x_test),len(y))
#获取输入样本的embedding表示
sentences,max_document_length=data_helpers.padding_sentences(x_test,'<PADDING>')
print(len(sentences))
x=np.array(word2vec_helpers.embedding_sentences(sentences,embedding_size=FLAGS.embedding_dim,file_to_save=os.path.join(out_dir,'trained_word2vec.model')))

print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

# 保存参数
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
data_helpers.saveDict(params, training_params_file)

# Shuffle数据，随机打乱
np.random.seed(10)
print(len(y))
print(x[2])
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 划分训练集与测试集
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

#训练
with tf.Graph().as_default():
    #添加session配置项
    session_conf=tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess=tf.Session(config=session_conf)
    with sess.as_default():
        cnn=TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        #定义训练过程
        # global_step 记录梯度更新次数，更新一次加1
        global_step=tf.Variable(0,name="global_step",trainable=False)
        optimizer=tf.train.AdamOptimizer(1e-3)
        grads_and_vars=optimizer.compute_gradients(cnn.loss)
        train_op=optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        #跟踪梯度值和稀疏性 (可选)
        grad_summaries=[]
        for g,v in grads_and_vars:
            if g is not None:
                # summary.histogram:用来显示直方图信息
                grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name),g)
                #summary.scalar:用来显示标量信息
                sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged=tf.summary.merge(grad_summaries)

        # 输出模型目录
        print("Writing to {}\n".format(out_dir))

        #总结损失值和准确率
        loss_summary=tf.summary.scalar("loss",cnn.loss)
        acc_summary=tf.summary.scalar("accuracy",cnn.accuracy)

        #训练总结信息
        train_summary_op=tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged])
        train_summary_dir=os.path.join(out_dir,"summaries","train")
        # summary.FileWriter:指定一个文件用来保存图。
        train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)

        # 验证总结信息
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # checkpoint目录。若这个目录已经存在，所以我们需要创建它
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #用于模型保存的对象
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            单次训练
            """
            print("x.shape = {}".format(x_batch.shape))
            print("y.shape = {}".format(y_batch.shape))
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # 记录
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            验证集上验证模型
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        # 生成批次
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print(batches)
        # 循环训练
        for batch in batches:
            print(zip(*batch))
            bzp = zip(*batch)  #[x1,x2,...x64]  [y1,y2,...,y64]
            x_batch=[]
            y_batch=[]
            while True:
                try:
                    tup=bzp.__next__()
                    x_batch.append(tup[0][0])
                    y_batch.append(tup[0][1])
                except:
                    break
            x_batch=np.array(x_batch)
            y_batch=np.array(y_batch)

            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

