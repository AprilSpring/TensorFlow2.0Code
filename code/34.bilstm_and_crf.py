#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/21 10:55 上午

@annotaion: 
    bilstm+crf实现

@author: tinghai
"""

import tensorflow as tf
import tensorflow_addons as tf_ad
import argparse
import json
import logging
logger = logging.getLogger('default')


parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="./data/train.txt",help="train file")
parser.add_argument("--test_path", type=str, default="./data/test.txt",help="test file")
parser.add_argument("--output_dir", type=str, default="checkpoints/",help="output_dir")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=100,help="epoch")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
args = parser.parse_args()


#%% 1) model construction
class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(NerModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(0.5)

    # @tf.function
    def call(self, text,labels=None,training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            # 之前使用 tf.contrib.crf.crf_log_likelihood
            # 4个参数：
#             inputs：batch_size x max_seq_len x num_tags的三维矩阵，即是上一层结果的输入。
#             tag_indices：batch_size x max_seq_len的二维矩阵，是标签。
#             sequence_lengths：batch_size向量，序列长度。
#             transition_params：num_tags x num_tags的状态转移矩阵，可以为空
            # 2个输出：
#             log_likelihood：对数似然值。
#             transition_params：状态转移矩阵
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens


#%% 2) data process
def tokenize(filename, vocab2id, tag2id):
    contents = []
    labels = []
    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip() for elem in fr.readlines()][:500000]:
            try:
                if line != "end":
                    w,t = line.split()
                    content.append(vocab2id.get(w,0))
                    label.append(tag2id.get(t,0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post') #'post' 文本后补齐
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,labels


vocab2id = {}
tag2id = {}
text_sequences ,label_sequences = tokenize(args.train_path, vocab2id, tag2id)
#text_sequences: 每条文本是一个长度为seq_length的列表，每个列表中的每个字由一个长度为L的向量组成，且每个字转换为一个ID编码
#label_sequences: 每条文本对应一个长度为seq_length的列表，每个列表中的每个字对应一个实体识别的label

train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

model = NerModel(hidden_num = args.hidden_num, vocab_size = len(vocab2id), label_size= len(tag2id), embedding_size = args.embedding_size)
optimizer = tf.keras.optimizers.Adam(args.lr)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          args.output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)

#%% 3）train
# @tf.function
def train_one_step(text_batch, labels_batch):
  with tf.GradientTape() as tape:
      logits, text_lens, log_likelihood = model(text_batch, labels_batch,training=True)
      loss = - tf.reduce_mean(log_likelihood)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,logits, text_lens


def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy


best_acc = 0
step = 0
for epoch in range(args.epoch):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            logger.info('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
              best_acc = accuracy
              ckpt_manager.save()
              logger.info("model saved")


#%% 4) predict
optimizer = tf.keras.optimizers.Adam(args.lr)
model = NerModel(hidden_num = args.hidden_num, vocab_size =len(vocab2id), label_size = len(tag2id), embedding_size = args.embedding_size)
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model) # restore model
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))

text = input("input:")
test_dataset = tf.keras.preprocessing.sequence.pad_sequences([[vocab2id.get(char,0) for char in text]], padding='post')
logits, text_lens = model.predict(test_dataset)

paths = []
for logit, text_len in zip(logits, text_lens):
    viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
    paths.append(viterbi_path)
print(paths[0])
print([id2tag[id] for id in paths[0]])

entities_result = format_result(list(text), [id2tag[id] for id in paths[0]])
print(json.dumps(entities_result, indent=4, ensure_ascii=False))



