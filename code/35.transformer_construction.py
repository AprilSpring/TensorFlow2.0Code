#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/21 7:32 下午

@annotaion:
    如何构建Transformer? self-attention？multi-head attention?
    参考：https://zhuanlan.zhihu.com/p/64881836

@author: tinghai
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
# pip install tfds-nightly==1.0.2.dev201904090105


#%% 超参数设置
EPOCHS = 20
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH=40
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
max_seq_len = 40
dropout_rate = 0.1


#%% 数据集导入
# 使用到Portugese-English翻译数据集
# 该数据集包含大约50000个训练样例，1100个验证示例和2000个测试示例。
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']


#%% 数据集处理
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

input_vocab_size = tokenizer_pt.vocab_size + 2  # +2 添加起始和终止表示符
target_vocab_size = tokenizer_en.vocab_size + 2

# token转化测试
# sample_str = 'hello world, tensorflow 2'
# tokenized_str = tokenizer_en.encode(sample_str)
# print(tokenized_str)
# original_str = tokenizer_en.decode(tokenized_str)
# print(original_str)

# 添加起始和终止表示符
def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

def tf_encode(pt, en):
    # 使用tf.py_function对python的函数进行封装, 需要声明三个变量,
    # func就是python预处理的函数, inp就是函数的参数, Tout就是python函数返回的数据类型
    return tf.py_function(func=encode, inp=[pt, en], Tout=[tf.int64, tf.int64])

# 过滤掉长度超过40的数据
def filter_long_sent(x, y, max_length=MAX_LENGTH):
    # tf.logical_and(True, False)计算2项输入的逻辑输出结果
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


#%% 生成datasets
# 训练数据
train_dataset = train_examples.map(tf_encode) # 数据处理函数map操作
train_dataset = train_dataset.filter(filter_long_sent) # 过滤过长的数据
train_dataset = train_dataset.cache() # 使用缓存数据加速读入
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([40], [40])) # 填充为最大长度-90
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) # 设置预取数据
de_batch, en_batch = next(iter(train_dataset)) # 按batch查看数据

# 验证数据
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_long_sent)
val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([40], [40]))


#%% 位置编码
def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


#%% 计算context attention
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) #（.., seq_len_v, depth）

    return output, attention_weights


#%% 构造mutil-head attention层
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0

        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights


# 测试
# temp_mha = MutilHeadAttention(d_model=512, num_heads=8)
# y = tf.random.uniform((1, 60, 512))
# output, att = temp_mha(y, k=y, q=y, mask=None)
# print(output.shape, att.shape) #(1, 60, 512) (1, 8, 60, 60)


#%% feed-forward
def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

# 测试
# sample_fnn = point_wise_feed_forward_network(512, 2048)
# sample_fnn(tf.random.uniform((64, 50, 512))).shape #TensorShape([64, 50, 512])


#%% Layer标准化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


#%% Encoder单层构建
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ddf)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

# 测试
# sample_encoder_layer = EncoderLayer(512, 8, 2048)
# sample_encoder_layer_output = sample_encoder_layer(
#     tf.random.uniform((64, 43, 512)), False, None)
# print(sample_encoder_layer_output.shape) #TensorShape([64, 43, 512])


#%% Encoder模块构建（eg., 12层）
class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encode_layer = [EncoderLayer(d_model, n_heads, ddf, drop_rate) for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training, mark):
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding[:,:seq_len,:]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encode_layer[i](x, training, mark) #调用call方法

        return x

# 测试
# sample_encoder = Encoder(2, 512, 8, 1024, 5000, 200)
# sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)), False, None)
# print(sample_encoder_output.shape) #TensorShape([64, 120, 512])


#%% Decoder单层构建
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MutilHeadAttention(d_model, num_heads)
        self.mha2 = MutilHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, inputs, encode_out, training, look_ahead_mask, padding_mask):
        # masked muti-head attention
        att1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask) #inputs相当于query
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)

        # muti-head attention
        att2, att_weight2 = self.mha2(encode_out, encode_out, inputs, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(out1 + att2)

        # 前向网络
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3, att_weight1, att_weight2

# 测试
# sample_decoder_layer = DecoderLayer(512, 8, 2048)
# sample_decoder_layer_output, _, _ = sample_decoder_layer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,False, None, None)
# print(sample_decoder_layer_output.shape) #TensorShape([64, 50, 512])


#%% Decoder模块构建（eg., 12层）
class Decoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decoder_layers= [DecoderLayer(d_model, n_heads, ddf, drop_rate)
                             for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mark, padding_mark):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:,:seq_len,:]

        h = self.dropout(h, training=training)
#         print('--------------------\n',h, h.shape)
        # 叠加解码层
        for i in range(self.n_layers):
            h, att_w1, att_w2 = self.decoder_layers[i](h, encoder_out,
                                                   training, look_ahead_mark,
                                                   padding_mark)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att_w1
            attention_weights['decoder_layer{}_att_w2'.format(i+1)] = att_w2

        return h, attention_weights

# 测试
# sample_decoder = Decoder(2, 512,8,1024,5000, 200)
# sample_decoder_output, attn = sample_decoder(tf.random.uniform((64, 100)),
#                                             sample_encoder_output, False,
#                                             None, None)
# print(sample_decoder_output.shape, attn['decoder_layer1_att_w1'].shape) #(TensorShape([64, 100, 512]), TensorShape([64, 8, 100, 100]))


#%% Transfomer构建
class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                input_vocab_size, target_vocab_size,
                max_seq_len, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, d_model, n_heads,diff,
                              input_vocab_size, max_seq_len, drop_rate)

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                              target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, encode_padding_mask,
            look_ahead_mask, decode_padding_mask):

        encode_out = self.encoder(inputs, training, encode_padding_mask)
        print(encode_out.shape)
        decode_out, att_weights = self.decoder(targets, encode_out, training,
                                               look_ahead_mask, decode_padding_mask)
        print(decode_out.shape)
        final_out = self.final_layer(decode_out)

        return final_out, att_weights

# 测试
# temp_input = tf.random.uniform((64, 62))
# temp_target = tf.random.uniform((64, 26))
# sample_transformer = Transformer(n_layers=2,
#                                  d_model=512,
#                                  n_heads=8,
#                                  diff=1024,
#                                  input_vocab_size=8500,
#                                  target_vocab_size=8000,
#                                  max_seq_len=120)
# fn_out, _ = sample_transformer(temp_input,
#                                temp_target,
#                                training=False,
#                                code_padding_mask=None,
#                                look_ahead_mask=None,
#                                decode_padding_mask=None,)
# print(fn_out.shape)


#%% 构建transformer对象
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          max_seq_len, dropout_rate)


#%% 自定义学习速率（随着训练逐渐减少LR）
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learing_rate = CustomSchedule(d_model)


#%% 优化器
optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,
                                    beta_2=0.98, epsilon=1e-9)


#%% 损失函数
def loss_fun(y_ture, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


#%% 评价指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


#%% 创建checkpoint管理器
checkpoint_path = './checkpoint/train'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('last checkpoit restore')


#%% 掩码构建
# padding-mask: 为了避免输入中padding对句子语义的影响，需要将padding位mark掉，原来为0的padding项，mark后变为为1
def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)

# 测试
# create_padding_mark([[1,2,0,0,3],[3,4,5,0,0]])


# look-ahead mask: 对未预测的token进行掩码
def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部），这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)

# 测试
# create_look_ahead_mark(3)


def create_mask(inputs, targets):
    encode_padding_mask = create_padding_mark(inputs)

    # 这个掩码用于掩输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mark(inputs)

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])

    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask


#%% 单步训练
@tf.function
def train_step(inputs, targets):
    tar_inp = targets[:,:-1]
    tar_real = targets[:,1:]
    # 构造掩码
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs, tar_inp,
                                    True,
                                    encode_padding_mask,
                                    combined_mask,
                                    decode_padding_mask)
        loss = loss_fun(tar_real, predictions)

    # 求梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)


#%% 开始训练
for epoch in range(EPOCHS):
    start = time.time()

    # 重置记录项
    train_loss.reset_states()
    train_accuracy.reset_states()

    # inputs 葡萄牙语， targets英语
    for batch, (inputs, targets) in enumerate(train_dataset):
        # 训练
        train_step(inputs, targets)

        if batch % 500 == 0:
            print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                epoch+1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('epoch {}, save model at {}'.format(epoch+1, ckpt_save_path))

    print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
        epoch+1, train_loss.result(), train_accuracy.result()))

    print('time in 1 epoch:{} secs\n'.format(time.time()-start))


