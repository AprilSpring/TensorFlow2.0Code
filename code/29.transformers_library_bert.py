#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/13 10:59 上午

@annotaion:
    使用transformers实现文本预训练模型的微调（建议在colab中测试）
    参考：https://blog.csdn.net/zkbaba/article/details/103706031

@author: tinghai
"""
import tensorflow as tf
# pip install transformers
# transformers，包含配置类、分词器类、模型类

# 1）加载预训练模型及其分词器
from transformers import TFBertModel, BertTokenizer, \
    BertForSequenceClassification, glue_convert_examples_to_features,\
    TFGPT2Model, GPT2Tokenizer,\
    TFBertForSequenceClassification, TFDistilBertForSequenceClassification

# bert_model = TFBertModel.from_pretrained("bert-base-cased")  # Automatically loads the config
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# gpt2_model = TFGPT2Model.from_pretrained("gpt2")  # Automatically loads the config
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#
# model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased") #更小的bert模型
# bert_tokenizer = DistilbertTokenizer.from_pretrained("distilbert-base-uncased")


# 2）数据处理
import tensorflow_datasets
data, info = tensorflow_datasets.load("glue/mrpc", with_info=True)

train_dataset = data["train"]
validation_dataset = data["validation"]

num_train = info.splits['train'].num_examples
num_valid = info.splits['validation'].num_examples

BATCH_SIZE = 32
train_dataset = glue_convert_examples_to_features(train_dataset, bert_tokenizer, 128, 'mrpc')
validation_dataset = glue_convert_examples_to_features(validation_dataset, bert_tokenizer, 128, 'mrpc')
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat(-1)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

train_batch = next(iter(train_dataset))
print(train_batch[0].keys())

# 3）预训练模型微调
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_history = bert_model.fit(
    train_dataset,
    epochs=3,
    steps_per_epoch=num_train//BATCH_SIZE,
    validation_data=validation_dataset,
    validation_steps=num_valid//BATCH_SIZE
)


# 4）预训练模型存储和加载
FINE_TUNED_MODEL_DIR = "./data/"
model.save_pretrained(FINE_TUNED_MODEL_DIR)

saved_model = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR, from_tf=True)


# 5）预测
sentence_0 = "At least 12 people were killed in the battle last week."
sentence_1 = "At least 12 people lost their lives in last weeks fighting."
sentence_2 = "The fires burnt down the houses on the street."

inputs_1 = bert_tokenizer.encode_plus(sentence_0, sentence_1, return_tensors="pt")
inputs_2 = bert_tokenizer.encode_plus(sentence_0, sentence_2, return_tensors="pt")

pred_1 = saved_model(**inputs_1)[0].argmax().item()
pred_2 = saved_model(**inputs_2)[0].argmax().item()

def print_result(id1, id2, pred):
    if pred == 1:
        print("sentence_1 is a paraphrase of sentence_0")
    else:
        print("sentence_1 is not a paraphrase of sentence_0")

print_result(0, 1, pred_1)
print_result(0, 2, pred_2)




