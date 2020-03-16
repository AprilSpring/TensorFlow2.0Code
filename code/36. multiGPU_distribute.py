#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/3/16 4:43 下午

@annotaion: 多GPU并行训练
            参考：https://zhuanlan.zhihu.com/p/88165283

@author: tinghai
"""
import time
import tensorflow as tf


#%% 1）定义变量
EPOCHS = 10

# 创建一个MirroredStrategy分发数据和计算图
strategy = tf.distribute.MirroredStrategy()

# 获取可利用的GPU数量
GPU_available = strategy.num_replicas_in_sync)

batch_size_per_replica = 32

# Global batch size
GLOBAL_BATCH_SIZE = batch_size_per_replica * GPU_available

# Buffer size for data loader
BUFFER_SIZE = batch_size_per_replica * GPU_available * 16


#%% 2）生成datasets（数据处理等归一化内容在此处完成）
dataset = ...

# 使用strategy分发数据
dist_dataset = strategy.experimental_distribute_dataset(dataset)


#%% 3）创建模型、定义优化器、损失函数、检查点、训练过程等，都需要在strategy的scope下
with strategy.scope():
    # 定义模型
    model = create_model(...)

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # 检查点
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5, checkpoint_name='ckpt')

    # 定义损失
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(logits, labels):
        per_example_loss = loss_object(y_true=labels, y_pred=logits)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    # 定义训练过程
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = compute_loss(logits=logits, labels=labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss


#%% 4） 使用experimental_run_v2 执行分布式训练
with strategy.scope():
    @tf.function # 并没有试过去掉这个注解会造成什么后果
    def distributed_train_step(dataset_inputs, dataset_labels):
        per_replica_losses = strategy.experimental_run_v2(
            train_step, args=(dataset_inputs, dataset_labels)
        )
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


#%% 5）执行训练
with strategy.scope():
    for epoch in range(EPOCHS):
        # ==== Train ====
        start = time.time()
        total_loss = 0.0
        num_batches = 0

        for record in dist_dataset:
            x_train = record['x']
            y_train = record['y']

            total_loss += distributed_train_step(x_train, y_train)
            num_batches += 1
        train_loss = total_loss / num_batches
        end = time.time()
        print('[{}] Time for epoch {} / {} is {:0.4f} sec, loss {:0.4f}'.format(time.asctime(), epoch + 1, EPOCHS,
                                                                                end - start, train_loss))
        # ==== Save checkpoint and validate ====
        if (epoch + 1) % 10 == 0:
            checkpoint_save_path = checkpoint_manager.save()
            print('[{}] Checkpoint saved, for epoch {}, at {}'.format(time.asctime(), epoch + 1, checkpoint_save_path))


