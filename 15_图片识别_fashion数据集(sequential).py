#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :15_图片识别_fashion数据集(sequential)
# @Time      :2022/1/1 0001 19:14
# @Author    :ljxiong
# @Email     :ljxiong84@163.com

# 导入模块
import tensorflow as tf

# 加载数据集
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
# print(x_train.shape)

# 输入特征归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 逐层描述神经网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 配置训练参数
model.compile(optimizer='adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 执行训练
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

# 打印结果
model.summary()








