---
layout: post
title: tensorflow实战
date: 2018-06-30
categories: 深度学习
tags: [tensorflow, 深度学习]
description: tensorflow训练mnist模型。
---
tensorflow是谷歌大脑开源的一个深度学习框架，本文我们将详细介绍如何使用tensorflow训练一个深度学习模型。  
mnist是一个手写数字(0~9)的数据集，提供了6W张训练图片和1W张测试图片，每一张图片都拥有相应的标签。mnist的每一张图片是28*28的灰度图像，[mnist官网](http://yann.lecun.com/exdb/mnist/)。  

# 实验环境
- python 3.5
- tensorflow 1.4.0
- system: Deepin
- 1060 N卡(如果没有，不影响实验)

# 文件结构
- mnist
	- data	----> 保存mnist数据集
	- download_mnist.py ----> 用于下载mnist数据集，如果数据集已经存在，不会重复下载 
	- model.py ----> mnist训练模型
	- train_mnist.py ----> 完成模型的训练和测试

# 数据集下载
你可以从mnist的官网下载，分别下载下面4个文件保存到`data`目录下：  

- `train-images-idx3-ubyte.gz`: 训练图片
- `train-labels-idx1-ubyte.gz`: 训练标签
- `t10k-images-idx3-ubyte.gz`: 测试图片
- `t10k-labels-idx1-ubyte.gz`: 测试标签

也可以使用tensorflow提供的API进行下载。

下面详细讲解`download_mnist.py`文件

```python
#!/usr/bin/env python3
# coding=utf-8

import os
# 导入tensorflow提供的mnist操作的库
from tensorflow.examples.tutorials.mnist import input_data
# 如果目录不存在，创建一个
if not os.path.exists('data'):
    os.mkdir('data')
# 从data从读取mnist数据集，one_hot=True表明如果文件不存在会自动下载
mnist = input_data.read_data_sets('data/', one_hot=True)

# 这是测试mnist是否读取成功的代码
if __name__ == '__main__':
    print("训练集图片尺寸:", mnist.train.images.shape)
    print("训练集标签尺寸:", mnist.train.labels.shape)
    print("验证集图片尺寸:", mnist.validation.images.shape)
    print("验证集标签尺寸:", mnist.validation.labels.shape)
    print("测试集图片尺寸:", mnist.test.images.shape)
    print("测试集标签尺寸:", mnist.test.labels.shape)
    print("输出第一个验证集标签数据:", mnist.train.labels[0, :])
```
输出结果如下：

```python
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
训练集图片尺寸: (55000, 784)
训练集标签尺寸: (55000, 10)
验证集图片尺寸: (5000, 784)
验证集标签尺寸: (5000, 10)
测试集图片尺寸: (10000, 784)
测试集标签尺寸: (10000, 10)
输出第一个验证集标签数据: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
```
这里有一点需要提一下，上面虽然我们下载的文件只是包含了训练集和测试集，但是tensorflow把测试集有进一步拆分，分出来了5000张作为验证集。还有一点就是，mnist标签的表示并不是用(0~9)的数字表示，而是使用一组`1*10`的向量表示：3的表示方式是下标第三个的数字为1(下标从0开始)，其他为0，这种标记方式称之为`onehot`(你是我的唯一)。

# mnist训练模型
本节中会提及很多深度学习方面的术语，如果有不明白的地方，后面我有可能会专门写一些文章讲解，但是目前请自行百度。  
由于mnist一般术语深度学习方面的`Hello World`，因此文中我选择使用一个很简单的4层模型(方便训练和理解，而且效果也不会差)：2个卷积层+2个全连接层(最后一个带`dropout`)。  
上面我们提到，mnist的图片的尺寸`28*28*1=784`，并且每一张图片都会带有一个标签数据，而且由于深度学习一般会将数据集分批次输入模型进行训练(这样做的好处是：1、减小内存的压力，2、分批次训练可以快速修正深度学习的参数，因为每训练完一个批次，就可以修正一次参数，3、适当增加批的大小可以加快收敛)，我们称之为`batch_size`，因此输入模型的数据尺寸为`[batch_size, 784]`。

```flow
in=>start: 输入
reshape1=>start: 尺寸变换
cov1=>start: 卷积层1
cov2=>start: 卷积层2
reshape2=>start: 尺寸变换
fc1=>start: 全连接层1
fc2=>start: 全连接层2

in->reshape1->cov1->cov2->reshape2->fc1->fc2
```

由于第一层卷积层需要针对图像的元素进行卷积，所以需要将输入的尺寸`[batch_size, 784]`变换为`[batch_size, 28， 28， 1]`。  
而在全连接层中，我们需要将元素映射单个特征，然后根据神经元的连接权重，将卷积出来的特征对应到0~9之间的数字，所以需要将3维的卷积特征(加上batch_size是4维)转换为1位的特征(加上batch_size是2维)。  
下面上代码详细解释整个模型`model.py`: 

```python
#!/usr/bin/env python3
# coding=utf-8

import tensorflow as tf


def _convolution_layer(layer_name, input, neuron_num):
    input_shape = input.get_shape().as_list()   ## 获取输入图像的尺寸

    with tf.variable_scope(layer_name) as _:
        with tf.variable_scope('conv') as scope:
            ## 卷积层的weight，卷积核大小为5*5，均值为0.1
            weight = tf.get_variable(name='weight', shape=[5, 5, input_shape[-1], neuron_num],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                     dtype=tf.float32)
            biases = tf.get_variable(name='biases', shape=[neuron_num],
                                     initializer=tf.constant_initializer(0.1),
                                     dtype=tf.float32)

            conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        with tf.variable_scope('pool') as scope:
            ## 使用max_pool的方式计算池化
            out = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME', name=scope.name)
    return out


def _fc_layer(layer_name, input, neuron_num, keep_prob):
    input_shape = input.get_shape().as_list()

    with tf.variable_scope(layer_name) as scope:
        weight = tf.get_variable(name='weight', shape=[input_shape[-1], neuron_num],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[neuron_num],
                                 initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)

        ## 加上dropout防止过拟合
        out = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weight), biases)),
                            keep_prob=keep_prob,
                            name=scope.name)

        tf.summary.histogram(scope.name, out)
    return out


def _mnist_model1(input, image_shape, keep_prob):
    ## 2层卷积+2层全连接(第一层带dropout),后面需要自行添加softmax

    input = tf.reshape(input, shape=[-1, image_shape[0], image_shape[1], image_shape[2]])

    ## 第一层卷积
    convolution_layer1_out = _convolution_layer(layer_name='convolution_layer01',
                                                input=input, neuron_num=32)
    ## 第二层卷积
    convolution_layer2_out = _convolution_layer(layer_name='convolution_layer02',
                                                input=convolution_layer1_out,
                                                neuron_num=64)

    fc1_input_shape = convolution_layer2_out.get_shape().as_list()
    fc1_input = tf.reshape(convolution_layer2_out,
                           shape=[-1, fc1_input_shape[1] * fc1_input_shape[2] * fc1_input_shape[3]])
    ## 全连接层1
    fc_layer1_out = _fc_layer(layer_name='fc_layer01',
                              input=fc1_input,
                              neuron_num=512,
                              keep_prob=keep_prob)
    ## 全连接层2
    fc_layer2_out = _fc_layer(layer_name='fc_layer02',
                              input=fc_layer1_out,
                              neuron_num=10,
                              keep_prob=1.0)
    out = fc_layer2_out
    return out


def mnist_optimizer(logit, y_):
    ## 使用交叉熵损失函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    ## 使用梯度下降优化器
    return tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)


mnist_model = _mnist_model1
```
该模型使用两个卷积层提取图片特征，然后将特征送入第一个全连接层，该全连接层使用0.5的`dropout`防止过拟合，输出512个特征，然后送入第二个全连接层，最终输出10个类别概率。  
这里有必要简单介绍下`loss`的计算方式，我们使用上面的模型计算输入图片后，会输出一个`1*10`的输出结果(可能很多是错误的)，我们计算模型给出的结果和实际标签之间的距离，这个距离越小，说明模型计算的结果越正确，因此优化器的作用就是用来减少这个距离。  
而梯度下降算法的原理，想象我们在一座山上，那么下山最快的方式就是沿着坡度最大的方向的反方向，而梯度下降算法的原理就是，计算一个数据点的梯度，沿着梯度最小的方向调整参数。  
在完成模型设计以后，需要设置好模型的输入以及让tensorflow开始计算我们的模型`train_mnist.py`：

```python
#!/usr/bin/env python3
# coding=utf-8

from download_mnist import mnist
import model
import tensorflow as tf

# 使用模型输出和标签计算准确度
def accuracy(y, y_):
    return tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)),
        tf.float32))

if __name__ == '__main__':
    # 占位符
    images = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    # 获取定义好的模型和优化器
    logit = model.mnist_model(input=images, image_shape=[28, 28, 1], keep_prob = keep_prob)
    optimizer = model.mnist_optimizer(logit, labels)
    # 初始化变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        # 计算50000个批次，每个批次大小为64
        for step in range(50000):
            batch = mnist.train.next_batch(64)
            # 每计算1000个批次，输出一下在验证集上的准确率
            if step % 1000 ==0:
                val = mnist.validation.next_batch(64)
                train_accuracy = accuracy(logit, val[1]).eval(feed_dict={
                    images:val[0],
                    labels:val[1],
                    keep_prob:1.0
                })
                print('当前步数：%05d，验证集上准确率：%.05f%%'%(step, train_accuracy*100))
            # 这里我们需要运行优化器，因为优化器才是用来修改整个模型学习参数的
            sess.run(optimizer, feed_dict={
                images:batch[0],
                labels:batch[1],
                keep_prob:0.5
            })
        # 最后输出在测试集上的准确率
        print('测试集上准确率：%.06f%%'%(accuracy(logit, mnist.test.labels).eval(
            feed_dict={
                images: mnist.test.images,
                labels: mnist.test.labels,
                keep_prob: 1.0
            }))*100)
```
经过50000次的迭代，我们的模型可以在测试集上达到`97%`以上的准确率。

```
当前步数：00000，验证集上准确率：12.50000%
当前步数：01000，验证集上准确率：78.12500%
当前步数：02000，验证集上准确率：92.18750%
当前步数：03000，验证集上准确率：90.62500%
当前步数：04000，验证集上准确率：92.18750%
当前步数：05000，验证集上准确率：95.31250%
当前步数：06000，验证集上准确率：95.31250%
当前步数：07000，验证集上准确率：96.87500%
当前步数：08000，验证集上准确率：95.31250%
当前步数：09000，验证集上准确率：96.87500%
当前步数：10000，验证集上准确率：100.00000%
当前步数：11000，验证集上准确率：96.87500%
当前步数：12000，验证集上准确率：95.31250%
当前步数：13000，验证集上准确率：96.87500%
当前步数：14000，验证集上准确率：95.31250%
测试集上准确率：97.899997%
```

下一篇我会讲解如何使用tensorflow训练`cifar10`数据集，并使用训练好的模型去识别单张图片，see you next time。
