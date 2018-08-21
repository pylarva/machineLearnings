# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 定义下面会用到的参数和一个函数

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx])/w for idx, val in enumerate(a)]


# 1） 准备数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

plt.plot(train_X, train_Y, 'ro', label="Original data")
plt.legend()
plt.show()

# 2）创建模型

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

z = tf.multiply(X, W) + b

# 反向优化
# 生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))

# 学习率 调整参数的速度
leaning_rate = 0.01

# 封装好的梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(leaning_rate).minimize(cost)

# 3）迭代训练

# 初始化所有变量
init = tf.global_variables_initializer()
# 迭代次数
training_epochs = 20
display_step = 2

# 启动session
with tf.Session() as sess:
    # 网络节点运算
    sess.run(init)
    # plotdata = {"batchsize": [], "loss": []}

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            # feed机制将真实数据灌到占位符对应的位置feed_dict={X: x, Y: y}
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch+1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    print("Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fittedline")
    plt.legend()
    plt.show()

    # 训练中的状态值
    plotdata["avgloss"] = moving_average[plotdata["loss"]]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()


