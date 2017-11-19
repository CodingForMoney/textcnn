import tensorflow as tf
import numpy as np
import math

hiden_fc1_dim = 1024

class TCNNConfig(object):
    """CNN配置参数"""
    vocab_size = 20000
    embedding_size = 128    # 词向量维度

    sequence_count = 200
    sequence_length = 32

    num_classes = 10

    # filter2_size = embedding_size  # filter1 处理词向量， 宽度固定
    filter1_num = 16        # 对于词向量，我们分为16个feature

    # filter2_size = sequence_length # filter2 处理句子，将句子和为一句。
    filter2_num = 64        # 输出4 个feature  , 现在为 16 * 4 = 64

    fitler3_size = 4        # filter3 在句子向量中， filtersize为6，每次走3步。 这个需要为2的倍数。
    filter3_num = 128       # 输出两个feature , 现在为 64 * 2 = 128 , 则整个特征数量为 
    fitler3_pool_size = 4   # 默认设置 4 ： 1
    fitler3_pool_stride = 1  

    hiden_fc1_dim = 1024    # 暂时只有两层fc
    # l2_reg_lambda = 0.0    # 暂时没有l2优化



class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    """
        参数含义：
         sequence_count  短句数量
         sequence_length 句子长度 ，即词的数量
         num_classes 分类数量
         filter_size 卷积核大小，一个window覆盖多少个单词 ， 这里原文传入一个数组，想要分析不同卷积核大小的结果。 可能是这样。
                     这里设置为 4 ， 方便计算。
         num_filters 卷积核数量。
         vocab_size 词汇表数量
         embedding_size  词向量维度 设置为128
        
    """
    def __init__(
      self, config):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, config.sequence_count, config.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, config.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # l2_loss = tf.constant(0.0)
        # 将词汇 向量化，
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([config.vocab_size, config.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print(self.embedded_chars_expanded )


        # 所以这里，我们的卷积层是 3维的。 
        # 第一层卷积， 降低词向量维度， 由 128 - > 16
        with tf.name_scope("conv1"):
            # Convolution Layer 卷积层要减少变量啊 。。。
            # 长，宽，高 ， channel ，filter数量
            filter1_shape = [1, 1, config.embedding_size, 1,config.filter1_num]
            W1 = tf.Variable(tf.truncated_normal(filter1_shape, stddev=0.1), name="W1_conv")
            b1 = tf.Variable(tf.constant(0.1, shape=[config.filter1_num]), name="b1_conv")
            conv1 = tf.nn.conv3d(
                self.embedded_chars_expanded,
                W1,
                # batch NO , 长， 宽，高， channel
                strides=[1 , 1 , 1 , config.embedding_size, 1],
                padding="VALID",
                name="conv3d_1")
            print(conv1)
            # Apply nonlinearity
            self.h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu_conv1")
            print(self.h1)

        # 第二层卷积， 消灭 句子维度
        with tf.name_scope("conv2"):
            #  第二层去掉宽度
            filter2_shape = [1, config.sequence_length , 1 , config.filter1_num,config.filter2_num]
            W2 = tf.Variable(tf.truncated_normal(filter2_shape, stddev=0.1), name="W2_conv")
            b2 = tf.Variable(tf.constant(0.1, shape=[config.filter2_num]), name="b2_conv")
            conv2 = tf.nn.conv3d(
                self.h1,
                W2,
                # batch NO , 长， 宽，高， channel
                strides=[1 , 1 , config.sequence_length, 1 , 1],
                padding="VALID",
                name="conv3d_2")
            print(conv2)
            # Apply nonlinearity
            self.h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu_conv2")
            print(self.h2)
            # Maxpooling over the outputs
            # self.pool2 = tf.nn.max_pool3d(
            #     h2,
            #     # batch , 长，宽，高，channel
            #     ksize=[1,  config.sequence_length, 1 , 1, 1],
            #     strides=[1, 1, 1, 1, 1],
            #     padding='VALID',
            #     name="pool3d_1")
            # print(self.pool2)

        # 第三层卷积， 在段落层次， 卷积句子
        with tf.name_scope("conv3"):
            #  第二层去掉宽度
            filter3_shape = [config.fitler3_size, 1 , 1 , config.filter2_num,config.filter3_num]
            W3 = tf.Variable(tf.truncated_normal(filter3_shape, stddev=0.1), name="W3_conv")
            b3 = tf.Variable(tf.constant(0.1, shape=[config.filter3_num]), name="b3_conv")
            conv3 = tf.nn.conv3d(
                self.h2,
                W3,
                # batch NO , 长， 宽，高， channel
                strides=[ 1 , config.fitler3_size, 1, 1 , 1],
                padding="SAME",
                name="conv3d_3")
            print(conv3)
            # Apply nonlinearity
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu_conv3")
            print(h3)
            # Maxpooling over the outputs
            self.pool3 = tf.nn.max_pool3d(
                h3,
                # batch , 长，宽，高，channel
                ksize=[1,  config.fitler3_pool_size, 1 , 1, 1],
                strides=[1, config.fitler3_pool_stride, 1, 1, 1],
                padding='SAME',
                name="pool3d_1")
            print(self.pool3)

        with tf.name_scope("fc1"):
            cov_width = math.ceil(config.sequence_count / config.fitler3_size)
            cov_width = math.ceil( cov_width / config.fitler3_pool_stride)
            print(cov_width)
            max_pool_dimension = cov_width * config.filter3_num

            W_fc1 = tf.Variable(tf.truncated_normal([max_pool_dimension , config.hiden_fc1_dim ], stddev=0.1), name="W_fc1")
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[config.hiden_fc1_dim]), name="b_fc1")
            h_pool_flat = tf.reshape(self.pool3, [-1, max_pool_dimension])
            self.h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1,name="fc1")


        with tf.name_scope('dropout'):
            # 防止过拟合，添加 dropout ， dropout概率有keep_prob 这个参数进行输入
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W_out = tf.Variable(
            #     [hiden_fc1_dim, num_classes],
            #     name="W_out"
            #     # , initializer=tf.contrib.layers.xavier_initializer()
            #     )
            W_out = tf.Variable(tf.truncated_normal([config.hiden_fc1_dim , config.num_classes ], stddev=0.1), name="W_out")
            b_out = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name="b_out")
            # l2_loss += tf.nn.l2_loss(W_out)
            # l2_loss += tf.nn.l2_loss(b_out)
            self.scores = tf.nn.xw_plus_b(self.h_fc1_drop, W_out, b_out, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) 
            # + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
