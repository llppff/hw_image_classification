import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class VGG16_Mode():

    #定义卷积层
    def conv_layer(self, data, ksize, stride, name, w_biases = False,padding = "SAME"):
        with tf.variable_scope(name) as scope:
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape= ksize, initializer= w_init,name= 'w')
            biases = tf.Variable(tf.constant(0.0, shape=[ksize[3]], dtype=tf.float32), 'biases')
        if w_biases == False:
            cov = tf.nn.conv2d(input= data, filter= w, strides= stride, padding= padding)
        else:
            cov = tf.nn.conv2d(input= data,filter= w, stride= stride,padding= padding) + biases
        return cov

    #定义池化层
    def pool_layer(self, data, ksize, stride, name, padding= 'VALID'):
        with tf.variable_scope(name) as scope:
            max_pool =  tf.nn.max_pool(value= data, ksize= ksize, strides= stride,padding= padding)
        return max_pool

    #将每张图片展成一维向量
    def flatten(self,data):
        [a,b,c,d] = data.shape
        ft = tf.reshape(data,[-1,b*c*d])
        return ft


    # 定义全连接层
    def fc_layer(self,data,name,fc_dims):
        with tf.variable_scope(name) as scope:
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape=[data.shape[1],fc_dims],name= 'w',initializer=w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            fc = tf.nn.relu(tf.matmul(data,w)+ biases)
        return fc

    #最后一个输出层
    def finlaout_layer(self,data,name,fc_dims):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # tf.contrib.layers.xavier_initializer()函数返回一个用于初始化权重的初始化程序 “Xavier”，这个初始化器是用来保持每一层的梯度大小都差不多相同
            #返回初始化权重矩阵
            w_init = tf.contrib.layers.xavier_initializer()
            # tf.get_variable()获取一个已经存在的变量或者创建一个新的变量,参数ininializer：如果创建了则用它来初始化变量
            w = tf.get_variable(shape=[data.shape[1],fc_dims],name= 'w',initializer=w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            fc = tf.nn.softmax(tf.matmul(data,w)+ biases)
        return fc

    #构造模型
    def model_bulid(self, height, width, channel,classes):
        x = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype= tf.float32 ,shape=[None,classes])
        # mnist_image_size:28*28，下方实现卷积层的函数中padding参数取值为'SAME'时，意思是经过卷积后image尺寸不变，padding具体尺寸根据需要而变化
        conv1_1 = tf.nn.relu(self.conv_layer(x,ksize= [3,3,1,64],stride=[1,1,1,1],padding="SAME",name="conv1_1"))#(None,28,28,64)
        conv1_2 = tf.nn.relu(self.conv_layer(conv1_1,ksize=[3,3,64,64],stride=[1,1,1,1],padding="SAME",name="conv1_2"))#(None,28,28,64)
        pool1_1 = self.pool_layer(conv1_2,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool1_1")#（None,14,14,64)


        conv2_1 = tf.nn.relu(self.conv_layer(pool1_1,ksize=[3,3,64,128],stride=[1,1,1,1],padding="SAME",name="conv2_1"))#(None,14,14,128)
        conv2_2 = tf.nn.relu(self.conv_layer(conv2_1,ksize=[3,3,128,128],stride=[1,1,1,1],padding="SAME",name="conv2_2"))#(None,14,14,128)
        pool2_1 = self.pool_layer(conv2_2,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool2_1")#(None,7,7,128)

        conv3_1 = tf.nn.relu(self.conv_layer(pool2_1,ksize=[3,3,128,256],stride=[1,1,1,1],padding="SAME",name="conv3_1"))#(None,7,7,256)
        conv3_2 = tf.nn.relu(self.conv_layer(conv3_1,ksize=[3,3,256,256],stride=[1,1,1,1],padding="SAME",name="conv3_2"))#(None,7,7,256)
        conv3_3 = tf.nn.relu(self.conv_layer(conv3_2,ksize=[3,3,256,256],stride=[1,1,1,1],padding="SAME",name="conv3_3"))#(None,7,7,256)
        pool3_1 = self.pool_layer(conv3_3,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool3_1")#(None,4,4,256)

        conv4_1 = tf.nn.relu(self.conv_layer(pool3_1,ksize=[3,3,256,512],stride=[1,1,1,1],padding="SAME",name="conv4_1"))#(None,4,4,512)
        conv4_2 = tf.nn.relu(self.conv_layer(conv4_1,ksize=[3,3,512,512],stride=[1,1,1,1],padding="SAME",name="conv4_2"))#(None,4,4,512)
        conv4_3 = tf.nn.relu(self.conv_layer(conv4_2,ksize=[3,3,512,512],stride=[1,1,1,1],padding="SAME",name="conv4_3"))#(None,4,4,512)
        pool4_1 = self.pool_layer(conv4_3,ksize=[1,2,2,1],stride=[1,1,1,1],name="pool4_1")#(None,3,3,256)

        conv5_1 = tf.nn.relu(self.conv_layer(pool4_1, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_1"))#(None,3,3,512)
        conv5_2 = tf.nn.relu(self.conv_layer(conv5_1, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_2"))#(None,3,3,512)
        conv5_3 = tf.nn.relu(self.conv_layer(conv5_2, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_3"))#(None,3,3,512)
        pool5_1 = self.pool_layer(conv5_3,ksize=[1,3,3,1],stride=[1,1,1,1],name="pool5_1")#(None,1,1,512)

        # Flatten
        ft = self.flatten(pool5_1)#(None,512)

        # 三个全连接层,维度变化为：(fc 4096)=>(fc 4096)=>(fc classes)，这里classes是10
        fc1 = self.fc_layer(ft,fc_dims=4096,name="fc1")
        fc2 = self.fc_layer(fc1,fc_dims=4096,name="fc2")
        fc3 = self.fc_layer(fc2,fc_dims=1000,name="fc3")

        #最后一个输出层，计算softmax
        finaloutput = self.finlaout_layer(fc3,fc_dims=10,name="final")#(None,10)

        # 计算交叉熵损失
        loss = tf.losses.softmax_cross_entropy(y,finaloutput)

        # optimize
        LEARNING_RATE_BASE = 0.0001
        LEARNING_RATE_DECAY = 0.1
        LEARNING_RATE_STEP = 300
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                                   , gloabl_steps,
                                                   LEARNING_RATE_STEP,
                                                   LEARNING_RATE_DECAY,
                                                   staircase=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            #采用Adam优化器来进行优化
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # 最后一层的输出就是对应的预测结果，根据预测结果计算预测精度
        prediction_label = finaloutput
        correct_prediction = tf.equal(tf.argmax(prediction_label, 1), tf.argmax(y, 1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        correct_times_in_batch = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.int32))
        return dict(
            x=x,
            y=y,
            optimize=optimize,
            correct_prediction=prediction_label,
            correct_times_in_batch=correct_times_in_batch,
            cost=loss,
            accurary=accurary
        )

    #初始化session
    def init_sess(self):
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    #进行训练
    def train_network(self,graph,x_train,y_train):
        #tensorflow向graph中添加越来越多的节点
        tf.reset_default_graph()
        self.sess.run(graph['optimize'],feed_dict={graph['x']:x_train, graph['y']:y_train})

    def load_data_and_train(self):
        mnist = input_data.read_data_sets('mnist_sets', one_hot=True)
        g = self.model_bulid(28, 28, 1, 10)
        self.init_sess()
        for i in range(5000):
            batch_train_xs, batch_train_ys = mnist.train.next_batch(500)
            batch_train_xs = np.reshape(batch_train_xs,[-1,28,28,1])
            self.train_network(g, batch_train_xs, batch_train_ys)
            # 每100步为一个epoch
            if i % 100 == 0:
                print("epoch:" + str(i/100))
                print("train accurary: ",
                  self.sess.run(g['accurary'], feed_dict={g['x']: batch_train_xs, g['y']: batch_train_ys}))

                #测试集进行精度计算
                batch_test_xs, batch_test_ys = mnist.test.next_batch(100)
                batch_test_xs = np.reshape(batch_test_xs, [-1, 28, 28, 1])
                print("test accurary: ",
                          self.sess.run(g['accurary'], feed_dict={g['x']: batch_test_xs, g['y']: batch_test_ys}))

VGG = VGG16_Mode()
VGG.load_data_and_train()
