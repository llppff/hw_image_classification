import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class VGG16_Mode():
    """ create vgg16 network use tensorflow
        VGG16 network structure:
        (conv 3x3 64)=>(conv 3x3 64, pool/2)
        (conv 3x3 128)=>(conv 3x3 128, pool/2)
        (conv 3x3 256)=>(conv 3x3 256)=>(conv 3x3 256)=>(conv 3x3 256, pool/2)
        (conv 3x3 512)=>(conv 3x3 512)=>(conv 3x3 512)=>(conv 3x3 512, pool/2)
        (fc 4096)=>(fc 4096)=>(fc 1000)=>(fc classes)
    """

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

        # conv 1 ,if image Nx465x128x1 ,(conv 3x3 64)=>(conv 3x3 64, pool/2)
        conv1_1 = tf.nn.relu(self.conv_layer(x,ksize= [3,3,1,64],stride=[1,1,1,1],padding="SAME",name="conv1_1"))
        conv1_2 = tf.nn.relu(self.conv_layer(conv1_1,ksize=[3,3,64,64],stride=[1,1,1,1],padding="SAME",name="conv1_2")) # Nx465x128x1 ==>   Nx465x128x64
        pool1_1 = self.pool_layer(conv1_2,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool1_1") # N*232x64x64

        # conv 2,(conv 3x3 128)=>(conv 3x3 128, pool/2)
        conv2_1 = tf.nn.relu(self.conv_layer(pool1_1,ksize=[3,3,64,128],stride=[1,1,1,1],padding="SAME",name="conv2_1"))
        conv2_2 = tf.nn.relu(self.conv_layer(conv2_1,ksize=[3,3,128,128],stride=[1,1,1,1],padding="SAME",name="conv2_2")) # Nx232x64x128
        pool2_1 = self.pool_layer(conv2_2,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool2_1") # Nx116x32x128

        # conv 3,(conv 3x3 256)=>(conv 3x3 256)=>(conv 3x3 256)=>(conv 3x3 256, pool/2)
        conv3_1 = tf.nn.relu(self.conv_layer(pool2_1,ksize=[3,3,128,256],stride=[1,1,1,1],padding="SAME",name="conv3_1"))
        conv3_2 = tf.nn.relu(self.conv_layer(conv3_1,ksize=[3,3,256,256],stride=[1,1,1,1],padding="SAME",name="conv3_2"))
        conv3_3 = tf.nn.relu(self.conv_layer(conv3_2,ksize=[3,3,256,256],stride=[1,1,1,1],padding="SAME",name="conv3_3"))
        # conv3_4 = tf.nn.relu(self.conv_layer(conv3_3,ksize=[3,3,256,256],stride=[1,1,1,1],padding="SAME",name="conv3_4")) # NX116X32X256
        pool3_1 = self.pool_layer(conv3_3,ksize=[1,1,1,1],stride=[1,2,2,1],name="pool3_1") # Nx58x16x256

        #conv 4,(conv 3x3 512) = > (conv 3x3 512) = > (conv 3x3 512) = > (conv 3x3 512, pool / 2)
        conv4_1 = tf.nn.relu(self.conv_layer(pool3_1,ksize=[3,3,256,512],stride=[1,1,1,1],padding="SAME",name="conv4_1"))
        conv4_2 = tf.nn.relu(self.conv_layer(conv4_1,ksize=[3,3,512,512],stride=[1,1,1,1],padding="SAME",name="conv4_2"))
        conv4_3 = tf.nn.relu(self.conv_layer(conv4_2,ksize=[3,3,512,512],stride=[1,1,1,1],padding="SAME",name="conv4_3"))
        # conv4_4 = tf.nn.relu(self.conv_layer(conv4_3,ksize=[3,3,512,512],stride=[1,1,1,1],padding="SAME",name="conv4_4")) # Nx58x16x512
        pool4_1 = self.pool_layer(conv4_3,ksize=[1,2,2,1],stride=[1,1,1,1],name="pool4_1") # Nx29x8x512

        # # conv 4,(conv 3x3 512) = > (conv 3x3 512) = > (conv 3x3 512) = > (conv 3x3 512, pool / 2)
        conv5_1 = tf.nn.relu(self.conv_layer(pool4_1, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_1"))
        conv5_2 = tf.nn.relu(self.conv_layer(conv5_1, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_2"))
        conv5_3 = tf.nn.relu(self.conv_layer(conv5_2, ksize=[3, 3, 512, 512], stride=[1, 1, 1, 1], padding="SAME", name="conv5_3"))
        pool5_1 = self.pool_layer(conv5_3,ksize=[1,3,3,1],stride=[1,1,1,1],name="pool5_1") # Nx29x8x512

        # Flatten
        ft = self.flatten(pool4_1)

        # Dense layer,(fc 4096)=>(fc 4096)=>(fc classes)
        fc1 = self.fc_layer(ft,fc_dims=4096,name="fc1")
        fc2 = self.fc_layer(fc1,fc_dims=4096,name="fc2")
        fc3 = self.fc_layer(fc2,fc_dims=1000,name="fc3")


        finaloutput = self.finlaout_layer(fc3,fc_dims=10,name="final")

        # cost
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
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # prediction
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
        # Tensorfolw Adding more and more nodes to the previous graph results in a larger and larger memory footprint
        # reset graph
        tf.reset_default_graph()
        self.sess.run(graph['optimize'],feed_dict={graph['x']:x_train, graph['y']:y_train})

    def load_data(self):
        mnist = input_data.read_data_sets('mnist_sets', one_hot=True)
        g = self.model_bulid(28, 28, 1, 10)
        self.init_sess()
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(500)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            self.train_network(g, batch_xs, batch_ys)
            if i % 100 == 0:
                print("epoch:" + str(i/100))
                print("cost: ", self.sess.run(g['cost'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}), "accurary: ",
                  self.sess.run(g['accurary'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}))

VGG = VGG16_Mode()
VGG.load_data()
