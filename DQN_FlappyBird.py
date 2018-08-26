#coding=utf-8
from  __future__ import print_function

#导入一些必要的包
import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque



class Flappy_Bird(object):
    def __init__(self):
        self.GAME = 'bird'  # 网络训练过程中保存文件夹的名字
        self.ACTIONS = 2  # 一共有两种动作
        self.GAMMA = 0.99  # Qlearning中 对Q值更新的权重
        self.OBSERVE = 10000  # 训练之前需要探索OBSERVE步，之后再用minibatch对DQN进行训练
        self.EXPLORE = 3000000  # 该步数之后停止随机探索
        self.FINAL_EPSILON = 0.0001
        self.INITIAL_EPSILON = 0.0001
        self.epsilon = self.INITIAL_EPSILON
        self.REPLAY_MEMORY = 50000  # 之前训练需要记忆的步数
        self.BATCH = 32  # minibatch大小
        self.FRAME_PER_ACTION = 1  # 每一帧的动作数量
        self.time = 0
        self.stored_memory =deque() #保存训练数据的队列
        self.s,self.readout,self.fc1 = self.creat_network() #创建一个网络 其中s为输入状态，readout为网络输出表示输出的Q值
        self.optimizer ,self.y,self.a = self.creat_optimizer(self.readout) #创建优化方案，使用ADAOPTIMIZAL优化器
        self.game_state = game.GameState()
        self.sess = tf.Session()             # 创建类的TensorFlow会话
        self.start = self.init_step()        #对创建系统的初始状态
        self.Saver = self.restore_param()                 #重新导入系统的参数

    #初始化过程，进行第一次游戏等等，如读取已经保存的模型参数等
    def init_step(self):
        #获得游戏中的第一幅图像
        do_nothing = np.zeros(self.ACTIONS)
        do_nothing[0]=1
        x_t,r_0,terminal =self.game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_BGR2GRAY)
        ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
        return s_t



    def restore_param(self):
        Saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            Saver.restore(self.sess,checkpoint.model_checkpoint_path)
            print("Successfully loaded",checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        return Saver


    # 定义一个weight，其中命名空间为name，形状为shape
    def weight_variable(self,name, shape):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',
                                      shape=shape,
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

            return weights

    def bias_variable(self,name, shape):
        with tf.variable_scope(name)  as scope:
            biases = tf.get_variable('biaess',
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))

            return biases

    # 定义一个卷积层，命名空间为name，输入为x，卷积核为W，步长为stride,偏差为bias,激活函数默认为relu
    def conv2d(self,name, x, W, stride, bias):
        with tf.variable_scope(name) as scope:
            conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, bias)
            output = tf.nn.relu(pre_activation, name=scope.name)

            return output

    # 定义一个池化层，默认为max_pooling
    def max_pool_2x2(self,name, x):
        with tf.variable_scope(name) as scope:
            maxpool = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            return maxpool

    # 创建DQN
    def creat_network(self):
        # 网络的参数，权重
        W_conv1 = self.weight_variable('W_conv1', [8, 8, 4, 32])  # 第一层卷积层为8x8的卷积核，输入通道为4，输出通道为32
        b_conv1 = self.bias_variable('b_conv1', [32])

        W_conv2 = self.weight_variable('W_conv2', [4, 4, 32, 64])  # 第二层卷积层为4x4的卷积核,输入通道为32，输出通道为64
        b_conv2 = self.bias_variable('b_conv2', [64])

        W_conv3 = self.weight_variable('W_conv3', [3, 3, 64, 64])  # 第三层卷积层为3x3的卷积核，输入通道为64，输出通道为64
        b_conv3 = self.bias_variable('b_conv3', [64])

        W_fc1 = self.weight_variable('W_fc1', [1600, 512])
        b_fc1 = self.bias_variable('b_fc1', [512])

        W_fc2 = self.weight_variable('W_fc2', [512, self.ACTIONS])
        b_fc2 = self.bias_variable('b_fc2', [self.ACTIONS])

        s = tf.placeholder("float", [None, 80, 80, 4])  # 输入层，输入图像为80x80的4通道图像

        h_conv1 = self.conv2d('h_conv1', s, W_conv1, 4, b_conv1)  # 构造第一个卷积层输出为conv1
        h_pool1 = self.max_pool_2x2('h_pool1', h_conv1)

        h_conv2 = self.conv2d('h_conv2', h_pool1, W_conv2, 2, b_conv2)

        h_conv3 = self.conv2d('h_conv3', h_conv2, W_conv3, 1, b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600], 'h_conv3_flat')
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv3_flat, W_fc1), b_fc1, 'h_fc1'))

        readout = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, 'h_fc2')

        return s, readout, h_fc1

    def creat_optimizer(self,readout):
        action = tf.placeholder(tf.float32,[None,self.ACTIONS])
        y = tf.placeholder(tf.float32,[None])
        readout_action = tf.reduce_sum(tf.multiply(readout,action),reduction_indices=1)
        cost =tf.reduce_mean(tf.square(y-readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        return train_step,y,action
    #输入一个初始状态s_t，时间为t，之后进行游戏
    def process_game(self,s_t):
        #通过CNN运算得到Q值向量
        read_out_t = self.sess.run(self.readout,feed_dict={self.s:[s_t]})[0]
        a_t =np.zeros([self.ACTIONS])
        action_index =0
        if self.time % self.FRAME_PER_ACTION == 0:
            if random.random()<= self.epsilon:   #随机选择动作
                print("-----------随机选择动作--------------")
                action_index = random.randrange(self.ACTIONS)
                a_t[action_index]=1
            else:                          #选择Q值最大的动作
                action_index = np.argmax(read_out_t)
                a_t[random.randrange(self.ACTIONS)] = 1
        else:
            a_t[0] = 1 #如果没有到当前帧，那么不做动作

        #减少epsilon的值
        if self.epsilon>self.FINAL_EPSILON and self.time>self.OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON-self.FINAL_EPSILON)/self.EXPLORE

        #运行刚刚选择的动作，获得下一个奖励和动作
        x_t1_colored,r_t,terminal = self.game_state.frame_step(a_t)
        x_t1 =cv2.cvtColor(cv2.resize(x_t1_colored,(80,80)),cv2.COLOR_BGR2GRAY)
        ret,x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1,(80,80,1))
        #s_t1为下一帧的状态
        s_t1 = np.append(x_t1,s_t[:,:,:3],axis=2)

        #保存该状态为以后训练CNN准备 (原来状态+动作+奖励+下一次状态+游戏是否结束)
        self.stored_memory.append([s_t,a_t,r_t,s_t1,terminal])

        if(len(self.stored_memory)>self.REPLAY_MEMORY):
            self.stored_memory.popleft()


        if self.time>self.OBSERVE:
            #如果大于探索步，那么进行CNN训练
            self.train_network()

        self.time+=1   #步数增加1

        if self.time % 10000 == 0:
            self.Saver.save(self.sess,'saved_networks/'+self.GAME+'-dqn',global_step=self.time)

        #打印状态
        if self.time <= self.OBSERVE:
            state = "observe"
        elif self.time>self.OBSERVE and self.time <= self.OBSERVE+self.EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.time, "/ STATE", state,\
            "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", r_t,\
            "/ Q_MAX %e" % np.max(read_out_t))
        return s_t1





    def train_network(self,):
        minibatch =random.sample(self.stored_memory,self.BATCH) #队列中进行随机采样

        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        y_batch = []
        readout_j1_batch = self.sess.run(self.readout,feed_dict={self.s:s_j1_batch} )   #进行一次前向运算，算出下一个状态的Q值

        for i in range(0,len(minibatch)):
            terminal = minibatch[i][4]

            #如果游戏结束，那么只奖励当前分数
            if terminal:
                y_batch.append(r_batch[i])
            #否则奖励总奖励
            else:
                y_batch.append(r_batch[i]+self.GAMMA*np.max(readout_j1_batch[i]))

        #下面进行梯度下降
        self.sess.run(self.optimizer,feed_dict={
            self.y:y_batch,
            self.a:a_batch,
            self.s:s_j_batch
        })

def playGame():
    flappy_bird = Flappy_Bird()
    train_sta = flappy_bird.init_step()
    while "flappy_bird"!="angry_bird":
        train_sta = flappy_bird.process_game(train_sta)


if __name__ == '__main__':

    playGame()

