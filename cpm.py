import tensorflow as tf
import numpy as np


class CpmStage1:
    def __init__(self, inputs_x, mask_cpm, mask_hm, gt_hm, gt_cpm, stage_num=6, hm_channel_num=19, cpm_channel_num=38):
        self.inputs_x = inputs_x
        self.mask_cpm = mask_cpm
        self.mask_hm = mask_hm
        self.gt_hm = gt_hm
        self.gt_cpm = gt_cpm
        self.stage_num = stage_num
        # self.net = self.gen_net()
        self.cpm_channel_num = cpm_channel_num
        self.hm_channel_num = hm_channel_num

    def stage_1(self, inputs, out_channel_num):
    # with tf.name_scope("conv1_stage1"):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("pool1_stage1"):
    #     net = tf.layers.max_pooling2d(inputs=net,
    #                                   pool_size=2,
    #                                   strides=2)

    # with tf.name_scope("conv2_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("pool2_stage1"):
    #     net = tf.layers.max_pooling2d(inputs=net,
    #                                   pool_size=2,
    #                                   strides=2)

    # with tf.name_scope("conv3_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("pool3_stage1"):
    #     net = tf.layers.max_pooling2d(inputs=net,
    #                                   pool_size=2,
    #                                   strides=2)

    # with tf.name_scope("conv4_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=512,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv5_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=out_channel_num,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        return net

    def stage_t(self, inputs, out_channel_num):
    # with tf.name_scope("conv1_stage1"):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv2_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv3_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv4_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv5_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv6_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())

    # with tf.name_scope("conv7_stage1"):
        net = tf.layers.conv2d(inputs=net,
                               filters=out_channel_num,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        return net

    def gen_net(self):
        cpm_loss = []
        hm_loss = []
        with tf.name_scope("stage1"):
            cpm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.cpm_channel_num)
            hm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.hm_channel_num)
            cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
            hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
            net = tf.concat([hm_net, self.inputs_x], 3)

        for i in range(self.stage_num-1):
            with tf.name_scope("stage2"):
                hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num)
                cpm_net = self.stage_t(inputs=net, out_channel_num=self.cpm_channel_num)
                hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
                cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
                if i < self.stage_num - 2:
                    net = tf.concat([hm_net, self.inputs_x], 3)

        with tf.name_scope("loss"):
            total_loss = tf.reduce_sum(hm_loss)#  + tf.reduce_sum(cpm_loss)
        tf.summary.scalar("loss", total_loss)

        # tf.summary.image('input', img_normalized, max_outputs=2)

        tf.summary.image('hm', self.gt_hm[:, :, :, 0:1], max_outputs=2)
        tf.summary.image('hm_pre', hm_net[:, :, :, 0:1], max_outputs=2)

        tf.summary.image('cpm', self.gt_cpm[:, :, :, 0:1], max_outputs=2)
        tf.summary.image('cpm_pre', cpm_net[:, :, :, 0:1], max_outputs=2)
        return hm_net, cpm_net, total_loss

    def get_loss(self, pre_y, gt_y, mask_type):
        if mask_type == 'cpm':
            return tf.reduce_sum(tf.square(pre_y - gt_y) * self.mask_cpm)
        if mask_type == 'hm':
            return tf.reduce_sum(tf.square(gt_y - pre_y) * self.mask_hm)

