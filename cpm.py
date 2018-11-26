import tensorflow as tf

class CpmStage1:
    def __init__(self, inputs_x, mask_cpm, mask_hm, gt_hm, gt_cpm, stage_num=6, hm_channel_num=19, cpm_channel_num=38):
        self.inputs_x = inputs_x
        self.mask_cpm = mask_cpm
        self.mask_hm = mask_hm
        self.gt_hm = gt_hm
        self.gt_cpm = gt_cpm
        self.stage_num = stage_num
        self.cpm_channel_num = cpm_channel_num
        self.hm_channel_num = hm_channel_num

    def stage_1(self, inputs, out_channel_num):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer(stddev=0.1),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer(stddev=0.1),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=3,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer(stddev=0.1),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.conv2d(inputs=net,
                               filters=512,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer(stddev=0.1),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        net = tf.layers.conv2d(inputs=net,
                               filters=out_channel_num,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer(stddev=0.1),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        return net

    def stage_t(self, inputs, out_channel_num):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=7,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               padding="same",
                               kernel_size=1,
                               activation="relu",
                               bias_initializer=tf.random_normal_initializer())
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
        cpm_pre = []
        hm_pre = []

        with tf.variable_scope('stage1'):
            cpm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.cpm_channel_num)
            hm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.hm_channel_num)
            cpm_pre.append(cpm_net)
            hm_pre.append(hm_net)
            cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
            hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
            net = tf.concat([hm_net, cpm_net, self.inputs_x], 3)

        with tf.variable_scope('staget'):
            for i in range(self.stage_num - 1):
                hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num)
                cpm_net = self.stage_t(inputs=net, out_channel_num=self.cpm_channel_num)
                cpm_pre.append(cpm_net)
                hm_pre.append(hm_net)
                hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
                cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
                if i < self.stage_num - 2:
                    net = tf.concat([hm_net, cpm_net, self.inputs_x], 3)

        with tf.name_scope("loss"):
            total_loss = tf.reduce_sum(hm_loss) + tf.reduce_sum(cpm_loss)
        tf.summary.scalar("loss", total_loss)

        tf.summary.image('hm', self.gt_hm[:, :, :, 0:1], max_outputs=4)
        tf.summary.image('hm_pre', hm_net[:, :, :, 0:1], max_outputs=4)

        return hm_pre, cpm_pre, total_loss


    # test code
    # def gen_net(self):
    #     cpm_loss = []
    #     hm_loss = []
    #     cpm_pre = []
    #     hm_pre = []
    #
    #     with tf.variable_scope('stage1'):
    #         with tf.name_scope('stage1'):
    #             cpm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.cpm_channel_num)
    #             hm_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.hm_channel_num)
    #             cpm_pre.append(cpm_net)
    #             hm_pre.append(hm_net)
    #             cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
    #             hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
    #             net = tf.concat([hm_net, cpm_net, self.inputs_x], 3)
    #
    #     with tf.variable_scope('staget'):
    #         for i in range(self.stage_num - 1):
    #             with tf.name_scope("staget"):
    #                 hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num)
    #                 cpm_net = self.stage_t(inputs=net, out_channel_num=self.cpm_channel_num)
    #                 cpm_pre.append(cpm_net)
    #                 hm_pre.append(hm_net)
    #                 hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
    #                 cpm_loss.append(self.get_loss(cpm_net, self.gt_cpm, mask_type='cpm'))
    #                 if i < self.stage_num - 2:
    #                     net = tf.concat([hm_net, cpm_net, self.inputs_x], 3)
    #
    #     with tf.name_scope("loss"):
    #         total_loss = tf.reduce_sum(hm_loss) + tf.reduce_sum(cpm_loss)
    #     tf.summary.scalar("loss", total_loss)
    #
    #     tf.summary.image('hm', self.gt_hm[:, :, :, 0:1], max_outputs=4)
    #     tf.summary.image('hm_pre', hm_net[:, :, :, 0:1], max_outputs=4)
    #
    #     return hm_pre, cpm_pre, total_loss

    def get_loss(self, pre_y, gt_y, mask_type):
        if mask_type == 'cpm':
            return tf.reduce_mean(tf.reduce_sum(tf.square(gt_y - pre_y) * self.mask_cpm, axis=[1, 2, 3]))
        if mask_type == 'hm':
            return tf.reduce_mean(tf.reduce_sum(tf.square(gt_y - pre_y) * self.mask_hm, axis=[1, 2, 3]))
