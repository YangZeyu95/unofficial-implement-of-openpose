import tensorflow as tf


def conv2(inputs, filters, kernel_size, name, padding='SAME', act=True, normalization=True):
    channels_in = inputs[0, 0, 0, :].get_shape().as_list()[0]
    with tf.variable_scope(name) as scope:
        # w = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, channels_in, filters], stddev=0.1))
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, channels_in, filters], trainable=True,
                            initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.Variable(tf.truncated_normal(shape=[filters], stddev=0.1))
        b = tf.get_variable('biases', shape=[filters], trainable=True,
                            initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding=padding)
        output = tf.nn.bias_add(conv, b)
        # if normalization:
        #     axis = list(range(len(output.get_shape()) - 1))
        #     mean, variance = tf.nn.moments(conv, axes=axis)
        #     output = tf.nn.batch_normalization(output, mean, variance, offset=None, scale=None, variance_epsilon=0.0001)
        if act:
            output = tf.nn.relu(output, name=scope.name)
    tf.summary.histogram('conv', conv)
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('output', output)
    return output

def gen_network(input):
    cpm_out = []
    hm_out = []
    with tf.variable_scope('train_layers'):
        add_layer = conv2(inputs=input, filters=256, kernel_size=3, name='add_layer_1')
        add_layer = conv2(inputs=add_layer, filters=128, kernel_size=3, name='add_layer_2')

        stage1_l1 = conv2(inputs=add_layer, filters=128, kernel_size=3, name='stage1_conv1_l1')
        stage1_l1 = conv2(inputs=stage1_l1, filters=128, kernel_size=3, name='stage1_conv2_l1')
        stage1_l1 = conv2(inputs=stage1_l1, filters=128, kernel_size=3, name='stage1_conv3_l1')
        stage1_l1 = conv2(inputs=stage1_l1, filters=512, kernel_size=1, name='stage1_conv4_l1')
        stage1_l1 = conv2(inputs=stage1_l1, filters=38, kernel_size=1, act=False, name='stage1_conv5_l1')

        stage1_l2 = conv2(inputs=add_layer, filters=128, kernel_size=3, name='stage1_conv1_l2')
        stage1_l2 = conv2(inputs=stage1_l2, filters=128, kernel_size=3, name='stage1_conv2_l2')
        stage1_l2 = conv2(inputs=stage1_l2, filters=128, kernel_size=3, name='stage1_conv3_l2')
        stage1_l2 = conv2(inputs=stage1_l2, filters=512, kernel_size=1, name='stage1_conv4_l2')
        stage1_l2 = conv2(inputs=stage1_l2, filters=19, kernel_size=1, act=False, name='stage1_conv5_l2')

        concat_stage_1 = tf.concat([stage1_l1, stage1_l2, add_layer], axis=3, name='concat_stage_1')

        stage2_l1 = conv2(inputs=concat_stage_1, filters=128, kernel_size=7, name='stage2_conv1_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=128, kernel_size=7, name='stage2_conv2_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=128, kernel_size=7, name='stage2_conv3_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=128, kernel_size=7, name='stage2_conv4_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=128, kernel_size=7, name='stage2_conv5_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=128, kernel_size=1, name='stage2_conv6_l1')
        stage2_l1 = conv2(inputs=stage2_l1, filters=38, kernel_size=1, act=False, name='stage2_conv7_l1')

        stage2_l2 = conv2(inputs=concat_stage_1, filters=128, kernel_size=7, name='stage2_conv1_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=128, kernel_size=7, name='stage2_conv2_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=128, kernel_size=7, name='stage2_conv3_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=128, kernel_size=7, name='stage2_conv4_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=128, kernel_size=7, name='stage2_conv5_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=128, kernel_size=1, name='stage2_conv6_l2')
        stage2_l2 = conv2(inputs=stage2_l2, filters=19, kernel_size=1, act=False, name='stage2_conv7_l2')
        cpm_out.append(stage2_l1)
        hm_out.append(stage2_l2)
        concat_stage_2 = tf.concat([stage2_l1, stage2_l2, add_layer], axis=3, name='concat_stage_2')

        stage3_l1 = conv2(inputs=concat_stage_2, filters=128, kernel_size=7, name='stage3_conv1_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=128, kernel_size=7, name='stage3_conv2_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=128, kernel_size=7, name='stage3_conv3_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=128, kernel_size=7, name='stage3_conv4_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=128, kernel_size=7, name='stage3_conv5_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=128, kernel_size=1, name='stage3_conv6_l1')
        stage3_l1 = conv2(inputs=stage3_l1, filters=38, kernel_size=1, act=False, name='stage3_conv7_l1')

        stage3_l2 = conv2(inputs=concat_stage_2, filters=128, kernel_size=7, name='stage3_conv1_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=128, kernel_size=7, name='stage3_conv2_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=128, kernel_size=7, name='stage3_conv3_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=128, kernel_size=7, name='stage3_conv4_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=128, kernel_size=7, name='stage3_conv5_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=128, kernel_size=1, name='stage3_conv6_l2')
        stage3_l2 = conv2(inputs=stage3_l2, filters=19, kernel_size=1, act=False, name='stage3_conv7_l2')
        cpm_out.append(stage3_l1)
        hm_out.append(stage3_l2)
        concat_stage_3 = tf.concat([stage3_l1, stage3_l2, add_layer], axis=3, name='concat_stage_3')

        stage4_l1 = conv2(inputs=concat_stage_3, filters=128, kernel_size=7, name='stage4_conv1_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=128, kernel_size=7, name='stage4_conv2_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=128, kernel_size=7, name='stage4_conv3_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=128, kernel_size=7, name='stage4_conv4_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=128, kernel_size=7, name='stage4_conv5_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=128, kernel_size=1, name='stage4_conv6_l1')
        stage4_l1 = conv2(inputs=stage4_l1, filters=38, kernel_size=1, act=False, name='stage4_conv7_l1')

        stage4_l2 = conv2(inputs=concat_stage_3, filters=128, kernel_size=7, name='stage4_conv1_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=128, kernel_size=7, name='stage4_conv2_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=128, kernel_size=7, name='stage4_conv3_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=128, kernel_size=7, name='stage4_conv4_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=128, kernel_size=7, name='stage4_conv5_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=128, kernel_size=1, name='stage4_conv6_l2')
        stage4_l2 = conv2(inputs=stage4_l2, filters=19, kernel_size=1, act=False, name='stage4_conv7_l2')
        cpm_out.append(stage4_l1)
        hm_out.append(stage4_l2)
        concat_stage_4 = tf.concat([stage4_l1, stage4_l2, add_layer], axis=3, name='concat_stage_4')

        stage5_l1 = conv2(inputs=concat_stage_4, filters=128, kernel_size=7, name='stage5_conv1_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=128, kernel_size=7, name='stage5_conv2_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=128, kernel_size=7, name='stage5_conv3_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=128, kernel_size=7, name='stage5_conv4_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=128, kernel_size=7, name='stage5_conv5_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=128, kernel_size=1, name='stage5_conv6_l1')
        stage5_l1 = conv2(inputs=stage5_l1, filters=38, kernel_size=1, act=False, name='stage5_conv7_l1')

        stage5_l2 = conv2(inputs=concat_stage_4, filters=128, kernel_size=7, name='stage5_conv1_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=128, kernel_size=7, name='stage5_conv2_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=128, kernel_size=7, name='stage5_conv3_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=128, kernel_size=7, name='stage5_conv4_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=128, kernel_size=7, name='stage5_conv5_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=128, kernel_size=1, name='stage5_conv6_l2')
        stage5_l2 = conv2(inputs=stage5_l2, filters=19, kernel_size=1, act=False, name='stage5_conv7_l2')
        cpm_out.append(stage5_l1)
        hm_out.append(stage5_l2)
        concat_stage_5 = tf.concat([stage5_l1, stage5_l2, add_layer], axis=3, name='concat_stage_5')

        stage6_l1 = conv2(inputs=concat_stage_5, filters=128, kernel_size=7, name='stage6_conv1_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=128, kernel_size=7, name='stage6_conv2_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=128, kernel_size=7, name='stage6_conv3_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=128, kernel_size=7, name='stage6_conv4_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=128, kernel_size=7, name='stage6_conv5_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=128, kernel_size=1, name='stage6_conv6_l1')
        stage6_l1 = conv2(inputs=stage6_l1, filters=38, kernel_size=1, act=False, name='stage6_conv7_l1')

        stage6_l2 = conv2(inputs=concat_stage_5, filters=128, kernel_size=7, name='stage6_conv1_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=128, kernel_size=7, name='stage6_conv2_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=128, kernel_size=7, name='stage6_conv3_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=128, kernel_size=7, name='stage6_conv4_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=128, kernel_size=7, name='stage6_conv5_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=128, kernel_size=1, name='stage6_conv6_l2')
        stage6_l2 = conv2(inputs=stage6_l2, filters=19, kernel_size=1, act=False, name='stage6_conv7_l2')
        cpm_out.append(stage6_l1)
        hm_out.append(stage6_l2)
        concat_stage_6 = tf.concat([stage6_l1, stage6_l2, add_layer], axis=3, name='concat_stage_6')

        stage7_l1 = conv2(inputs=concat_stage_6, filters=128, kernel_size=7, name='stage7_conv1_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=128, kernel_size=7, name='stage7_conv2_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=128, kernel_size=7, name='stage7_conv3_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=128, kernel_size=7, name='stage7_conv4_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=128, kernel_size=7, name='stage7_conv5_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=128, kernel_size=1, name='stage7_conv6_l1')
        stage7_l1 = conv2(inputs=stage7_l1, filters=38, kernel_size=1, act=False, name='stage7_conv7_l1')

        stage7_l2 = conv2(inputs=concat_stage_6, filters=128, kernel_size=7, name='stage7_conv1_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=128, kernel_size=7, name='stage7_conv2_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=128, kernel_size=7, name='stage7_conv3_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=128, kernel_size=7, name='stage7_conv4_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=128, kernel_size=7, name='stage7_conv5_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=128, kernel_size=1, name='stage7_conv6_l2')
        stage7_l2 = conv2(inputs=stage7_l2, filters=19, kernel_size=1, act=False, name='stage7_conv7_l2')
        cpm_out.append(stage7_l1)
        hm_out.append(stage7_l2)
        openpose = tf.concat([stage7_l1, stage7_l2], axis=3, name='concat_stage_7')

    return hm_out, cpm_out, add_layer
