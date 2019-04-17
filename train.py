import os
import time
import logging
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.contrib import slim
import vgg
from cpm import PafNet
from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale


def train():
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--batch_size', type=str, default=10)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--annot_path', type=str,
                        default='/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
                                'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/')
    parser.add_argument('--img_path', type=str,
                        default='/run/user/1000/gvfs/smb-share:server=server,share=data/yzy/dataset/'
                                'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/')
    # parser.add_argument('--annot_path_val', type=str,
    #                     default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
    #                             'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/'
    #                             'person_keypoints_val2017.json')
    # parser.add_argument('--img_path_val', type=str,
    #                     default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
    #                             'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/val2017/')
    parser.add_argument('--save_checkpoint_frequency', type=str, default=1000)
    parser.add_argument('--save_summary_frequency', type=str, default=100)
    parser.add_argument('--stage_num', type=str, default=6)
    parser.add_argument('--hm_channels', type=str, default=19)
    parser.add_argument('--paf_channels', type=str, default=38)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--max_echos', type=str, default=5)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--loss_func', type=str, default='l2')
    args = parser.parse_args()

    if not args.continue_training:
        start_time = time.localtime(time.time())
        checkpoint_path = args.checkpoint_path + ('%d-%d-%d-%d-%d-%d' % start_time[0:6])
        os.mkdir(checkpoint_path)
    else:
        checkpoint_path = args.checkpoint_path

    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(checkpoint_path + '/train_log.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('checkpoint_path: ' + checkpoint_path)

    # define input placeholder
    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[args.batch_size, 368, 368, 3])
        # mask_hm = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 46, 46, args.hm_channels])
        # mask_paf = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 46, 46, args.paf_channels])
        hm = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 46, 46, args.hm_channels])
        paf = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 46, 46, args.paf_channels])

    # defien data loader
    logger.info('initializing data loader...')
    set_network_input_wh(args.input_width, args.input_height)
    scale = 8
    set_network_scale(scale)
    df = get_dataflow_batch(args.annot_path, True, args.batch_size, img_path=args.img_path)
    steps_per_echo = df.size()
    enqueuer = DataFlowToQueue(df, [raw_img, hm, paf], queue_size=100)
    q_inp, q_heat, q_vect = enqueuer.dequeue()
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, 1), tf.split(q_heat, 1), tf.split(q_vect, 1)
    img_normalized = q_inp_split[0] / 255 - 0.5  # [-0.5, 0.5]

    df_valid = get_dataflow_batch(args.annot_path, False, args.batch_size, img_path=args.img_path)
    df_valid.reset_state()
    validation_cache = []

    logger.info('initializing model...')
    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    net = PafNet(inputs_x=vgg_outputs, stage_num=args.stage_num, hm_channel_num=args.hm_channels, use_bn=args.use_bn)
    hm_pre, paf_pre, added_layers_out = net.gen_net()

    # two kinds of loss
    losses = []
    with tf.name_scope('loss'):
        for idx, (l1, l2), in enumerate(zip(hm_pre, paf_pre)):
            if args.loss_func == 'square':
                hm_loss = tf.reduce_sum(tf.square(tf.concat(l1, axis=0) - q_heat_split[0]))
                paf_loss = tf.reduce_sum(tf.square(tf.concat(l2, axis=0) - q_vect_split[0]))
                losses.append(tf.reduce_sum([hm_loss, paf_loss]))
                logger.info('use square loss')
            else:
                hm_loss = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_heat_split[0])
                paf_loss = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_vect_split[0])
                losses.append(tf.reduce_mean([hm_loss, paf_loss]))
                logger.info('use l2 loss')
        loss = tf.reduce_sum(losses) / args.batch_size

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(1e-4, global_step, steps_per_echo, 0.5, staircase=True)
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_vgg:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8).minimize(loss=loss,
                                                                                           global_step=global_step,
                                                                                           var_list=trainable_var_list)
    logger.info('initialize saver...')
    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)

    logger.info('initialize tensorboard')
    tf.summary.scalar("lr", learning_rate)
    tf.summary.scalar("loss2", loss)
    tf.summary.histogram('img_normalized', img_normalized)
    tf.summary.histogram('vgg_outputs', vgg_outputs)
    tf.summary.histogram('added_layers_out', added_layers_out)
    tf.summary.image('vgg_out', tf.transpose(vgg_outputs[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=512)
    tf.summary.image('added_layers_out', tf.transpose(added_layers_out[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=128)
    tf.summary.image('paf_gt', tf.transpose(q_vect_split[0][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=38)
    tf.summary.image('hm_gt', tf.transpose(q_heat_split[0][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=19)
    for i in range(args.stage_num):
        tf.summary.image('hm_pre_stage_%d' % i, tf.transpose(hm_pre[i][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=19)
        tf.summary.image('paf_pre_stage_%d' % i, tf.transpose(paf_pre[i][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=38)
    tf.summary.image('input', img_normalized, max_outputs=4)

    logger.info('initialize session...')
    merged = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(checkpoint_path, sess.graph)
        sess.run(tf.group(tf.global_variables_initializer()))
        if args.backbone_net_ckpt_path is not None:
            logger.info('restoring vgg weights from %s' % args.backbone_net_ckpt_path)
            restorer.restore(sess, args.backbone_net_ckpt_path)
        if args.continue_training:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
            logger.info('restoring from checkpoint...')
        logger.info('start training...')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()
        while True:
            best_checkpoint = float('inf')
            for _ in tqdm(range(steps_per_echo),):
                total_loss, _, gs_num = sess.run([loss, train, global_step])
                echo = gs_num / steps_per_echo

                if gs_num % args.save_summary_frequency == 0:
                    total_loss, gs_num, summary, lr = sess.run([loss, global_step, merged, learning_rate])
                    writer.add_summary(summary, gs_num)
                    logger.info('echos=%f, setp=%d, total_loss=%f, lr=%f' % (echo, gs_num, total_loss, lr))

                if gs_num % args.save_checkpoint_frequency == 0:
                    valid_loss = 0
                    if len(validation_cache) == 0:
                        for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                            validation_cache.append((images_test, heatmaps, vectmaps))
                        df_valid.reset_state()
                        del df_valid
                        df_valid = None

                    for images_test, heatmaps, vectmaps in validation_cache:
                        valid_loss += sess.run(loss, feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps})

                    if valid_loss / len(validation_cache) <= best_checkpoint:
                        best_checkpoint = valid_loss / len(validation_cache)
                        saver.save(sess, save_path=checkpoint_path + '/' + 'model', global_step=gs_num)
                        logger.info('best_checkpoint = %f, saving checkpoint to ' % best_checkpoint + checkpoint_path + '/' + 'model-%d' % gs_num)

                    else:
                        logger.info('loss = %f drop' % (valid_loss / len(validation_cache)))

                if echo >= args.max_echos:
                    sess.close()
                    return 0


if __name__ == '__main__':
    train()
