import argparse
import tensorflow as tf
import os
from dataset import get_dataflow, batch_dataflow
from dataflow import COCODataPaths
from tensorflow.contrib import slim
import vgg
from cpm import CpmStage1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--batch_size', type=str, default=1)
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints/train/')
    parser.add_argument('--backbone_net_ckpt_path', type=str,
                        default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--annot_path_train', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/'
                        'person_keypoints_train2017.json')
    parser.add_argument('--img_path_train', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/train2017/')
    parser.add_argument('--annot_path_val', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/'
                        'person_keypoints_val2017.json')
    parser.add_argument('--img_path_val', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/val2017/')
    parser.add_argument('--stage_num', type=str, default=1)
    args = parser.parse_args()
    echos = 100

    for i in os.listdir(args.checkpoint_path):
        os.remove(args.checkpoint_path + i)

    # get training data
    coco_data_val = COCODataPaths(annot_path=args.annot_path_val, img_dir=args.img_path_val)
    df = get_dataflow(coco_data_val)
    batch_df = batch_dataflow(df, args.batch_size)

    # define input placeholder
    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, 368, 368, 3])
        mask_hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 19])
        mask_cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])
        hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 19])
        cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])

    img_normalized = raw_img / 255 - 0.5  # [-0.5, 0.5]

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    net = CpmStage1(inputs_x=vgg_outputs, mask_cpm=mask_cpm,
                    mask_hm=mask_hm, gt_hm=hm, gt_cpm=cpm, stage_num=args.stage_num)
    hm_pre, cpm_pre, loss = net.gen_net()
    tf.summary.histogram('hm_pre', hm_pre)
    tf.summary.histogram('hm_gt', hm)
    # 这个loss是其他版本代码里的实现
    losses = []
    with tf.name_scope('loss'):
        for idx, (l1, l2) in enumerate(zip(cpm_pre, hm_pre)):
            loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - mask_cpm)
            loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - mask_hm)
            losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        loss_2 = tf.reduce_sum(losses) / args.batch_size

    tf.summary.scalar("loss2", loss_2)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(5e-4, global_step, 1000, 0.9, staircase=True)
    tf.summary.scalar("lr", learning_rate)

    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                               global_step=global_step,
                                                               var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                           scope='stage1'),
                                                                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                           scope='staget'),
                                                                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                           scope='add_layers')])# ,
                                                                         # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                         #                   scope='vgg_19')])
    # tf.train.AdagradOptimizer(learning_rate=0.1).minimize()
    # get vgg19 restorer
    variables_in_checkpoint = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')
    restorer = tf.train.Saver(variables_in_checkpoint)

    tf.summary.image('vgg_out', tf.transpose(vgg_outputs[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=512)
    tf.summary.image('cpm_gt', tf.transpose(cpm[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=38)
    tf.summary.image('hm_gt', tf.transpose(hm[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=19)
    for i in range(args.stage_num):
        tf.summary.image('hm_pre_stage_%d' % i, tf.transpose(hm_pre[i][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=19)
        tf.summary.image('cpm_pre_stage_%d' % i, tf.transpose(cpm_pre[i][0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=38)
    tf.summary.image('input', raw_img, max_outputs=4)
    tf.summary.image('hm_mask', tf.transpose(mask_hm[0:1, :, :, :], perm=[3, 1, 2, 0]), max_outputs=19)
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(args.checkpoint_path, sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # sess.run(tf.local_variables_initializer())
        restorer.restore(sess, args.backbone_net_ckpt_path)
        for echo in range(echos):
            for i, data in enumerate(batch_df):
                for p in range(20000):
                    total_loss, _, summary, hm_out, m_hm = sess.run([loss, train, merged, hm_pre, mask_hm],
                                                                    feed_dict={raw_img: data[0],
                                                                               mask_cpm: data[1],
                                                                               mask_hm: data[2],
                                                                               cpm: data[3],
                                                                               hm: data[4]})
                    # loss_img = total_loss[0, :, :, 0]
                    print(str(total_loss) + ' ' +
                          str(echo * len(batch_df) + i + p))
                    writer.add_summary(summary, echo * len(batch_df) + i + p)
