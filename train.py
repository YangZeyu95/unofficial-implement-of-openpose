import argparse
import tensorflow as tf
import os
from dataset import get_dataflow, batch_dataflow
from dataflow import COCODataPaths
import cv2
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
    args = parser.parse_args()
    echos = 100

    # batch_size = 2
    # checkpoint_path = 'checkpoints/train/'
    # vgg19_ckpt_path = "checkpoints/vgg/vgg_19.ckpt"
    # curr_dir = os.path.dirname(__file__)
    # annot_path_train = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
    #                  'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/' \
    #                  'person_keypoints_train2017.json'
    # img_dir_train = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
    #               'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/train2017/'
    # annot_path_val = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
    #                  'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/' \
    #                  'person_keypoints_val2017.json'
    # img_dir_val = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
    #               'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/val2017/'

    # coco_data_train = COCODataPaths(annot_path=args.annot_path_train, img_dir=args.img_path_train)
    coco_data_val = COCODataPaths(
        annot_path=args.annot_path_val, img_dir=args.img_path_val)
    df = get_dataflow(coco_data_val)
    train_samples = df.size()
    batch_df = batch_dataflow(df, args.batch_size)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, 368, 368, 3])
        mask_hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 19])
        mask_cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])
        hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 1])
        cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])

    img_normalized = raw_img / 255 - 0.5  # [-0.5, 0.5]
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    net = CpmStage1(inputs_x=vgg_outputs, mask_cpm=mask_cpm,
                    mask_hm=mask_hm, gt_hm=hm, gt_cpm=cpm, stage_num=2)
    hm_pre, loss = net.gen_net()
    # loss = tf.div(tf.div(loss, args.batch_size), 2)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        1e-5, global_step, 5000, 0.9, staircase=True)
    tf.summary.scalar("lr", learning_rate)
    loss = tf.reduce_sum(tf.square(hm_pre - hm))
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate).minimize(
            loss=loss, global_step=global_step, var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stage1'), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='add_layers')])

    tf.summary.image('input', img_normalized, max_outputs=2)
    tf.summary.histogram('all', tf.get_collection(tf.GraphKeys))
    #
    # tf.summary.image('hm', hm[:, :, :, 0:1], max_outputs=2)
    # tf.summary.image('hm_pre', hm_pre[:, :, :, 0:1], max_outputs=2)
    #
    # tf.summary.image('cpm', cpm[:, :, :, 0:1], max_outputs=2)
    # tf.summary.image('cpm_pre', cpm_pre[:, :, :, 0:1], max_outputs=2)
    # variables_can_be_restored_list = []
    # variables_can_be_restored_tup = tf.train.list_variables(vgg19_ckpt_path)
    # for variables_can_be_restored in variables_can_be_restored_tup:
    #     variables_can_be_restored_list.append(variables_can_be_restored[0])
    variables_in_checkpoint = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')
    restorer = tf.train.Saver(variables_in_checkpoint)
    merged = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(args.checkpoint_path, sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        print('restoring from vgg19')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        for echo in range(echos):
            for i, data in enumerate(batch_df):
                # sess.run(train, feed_dict={raw_img: data[0],
                #                            mask_cpm: data[1],
                #                            mask_hm: data[2],
                #                            cpm: data[3],
                #                            hm: data[4]})
                # if i == 2:
                for p in range(20000):
                    total_loss, _, summary, hm_out = sess.run([loss, train, merged, hm_pre], feed_dict={raw_img: data[0],
                                                                                        mask_cpm: data[1],
                                                                                        mask_hm: data[2],
                                                                                        cpm: data[3],
                                                                                        hm: data[4][:, :, :, 0:1]})
                    print(str(total_loss) + ' ' +
                          str(echo * len(batch_df) + i + p))
                    writer.add_summary(summary, echo * len(batch_df) + i + p)
                # cv2.imshow("j", data[0][0][1][:][:][:])
            # cv2.waitKey(0)
