import train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--batch_size', type=str, default=10)
    parser.add_argument('--not_continue_training', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--annot_path_train', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/')
    parser.add_argument('--img_path_train', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/')
    parser.add_argument('--annot_path_val', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/'
                        'person_keypoints_val2017.json')
    parser.add_argument('--img_path_val', type=str, default='/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/'
                        'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/val2017/')
    parser.add_argument('--save_checkpoint_frequency', type=str, default=1000)
    parser.add_argument('--save_summary_frequency', type=str, default=100)
    parser.add_argument('--stage_num', type=str, default=6)
    parser.add_argument('--hm_channels', type=str, default=19)
    parser.add_argument('--cpm_channels', type=str, default=38)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--img_path', type=str, default='images/ski.jpg')
    parser.add_argument('--max_echos', type=str, default=5)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--loss_func', type=str, default='square')
    args = parser.parse_args()

    # train.train(args=args, loss_func='org', use_bn=False)
    train.train()
