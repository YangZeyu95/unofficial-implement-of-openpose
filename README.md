# Unofficial-Implement-of-Openpose
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/IMG_4030.GIF", width="720">
</p>　　

An easy implement for openpose using TensorFlow.

Only basic python is used, so the code is easy to understand.

You can check the graph, internal outputs of every stage and histogram of every layer in tensorboard.

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose.

The Dataloader and Post-processing code is from [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation).

Python 3.6　　
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/graph_run%3D.png", width="720">
</p>　

## Training
1. Download vgg19 weights file [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) or 链接: https://pan.baidu.com/s/1ZxWKVPe4hrEhDxOpjLiUKA 提取码: widj and uzip to 'checkpoints/vgg/' (please create the path yourself).
2. Download COCO2017: 2017 Train images, 2017 Val images and 2017 Train/Val annotations [here](http://cocodataset.org/#download).  
make sure have this structure:  
-COCO/  
　-images/  
　　-train2017/  
　　-val2017/  
　-annotations/    

3. Specify the '--annot_path_train' and '--img_path_train' in train.py to your own 'COCO/annotations/' and 'COCO/images/'.
4. run train.py `python train.py` and install requirements follow the error and run again.
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/loss2%20(1).svg", width="720">
</p>　　
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/Screenshot%202018-12-12%20at%2010.21.55%20PM.png", width="720">
</p>　　

## Test
Specify the --checkpoint_path and --img_path in run.py to your path. If you want use webcam, set --run_model webcam.   
`python run.py`  
pretrained model will be uploaded soon.　　


