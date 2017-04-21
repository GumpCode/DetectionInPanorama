#!/bin/bash
# **********************************************************
# * Author        : Gump from CQU
# * Email         : gumpglh@qq.com
# * Create time   : 2017-04-21 13:48
# * Last modified : 2017-04-21 13:48
# * Filename      : detect_for_panorama.sh
# * Description   : 
# * Copyright © gumpglh. All rights reserved.
# **********************************************************
caffe_root_dir=/home/ganlinhao/GumpCode/DetectionInPanorama

cd $caffe_root_dir

confidence_threshold=0.8
model_file=$caffe_root_dir/models/VOC0712/SSD_300x300/deploy.prototxt
weight_file=$caffe_root_dir/models/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
type=full
imglist_file=$caffe_root_dir/examples/detect_for_panorama/images/full/images_list.txt

./build/examples/detect_for_panorama/detect4panorama.bin --confidence_threshold=$confidence_threshold $model_file $weight_file $imglist_file
