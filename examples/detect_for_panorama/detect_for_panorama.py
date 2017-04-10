#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Gump from CQU
# * Email         : gumpglh@qq.com
# * Create time   : 2017-04-10 16:02
# * Last modified : 2017-04-10 16:02
# * Filename      : detect_for_panorama.py
# * Description   :
# * Copyright © gumpglh. All rights reserved.
# **********************************************************

import os
import sys
import glob
import subprocess

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

caffe_root = "."
model_file = "{}/models/VOC0712/SSD_300x300/deploy.prototxt".format(caffe_root)
weight_file = "{}/models/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel".format(caffe_root)
imglist_file = "{}/examples/detect_for_panorama/images/images_list.txt".format(caffe_root)
output_file = "{}/examples/detect_for_panorama/images/output.txt".format(caffe_root)
images_dir = "{}/examples/detect_for_panorama/images".format(caffe_root)
confidence_threshold = 0.6

with open(imglist_file, 'w') as f:
    images = glob.glob("{}/*.jpg".format(images_dir))
    for im in images:
        f.write(im + '\n')

cmd = "./build/examples/detect_for_panorama/detect4panorama.bin --out_file={} --confidence_threshold={} {} {} {}".format(
        output_file, confidence_threshold, model_file, weight_file, imglist_file)
print cmd
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output = process.communicate()[0]
