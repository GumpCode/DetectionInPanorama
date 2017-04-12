#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Gump from CQU
# * Email         : gumpglh@qq.com
# * Create time   : 2017-04-11 15:57
# * Last modified : 2017-04-11 15:57
# * Filename      : draw_detections.py
# * Description   :
# * Copyright © gumpglh. All rights reserved.
# **********************************************************

import os

caffe_root = "."
detections_list_file = "{}/examples/detect_for_panorama/images/output.txt".

with open(detections_list_file, 'r') as f:
    f.readlines()
