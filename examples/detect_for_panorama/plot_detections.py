'''
Plot the detection results output by ssd_detect.cpp.
'''

import argparse
from collections import OrderedDict
from google.protobuf import text_format
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
import sys

import caffe
from caffe.proto import caffe_pb2

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def showResults(img_file, results, labelmap=None, save_dir=None):
    if not os.path.exists(img_file):
        print "{} does not exist".format(img_file)
        return
    img = io.imread(img_file)
    plt.clf()
    plt.imshow(img)
    plt.axis('off');
    ax = plt.gca()
    if labelmap:
        # generate same number of colors as classes in labelmap.
        num_classes = len(labelmap.item)
    else:
        # generate 20 colors.
        num_classes = 20
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    for res in results:
        label = res['label']
        name = "class " + str(label)
        if labelmap:
            name = get_labelname(labelmap, label)[0]
        color = colors[label % num_classes]
        bbox = res['bbox']
        coords = (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        if 'score' in res:
            score = res['score']
            display_text = '%s: %.2f' % (name, score)
        else:
            display_text = name
        #ax.text(bbox[0], bbox[1], display_text, bbox={'facecolor':color, 'alpha':0.5})
    plt.savefig(save_dir + img_file.split('/')[-1])

if __name__ == "__main__":
    img_results = OrderedDict()
    caffe_root = '.'
    examples_root = "{}/examples/detect_for_panorama".format(caffe_root)
    img_dir = "{}/images".format(examples_root)

    full_result_file = "{}/images/full/output.txt".format(examples_root)
    part_result_file = "{}/images/part/output.txt".format(examples_root)
    file_list = [full_result_file, part_result_file]

    labelmap_file = "{}/data/VOC0712/labelmap_voc.prototxt".format(caffe_root)
    save_dir = "{}/images/output/".format(examples_root)
    labelmap = None
    if labelmap_file and os.path.exists(labelmap_file):
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
    for f in file_list:
        result_file = f
        with open(result_file, "r") as f:
            for line in f.readlines():
                img_name, label, score, xmin, ymin, xmax, ymax = line.strip("\n").split()
                #print img_name, label, score, xmin
                if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                   pass
                else:
                    img_file = img_name
                    result = dict()
                    result["label"] = int(label)
                    result["score"] = float(score)
                    result["bbox"] = [float(xmin), float(ymin), float(xmax), float(ymax)]
                    if img_file not in img_results:
                        img_results[img_file] = [result]
                    else:
                        img_results[img_file].append(result)
        for img_file, results in img_results.iteritems():
            showResults(img_file, results, labelmap, save_dir)
