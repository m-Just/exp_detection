from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform

from utils import im_processing, text_processing, eval_tools
from models import processing_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
query_file = './exp-referit/data/referit_query_trainval.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

# Saving directory
data_folder = './exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'

# Sample selection params
pos_iou = .7
neg_iou = 1e-6
neg_to_pos_ratio = 1.0

# Model Param
N = 1
T = 20
input_H = 320
input_W = 320

################################################################################
# Load annotations and bounding box proposals
################################################################################

query_dict = json.load(open(query_file))
bbox_dict = json.load(open(bbox_file))
imcrop_dict = json.load(open(imcrop_file))
imsize_dict = json.load(open(imsize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

# Object proposals
bbox_proposal_dict = {}
for imname in imlist:
    bboxes = np.loadtxt(bbox_proposal_dir + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
    bbox_proposal_dict[imname] = bboxes

################################################################################
# Load training data
################################################################################

# Gather training sample per image
training_samples = []
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    imsize = imsize_dict[imname]
    for imcrop_name in this_imcrop_names:
        if not imcrop_name in query_dict:
            continue
        gt_bbox = np.array(bbox_dict[imcrop_name]).reshape((1, 4))
        this_descriptions = query_dict[imcrop_name]

        for description in this_descriptions:
            sample = (imname, imsize, gt_bbox, description)
            training_samples.append(sample)

# Shuffle samples
np.random.seed(3)
perm_idx = np.random.permutation(len(training_samples))
shuffled_training_samples = [training_samples[n] for n in perm_idx]
del training_samples
print('#total sample =', len(training_samples))

num_batch = len(shuffled_training_samples) // N
print('#total batch = %d' % num_batch)

################################################################################
# Save training samples to disk
################################################################################

text_seq_batch = np.zeros((T, N), dtype=np.int32)
imcrop_batch = np.zeros((N, input_H, input_W, 3), dtype=np.uint8)
spatial_batch = np.zeros((N, 8), dtype=np.float32)
imsize_batch = np.zeros((N, 2), dtype=np.float32)
gt_box_batch = np.zeros((N, 5), dtype=np.float32)   # (x1, y1, x2, y2, cls)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        imname, imsize, sample_bbox, description = shuffled_training_samples[n_sample]
        im = skimage.io.imread(image_dir + imname)
        xmin, ymin, xmax, ymax = sample_bbox

        imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
        imcrop = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [input_H, input_W]))
        spatial_feat = processing_tools.spatial_feature_from_bbox(sample_bbox, imsize)
        text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)

        idx = n_sample - batch_begin
        text_seq_batch[:, idx] = text_seq
        imcrop_batch[idx, ...] = imcrop
        spatial_batch[idx, ...] = spatial_feat
        imsize_batch[idx, ...] = np.array(imsize[::-1], dtype=np.float32) # result size format is height x width
        gt_box_batch[idx, ...] = np.array([xmin, ymin, xmax, ymax, 1], dtype=np.float32)
            # TODO here labels = 1 or 0 (object or non-object)

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
        text_seq_batch=text_seq_batch,
        imcrop_batch=imcrop_batch,
        spatial_batch=spatial_batch,
        imsize_batch=imsize_batch,
        gt_box_batch=gt_box_batch)
