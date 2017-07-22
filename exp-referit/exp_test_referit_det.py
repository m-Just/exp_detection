from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from networks import text_objseg_model as segmodel
from util import im_processing, text_processing, eval_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
query_file = './exp-referit/data/referit_query_test.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

pretrained_model = './exp-referit/tfmodel/referit_fc8_det_iter_60000.tfmodel'

# Model Params
T = 20
N = 100
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim

# Evaluation Param
correct_iou_thresh = 0.5
use_nms = False
nms_thresh = 0.3

################################################################################
# Evaluation network
################################################################################

# Inputs
text_seq_batch = np.zeros((T, N), dtype=np.int32)
imcrop_batch = np.zeros((N, input_H, input_W, 3), dtype=np.uint8)
imsize_batch = np.zeros((N, 2), dtype=np.float32)
gt_box_batch = np.zeros((N, 5), dtype=np.float32)   # (x1, y1, x2, y2, cls)

# Outputs
scores = segmodel.text_objseg_region(text_seq_batch, imcrop_batch, imsize_batch,
    gt_box_batch, num_vocab, embed_dim, lstm_dim, rpn_feat_dim,
    mlp_dropout=mlp_dropout, is_training=is_bn_training)

# Load pretrained model
snapshot_saver = tf.train.Saver()
sess = tf.Session()
snapshot_saver.restore(sess, pretrained_model)

################################################################################
# Load annotations and bounding box proposals
################################################################################

query_dict = json.load(open(query_file))
bbox_dict = json.load(open(bbox_file))
imcrop_dict = json.load(open(imcrop_file))
imsize_dict = json.load(open(imsize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Flatten the annotations
################################################################################

flat_query_dict = {imname: [] for imname in imlist}
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    for imcrop_name in this_imcrop_names:
        gt_bbox = bbox_dict[imcrop_name]
        if imcrop_name not in query_dict:
            continue
        this_descriptions = query_dict[imcrop_name]
        for description in this_descriptions:
            flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

################################################################################
# Testing
################################################################################

eval_bbox_num_list = [1, 10, 100]
bbox_correct = np.zeros(len(eval_bbox_num_list), dtype=np.int32)
bbox_total = 0

# Pre-allocate arrays
imcrop_val = np.zeros((N, 224, 224, 3), dtype=np.float32)
spatial_val = np.zeros((N, 8), dtype=np.float32)
text_seq_val = np.zeros((T, 1), dtype=np.int32)
lstm_top_val = np.zeros((N, D_text))

num_im = len(imlist)
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    imsize = imsize_dict[imname]

    # # Extract visual features from all proposals
    # im = skimage.io.imread(image_dir + imname)
    # if im.ndim == 2:
    #     im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
    # imcrop_val[:num_proposal, ...] = im_processing.crop_bboxes_subtract_mean(
    #     im, bbox_proposals, 224, vgg_net.channel_mean)

    # Extract textual features from sentences
    for imcrop_name, gt_bbox, description in flat_query_dict[imname]:
        text_seq_val[:, 0] = text_processing.preprocess_sentence(description, vocab_dict, T)

        feed_dict = {
            text_seq_batch: text_seq_val,
            imcrop_batch  : imcrop_val,
            imsize_batch  : imsize_val,
            gt_box_batch  : gt_box_val,
        }
