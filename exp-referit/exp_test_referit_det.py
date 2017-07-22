from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from networks import text_objseg_model as segmodel
from utils import im_processing, text_processing, eval_tools

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

pretrained_model = './exp-referit/tfmodel/referit_rpn_det_iter_60000.tfmodel'

# Model Params
T = 20
N = 1   # TODO changed from 100 to 1
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
rpn_feat_dim = 512
input_H = 320
input_W = 320

# Evaluation Param
correct_iou_thresh = 0.5
use_nms = False
nms_thresh = 0.3

################################################################################
# Evaluation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])
imsize_batch = tf.placeholder(tf.float32, [N, 2])
gt_box_batch = tf.placeholder(tf.float32, [N, 5])

# Outputs
net = segmodel.text_objseg_region(text_seq_batch, imcrop_batch, imsize_batch,
    gt_box_batch, num_vocab, embed_dim, lstm_dim, rpn_feat_dim,
    is_training=False)

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

rpn_bbox_pred = tf.reshape(net.get_output('rpn_bbox_pred'), [-1, 4])    # TODO gather according to rpn_label ?
rpn_cls_score = tf.reshape(net.get_output('rpn_cls_score_reshape'), [-1, 2])

eval_bbox_num_list = [1, 10, 100]
bbox_correct = np.zeros(len(eval_bbox_num_list), dtype=np.int32)
bbox_total = 0

num_im = len(imlist)
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    imsize_val = imsize_dict[imname]

    # Extract visual features from image
    im = skimage.io.imread(image_dir + imname)
    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]
    imcrop_val = processed_im[..., ::-1] - segmodel.IMG_MEAN

    # Extract textual features from sentences
    for imcrop_name, gt_bbox, description in flat_query_dict[imname]:
        text_seq_val = text_processing.preprocess_sentence(description, vocab_dict, T)

        feed_dict = {
            text_seq_batch: text_seq_val,
            imcrop_batch  : imcrop_val,
            imsize_batch  : imsize_val,
            gt_box_batch  : np.array([0, 0, 0, 0, 1], dtype=np.float32)
        }   # TODO gt_box_batch is here only for rpn anchor layer's required parameter,
            # which is unused in the inferences, and should be removed later

        bbox_pred, score = sess.run([rpn_bbox_pred, rpn_cls_score], feed_dict=feed_dict)

        pos_label = np.ones(len(bbox_pred), dtype=np.int32)
        pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=pos_label, logits=score).eval(session=sess)
        predictions = np.argsort(pos_loss)

        for i in predictions:
            print(bbox_pred[i])
