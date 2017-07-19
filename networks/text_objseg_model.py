from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# from tensorflow.python.ops.nn import dropout as drop
# from utils.cnn import conv_layer as conv
from utils.cnn import conv_relu_layer as conv_relu
# from utils.cnn import deconv_layer as deconv
# from utils.cnn import fc_layer as fc
# from utils.cnn import fc_relu_layer as fc_relu
from models.processing_tools import *
from networks import lstm_net
from networks.rpn_model import RPN
from deeplab_resnet.model import DeepLabResNetModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def text_objseg_region(text_seq_batch, imcrop_batch, spatial_batch,
    imsize_batch, gt_box_batch, num_vocab, embed_dim, lstm_dim, rpn_feat_dim,
    mlp_dropout, is_training):

    # Language feature (LSTM hidden state)
    feat_lang = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

    # deeplab101
    resnet = DeepLabResNetModel({'data': imcrop_batch},
        is_training=is_training, num_classes=embed_dim)
    feat_vis = resnet.layers['fc1_voc12']

    # Reshape and tile LSTM top
    featmap_H, featmap_W = feat_vis.get_shape().as_list()[1:3]
    N, D_text = feat_lang.get_shape().as_list()
    feat_lang = tf.tile(tf.reshape(feat_lang, [N, 1, 1, D_text]),
        [1, featmap_H, featmap_W, 1])

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them along axis 3 (channel dimension)
    spatial_batch = tf.convert_to_tensor(generate_spatial_batch(N, featmap_H, featmap_W))
    feat_all = tf.concat(values=[tf.nn.l2_normalize(feat_lang, 3),
                                 tf.nn.l2_normalize(feat_vis, 3),
                                 spatial_batch], axis=3)

    # conv all feats as RPN input
    feat_all_conv = conv_relu('feat_all_conv', feat_all,
                              kernel_size=3, stride=1, output_dim=rpn_feat_dim)
                              # TODO test for different output dim

    # feed feature maps to RPN
    rpn_net = RPN({'gt_boxes': gt_box_batch, 'im_info': imsize_batch},
        feat_all_conv, rpn_feat_dim, trainable=True)

    return rpn_net

    # MLP Classifier over concatenate feature
    # with tf.variable_scope('classifier'):
    #     mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
    #     if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
    #     mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)
    #
    # return mlp_l2
