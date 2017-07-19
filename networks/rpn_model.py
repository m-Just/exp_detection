import tensorflow as tf
from networks.network import Network

# define
n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class RegionProposalNetwork(Network):
    def __init__(self, inputs, feat_name, feat_dim, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)

        self.feat_name = feat_name
        self.feat_dim = feat_dim
        self.trainable = trainable

        self.setup()

    def setup(self):
        #========= RPN ============
        (self.feed(self.feat_name)
             .conv(3, 3, self.feat_dim, 1, 1, name='rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*2, 1, 1, padding='VALID',
                relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))

        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*4, 1, 1, padding='VALID',
                relu=False, name='rpn_bbox_pred'))

        # #========= RoI Proposal ============
        # (self.feed('rpn_cls_score')
        #      .reshape_layer(2, name='rpn_cls_score_reshape')
        #      .softmax(name='rpn_cls_prob'))
        #
        # (self.feed('rpn_cls_prob')
        #      .reshape_layer(len(anchor_scales)*3*2, name='rpn_cls_prob_reshape'))
        #
        # (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
        #      .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))
        #
        # (self.feed('rpn_rois', 'gt_boxes')
        #      .proposal_target_layer(n_classes, name='roi-data'))
        #
        # #========= RCNN ============
        # (self.feed(self.feat_name, 'roi-data')
        #      .roi_pool(7, 7, 1.0/16, name='pool_5')
        #      .fc(4096, name='fc6')
        #      .dropout(0.5, name='drop6')
        #      .fc(4096, name='fc7')
        #      .dropout(0.5, name='drop7')
        #      .fc(n_classes*4, relu=False, name='bbox_pred'))
        #
        # # classifier
        # (self.feed('drop7')
        #      .fc(n_classes, relu=False, name='cls_score')
        #      .softmax(name='cls_prob'))
