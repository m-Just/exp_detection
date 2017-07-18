from __future__ import absolute_import, division, print_function

import sys
import os
import tensorflow as tf
import numpy as np

from networks import text_objseg_model as segmodel
from utils import data_reader
from utils import loss
from utils import rcnn

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
N = 10  # TODO test for different batch sizes
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
rpn_feat_dim = 512
is_bn_training = False

# Initialization Params
pretrained_params = '../deeplab_resnet/models/deeplab_resnet_init.ckpt'

# Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_lr = 0.01
lr_decay_step = 10000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.9
max_iter = 25000

fix_convnet = True
vgg_dropout = False
mlp_dropout = False
vgg_lr_mult = 1.

# Data Params
data_folder = './exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'

# Snapshot Params
snapshot = 5000
snapshot_file = './exp-referit/tfmodel/referit_rpn_det_iter_%d.tfmodel'

################################################################################
# Parsed Arguments
################################################################################

parser = argparse.ArgumentParser(description="Localizaton with Deeplab101-RPN")
parser.add_argument("--gpu", type=str, default=0,
                    help="Which gpu to use.")
# parser.add_argument("--batch_size", type=int, default=N,
#                     help="Number of images sent to the network in one step.")
parser.add_argument("--start_lr", type=float, default=start_lr,
                    help="Start learning rate.")
parser.add_argument("--end_lr", type=float, default=end_lr,
                    help="End learning rate.")
parser.add_argument("--max_iter", type=int, default=max_iter,
                    help="Number of training iterations.")
parser.add_argument("--fix_convnet", type=int, default=fix_convnet,
                    help="Whether keep the conv5_x layers fixed.")
parser.add_argument("--deeplab_lr_mult", type=float, default=deeplab_lr_mult,
                    help="Learning rate multiplier for fine-tuning deeplab network.")

args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
spatial_batch = tf.placeholder(tf.float32, [N, 8])
label_batch = tf.placeholder(tf.float32, [N, 1])
imsize_batch = tf.placeholder(tf.float32, [N, 2])
gt_box_batch = tf.placeholder(tf.float32, [N, 5])

# Outputs
net = segmodel.text_objseg_region(text_seq_batch, imcrop_batch,
    spatial_batch, imsize_batch, gt_box_batch, num_vocab, embed_dim,
    lstm_dim, rpn_feat_dim, vgg_dropout=vgg_dropout, mlp_dropout=mlp_dropout)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################

# Only train the last layers of convnet and keep conv layers fixed
if args.fix_convnet == 1:
    train_var_list = [var for var in tf.trainable_variables()
                        if var.name.startswith('fc1_voc12') or var.name.startswith('classifier')
                        or var.name.startswith('word_embedding') or var.name.startswith('lstm')
                        or var.name.startswith('rpn')]
else: # also train the conv5_x layers
    train_var_list = [var for var in tf.trainable_variables()
                        if var.name.startswith('fc1_voc12') or var.name.startswith('classifier')
                        or var.name.startswith('word_embedding') or var.name.startswith('lstm')
                        or var.name.startswith('res5') or var.name.startswith('bn5')
                        or var.name.startswith('rpn')]
                        # TODO check if all rpn layers name starts with rpn

# Add regularization to weight matrices (excluding bias)
# TODO confirm if rpn weights should be regularized
reg_var_list = [var for var in tf.trainable_variables()
                if (var in train_var_list) and not var.name.startswith('rpn') and
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.name)
print('Done.')

# Collect learning rate for trainable variables
var_lr_mult = {var: (args.deeplab_lr_mult if var.name.startswith('res5')
                or var.name.startswith('bn5')
                else 1.0)
                for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

# bounding box regression L1 loss
rpn_bbox_pred = net.get_output('rpn_bbox_pred')
rpn_bbox_targets = tf.transpose(net.get_output('rpn-data')[1],[0,2,3,1])
rpn_bbox_inside_weights = tf.transpose(net.get_output('rpn-data')[2],[0,2,3,1])
rpn_bbox_outside_weights = tf.transpose(net.get_output('rpn-data')[3],[0,2,3,1])

rpn_smooth_l1 = rcnn.modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets,
    rpn_bbox_inside_weights, rpn_bbox_outside_weights)
rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

# regularization L2 loss
reg_loss = loss.l2_regularization_loss(reg_var_list, weight_decay)

total_loss = rpn_loss_box + reg_loss

################################################################################
# Solver
################################################################################

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.polynomial_decay(args.start_lr, global_step, args.max_iter,
    end_learning_rate = args.end_lr, power = 0.9)
solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
# Compute gradients
grads_and_vars = solver.compute_gradients(total_loss, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)

################################################################################
# Initialize parameters and load data
################################################################################

# Load training data
reader = data_reader.DataReader(data_folder, data_prefix)

# Run Initialization operations
snapshot_saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Load pretrained ResNet and randomly initialize the last layer
restored_var = [var for var in tf.global_variables() if 'fc1_voc12' not in var.name]
snapshot_loader = tf.train.Saver(var_list=restored_var)
print('Loading deeplab101 weights...')
snapshot_loader.restore(sess, pretrained_params)
print('Done.')

################################################################################
# Optimization loop
################################################################################

rpn_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

for n_iter in range(args.max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val = batch['text_seq_batch']
    imcrop_val = batch['imcrop_batch'].astype(np.float32)
    imcrop_val = imcrop_val[:,:,:,::-1] - segmodel.IMG_MEAN
    spatial_batch_val = batch['spatial_batch']
    label_val = batch['label_batch'].astype(np.float32)
    imsize_val = batch['imsize_batch'].astype(np.float32)
    gt_box_val = batch['gt_box_batch'].astype(np.float32)

    feed_dict = {
        text_seq_batch: text_seq_val,
        imcrop_batch  : imcrop_val,
        spatial_batch : spatial_batch_val,
        label_batch   : label_val,
        imsize_batch  : imsize_val,
        gt_box_batch  : gt_box_val,
    }

    # Forward and Backward pass
    scores_val, rpn_loss_val, _, lr_val = \
    sess.run([scores, rpn_loss_box, train_step, learning_rate], feed_dict=feed_dict)

    rpn_loss_avg = decay * rpn_loss_avg + (1 - decay) * rpn_loss_val
    print('\titer = %d, rpn_reg_loss (cur) = %f, rpn_reg_loss (avg) = %f, lr = %f'
        % (n_iter, rpn_loss_val, rpn_loss_avg, lr_val))

    # TODO Accuracy

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) == args.max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))