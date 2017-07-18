import tensorflow as tf

def modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
    ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
    SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
    |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
    tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul
