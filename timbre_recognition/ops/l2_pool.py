import tensorflow as tf
from timbre_recognition.ops.arg_scope import add_arg_scope


@add_arg_scope
def l2_pool(inputs, ksize, strides, padding, data_format='NHWC', name=None):
  """Peforms L2 pooling on the input. Does not support pooling in the batch 
  dimension.
  """
  channels_in = inputs.get_shape()[3]
  channels_out = channels_in // ksize[3]
  kernel_size = ksize[1:3] + [channels_in] + [channels_out]
  kernel = tf.constant(1., dtype=inputs.dtype.name, shape=kernel_size)
  return tf.sqrt(tf.nn.conv2d(tf.square(inputs), kernel, strides=strides, 
                              padding=padding, data_format=data_format, 
                              name=name))