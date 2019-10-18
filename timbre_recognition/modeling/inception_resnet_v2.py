"""Implementation of the Inception-v4 network architecture for 1-D data.

As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from timbre_recognition.ops.arg_scope import *
from timbre_recognition.ops.l2_pool import l2_pool

# Decorate ops with add_arg_scope
tf.nn.conv2d = add_arg_scope(tf.nn.conv2d)
tf.nn.avg_pool = add_arg_scope(tf.nn.avg_pool)
tf.nn.max_pool = add_arg_scope(tf.nn.max_pool)
tf.nn.batch_normalization = add_arg_scope(tf.nn.batch_normalization)
tf.nn.dropout = add_arg_scope(tf.nn.dropout)

# Number of kernels for the Reduction-A module
k, l, m, n = 256, 256, 384, 384

"""
Weights for conv_2d have shape = [filter_height, filter_width, in_channels, 
out_channels]and are initialized as follows:

  tf.random.normal(shape) / tf.math.sqrt(filter_height * filter_width * in_channels)
  
where filter_height, filter_width, and in_channels are set as floats for 
tf.math.sqrt
"""


def init_inception_resnet_kernels(shape, embed_dim):
  inputs = tf.zeros(shape)
  kernel_module = tf.Module()
  embeddings = inception_resnet_v2(inputs, kernel_module, embed_dim=embed_dim)
  batch_hard_triplet_loss(labels=[1], embeddings=embeddings, margin=0, squared=True, distance='mahalanobis', 
                          kernel_module=kernel_module)
  return kernel_module


def init_inception_resnet_stem_kernels(scope=''):
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    # 1 x n x 1
    # Consider making this kernel length much bigger (100-200)
    kernel_module.Conv2d_1a_20 = tf.Variable(initial_value=tf.random.normal(shape=[1, 20, 1, 32]) / tf.math.sqrt(20.),
                                              name='Conv2d_1a_20')
    # 1 x n x 32
    kernel_module.Conv2d_2a_20 = tf.Variable(initial_value=tf.random.normal(shape=[1, 20, 32, 32]) / tf.math.sqrt(20. * 32.),
                                              name='Conv2d_2a_20')
    # 1 x n x 32
    kernel_module.Conv2d_2b_20 = tf.Variable(initial_value=tf.random.normal(shape=[1, 20, 32, 64]) / tf.math.sqrt(20. * 32.),
                                              name='Conv2d_2b_20')
    # 1 x n x 64
    with tf.name_scope('Mixed_3a'):
      kernel_module.Mixed_3a = tf.Module()
      with tf.name_scope('Branch_1'):
        kernel_module.Mixed_3a.Branch_1 = tf.Module()
        kernel_module.Mixed_3a.Branch_1.Conv2d_0a_20 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 20, 64, 96]) / tf.math.sqrt(20. * 64.), name='Conv2d_0a_20')
        
    # 1 x n x 160
    with tf.name_scope('Mixed_4a'):
      kernel_module.Mixed_4a = tf.Module()
      with tf.name_scope('Branch_0'):
        kernel_module.Mixed_4a.Branch_0 = tf.Module()
        kernel_module.Mixed_4a.Branch_0.Conv2d_0a_1 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 1, 160, 64]) / tf.math.sqrt(160.), name='Conv2d_0a_1')
        kernel_module.Mixed_4a.Branch_0.Conv2d_1a_20 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 20, 64, 96]) / tf.math.sqrt(20. * 64.), name='Conv2d_1a_20')
      with tf.name_scope('Branch_1'):
        kernel_module.Mixed_4a.Branch_1 = tf.Module()
        kernel_module.Mixed_4a.Branch_1.Conv2d_0a_1 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 1, 160, 64]) / tf.math.sqrt(160.), name='Conv2d_0a_1')
        kernel_module.Mixed_4a.Branch_1.Conv2d_0b_20 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 50, 64, 64]) / tf.math.sqrt(20. * 64.), name='Conv2d_0b_20')
        kernel_module.Mixed_4a.Branch_1.Conv2d_1a_20 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 20, 64, 96]) / tf.math.sqrt(20. * 64.), name='Conv2d_1a_20')
        
    # 1 x n x 192
    with tf.name_scope('Mixed_5a'):
      kernel_module.Mixed_5a = tf.Module()
      with tf.name_scope('Branch_0'):
        kernel_module.Mixed_5a.Branch_0 = tf.Module()
        kernel_module.Mixed_5a.Branch_0.Conv2d_1a_20 = tf.Variable(
          initial_value=tf.random.normal(shape=[1, 20, 192, 192]) / tf.math.sqrt(20. * 192.), name='Conv2d_1a_20')
    return kernel_module

  
def init_inception_resnet_a_kernels(channels_in, scope):
  float_channels_in = tf.dtypes.cast(channels_in, tf.float32)                                        
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    with tf.name_scope('Branch_0'):
      kernel_module.Branch_0 = tf.Module()
      kernel_module.Branch_0.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 32]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
    with tf.name_scope('Branch_1'):
      kernel_module.Branch_1 = tf.Module()
      kernel_module.Branch_1.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 32]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_1.Conv2d_0b_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 32, 32]) / tf.math.sqrt(20. * 32.), name='Conv2d_0b_20')
    with tf.name_scope('Branch_2'):
      kernel_module.Branch_2 = tf.Module()
      kernel_module.Branch_2.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 32]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_2.Conv2d_0b_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 32, 48]) / tf.math.sqrt(20. * 32.), name='Conv2d_0b_20')
      kernel_module.Branch_2.Conv2d_0c_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 48, 64]) / tf.math.sqrt(20. * 48.), name='Conv2d_0c_20')
    with tf.name_scope('Up'):
      kernel_module.Up = tf.Module()
      kernel_module.Up.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, 128, 384]) / tf.math.sqrt(128.), name='Conv2d_0a_1')
    return kernel_module
    
    
def init_reduction_a_kernels(channels_in, scope):
  float_channels_in = tf.dtypes.cast(channels_in, tf.float32)
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    with tf.name_scope('Branch_0') as scope:
      kernel_module.Branch_0 = tf.Module()
      kernel_module.Branch_0.Conv2d_1a_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, channels_in, n]) / tf.math.sqrt(20. * float_channels_in), name='Conv2d_1a_20')
    with tf.name_scope('Branch_1') as scope:
      kernel_module.Branch_1 = tf.Module()
      kernel_module.Branch_1.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, k]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_1.Conv2d_0b_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, k, l]) / tf.math.sqrt(20. * k), name='Conv2d_0b_20')
      kernel_module.Branch_1.Conv2d_1a_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, l, m]) / tf.math.sqrt(20. * l), name='Conv2d_1a_20')
    return kernel_module
   
    
def init_inception_resnet_b_kernels(channels_in, scope):
  float_channels_in = tf.dtypes.cast(channels_in, tf.float32)
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    with tf.name_scope('Branch_0') as scope:
      kernel_module.Branch_0 = tf.Module()
      kernel_module.Branch_0.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 192]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
    with tf.name_scope('Branch_1'):
      kernel_module.Branch_1 = tf.Module()
      kernel_module.Branch_1.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 128]) / tf.math.sqrt(float_channels_in),name='Conv2d_0a_1')
      kernel_module.Branch_1.Conv2d_0b_50 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 50, 128, 192]) / tf.math.sqrt(50. * 192.), name='Conv2d_0b_50')
    with tf.name_scope('Up'):
      kernel_module.Up = tf.Module()
      kernel_module.Up.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, 384, 1152]) / tf.math.sqrt(384.), name='Conv2d_0a_1')
    return kernel_module
  
  
def init_reduction_b_kernels(channels_in, scope):
  float_channels_in = tf.dtypes.cast(channels_in, tf.float32)
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    with tf.name_scope('Branch_1') as scope:
      kernel_module.Branch_1 = tf.Module()
      kernel_module.Branch_1.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 256]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_1.Conv2d_1a_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 256, 384]) / tf.math.sqrt(20. * 256.), name='Conv2d_1a_20')
    with tf.name_scope('Branch_2') as scope:
      kernel_module.Branch_2 = tf.Module()
      kernel_module.Branch_2.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 256]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_2.Conv2d_1a_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 256, 288]) / tf.math.sqrt(20. * 256.), name='Conv2d_1a_20')
    with tf.name_scope('Branch_3') as scope:
      kernel_module.Branch_3 = tf.Module()
      kernel_module.Branch_3.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 256]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_3.Conv2d_0b_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 256, 288]) / tf.math.sqrt(20. * 256.), name='Conv2d_0b_20')
      kernel_module.Branch_3.Conv2d_1a_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 288, 320]) / tf.math.sqrt(20. * 288.), name='Conv2d_1a_20')
    return kernel_module

  
def init_inception_resnet_c_kernels(channels_in, scope):
  float_channels_in = tf.dtypes.cast(channels_in, tf.float32)
  with tf.name_scope(scope):
    kernel_module = tf.Module()
    with tf.name_scope('Branch_0') as scope:
      kernel_module.Branch_0 = tf.Module()
      kernel_module.Branch_0.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 192]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
    with tf.name_scope('Branch_1') as scope:
      kernel_module.Branch_1 = tf.Module()
      kernel_module.Branch_1.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, channels_in, 192]) / tf.math.sqrt(float_channels_in), name='Conv2d_0a_1')
      kernel_module.Branch_1.Conv2d_0b_20 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 20, 192, 256]) / tf.math.sqrt(20. * 384.), name='Conv2d_0b_20')
    with tf.name_scope('Up'):
      kernel_module.Up = tf.Module()
      kernel_module.Up.Conv2d_0a_1 = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, 448, 2144]) / tf.math.sqrt(448.), name='Conv2d_0a_1')
    return kernel_module
  
  
def init_dense_weights(shape):
  kernel_module = tf.Module()
  kernel_module.Weights = tf.Variable(
    initial_value=tf.random.normal(shape=shape) / tf.math.sqrt(shape[0] * shape[1] * 1.), name='Dense')
  return kernel_module

def norm_and_activate(x):
  mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
  x = tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None,
                            variance_epsilon=1.e-5)
  x = tf.keras.activations.relu(x)
  return x
  
  
def block_inception_resnet_a(inputs, kernel_module, scale, scope):
  with tf.name_scope(scope):
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool], 
                   strides=[1, 1, 1, 1], padding='SAME'):
      with tf.name_scope('Branch_0') as name_scope:
        branch_0 = tf.nn.conv2d(inputs, kernel_module.Branch_0.Conv2d_0a_1)
      with tf.name_scope('Branch_1'):
        branch_1 = tf.nn.conv2d(inputs, kernel_module.Branch_1.Conv2d_0a_1)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_0b_20)
      with tf.name_scope('Branch_2'):
        branch_2 = tf.nn.conv2d(inputs, kernel_module.Branch_2.Conv2d_0a_1)
        branch_2 = tf.nn.conv2d(branch_2, kernel_module.Branch_2.Conv2d_0b_20)
        branch_2 = tf.nn.conv2d(branch_2, kernel_module.Branch_2.Conv2d_0c_20)
      combined = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      combined = tf.nn.conv2d(combined, kernel_module.Up.Conv2d_0a_1)
      combined *= scale
      combined += inputs
      return norm_and_activate(combined)

    
def block_reduction_a(inputs, kernel_module, scope):
  channels_in = inputs.get_shape()[3]
  with tf.name_scope(scope):
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool], 
                   strides=[1, 1, 1, 1], padding='SAME'):
      with tf.name_scope('Branch_0'):
        branch_0 = tf.nn.conv2d(inputs, kernel_module.Branch_0.Conv2d_1a_20,
                                strides=[1, 1, 4, 1], padding='VALID')
      with tf.name_scope('Branch_1'):
        branch_1 = tf.nn.conv2d(inputs, kernel_module.Branch_1.Conv2d_0a_1)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_0b_20)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_1a_20,
                                strides=[1, 1, 4, 1], padding='VALID')
      with tf.name_scope('Branch_2'):
        branch_2 = tf.nn.max_pool(inputs, [1, 1, 20, 1], 
                                  strides=[1, 1, 4, 1], padding='VALID')
      combined = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      return norm_and_activate(combined)

    
def block_inception_resnet_b(inputs, kernel_module, scale, scope):
  with tf.name_scope(scope):
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool],
                   strides=[1, 1, 1, 1], padding='SAME'):
      with tf.name_scope('Branch_0') as scope:
        branch_0 = tf.nn.conv2d(inputs, kernel_module.Branch_0.Conv2d_0a_1)
      with tf.name_scope('Branch_1'):
        branch_1 = tf.nn.conv2d(inputs, kernel_module.Branch_1.Conv2d_0a_1)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_0b_50)
      combined = tf.concat(axis=3, values=[branch_0, branch_1])
      combined = tf.nn.conv2d(combined, kernel_module.Up.Conv2d_0a_1)
      combined *= scale
      combined += inputs
      return norm_and_activate(combined)

    
def block_reduction_b(inputs, kernel_module, scope):
  channels_in = inputs.get_shape()[3]
  with tf.name_scope(scope):
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool], 
                   strides=[1, 1, 1, 1], padding='SAME'):
      with tf.name_scope('Branch_0'):
        branch_0 = tf.nn.max_pool(inputs, [1, 1, 20, 1],
                                  strides=[1, 1, 4, 1], padding='VALID')
      with tf.name_scope('Branch_1'):
        branch_1 = tf.nn.conv2d(inputs, kernel_module.Branch_1.Conv2d_0a_1)
        branch_1 = norm_and_activate(branch_1)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_1a_20,
                                strides=[1, 1, 4, 1], padding='VALID')
      with tf.name_scope('Branch_2'):
        branch_2 = tf.nn.conv2d(inputs, kernel_module.Branch_2.Conv2d_0a_1)
        branch_2 = norm_and_activate(branch_2)
        branch_2 = tf.nn.conv2d(branch_2, kernel_module.Branch_2.Conv2d_1a_20,
                                strides=[1, 1, 4, 1], padding='VALID')
      with tf.name_scope('Branch_3'):
        branch_3 = tf.nn.conv2d(inputs, kernel_module.Branch_3.Conv2d_0a_1)
        branch_3 = tf.nn.conv2d(branch_3, kernel_module.Branch_3.Conv2d_0b_20)
        branch_3 = tf.nn.conv2d(branch_3, kernel_module.Branch_3.Conv2d_1a_20,
                                strides=[1, 1, 4, 1], padding='VALID')
      combined = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      return norm_and_activate(combined)

    
def block_inception_resnet_c(inputs, kernel_module, scale, scope):
  """ The structure of this block breaks from the paper. The parallel
  convolutions within branches are flattened into one single convolution because
  parallel 1xn and nx1 convolutions are not possible with 1-D data. 
  """
  channels_in = inputs.get_shape()[3]
  with tf.name_scope(scope):
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool], 
                   strides=[1, 1, 1, 1], padding='SAME'):
      with tf.name_scope('Branch_0'):
        branch_0 = tf.nn.conv2d(inputs, kernel_module.Branch_0.Conv2d_0a_1)
      with tf.name_scope('Branch_1'):
        branch_1 = tf.nn.conv2d(inputs, kernel_module.Branch_1.Conv2d_0a_1)
        branch_1 = tf.nn.conv2d(branch_1, kernel_module.Branch_1.Conv2d_0b_20)
      combined = tf.concat(axis=3, values=[branch_0, branch_1])
      combined = tf.nn.conv2d(combined, kernel_module.Up.Conv2d_0a_1)
      combined *= scale
      combined += inputs
      return norm_and_activate(combined)

    
def inception_resnet_v2_base(inputs, kernel_module, final_endpoint='Inception_Resnet_C3_Block', scope=None):
  """Builds the Inception-Resnet-v2 network up to the final endpoint (inclusive).

  Args:
    - inputs: A 4-D `tf.float32` `Tensor` with size [batch_size, height, weight, 
      channels]. Height and channels should be 1 for the input.
    - kernel_module: a `KernelModule` object to save weights to.
    - final_endpoint: A `string` specifying the endpoint to construct the network
      up to. Must be one of the following: ['Conv2d_1a_20, 'Conv2d_2a_20', 
      'Conv2d_2b_20', 'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Inception_Resnet_A1_Block',
      'Inception_Resnet_A2_Block', 'Inception_Resnet_A3_Block', 'Inception_Resnet_A4_Block',
      'Inception_Resnet_A5_Block', 'Reduction_A'_Block, 'Inception_Resnet_B1'_Block, 
      'Inception_Resnet_B2_Block', 'Inception_Resnet_B3'_Block, 'Inception_Resnet_B4'_Block,
      'Inception_Resnet_B5'_Block, 'Inception_Resnet_B6'_Block, 'Inception_Resnet_B7_Block', 
      'Inception_Resnet_B8_Block', 'Inception_Resnet_B9_Block', 'Inception_Resnet_B10_Block',
      'Reduction_B_Block', 'Inception_Resnet_C1_Block', 'Inception_Resnet_C2_Block', 
      'Inception_Resnet_C3_Block', 'Inception_Resnet_C4_Block', 'Inception_Resnet_C5_Block'].
    - scope: Optional `string` or `VariableScope` specifying the name of a 
      `name_scope` 
  
  Returns:
    - logits: A `Tensor` of type `tf.float32` of logits output from the model.
    - endpoints: A `dict` containing the set of endpoints from the model

  Raises:
    - ValueError: If final_endpoint is not set to one of the predefined values.
  """
  
  # NOTE: batch size is left out of the tensor dimensions shown in the following comments
  
  endpoints = {}

  def add_and_check_final(name, net):
    endpoints[name] = net
    return name == final_endpoint
  
  with tf.name_scope(scope):
    if getattr(kernel_module, 'Stem', None) is None:
      setattr(kernel_module, 'Stem',
              init_inception_resnet_stem_kernels())
    stem_kernels = getattr(kernel_module, 'Stem')
    
    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool],
                   strides=[1, 1, 1, 1], padding='SAME'):
      # 1 x n x 1
      net = tf.nn.conv2d(inputs, stem_kernels.Conv2d_1a_20, strides=[1, 1, 4, 1], 
                         padding='VALID')
      net = norm_and_activate(net)
      if add_and_check_final('Conv2d_1a_20', net): return net, endpoints

      # 1 x n x 32
      net = tf.nn.conv2d(net, stem_kernels.Conv2d_2a_20, padding='VALID')
      net = norm_and_activate(net)
      if add_and_check_final('Conv2d_2a_20', net): return net, endpoints
      
      # 1 x n x 32
      net = tf.nn.conv2d(net, stem_kernels.Conv2d_2b_20)
      net = norm_and_activate(net)
      if add_and_check_final('Conv2d_2b_20', net): return net, endpoints

      # 1 x n x 64
      with tf.name_scope('Mixed_3a'):
        with tf.name_scope('Branch_0'):
          branch_0 = tf.nn.max_pool(net, [1, 1, 20, 1], strides=[1, 1, 4, 1], 
                                    padding='VALID')
          branch_0 = norm_and_activate(branch_0)
        with tf.name_scope('Branch_1'):
          branch_1 = tf.nn.conv2d(net, stem_kernels.Mixed_3a.Branch_1.Conv2d_0a_20, strides=[1, 1, 4, 1], 
                                  padding='VALID')
          branch_1 = norm_and_activate(branch_1)
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_3a', net): return net, endpoints

      # 1 x n x 160
      with tf.name_scope('Mixed_4a'):
        with tf.name_scope('Branch_0'):
          branch_0 = tf.nn.conv2d(net, stem_kernels.Mixed_4a.Branch_0.Conv2d_0a_1)
          branch_0 = norm_and_activate(branch_0)
          branch_0 = tf.nn.conv2d(branch_0, stem_kernels.Mixed_4a.Branch_0.Conv2d_1a_20, strides=[1, 1, 4, 1],
                                  padding='VALID')
          branch_0 = norm_and_activate(branch_0)
        with tf.name_scope('Branch_1'):
          branch_1 = tf.nn.conv2d(net, stem_kernels.Mixed_4a.Branch_1.Conv2d_0a_1)
          branch_1 = norm_and_activate(branch_1)
          branch_1 = tf.nn.conv2d(branch_1, stem_kernels.Mixed_4a.Branch_1.Conv2d_0b_20)
          branch_1 = norm_and_activate(branch_1)
          branch_1 = tf.nn.conv2d(branch_1, stem_kernels.Mixed_4a.Branch_1.Conv2d_1a_20, strides=[1, 1, 4, 1],
                                  padding='VALID')
          branch_1 = norm_and_activate(branch_1)
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_4a', net): return net, endpoints

      # 1 x n x 192
      with tf.name_scope('Mixed_5a'):
        with tf.name_scope('Branch_0'):
          branch_0 = tf.nn.conv2d(net, stem_kernels.Mixed_5a.Branch_0.Conv2d_1a_20,
                                  strides=[1, 1, 4, 1], padding='VALID')
          branch_0 = norm_and_activate(branch_0)
        with tf.name_scope('Branch_1'):
          branch_1 = tf.nn.max_pool(net, [1, 1, 20, 1], strides=[1, 1, 4, 1],
                                    padding='VALID')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_5a', net): return net, endpoints

      # 1 x n x 384
      # 5 x Inception-A blocks
      for i in range(5):
        block_scope = 'Inception_Resnet_A' + str(i+1) + '_Block'
        if getattr(kernel_module, block_scope, None) is None:
          setattr(kernel_module, block_scope,
                  init_inception_resnet_a_kernels(net.get_shape()[3], block_scope))
        net = block_inception_resnet_a(net, getattr(kernel_module, block_scope), 0.17, block_scope)
        if add_and_check_final(block_scope, net): return net, endpoints

      # 1 x n x 384
      # Reduction-A block
      block_scope = 'Reduction_A_Block'
      if getattr(kernel_module, block_scope, None) is None:
        setattr(kernel_module, block_scope,
                init_reduction_a_kernels(net.get_shape()[3], block_scope))
      net = block_reduction_a(net, getattr(kernel_module, block_scope), block_scope)
      if add_and_check_final(block_scope, net): return net, endpoints

      # 1 x n x 1152
      # 10 x Inception-B blocks
      for i in range(10):
        block_scope = 'Inception_Resnet_B' + str(i+1) + '_Block'
        if getattr(kernel_module, block_scope, None) is None:
          setattr(kernel_module, block_scope,
                  init_inception_resnet_b_kernels(net.get_shape()[3], block_scope))
        net = block_inception_resnet_b(net, getattr(kernel_module, block_scope), 0.1, block_scope)
        if add_and_check_final(block_scope, net): return net, endpoints

      # 1 x n x 1152
      # Reduction-B block
      block_scope = 'Reduction_B_Block'
      if getattr(kernel_module, block_scope, None) is None:
        setattr(kernel_module, block_scope,
                init_reduction_b_kernels(net.get_shape()[3], block_scope))
      net = block_reduction_b(net, getattr(kernel_module, block_scope), block_scope)
      if add_and_check_final(block_scope, net): return net, endpoints

      # 1 x n x 2144
      # 5 x Inception-C blocks
      for i in range(5):
        block_scope = 'Inception_Resnet_C' + str(i+1) + '_Block'
        if getattr(kernel_module, block_scope, None) is None:
          setattr(kernel_module, block_scope,
                  init_inception_resnet_c_kernels(net.get_shape()[3], block_scope))
        net = block_inception_resnet_c(net, getattr(kernel_module, block_scope), 0.2, block_scope)
        if add_and_check_final(block_scope, net): return net, endpoints
  raise ValueError('Unknown final endpoint %s' % final_endpoint)

  
def inception_resnet_v2(inputs, kernel_module, num_classes=50, embed_dim=None,
                 is_training=True,
                 dropout_prob=0.2, 
                 scope='InceptionResnetV2',
                 create_aux_logits=True):
  """Builds the Inception-ResNet-v2 model

  Args:
    - inputs: A 4-D `tf.float32` `Tensor` with shape [batch_size, height, width, 
      channels]. Height and channels should be 1 for the input. 
    - kernel_module: A `tf.Module` to store kernels. This will be populated with
      every weight in the newtork
    - num_classes: An `int` specifying the number of predicted classes. If 0 or
      None, the logits layer is omitted and the input features to the logits
      layer (before dropout) are returned instead.
    - embed_dim: An `int` specifying the dimensionality of the embedding space
    - is_training: A `bool` specifying whether is training or not
    - dropoutp_prob: A `float` specifying the probability a neron is dropped
      before the final layer.
    - reuse: Must be True, None, or tf.AUTO_REUSE. Specifies whether the network
      and its variables should be reused. To be able to reuse, 'scope' must be 
      given.
    - scope: Optional `string` or `VariableScope` specifying the name of a 
      `name_scope`.
    create_aux_logits: A `bool` specifying whether to include the auxiliary 
      logits.
    
  Returns:
    - embedding: A 2-D `tf.float32` `Tensor` embedding with shape [batch_size,
      embed_dim]
    - endpoints: A `dict` containing the set of endpoints from the model
  """
  
  # NOTE: batch size is left out of the tensor dimensions shown in the following comments.
  # It is in the 0 index, e.g. 1 x n x 1536 is actually batch_size 1 x n x 1536
  
  endpoints = {}
  
  with tf.name_scope(scope) as scope:
    net, endpoints = inception_resnet_v2_base(inputs, kernel_module, scope=scope)

    with arg_scope([tf.nn.conv2d, tf.nn.avg_pool, tf.nn.max_pool, l2_pool],
               strides=[1, 1, 1, 1], padding='SAME'):
      # Auxiliary Head logits
      pass
      # Pool down to 1x1
      # Dropout
      # FC to 128
      # L2 normalize
      with tf.name_scope('Logits'):
        # 1 x n x 1536
        ksize = net.get_shape()[1:3]
        if ksize.is_fully_defined():
          net = l2_pool(net, [1] + ksize.as_list() + [1], padding='VALID')
        else:
          net = tf.reduce_mean(net, 2, keepdims=True)

        endpoints['global_pool'] = net

        # 1 x 1 x 1536
        net = tf.nn.dropout(net, dropout_prob)
        net = tf.squeeze(net, [2])
        net = tf.squeeze(net, [1])

        # 1536
        if getattr(kernel_module, 'Dense', None) is None:
          setattr(kernel_module, 'Dense',
                  init_dense_weights([net.shape[-1], embed_dim]))
        logits = tf.linalg.matmul(net, getattr(kernel_module, 'Dense').Weights)
        endpoints['Logits'] = logits
        embedding = tf.nn.l2_normalize(logits)
        endpoints['Embedding'] = embedding
  return embedding, endpoints

        




















        
                                 
  
