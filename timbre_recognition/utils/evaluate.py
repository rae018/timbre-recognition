from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import random
from timbre_recognition.modeling.inception_resnet_v2 import *
from timbre_recognition.utils.io import *
from timbre_recognition.configs.config import *
from timbre_recognition.ops.triplet_loss import *


def is_congruent(x, y, M, threshold):
  """Computes the mahalanobis distance between x and y using distance
  metric M, and returns whether that disance is strictly within the 
  passed threshold.
  
  Args:
    - x: A `float32` `Tensor` of an input embedding.
    - y: A `float32` `Tensor` of an input embedding.
    - M: A `float32` distance metric used in the mahalanobis distance.
    - threshold: A `float32` distance threshold for congruence.
    
  Returns:
    - congruent: A `bool` specifying whether y is congruent to x
    
    """
  x = tf.expand_dims(x, 0)
  y = tf.expand_dims(y, 0)

  distance = tf.matmul(tf.matmul(x-y, M), tf.transpose(x-y))

  return distance < threshold
  
def restore_kernel_module_from_checkpoint(network, dataset, checkpoint):
  shape = []
  if (dataset == 'esc-50'):
    shape = [1, 1, 220500, 1]
  
  kernel_module = init_inception_resnet_kernels(shape, cfg.MODEL.EMBED_DIM)
  ckpt = tf.train.Checkpoint(kernel_module=kernel_module)
  ckpt.restore(checkpoint)
  
  return kernel_module

def choose_negatives(keep_mask, num_keep):
  """Randomly selects negative pairs to use for evaluation
  
  Args:
    - keep_mask: An `int32` `Tensor` of boolean values (0,1) with all positive pairs
      1 and all negative pairs 0.
    - num_keep: An `int` specifying the number of negative pairs to keep for evaluation.
    
  Rreturns:
    - keep_mask: An `int` `Tensor`, the same input tensor except with exactly num_keep
      negative pairs turned on (set to 1). These pairs are selected randomly.
  """
  shape = keep_mask.shape
  
  # Form list of indices of all negative pairs
  negatives = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      if not keep_mask[i, j]:
        negatives += [(i,j)]
        
  keep_mask = keep_mask.numpy()
  # Randomly select negative pairs and turn them on
  keep_negatives = random.sample(negatives, num_keep)
  for i, j in keep_negatives:
    keep_mask[i, j] = 1
  
  return tf.convert_to_tensor(keep_mask)
  
def compute_accuracy(distances, truth_mask, keep_mask, threshold):
  """Returns the percent accuracy of distances under threshold, used truth_mask
  as GT-reference.
  
  Args:
    - distances: A `float32` `Tensor` of shape [batch_size, batch_size] containing 
      every computed pairwise distance. Index [i, j] contains the distance between 
      sample i and j.
    - trugh_mask: A `bool` `Tensor` of shape [batch_size, batch_size] specifying whether
      truth for every pair. Index [i, j] is True iff class(i) == class(j).
    - keep_mask: A `int32` `Tensor` specifying which pairs to use when computing accuracy.
      Because most pairs are negative, to get an even split between positive and negative
      pairs for testing we use a subset of the negative pairs.
    - threshold: A `float32` threshold of distance. A distance will be considered a 
      positive pair iff it is less than threshold.
  
  Returns:
    - accuracy: A `float32` of the percent accuracy of distances under threshold compared
      to truth_mask within keep_mask
  """
  congruent_mask = distances < threshold
  correct = tf.math.logical_not(tf.math.logical_xor(congruent_mask, truth_mask))
  correct = tf.dtypes.cast(correct, tf.dtypes.int32)
  # Remove the diagonal since it is trivial (any sample will have distance 0 from itself)
  # The lower triangle may not be symmetric since matrix multiplication is not commutative
  correct -= tf.linalg.band_part(correct, 0, 0)
  correct *= keep_mask
  accuracy = tf.reduce_sum(correct) / tf.reduce_sum(keep_mask)
 
  return accuracy
  
def evaluate_model(network, dataset, path, kernel_module, threshold):
  """Evaluates the accuracy of a model with test data.
  
  Args:
    - network: A `string` specifying what network is being evaluated
    - dataset: A `string` specifying what dataset to use for testing
    - path: A `string` specifying the path to the dataset
    - kernel_module: a `tf.Module` contained the trained parameters of
      the network.
    - threshold: A `float32` distance threshold for congruence, or an 
      iterable list of such thresholds (`np.ndarray` or `list`).
      
  Returns:
    - accuracy: A `float32` of the accuracy of the model, or a `list` of
      accuracies if a list of thresholds was passed in.
  """
  data, labels = load_esc50_test_set(path)
  labels = tf.strings.to_number(labels[:, 3], tf.dtypes.int32)
  
  dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(cfg.TRAIN.BATCH_SIZE)
  dataset_iter = iter(dataset)
  batch_data, batch_labels = next(dataset_iter, (None, None))
  
  # Init embeddings
  embeddings, endpoints = inception_resnet_v2(batch_data, kernel_module, embed_dim=cfg.MODEL.EMBED_DIM)
  batch_data, batch_labels = next(dataset_iter, (None, None))
  
  while batch_data is not None:
    batch_embeddings, endpoints = inception_resnet_v2(batch_data, kernel_module, embed_dim=cfg.MODEL.EMBED_DIM)
    embeddings = tf.concat([embeddings, batch_embeddings], axis=0)
    batch_data, batch_labels = next(dataset_iter, (None, None))
    
  distances = pairwise_mahalanobis_distances(tf.convert_to_tensor(embeddings), kernel_module, squared=True)
  
  # This matrix is symmetric
  truth_mask = tf.convert_to_tensor([tf.math.equal(labels, x) for x in labels])
  
  keep_mask = tf.dtypes.cast(truth_mask, tf.dtypes.int32)
  
  # Number of positive pairs without the diagonal
  num_positive = tf.reduce_sum(keep_mask) - len(labels)
  
  # Keep the same number of negatives as positive for an even representation
  keep_mask = choose_negatives(keep_mask, num_positive.numpy())
  
  # Don't keep disgonals since they're trivial
  keep_mask -= tf.linalg.band_part(keep_mask, 0, 0)
  
  if (type(threshold) == int):
    return compute_accuracy(distances, truth_mask, keep_mask, threshold)
  else:
    accuracies = []
    for t in threshold:
      accuracy = compute_accuracy(distances, truth_mask, keep_mask, t)
      accuracies += [accuracy]
    
    return accuracies


