from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

def _pairwise_l2_distances(embeddings, squared=False):
  """Compute the 2-D matrix of l2 distances between all the embeddings.
  
  Args:
    - embeddings: A 2-D `Tensor` of shape [batch_size, embed_dim]
    - squared: A `bool`. If True, output is the pairwise squared euclidean 
      distance matrix. If False, output is the pairwise euclidean distance 
      matrix.
      
  Returns:
    - distances: A `Tensor` of shape [batch_size, batch_size] containing
      pairwise l2 distances.
  """
  # Get the dot product between all embeddings
  # shape [batch_size, batch_size]
  dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

  # Get squared l2 norm for each embedding. We can just take the diagonal of `dot_product`.
  # This also provides more numerical stability (the diagonal of the result will be exactly 0).
  # shape (batch_size,)
  square_norm = tf.linalg.tensor_diag_part(dot_product)

  # Compute the pairwise distance matrix:
  # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
  # shape [batch_size, batch_size]
  distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

  # Because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)

  if not squared:
    # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
    # we need to add a small epsilon where distances == 0.0
    mask = tf.dtypes.cast(tf.equal(distances, 0.0), tf.float32)
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)

  return distances

def init_distance_metric(embed_dim):
  kernel_module = tf.Module()
  kernel_module.Distance_Metric = tf.Variable(
    initial_value=tf.random.uniform(shape=[embed_dim, embed_dim]), name='DistanceMetric')
  return kernel_module

def pairwise_mahalanobis_distances_classes(embeddings, class_centers, kernel_module, squared=False):
  """Compute the 2-D matrix of mahalanobis distances between embeddings and class centers
  
  Args:
    - embeddings: A 2-D `Tensor` of shape [batch_size, embed_dim]
    - class_centers: A 2-D `Tensor` of shape [num_classes, embed_dim]
    - kernel_module: A `tf.Module` for distance metric
    - squared: A `bool`. If True, output is the pairwise equared euclidean
      distance matrix. If False, output is the pairwise euclidean distance
      matrix.
    
    Returns:
      - distances: A `Tensor` of shape [batch_size, num_classes] containing
        pairwise mahalanobis distances.
    """
  embed_dim = embeddings.get_shape()[1]
  M = getattr(kernel_module, 'MahalanobisDistance').Distance_Metric # At this point M should be set
  
  # (a-b)'M(a-b) = a'Ma - a'Mb - b'Ma + b'Mb
  # a's are the embeddings, b's are the class centers
  aMb = tf.matmul(tf.matmul(embeddings, M), tf.transpose(class_centers))
  bMa = tf.transpose(tf.matmul(tf.matmul(class_centers, M), tf.transpose(embeddings)))
  aMa = tf.linalg.tensor_diag_part(tf.matmul(tf.matmul(embeddings, M), tf.transpose(embeddings)))
  bMb = tf.linalg.tensor_diag_part(tf.matmul(tf.matmul(class_centers, M), tf.transpose(class_centers)))
  
  distances = tf.expand_dims(aMa, 1) - aMb - bMa + tf.expand_dims(bMb, 0)
  
  # Because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)
  
  if not squared:
    # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
    # we need to add a small epsilon where distances == 0.0
    mask = tf.dtypes.cast(tf.equal(distances, 0.0), tf.float32)
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)

  return distances
  
  
def pairwise_mahalanobis_distances(embeddings, kernel_module, squared=False):
  """Compute the 2-D matrix of mahalanobis distances between all the embeddings.
  
  Args:
    - embeddings: A 2-D `Tensor` of shape [batch_size, embed_dim]
    - kernel_module: A `tf.Module` for distance metric
    - squared: A `bool`. If True, output is the pairwise equared euclidean
      distance matrix. If False, output is the pairwise euclidean distance
      matrix.
    
    Returns:
      - distances: A `Tensor` of shape [batch_size, batch_size] containing
        pairwise mahalanobis distances.
    """
  embed_dim = embeddings.get_shape()[1] # Possibly replace with cfg.embed_dim
  
  if getattr(kernel_module, 'MahalanobisDistance', None) is None:
    setattr(kernel_module, 'MahalanobisDistance', init_distance_metric(embed_dim))
  M = getattr(kernel_module, 'MahalanobisDistance').Distance_Metric
  
  # Get the dot product between all embeddings
  # Computes a'Mb for all embedding pairs (a, b)
  # Diagonal is a'Ma
  # shape [batch_size, batch_size]
  dot_product = tf.matmul(tf.matmul(embeddings, M), tf.transpose(embeddings))
  
  square_norm = tf.linalg.tensor_diag_part(dot_product)
  
  # Compute the pairwise distance matrix:
  # a'Ma - a'Mb - b'Ma + b'Mb
  # shape [batch_size, batch_size]
  distances = tf.expand_dims(square_norm, 1) - dot_product - tf.transpose(dot_product) + tf.expand_dims(square_norm, 0)
  
  # Because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)
  
  if not squared:
    # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
    # we need to add a small epsilon where distances == 0.0
    mask = tf.dtypes.cast(tf.equal(distances, 0.0), tf.float32)
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)

  return distances


def get_anchor_positive_triplet_mask(labels):
  """Return a 2-D mask where mask[a, p] is True iff a and p are distinct and have same label.

  Args:
    - labels: A tf.int32 `Tensor` with shape [batch_size]

  Returns:
    - mask: A tf.bool `Tensor` with shape [batch_size, batch_size]
  """
  # Check that i and j are distinct
  indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
  indices_not_equal = tf.logical_not(indices_equal)

  # Check if labels[i] == labels[j]
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

  # Combine the two masks
  mask = tf.logical_and(indices_not_equal, labels_equal)

  return mask


def get_anchor_negative_triplet_mask(labels):
  """Return a 2-D mask where mask[a, n] is True iff a and n have distinct labels.

  Args:
    - labels: tf.int32 `Tensor` with shape [batch_size]

  Returns:
    - mask: tf.bool `Tensor` with shape [batch_size, batch_size]
  """
  # Check if labels[i] != labels[k]
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

  mask = tf.logical_not(labels_equal)

  return mask


def get_triplet_mask(labels):
  """Return a 3-D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
  A triplet (i, j, k) is valid if:
    - i, j, k are distinct
    - labels[i] == labels[j] and labels[i] != labels[k]

  Args:
    - labels: tf.int32 `Tensor` with shape [batch_size]
  """
  # Check that i, j and k are distinct
  indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
  indices_not_equal = tf.logical_not(indices_equal)
  i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
  i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
  j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

  distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


  # Check if labels[i] == labels[j] and labels[i] != labels[k]
  label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
  i_equal_j = tf.expand_dims(label_equal, 2)
  i_equal_k = tf.expand_dims(label_equal, 1)

  valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

  # Combine the two masks
  mask = tf.logical_and(distinct_indices, valid_labels)

  return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
  """Build the triplet loss over a batch of embeddings.
  We generate all the valid triplets and average the loss over the positive ones.

  Args:
    - labels: labels of the batch, of size [batch_size]
    - embeddings: tensor of shape [batch_size, embed_dim]
    - margin: margin for triplet loss
    - squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
      If false, output is the pairwise euclidean distance matrix.

  Returns:
    - triplet_loss: scalar tensor containing the triplet loss
  """
  # Get the pairwise distance matrix
  pairwise_dist = pairwise_l2_distances(embeddings, squared=squared)

  # shape [batch_size, batch_size, 1]
  anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
  assert anchor_positive_dist.shape[2] == 1, '{}'.format(anchor_positive_dist.shape)
  # shape [batch_size, 1, batch_size]
  anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
  assert anchor_negative_dist.shape[1] == 1, '{}'.format(anchor_negative_dist.shape)

  # Compute a 3-D tensor of size [batch_size, batch_size, batch_size]
  # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
  # Uses broadcasting where the 1st argument has shape [batch_size, batch_size, 1]
  # and the 2nd [batch_size, 1, batch_size]
  triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

  # Put to zero the invalid triplets
  # (where label(a) != label(p) or label(n) == label(a) or a == p)
  mask = get_triplet_mask(labels)
  mask = tf.dtypes.cast(mask, tf.float32)
  triplet_loss = tf.multiply(mask, triplet_loss)

  # Remove negative losses (i.e. the easy triplets)
  triplet_loss = tf.maximum(triplet_loss, 0.0)

  # Count number of positive triplets (where triplet_loss > 0)
  valid_triplets = tf.dtypes.case(tf.greater(triplet_loss, 1e-16), tf.float32)
  num_positive_triplets = tf.reduce_sum(valid_triplets)
  num_valid_triplets = tf.reduce_sum(mask)
  fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

  # Get final mean triplet loss over the positive valid triplets
  triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

  return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, kernel_module=None):
  """Build the triplet loss over a batch of embeddings.
  For each anchor, we get the hardest positive and hardest negative to form a triplet.

  Args:
    - labels: labels of the batch, of size [batch_size]
    - embeddings: tensor of shape [batch_size, embed_dim]
    - margin: margin for triplet loss
    - squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
      If false, output is the pairwise euclidean distance matrix.
    - distance: A `string` specifying what distance function to use. Can be
      'eculidian' or 'mahalanobis'.
    - kernel_module: A `tf.Module` containing the variables of the network. Only needed
      if the arg distance is set to 'mahalanobis'.

  Returns:
    - triplet_loss: scalar tensor containing the triplet loss
  """
  # Get the pairwise distance matrix
  pairwise_dist = pairwise_mahalanobis_distances(embeddings, kernel_module, squared=squared)

  # For each anchor, get the hardest positive
  # First, we need to get a mask for every valid positive (they should have same label)
  mask_anchor_positive = get_anchor_positive_triplet_mask(labels)
  mask_anchor_positive = tf.dtypes.cast(mask_anchor_positive, tf.float32)

  # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
  anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

  # shape [batch_size, 1]
  hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
  tf.summary.scalar('hardest_positive_dist', tf.reduce_mean(hardest_positive_dist))

  # For each anchor, get the hardest negative
  # First, we need to get a mask for every valid negative (they should have different labels)
  mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
  mask_anchor_negative = tf.dtypes.cast(mask_anchor_negative, tf.float32)

  # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
  max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
  anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

  # shape [batch_size]
  hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
  tf.summary.scalar('hardest_negative_dist', tf.reduce_mean(hardest_negative_dist))

  # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
  triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

  # Get final mean triplet loss
  triplet_loss = tf.reduce_mean(triplet_loss)

  return triplet_loss


def batch_triplet_semihard_loss(labels, embeddings, margin, kernel_module=None):
  """Computes the triplet loss with semi-hard negative mining.
  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.
  Returns:
    triplet_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_mahalanobis_distances(embeddings, kernel_module, squared=True)
  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  # Compute the mask.
  pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
  mask = math_ops.logical_and(
    array_ops.tile(adjacency_not, [batch_size, 1]),
    math_ops.greater(
      pdist_matrix_tile, array_ops.reshape(
        array_ops.transpose(pdist_matrix), [-1, 1])))
  mask_final = array_ops.reshape(
    math_ops.greater(
      math_ops.reduce_sum(
        math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
    0.0), [batch_size, batch_size])
  mask_final = array_ops.transpose(mask_final)

  adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  mask = math_ops.cast(mask, dtype=dtypes.float32)

  # negatives_outside: smallest D_an where D_an > D_ap.
  negatives_outside = array_ops.reshape(
    masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = array_ops.transpose(negatives_outside)

  # negatives_inside: largest D_an.
  negatives_inside = array_ops.tile(
    masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
  semi_hard_negatives = array_ops.where(
    mask_final, negatives_outside, negatives_inside)

  loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = math_ops.cast(
    adjacency, dtype=dtypes.float32) - array_ops.diag(
      array_ops.ones([batch_size]))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = math_ops.reduce_sum(mask_positives)

  triplet_loss = math_ops.truediv(
    math_ops.reduce_sum(
      math_ops.maximum(
        math_ops.multiply(loss_mat, mask_positives), 0.0)),
    num_positives,
    name='triplet_semihard_loss')

  return triplet_loss
  
  
# ==============================================================================
# Straight from TensorFlow source:
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py
  
  
def pairwise_distance(feature, squared=False):
  """Computes the pairwise distance matrix with numerical stability.
  output[i, j] = || feature[i, :] - feature[j, :] ||_2
  Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
  Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
  """
  pairwise_distances_squared = math_ops.add(
    math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
    math_ops.reduce_sum(
      math_ops.square(array_ops.transpose(feature)),
      axis=[0],
      keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                              array_ops.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
    pairwise_distances = pairwise_distances_squared
  else:
    pairwise_distances = math_ops.sqrt(
      pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

  # Undo conditionally adding 1e-16.
  pairwise_distances = math_ops.multiply(
    pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

  num_data = array_ops.shape(feature)[0]
  # Explicitly set diagonals to zero.
  mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
    array_ops.ones([num_data]))
  pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
  return pairwise_distances


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.
  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
  masked_maximums = math_ops.reduce_max(
    math_ops.multiply(data - axis_minimums, mask), dim,
    keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
  masked_minimums = math_ops.reduce_min(
    math_ops.multiply(data - axis_maximums, mask), dim,
    keepdims=True) + axis_maximums
  return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
  """Computes the triplet loss with semi-hard negative mining.
  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.
  Returns:
    triplet_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_distance(embeddings, squared=True)
  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  # Compute the mask.
  pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
  mask = math_ops.logical_and(
    array_ops.tile(adjacency_not, [batch_size, 1]),
    math_ops.greater(
      pdist_matrix_tile, array_ops.reshape(
        array_ops.transpose(pdist_matrix), [-1, 1])))
  mask_final = array_ops.reshape(
    math_ops.greater(
      math_ops.reduce_sum(
        math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
    0.0), [batch_size, batch_size])
  mask_final = array_ops.transpose(mask_final)

  adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  mask = math_ops.cast(mask, dtype=dtypes.float32)

  # negatives_outside: smallest D_an where D_an > D_ap.
  negatives_outside = array_ops.reshape(
    masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = array_ops.transpose(negatives_outside)

  # negatives_inside: largest D_an.
  negatives_inside = array_ops.tile(
    masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
  semi_hard_negatives = array_ops.where(
    mask_final, negatives_outside, negatives_inside)

  loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = math_ops.cast(
    adjacency, dtype=dtypes.float32) - array_ops.diag(
      array_ops.ones([batch_size]))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = math_ops.reduce_sum(mask_positives)

  triplet_loss = math_ops.truediv(
    math_ops.reduce_sum(
      math_ops.maximum(
        math_ops.multiply(loss_mat, mask_positives), 0.0)),
    num_positives,
    name='triplet_semihard_loss')

  return triplet_loss