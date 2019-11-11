from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from timbre_recognition.modeling.inception_resnet_v2 import *
from timbre_recognition.ops.triplet_loss import *
from timbre_recognition.configs.config import cfg
import tensorflow as tf


def inception_resnet_v2_train(inputs, labels, kernel_module, step):
  # Variables that affect learning rate.
  decay_steps = int(cfg.TRAIN.NUM_BATCHES_PER_EPOCH * cfg.TRAIN.NUM_EPOCHS_PER_DECAY)
  
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
    cfg.TRAIN.BASE_LR, decay_steps, cfg.TRAIN.LR_DECAY_RATE, staircase=True)(step)

  with tf.GradientTape() as tape:
    embeddings, endpoints = inception_resnet_v2(inputs, kernel_module, cfg.MODEL.NUM_CLASSES, cfg.MODEL.EMBED_DIM)
    class_labels = tf.strings.to_number(labels[:, 3], tf.dtypes.int32)
    triplet_loss = batch_triplet_semihard_loss(class_labels, embeddings, cfg.TRAIN.LOSS_MARGIN, kernel_module)
    print("Loss:", triplet_loss.numpy())
    
  optimizer = tf.keras.optimizers.SGD(lr)
  gradients = tape.gradient(triplet_loss, kernel_module.trainable_variables)
  optimizer.apply_gradients(zip(gradients, kernel_module.trainable_variables))
  
  
def add_batch_embeddings(batch_data, batch_labels, kernel_module, class_centers):
  class_labels = tf.strings.to_number(batch_labels[:, 3], tf.dtypes.int32)
  embeddings, endpoints = inception_resnet_v2(batch_data, kernel_module, embed_dim=cfg.MODEL.EMBED_DIM)
  
  order = tf.argsort(class_labels)
  values = tf.gather(embeddings, order)
  rowids = tf.gather(class_labels, order)
  # This is to enfore the first dimension is 50, instead of just the highest
  # class number in the batch
  extension = tf.zeros([1, cfg.MODEL.EMBED_DIM])
  values = tf.concat([values, extension], 0)
  rowids = tf.concat([rowids, [cfg.DATASET.NUM_CLASSES]], 0)
  
  sorted_embeddings = tf.RaggedTensor.from_value_rowids(values=values, value_rowids=rowids)
  class_centers = tf.concat([class_centers, sorted_embeddings], axis=1)
  
  return class_centers