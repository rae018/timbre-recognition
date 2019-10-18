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
    triplet_loss = batch_hard_triplet_loss(class_labels, embeddings, cfg.TRAIN.LOSS_MARGIN, True, 'mahalanobis', kernel_module)
    print("Loss:", triplet_loss.numpy())
    
  optimizer = tf.keras.optimizers.SGD(lr)
  gradients = tape.gradient(triplet_loss, kernel_module.trainable_variables)
  optimizer.apply_gradients(zip(gradients, kernel_module.trainable_variables))
  
  # Possibly add exponential moving average and add summaries
  


