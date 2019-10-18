from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from timbre_recognition.modeling.inception_resnet_v2_train import *
from timbre_recognition.utils.io import *
from timbre_recognition.configs.config import *
import tensorflow as tf

config_file = 'timbre_recognition/configs/esc50.yaml'
data_directory = 'timbre_recognition/datasets/ESC-50/audio/'

def main():
  print('Loading configuration file...')
  merge_cfg_from_file(config_file)
  print('Configurations loaded')
  
  data, labels = load_esc50_dataset(data_directory)
  kernel_module = tf.Module()
  ckpt = tf.train.Checkpoint(kernel_module=kernel_module)
  manager = tf.train.CheckpointManager(ckpt, cfg.MODEL.OUTPUT_DIR, max_to_keep=None)
  
  print('Starting training')
  for epoch in range(cfg.TRAIN.NUM_EPOCHS):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(cfg.DATASET.BUFFER_SIZE).batch(cfg.TRAIN.BATCH_SIZE)
    batch_data, batch_labels = next(iter(dataset), (None, None))
    while batch_data is not None:
      inception_resnet_v2_train(batch_data, batch_labels, kernel_module, epoch)
      batch_data, batch_labels = next(iter(dataset), (None, None))
    ckpt = tf.train.Checkpoint(kernel_module=kernel_module)
    manager.save()
    print('Checkpoint for epoch {} saved'.format(epoch))

if __name__ == '__main__':
  main()