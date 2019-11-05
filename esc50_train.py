#!timbre_recognition/bin/python3
#SBATCH -p csci                                             
#SBATCH -n 2                                                          
#SBATCH --gres=gpu:4 # number of GPUs                                             
#SBATCH -o slurm.%N.%j.stdout.txt # STDOUT                                           
#SBATCH -e slurm.%N.%j.stderr.txt # STDERR                                        
#SBATCH --mail-user=rae018@bucknell.edu # address to email        
#SBATCH --mail-type=ALL 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# So slurm can find relative packages    
import sys                                                           
import os                                                                                          
sys.path.append(os.getcwd())

from timbre_recognition.modeling.inception_resnet_v2_train import *
from timbre_recognition.utils.io import *
from timbre_recognition.utils.evaluate import *
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
  
  print('STARTING TRAINING')
  for epoch in range(cfg.TRAIN.NUM_EPOCHS):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(cfg.DATASET.BUFFER_SIZE).batch(cfg.TRAIN.BATCH_SIZE)
    dataset_iter = iter(dataset)
    batch_data, batch_labels = next(dataset_iter, (None, None))
    while batch_data is not None:
      inception_resnet_v2_train(batch_data, batch_labels, kernel_module, epoch)
      batch_data, batch_labels = next(dataset_iter, (None, None))
    ckpt = tf.train.Checkpoint(kernel_module=kernel_module)
    manager.save()
    print('Checkpoint for epoch {} saved'.format(epoch))
  print('TRAINING COMPLETE')
   
  print('STARTING EVALUATION')
  thresholds = tf.linspace(0.0, 2.0, 21).numpy()
  accuracies = evaluate_model('inception_resnet_v2', 'esc-50', cfg.DATASET.PATH, kernel_module, thresholds)
  for i in range(len(thresholds)):
    print('Accuracy for threshold {}: {}'.format(thresholds[i], accuracies[i]))
  print('EVAULATION COMPLETE')
    

if __name__ == '__main__':
  main()
