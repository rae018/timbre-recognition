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

def main():
  print('Loading configuration file...')
  merge_cfg_from_file(config_file)
  print('Configurations loaded')
  
  checkpoint = cfg.MODEL.OUTPUT_DIR + 'ckpt-800'
  kernel_module = restore_kernel_module_from_checkpoint('inception_resnet_v2', 'esc-50', checkpoint)
  data, labels = load_esc50_dataset(cfg.DATASET.PATH)
   
  print('FINDING CLASS CENTERS')
  init = tf.zeros([1, 128])
  class_centers = tf.RaggedTensor.from_value_rowids(values=init, value_rowids=tf.constant([50]))
  dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(cfg.DATASET.BUFFER_SIZE).batch(cfg.TRAIN.BATCH_SIZE)
  dataset_iter = iter(dataset)
  batch_data, batch_labels = next(dataset_iter, (None, None))
  while batch_data is not None:
    class_centers = add_batch_embeddings(batch_data, batch_labels, kernel_module, class_centers)
    batch_data, batch_labels = next(dataset_iter, (None, None))
  class_centers = tf.reduce_mean(class_centers, 1)[:50]
  print('CLASS CENTERS COMPLETE')
  
  print('STARTING EVALUATION')
  data, labels = load_esc50_test_set(cfg.DATASET.PATH)
  labels = tf.strings.to_number(labels[:, 3], tf.dtypes.int32)
  
  dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(cfg.TRAIN.BATCH_SIZE)
  dataset_iter = iter(dataset)
  batch_data, batch_labels = next(dataset_iter, (None, None))
  
  accuracies = []
  
  while batch_data is not None:
    embeddings, endpoints = inception_resnet_v2(batch_data, kernel_module, embed_dim=cfg.MODEL.EMBED_DIM)
    distances = pairwise_mahalanobis_distances(embeddings, class_centers, kernel_module)
    # shape [batch_size]
    predictions = tf.argsort(distances, 1)[:,0] # 0 index holds class center closest to embedding, taken along entire batch
    results = tf.math.equal(batch_labels, predictions)
    num_correct = tf.math.count_nonzero(results)
    batch_accuracy = num_correct / len(results)
    accuracies += [batch_accuracy]
    
    batch_data, batch_labels = next(dataset_iter, (None, None))
  accuracy = tf.reduce_mean(accuracies)
  print('EVAULATION COMPLETE')
  print("Accuracy: {}".format(accuracy))
    

if __name__ == '__main__':
  main()