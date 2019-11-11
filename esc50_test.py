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
   
  print('STARTING EVALUATION')
  thresholds = tf.linspace(0.0, 2.0, 21).numpy()
  accuracies = evaluate_model('inception_resnet_v2', 'esc-50', cfg.DATASET.PATH, kernel_module, thresholds)
  for i in range(len(thresholds)):
    print('Accuracy for threshold {}: {}'.format(thresholds[i], accuracies[i]))
  print('EVAULATION COMPLETE')
    

if __name__ == '__main__':
  main()