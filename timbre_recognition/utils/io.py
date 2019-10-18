import numpy as np
import tensorflow as tf
import librosa
import re

# ---------------------------------------------------------------------------- #
# Model cofigurations
# ---------------------------------------------------------------------------- #

def load_wav_file_tf(path, desired_channels=-1):
  """Pure tensorflow implementation to load a single .wav file. 

  Args:
    - path: A `string` or `string` `Tensor` specifying the name of the .wav file 
      to read.
    - desired_channels: An `int` specifying the number of desired channels to 
      read. 

  Returns:
    - audio: A `float32` `ndarray` of shape (num_samples, desired_channels)
      containing the audio samples, normalized on the intervail [-1, 1]
    - sample_rate: An `int32` specifying the sampling frequency of the audio.
  """
  if not tf.is_tensor(path):
    path = tf.convert_to_tensor(path)
  file = tf.io.read_file(tf.convert_to_tensor(path))
  y, sr = tf.audio.decode_wav(file, desired_channels=desired_channels)
  y = tf.expand_dims(y, 0)
  y = tf.expand_dims(y, 1)
  return y, sr

def load_wav_file_librosa(path, sr=None, mono=False):
  """Librosa implementation to load a single .wav file. The main reason of 
  using this over the tensorflow implementation is ability to resample to 
  a desired rate.

  Args:
    - path: A `string` specifying the name of the .wav file to read
    - sr: A `int` specifying the target sampling rate. If None the 
      native sampling rate is used. 
    - mono: A `bool` that if true will convert multi-channel to mono.

  Returns:
    - audio: A `float32` `ndarray` of shape (num_samples, desired_channels)
      containing the audio samples, normalized on the intervail [-1, 1]
    - sample_rate: An `int32` specifying the sampling frequency of the audio.
  """
  y, sr = librosa.load(path, sr=sr, mono=mono)
  y = tf.convert_to_tensor(y)
  y = tf.expand_dims(y, 0)
  y = tf.expand_dims(y, 1)
  y = tf.expand_dims(y, 3)
  return y, sr

def parse_esc50_filename(filename):
  """Parses a ESC-50 data filename into its fold, source number, take, and
  target values.
  
  Args:
    - filename: A `string` or `bytes` specifying the name or path of an ESC-50 
      audio file.
  Returns:
    - fold: A `string` specifying the index of the cross-validation fold.
    - src_number: A `string` specifying the ID of the original Freesound clip.
    - take: A `string` specifying a letter disambiguating between different 
      fragments from the same Freesound clip.
    - target: A `string` specifying the class in numeric format [0, 49].
  """
  if type(filename) == str:
    filename = re.split('/', filename)[-1]
    fold, src_number, take, target = re.split('-|\.', filename)[:4]
  elif type(filename) == bytes:
    filename = re.split(b'/', filename)[-1].decode('utf-8')
    fold, src_number, take, target = re.split('-|\.', filename)[:4]
  return fold, src_number, take, target

def load_esc50_dataset(path):
  """Pure tensorflow implementation to load the entire esc50 dataset. 
  
  Args:
    - directory: A `string` containing the path to the dataset
    
  Returns:
    - data: A `float32` `Tensor` with shape  [dataset_size, height, width,
      channels]. Width is just the number of samples. The dataset_size should
      be 2000, and channels and height should be 1.
    - labels: A `string` `Tensor` with shape [dataset_size, 4]
  
  """
  filenames = tf.io.match_filenames_once(path + '*.wav').value()
  data = tf.concat([load_wav_file_tf(x)[0] for x in filenames], axis=0)
  labels = tf.convert_to_tensor([parse_esc50_filename(x.numpy()) for x in filenames])
  return data, labels  