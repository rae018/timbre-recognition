# Most of this code was taken from Facebook Detectron:
# https://github.com/facebookresearch/Detectron
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from ast import literal_eval
import numpy as np
import yaml

class AttrDict(dict):  
  IMMUTABLE = '__immutable__'
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__[AttrDict.IMMUTABLE] = False

  def __getattr__(self, name):
    if name in self.__dict__:
      return self.__dict__[name]
    elif name in self:
      return self[name]
    else:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    if not self.__dict__[AttrDict.IMMUTABLE]:
      if name in self.__dict__:
        self.__dict__[name] = value
      else:
        self[name] = value
    else:
      raise AttributeError(
        'Attempted to set "{}" to "{}", but AttrDict is immutable'.
        format(name, value)
      )
      
  def immutable(self, is_immutable):
    """Set immutability to is_immutable and recursively apply the setting
    to all nested AttrDicts.
    """
    self.__dict__[AttrDict.IMMUTABLE] = is_immutable
    # Recursively set immutable state
    for v in self.__dict__.values():
      if isinstance(v, AttrDict):
        v.immutable(is_immutable)
    for v in self.values():
      if isinstance(v, AttrDict):
        v.immutable(is_immutable)

  def is_immutable(self):
    return self.__dict__[AttrDict.IMMUTABLE]
            
__C = AttrDict()

cfg = __C

# ---------------------------------------------------------------------------- #
# Dataset cofigurations
# ---------------------------------------------------------------------------- #

__C.DATASET = AttrDict()

__C.DATASET.BUFFER_SIZE = 2000

__C.DATASET.PATH = ''

# ---------------------------------------------------------------------------- #
# Model cofigurations
# ---------------------------------------------------------------------------- #

__C.MODEL = AttrDict()

# File to save model to
__C.MODEL.OUTPUT_DIR = ''

# The backbone conv body of the network
__C.MODEL.CONV_BODY = ''

# Number of classes is the dataset
__C.MODEL.NUM_CLASSES = -1

__C.MODEL.EMBED_DIM = 128

# ---------------------------------------------------------------------------- #
# Training cofigurations
# ---------------------------------------------------------------------------- #

__C.TRAIN = AttrDict()

# Is training
__C.TRAIN.TRAINING = True

# File to save weights to and initialize from
__C.TRAIN.WEIGHTS = '' # weight filename

# Batch size
__C.TRAIN.BATCH_SIZE = 64

# Base learning rate used in training
__C.TRAIN.BASE_LR = 0.001

__C.TRAIN.LR_DECAY_RATE = 0.001

# Interval at which to save model checkpoints
__C.TRAIN.CHECKPOINT_PERIOD = 1000

__C.TRAIN.NUM_EPOCHS = 200

# This is dataset size / batch size
__C.TRAIN.NUM_BATCHES_PER_EPOCH = 32

__C.TRAIN.NUM_EPOCHS_PER_DECAY = 100

__C.TRAIN.LOSS_MARGIN = 0.1

# ---------------------------------------------------------------------------- #
# Evaulate cofigurations
# ---------------------------------------------------------------------------- #

__C.EVALUATE = AttrDict()

__C.EVALUATE.THRESHOLD = 0.1

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPRECATED_KEYS = set(
  {
  }
)


# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
}


# ---------------------------------------------------------------------------- #
# Renamed modules
# If a module containing a data structure used in the config (e.g. AttrDict)
# is renamed/moved and you don't want to break loading of existing yaml configs
# (e.g. from weights files) you can specify the renamed module below.
# ---------------------------------------------------------------------------- #
_RENAMED_MODULES = {
}


def assert_and_infer_cfg(make_immutable=True):
  """Call this function in your script after you have finished setting all cfg
  values that are necessary (e.g., merging a config from a file, merging
  command line config options, etc.). By default, this function will also
  mark the global cfg as immutable to prevent changing the global cfg settings
  during script execution (which can lead to hard to debug errors or code
  that's harder to understand than is necessary).
  """
  if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
      __C.RPN.RPN_ON = True
  if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
      __C.TEST.PRECOMPUTED_PROPOSALS = False
  if make_immutable:
      cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
  """Load a yaml config file and merge it into the global config."""
  with open(cfg_filename, 'r') as f:
    yaml_cfg = AttrDict(yaml.load(f))
  _merge_a_into_b(yaml_cfg, __C)
  

def _merge_a_into_b(a, b, stack=None):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  assert isinstance(a, AttrDict), \
    '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
  assert isinstance(b, AttrDict), \
    '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

  for k, v_ in a.items():
    full_key = '.'.join(stack) + '.' + k if stack is not None else k
    # a must specify keys that are in b
    if k not in b:
      if _key_is_deprecated(full_key):
        continue
      elif _key_is_renamed(full_key):
        _raise_key_rename_error(full_key)
      else:
        raise KeyError('Non-existent config key: {}'.format(full_key))

    v = copy.deepcopy(v_)
    v = _decode_cfg_value(v)
    v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

    # Recursively merge dicts
    if isinstance(v, AttrDict):
      try:
        stack_push = [k] if stack is None else stack + [k]
        _merge_a_into_b(v, b[k], stack=stack_push)
      except BaseException:
        raise
    else:
      b[k] = v


def _key_is_deprecated(full_key):
  if full_key in _DEPRECATED_KEYS:
    logger.warn(
      'Deprecated config key (ignoring): {}'.format(full_key)
    )
    return True
  return False


def _key_is_renamed(full_key):
  return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
  new_key = _RENAMED_KEYS[full_key]
  if isinstance(new_key, tuple):
    msg = ' Note: ' + new_key[1]
    new_key = new_key[0]
  else:
    msg = ''
  raise KeyError(
    'Key {} was renamed to {}; please update your config.{}'.
    format(full_key, new_key, msg)
  )


def _decode_cfg_value(v):
  """Decodes a raw config value (e.g., from a yaml config files or command
  line argument) into a Python object.
  """
  # Configs parsed from raw yaml will contain dictionary keys that need to be
  # converted to AttrDict objects
  if isinstance(v, dict):
    return AttrDict(v)
  # All remaining processing is only applied to strings
  if not isinstance(v, str):
    return v
  # Try to interpret `v` as a:
  #   string, number, tuple, list, dict, boolean, or None
  try:
    v = literal_eval(v)
  # The following two excepts allow v to pass through when it represents a
  # string.
  #
  # Longer explanation:
  # The type of v is always a string (before calling literal_eval), but
  # sometimes it *represents* a string and other times a data structure, like
  # a list. In the case that v represents a string, what we got back from the
  # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
  # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
  # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
  # will raise a SyntaxError.
  except ValueError:
    pass
  except SyntaxError:
    pass
  return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
  """Checks that `value_a`, which is intended to replace `value_b` is of the
  right type. The type is correct if it matches exactly or is one of a few
  cases in which the type can be easily coerced.
  """
  # The types must match (with some exceptions)
  type_b = type(value_b)
  type_a = type(value_a)
  if type_a is type_b:
    return value_a

  # Exceptions: numpy arrays, strings, tuple<->list
  if isinstance(value_b, np.ndarray):
    value_a = np.array(value_a, dtype=value_b.dtype)
  elif isinstance(value_b, str):
    value_a = str(value_a)
  elif isinstance(value_a, tuple) and isinstance(value_b, list):
    value_a = list(value_a)
  elif isinstance(value_a, list) and isinstance(value_b, tuple):
    value_a = tuple(value_a)
  else:
    raise ValueError(
      'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
      'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
    )
  return value_a
