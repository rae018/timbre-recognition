import tensorflow as tf


def print_module_tree(module, level=0):
  offset = '  ' * level
  for k, v in module.__dict__.items():
    if type(v) is tf.Module:
      print(offset + k + ':')
      print_module_tree(v, level+1)
    elif tf.is_tensor(v):
      print(offset + k)
  
  
def compute_num_params(module):
  """Counts the number of parameters in a module
  
  Args:
    - module: A `tf.Module` containing the parameters to count
  
  Returns:
    - params: An `int32` `tf.Tensor` speicifying the number of params
  """
  params = 0
  for k, v in module.__dict__.items():
    if type(v) is tf.Module:
      params += compute_num_params(v)
    elif tf.is_tensor(v):
      params += tf.reduce_prod(v.shape)
  return params