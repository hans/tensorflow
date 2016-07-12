from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), "floaty_ops_impl.so"))

floaty_gather = _module.floaty_gather
floaty_scatter_update = _module.floaty_scatter_update


@ops.RegisterShape("FloatyGather")
def _FloatyGatherShape(op):
  """Shape function for array_ops.gather."""
  params_shape = op.inputs[0].get_shape()
  indices_shape = op.inputs[1].get_shape()
  return [indices_shape.concatenate(params_shape[1:])]


@ops.RegisterGradient("FloatyGather")
def _FloatyGatherGrad(op, grad):
  if op.inputs[0].get_shape().is_fully_defined():
    dense_shape = constant_op.constant(op.inputs[0].get_shape().as_list())
    values_shape = [-1] + op.inputs[0].get_shape()[1:].as_list()
  else:
    # op.inputs[0] can be large, so colocate the shape calculation with it.
    with ops.colocate_with(op.inputs[0]):
      dense_shape = array_ops.shape(op.inputs[0])
      values_shape = array_ops.concat(0, [[-1], dense_shape[1:]])

  values = array_ops.reshape(grad, values_shape)
  indices = math_ops.to_int32(array_ops.reshape(op.inputs[1], [-1]))
  return [ops.IndexedSlices(values, indices, dense_shape), None]

