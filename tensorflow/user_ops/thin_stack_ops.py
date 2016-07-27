from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.platform import resource_loader

from tensorflow.user_ops import floaty_ops


try:
  # Try to load the GPU-enabled library first.
  _module = load_op_library("/tf-dev/bazel-bin/tensorflow/user_ops/libthin_stack_ops_impl_gpu.so")
except Exception, e:
  print(e)
  # Load CPU only.
  _module = load_op_library(os.path.join(resource_loader.get_data_files_path(), "thin_stack_ops_impl.so"))

thin_stack_lookup = _module.thin_stack_lookup
_thin_stack_lookup_gradient_impl = _module.thin_stack_lookup_grad
thin_stack_update = _module.thin_stack_update


@ops.RegisterShape("ThinStackLookup")
def _thin_stack_lookup_shape(op):
    batch_size = op.inputs[3].get_shape()[0]
    model_dim = op.inputs[0].get_shape()[1]
    embedding_dim = op.inputs[1].get_shape()[1]

    stack_el_shape = TensorShape((batch_size, model_dim))
    buf_el_shape = TensorShape((batch_size, embedding_dim))
    stack2_ptrs_shape = TensorShape((batch_size,))

    return [stack_el_shape, stack_el_shape, buf_el_shape, stack2_ptrs_shape]


def _fetch_buffer_cursors(buffer_cursors):
  """
  Given some Tensor which is a function of the `buffer_cursors` variable,
  return the original buffer_cursors variable contained within the op.
  """

  while buffer_cursors.op.type != "Variable":
      if buffer_cursors.op.type == "ThinStackUpdate":
        buffer_cursors = buffer_cursors.op.inputs[5]
      elif buffer_cursors.op.type == "ThinStackLookup":
        buffer_cursors = buffer_cursors.op.inputs[4]
      elif buffer_cursors.op.type == "Identity":
        buffer_cursors = buffer_cursors.op.inputs[0]
      else:
        raise RuntimeError("unknown op")

  return buffer_cursors


@ops.RegisterGradient("ThinStackLookup")
def _thin_stack_lookup_gradient(op, grad_stack1, grad_stack2, grad_buf_top, _):
    grad_stack1 = tf.Print(grad_stack1, [grad_stack1], "grad_stack1")
    grad_stack2 = tf.Print(grad_stack2, [grad_stack2], "grad_stack2")
    grad_buf_top = tf.Print(grad_buf_top, [grad_buf_top], "grad_buf_top")

    stack, buffer, _, _, buffer_cursors, transitions = op.inputs
    buffer_cursors = _fetch_buffer_cursors(buffer_cursors)

    stack2_ptrs = op.outputs[3]
    t = op.get_attr("timestep")

    batch_size = buffer_cursors.get_shape().as_list()[0]
    num_tokens = buffer.get_shape().as_list()[0] / batch_size
    batch_range = math_ops.range(batch_size)
    batch_range_i = tf.to_float(batch_range)

    grad_stack = gen_state_ops._temporary_variable(stack.get_shape().as_list(), tf.float32, "grad_stack%i" % t)
    grad_buffer = gen_state_ops._temporary_variable(buffer.get_shape().as_list(), tf.float32, "grad_buffer%i" % t)
    grad_stack = tf.assign(grad_stack, tf.zeros_like(grad_stack))
    grad_buffer = tf.assign(grad_buffer, tf.zeros_like(grad_buffer))

    updates = []

    # Write grad_stack1 into block (t - 1)
    if t >= 1:
      in_cursors = (t - 1) * batch_size + batch_range
      grad_stack = tf.scatter_add(grad_stack, in_cursors, grad_stack1)

    # Write grad_stack2 using stored lookup pointers
    grad_stack = floaty_ops.floaty_scatter_add(grad_stack, stack2_ptrs * batch_size + batch_range_i, grad_stack2)

    # Write grad_buf_top using buffer_cursors
    buffer_cursors = tf.assign(buffer_cursors, buffer_cursors - 1.0 + transitions)
    with tf.control_dependencies([buffer_cursors]):
      buffer_ptrs = tf.minimum((float) (num_tokens * batch_size) - 1.0,
                               buffer_cursors * batch_size + batch_range_i)
      grad_buffer = floaty_ops.floaty_scatter_add(grad_buffer, buffer_ptrs, grad_buf_top)

    with tf.control_dependencies([grad_stack, grad_buffer]):
      grad_stack = gen_state_ops._destroy_temporary_variable(grad_stack, "grad_stack%i" % t)
      grad_buffer = gen_state_ops._destroy_temporary_variable(grad_buffer, "grad_buffer%i" % t)
      grad_stack = tf.Print(grad_stack, [grad_stack], "grad_stack", summarize=100)
      grad_buffer = tf.Print(grad_buffer, [grad_buffer], "grad_buffer", summarize=100)

      with tf.control_dependencies([grad_stack, grad_buffer]):
        return grad_stack, grad_buffer, None, None, None, None

# Deprecated custom gradient op.
#@ops.RegisterGradient("ThinStackLookup")
def _thin_stack_lookup_metal_gradient(op, stack1_grad, stack2_grad, buf_top_grad, _):
    stack, buffer, _, _, buffer_cursors, transitions = op.inputs
    stack2_ptrs = op.outputs[3]
    timestep = op.get_attr("timestep")

    # HACK: Recover original Variable instances from op chain
    while stack.op.type != "Variable":
      stack = stack.op.inputs[0]
    while buffer.op.type != "Variable":
      assert buffer.op.type == "Identity"
      buffer = buffer.op.inputs[0]
    buffer_cursors = _fetch_buffer_cursors(buffer_cursors)

    updates = _thin_stack_lookup_gradient_impl(
            stack, buffer, stack2_ptrs, buffer_cursors,
            stack1_grad, stack2_grad, buf_top_grad, transitions,
            timestep)

    with ops.control_dependencies(updates):
        buffer = tf.Print(buffer, [buffer], "buffer", summarize=100)
        return tf.identity(stack), tf.identity(buffer), None, None, None, None


@ops.RegisterShape("ThinStackUpdate")
def _thin_stack_update_shape(op):
    _, _, stack, queue, cursors, buffer_cursors = op.inputs
    return [stack.get_shape(), queue.get_shape(), cursors.get_shape(),
            buffer_cursors.get_shape()]


@ops.RegisterGradient("ThinStackUpdate")
def _thin_stack_update_gradient(op, stack_grad, *rest):
    batch_size = op.inputs[4].get_shape().as_list()[0]
    t = op.get_attr("timestep")

    stack_grad = tf.Print(stack_grad, [stack_grad], "---------HERE", summarize=100)#[t * batch_size:(t + 1)*batch_size, :]], "---------------HERE", summarize=100)
#    input_grad = array_ops.slice(stack_grad, [t * batch_size, 0], [batch_size, -1])
    input_grad = stack_grad[t * batch_size:(t + 1) * batch_size, :]

    return input_grad, None, stack_grad, None, None, None
