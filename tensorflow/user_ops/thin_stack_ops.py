from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import resource_loader


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


@ops.RegisterGradient("ThinStackLookup")
def _thin_stack_lookup_gradient(op, grad):
    stack, buffer, _, _, buffer_cursors = op.inputs[:5]
    stack2_ptrs = op.outputs[3]
    timestep = op.get_attr("timestep")
    stack1_grad, stack2_grad, buf_top_grad, _ = grad

    # HACK
    stack = stack.op.inputs[0]
    buffer = buffer.op.inputs[0]

    stack_grad, buffer_grad = _thin_stack_lookup_gradient_impl(
            stack, buffer, stack2_ptrs, buffer_cursors,
            stack1_grad, stack2_grad, buf_top_grad, timestep)

    with ops.control_dependencies([stack_grad, buffer_grad]):
        return stack_grad, buffer_grad, None, None, None


@ops.RegisterShape("ThinStackUpdate")
def _thin_stack_update_shape(op):
    _, _, stack, queue, cursors, buffer_cursors = op.inputs
    return [stack.get_shape(), queue.get_shape(), cursors.get_shape(),
            buffer_cursors.get_shape()]


@ops.RegisterGradient("ThinStackUpdate")
def _thin_stack_update_gradient(op, grad):
    batch_size = op.inputs[4].get_shape()[0]
    t = op.inputs[6]

    stack_grad = grad[0]
    input_grad = stack_grad[t * batch_size:(t + 1) * batch_size]

    return input_grad, None, None, None, None, None, None


