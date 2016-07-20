from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


_module = load_op_library(os.path.join(resource_loader.get_data_files_path(), "thin_stack_ops_impl.so"))

thin_stack_lookup = _module.thin_stack_lookup
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


@ops.RegisterShape("ThinStackUpdate")
def _thin_stack_update_shape(op):
    _, _, _, stack, queue, cursors, buffer_cursors, _ = op.inputs
    return [stack.get_shape(), queue.get_shape(), cursors.get_shape(),
            buffer_cursors.get_shape()]


# TODO grad definitions: just invoke grad ops
