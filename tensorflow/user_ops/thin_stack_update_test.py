from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.user_ops import thin_stack_ops as ts


class ThinStackUpdateTest(test.TestCase):
  use_gpu = False

  def testInitialUpdate(self):
    pass # TODO

  def testIntermediateUpdate(self):
    """Test a standard update somewhere in the middle of a stack recurrence."""
    batch_size = 2
    model_dim = 5
    embedding_dim = 5
    num_timesteps = 5

    num_tokens = (num_timesteps + 1) / 2

    with self.test_session(use_gpu=self.use_gpu):
      # Example 1: S S R S
      # Example 2: S S S R
      #                  ^
      # we are running lookup at the above timestep

      stack = constant_op.constant([[-1., -1., -1., -1., -1.],
                                    [ 1.,  1.,  1.,  1.,  1.],
                                    [-2., -2., -2., -2., -2.],
                                    [ 2.,  2.,  2.,  2.,  2.],
                                    [-3., -3., -3., -3., -3.],
                                    [ 3.,  3.,  3.,  3.,  3.],
                                    [ 0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.]])
      buffer = constant_op.constant([[-1., -1., -1., -1., -1.],
                                     [ 1.,  1.,  1.,  1.,  1.],
                                     [-2., -2., -2., -2., -2.],
                                     [ 2.,  2.,  2.,  2.,  2.],
                                     [-3., -3., -3., -3., -3.],
                                     [ 3.,  3.,  3.,  3.,  3.]])
      queue = constant_op.constant([2., 0.,
                                    0., 1.,
                                    0., 2.,
                                    0., 0.,
                                    0., 0.])
      cursors = constant_op.constant([0., 2.])
      buffer_cursors = constant_op.constant([2., 3.])
      t = constant_op.constant(3)

      stack_val = stack.eval()
      buffer_val = buffer.eval()

      shift_in = constant_op.constant([buffer_val[2], buffer_val[3]])
      reduce_in = constant_op.constant([stack_val[4] + stack_val[0],
                                        stack_val[5] + stack_val[3]])
      transitions = constant_op.constant([0., 1.])

      stack_next, queue_next, cursors_next, buffer_cursors_next = \
          ts.thin_stack_update(shift_in, reduce_in, transitions,
                               stack, queue, cursors, buffer_cursors)


if __name__ == "__main__":
  test.main()

