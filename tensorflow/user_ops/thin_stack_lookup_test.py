from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.user_ops import thin_stack_ops as ts


class ThinStackLookupTest(test.TestCase):
  use_gpu = False

  def testInitialLookup(self):
    pass # TODO

  def testIntermediateLookup(self):
    """Test a standard lookup somewhere in the middle of a stack recurrence."""
    batch_size = 2
    model_dim = 5
    embedding_dim = 5
    num_timesteps = 5

    num_tokens = (num_timesteps + 1) / 2

    with self.test_session(use_gpu=self.use_gpu) as s:
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

      ret = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, t)
      stack1, stack2, buf_top, stack2_ptrs = s.run(ret)

    stack1_expected = stack_val[4:6]
    stack2_expected = np.array([stack_val[0], stack_val[3]])
    buf_top_expected = np.array([buffer_val[4], buffer_val[5]])
    stack2_ptrs_expected = np.array([0, 3])

    self.assertAllEqual(stack1_expected, stack1)
    self.assertAllEqual(stack2_expected, stack2)
    self.assertAllEqual(buf_top_expected, buf_top)
    self.assertAllClose(stack2_ptrs_expected, stack2_ptrs)


if __name__ == "__main__":
  test.main()

