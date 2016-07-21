from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.variables import initialize_variables, Variable
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

    with self.test_session(use_gpu=self.use_gpu) as s:
      # Example 1: S S R S
      # Example 2: S S S R
      #                  ^
      # we are running lookup at the above timestep

      stack = Variable([[-1., -1., -1., -1., -1.],
                        [ 1.,  1.,  1.,  1.,  1.],
                        [-2., -2., -2., -2., -2.],
                        [ 2.,  2.,  2.,  2.,  2.],
                        [-3., -3., -3., -3., -3.],
                        [ 3.,  3.,  3.,  3.,  3.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.]])
      buffer = Variable([[-1., -1., -1., -1., -1.],
                         [ 1.,  1.,  1.,  1.,  1.],
                         [-2., -2., -2., -2., -2.],
                         [ 2.,  2.,  2.,  2.,  2.],
                         [-3., -3., -3., -3., -3.],
                         [ 3.,  3.,  3.,  3.,  3.]])
      queue = Variable([2., 0.,
                        0., 1.,
                        0., 2.,
                        0., 0.,
                        0., 0.])
      cursors = Variable([0., 2.])
      buffer_cursors = Variable([2., 3.])
      t = constant_op.constant(3)

      s.run(initialize_variables([stack, buffer, queue, cursors, buffer_cursors]))

      stack_val = stack.eval()
      buffer_val = buffer.eval()

      shift_in = constant_op.constant(np.array([buffer_val[4], buffer_val[5]]))
      reduce_in = constant_op.constant(np.array([stack_val[4] + stack_val[0],
                                                 stack_val[5] + stack_val[3]]))
      transitions = tf.expand_dims(constant_op.constant([0., 1.]), 1)
      input_val = transitions * reduce_in + (1. - transitions) * shift_in


      ret = ts.thin_stack_update(input_val, transitions,
                                 stack, queue, cursors, buffer_cursors, t)
      stack_next, queue_next, cursors_next, buffer_cursors_next = s.run(ret)

    print(stack_next)
    print(queue_next)
    print(cursors_next)
    print(buffer_cursors_next)

    # TODO assert that we are sharing underlying data here?
    stack_expected = np.copy(stack_val)
    stack_expected[6] = buffer_val[4]
    stack_expected[7] = stack_val[5] + stack_val[3]

    queue_expected = np.array([2., 0.,
                               3., 3.,
                               0., 2., # NB: we didn't erase this, but it's okay
                               0., 0.,
                               0., 0.])
    cursors_expected = np.array([1., 1.])
    buffer_cursors_expected = np.array([3., 3.])

    self.assertAllEqual(stack_next, stack_expected)
    self.assertAllEqual(queue_next, queue_expected)
    self.assertAllEqual(cursors_next, cursors_expected)
    self.assertAllEqual(buffer_cursors_next, buffer_cursors_expected)


if __name__ == "__main__":
  test.main()

