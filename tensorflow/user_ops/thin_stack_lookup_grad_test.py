from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.variables import initialize_variables, Variable
from tensorflow.python.platform import test
from tensorflow.user_ops import thin_stack_ops as ts


class ThinStackLookupGradTest(test.TestCase):
  use_gpu = False

  def testInitialLookupGrad(self):
    pass # TODO

  def testIntermediateLookupGrad(self):
    """
    Test the gradient of a standard lookup somewhere in the middle of a stack
    recurrence.
    """

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

      s.run(initialize_variables([stack, buffer, queue, cursors, buffer_cursors]))

      stack_val = stack.eval()
      buffer_val = buffer.eval()

      lookup = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, timestep=3)

      #### GRADIENT

      stack1_grad = tf.random_uniform((batch_size, model_dim))
      stack2_grad = tf.random_uniform((batch_size, model_dim))
      buf_top_grad = tf.random_uniform((batch_size, model_dim))
      in_grads = (stack1_grad, stack2_grad, buf_top_grad, None)

      out_grads = ts._thin_stack_lookup_gradient(lookup[0].op, in_grads)
      out_grads = out_grads[:2]

      fetch = out_grads + (stack1_grad, stack2_grad, buf_top_grad)

      ret = s.run(fetch)

    print(ret)


class GpuThinStackLookupGradTest(ThinStackLookupGradTest):
  use_gpu = True


if __name__ == "__main__":
  test.main()

