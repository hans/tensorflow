from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.variables import initialize_variables, Variable
from tensorflow.python.platform import test
from tensorflow.user_ops import thin_stack_ops as ts


class IntegratedThinStackTest(test.TestCase):
  use_gpu = False

  def testIntegrated3Step(self):
    """
    Test the simplest possible transition sequence on a batch of random inputs.
    """

    batch_size = 2
    model_dim = 5
    embedding_dim = 5
    num_timesteps = 3

    num_tokens = (num_timesteps + 1) / 2

    with self.test_session(use_gpu=self.use_gpu) as s:
      stack = Variable(np.zeros((batch_size * num_timesteps, model_dim), dtype=np.float32), name="stack")
      buffer = Variable(np.random.random((batch_size * num_tokens, embedding_dim)).astype(np.float32), name="buffer")
      queue = Variable(np.zeros((batch_size * num_timesteps,), dtype=np.float32), name="queue")
      cursors = Variable(np.zeros((batch_size,), dtype=np.float32) - 1., name="cursors")
      buffer_cursors = Variable(np.zeros((batch_size,), dtype=np.float32), name="buffer_cursors")

      s.run(initialize_variables([stack, buffer, queue, cursors, buffer_cursors]))

      ######## Fprop test.
      b1, b2, top = self._run_ff(batch_size, stack, buffer, queue, cursors, buffer_cursors)
      b1_sim, b2_sim, top_sim = self._simulate_ff(buffer)

      ######## Bprop test.
      # Get some scalar error signal for grad calculation
      top, top_sim = tf.reduce_sum(top), tf.reduce_sum(top_sim)
      grad = tf.gradients(top, buffer)[0]
      grad_sim = tf.gradients(top_sim, buffer)[0]
      print(grad, grad_sim)

      ######## Run fetches.
      ret = s.run([top, top_sim, grad, grad_sim])
      top_, top_sim_, grad_, grad_sim_ = ret[:2]

    self.assertAllClose(top_, top_sim_)
    self.assertAllClose(grad_b1_, grad_b1_sim_)
    self.assertAllClose(grad_b2_, grad_b2_sim_)

  def _compose(self, stack1, stack2):
    return stack1 + stack2

  def _run_ff(self, batch_size, stack, buffer, queue, cursors, buffer_cursors):
    """
    Simulate a 3-step S S M sequence using the thin stack representation / ops.
    """

    transitions_shift = tf.zeros_like(cursors)
    transitions_reduce = tf.ones_like(cursors)

    # Shift.
    _, _, b1, p1 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, 0)
    print(tf.gradients(tf.reduce_sum(b1), buffer))
    updates = ts.thin_stack_update(b1, transitions_shift, stack, queue, cursors, buffer_cursors, 0)

    with tf.control_dependencies(updates):
      # Shift.
      _, _, b2, p2 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, 1)
      updates = ts.thin_stack_update(b2, transitions_shift, stack, queue, cursors, buffer_cursors, 1)

      with tf.control_dependencies(updates):
        # Reduce.
        s1_3, s2_3, _, p3 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, 2)
        updates = ts.thin_stack_update(self._compose(s1_3, s2_3),
                                       transitions_reduce, stack, queue, cursors, buffer_cursors, 2)

        with tf.control_dependencies(updates):
          stack_top = stack[batch_size * 2:, :]

    return b1, b2, stack_top

  def _simulate_ff(self, buffer):
    # Simulate S S M == compose(b1, b2)
    b1, b2 = buffer[:2, :], buffer[2:4, :]
    stack_top = self._compose(b1, b2)
    return b1, b2, stack_top


if __name__ == "__main__":
  test.main()
