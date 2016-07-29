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

  def _testIntegrated(self, batch_size, model_dim, num_timesteps, ff_fun, sim_fun):
    """
    Test the simplest possible transition sequence on a batch of random inputs.
    """

    tf.reset_default_graph()

    embedding_dim = model_dim
    num_tokens = (num_timesteps + 1) / 2

    with self.test_session(use_gpu=self.use_gpu) as s:
      stack = Variable(np.zeros((batch_size * num_timesteps, model_dim), dtype=np.float32), name="stack")
      buffer = Variable(np.random.random((batch_size * num_tokens, embedding_dim)).astype(np.float32), name="buffer")
      queue = Variable(np.zeros((batch_size * num_timesteps,), dtype=np.float32), name="queue")
      cursors = Variable(np.zeros((batch_size,), dtype=np.float32) - 1., name="cursors")
      buffer_cursors = Variable(np.zeros((batch_size,), dtype=np.float32), name="buffer_cursors")

      ######## Fprop test.
      top = ff_fun(batch_size, stack, buffer, queue, cursors, buffer_cursors)
      top_sim = sim_fun(buffer)

      s.run(initialize_variables(tf.all_variables()))

      ######## Bprop test.
      # Get some scalar error signal for grad calculation
      top, top_sim = tf.reduce_sum(top), tf.reduce_sum(top_sim)
      with tf.control_dependencies([top]):
        grad = tf.gradients(top, buffer)[0]
      grad_sim = tf.gradients(top_sim, buffer)[0]

      ######## Run fetches.
      ret = s.run([top, top_sim, grad, grad_sim])
      top_, top_sim_, grad_, grad_sim_ = ret[:4]

    self.assertAllClose(top_, top_sim_)
    self.assertAllClose(grad_, grad_sim_)

  def _compose(self, stack1, stack2):
    if not hasattr(self, "W"):
      batch_size, model_dim = stack1.get_shape().as_list()
      self.W = tf.get_variable("W", (model_dim * 2, model_dim))

    ret = tf.matmul(tf.concat(1, [stack1, stack2]), self.W)
    return ret

  def _run_ff_3(self, batch_size, stack, buffer, queue, cursors, buffer_cursors):
    """
    Simulate a 3-step S S M sequence using the thin stack representation / ops.
    """

    transitions_shift = tf.zeros_like(cursors)
    transitions_reduce = tf.ones_like(cursors)

    # Shift.
    _, _, b1, p1 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_shift, 0)
    updates = ts.thin_stack_update(b1, transitions_shift, stack, queue, cursors, buffer_cursors, 0)
    stack, queue, cursors, buffer_cursors = updates

    with tf.control_dependencies(updates):
      # Shift.
      _, _, b2, p2 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_shift, 1)
      updates = ts.thin_stack_update(b2, transitions_shift, stack, queue, cursors, buffer_cursors, 1)
      stack, queue, cursors, buffer_cursors = updates

      with tf.control_dependencies(updates):
        # Reduce.
        s1_3, s2_3, _, p3 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_reduce, 2)
        updates = ts.thin_stack_update(self._compose(s1_3, s2_3),
                                       transitions_reduce, stack, queue, cursors, buffer_cursors, 2)
        stack = updates[0]

        with tf.control_dependencies(updates):
          stack_top = stack[batch_size * 2:, :]

    return stack_top

  def _sim_ff_3(self, buffer):
    # Simulate S S M == compose(b2, b1)
    b1, b2 = buffer[:2, :], buffer[2:4, :]
    stack_top = self._compose(b2, b1)
    return stack_top

  def _run_ff_5(self, batch_size, stack, buffer, queue, cursors, buffer_cursors):
    """
    Simulate a 5-step S S M S M sequence using the thin stack representation.
    """

    transitions_shift = tf.zeros_like(cursors)
    transitions_reduce = tf.ones_like(cursors)

    # Shift.
    _, _, b1, p1 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_shift, 0)
    updates = ts.thin_stack_update(b1, transitions_shift, stack, queue, cursors, buffer_cursors, 0)
    stack, queue, cursors, buffer_cursors = updates

    with tf.control_dependencies(updates):
      # Shift.
      _, _, b2, p2 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_shift, 1)
      updates = ts.thin_stack_update(b2, transitions_shift, stack, queue, cursors, buffer_cursors, 1)
      stack, queue, cursors, buffer_cursors = updates

      with tf.control_dependencies(updates):
        # Reduce.
        s3_1, s3_2, _, p3 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_reduce, 2)
        updates = ts.thin_stack_update(self._compose(s3_1, s3_2),
                                       transitions_reduce, stack, queue, cursors, buffer_cursors, 2)
        stack, queue, cursors, buffer_cursors = updates

        with tf.control_dependencies(updates):
          # Shift.
          _, _, b4, p4 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_shift, 3)
          updates = ts.thin_stack_update(b4, transitions_shift, stack, queue, cursors, buffer_cursors, 3)
          stack, queue, cursors, buffer_cursors = updates

          with tf.control_dependencies(updates):
            s5_1, s5_2, _, p5 = ts.thin_stack_lookup(stack, buffer, queue, cursors, buffer_cursors, transitions_reduce, 4)
            updates = ts.thin_stack_update(self._compose(s5_1, s5_2),
                                           transitions_reduce, stack, queue, cursors, buffer_cursors, 4)
            stack = updates[0]

            with tf.control_dependencies(updates):
              stack_top = stack[batch_size * 4:, :]

      return stack_top

  def _sim_ff_5(self, buffer):
    b1, b2, b3 = buffer[:2, :], buffer[2:4, :], buffer[4:6, :]
    stack_top = self._compose(b3, self._compose(b2, b1))
    return stack_top

  def testIntegrated3Step(self):
    self._testIntegrated(2, 5, 3, self._run_ff_3, self._sim_ff_3)

  def testIntegrated5Step(self):
    self._testIntegrated(2, 5, 5, self._run_ff_5, self._sim_ff_5)


class GpuIntegratedThinStackTest(IntegratedThinStackTest):
    use_gpu = True


if __name__ == "__main__":
  test.main()
