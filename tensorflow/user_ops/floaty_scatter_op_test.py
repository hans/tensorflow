# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for floaty_ops.floaty_scatter"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.user_ops import floaty_ops


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


class ScatterTest(tf.test.TestCase):

  def _VariableRankTest(self, np_scatter, tf_scatter, vtype, itype, use_gpu,
                        repeat_indices=False):
    np.random.seed(8)
    with self.test_session(use_gpu=use_gpu):
      for indices_shape in (), (2,), (3, 7), (3, 4, 7):
        for extra_shape in (), (5,), (5, 9):
          # Generate random indices with no duplicates for easy numpy comparison
          size = np.prod(indices_shape, dtype=itype)
          first_dim = 3 * size
          indices = np.arange(first_dim)
          np.random.shuffle(indices)
          indices = indices[:size]
          if size > 1 and repeat_indices:
            # Add some random repeats.
            indices = indices[:size//2]
            for _ in range(size-size//2):
              # Randomly append some repeats.
              indices = np.append(indices, indices[np.random.randint(size//2)])
            np.random.shuffle(indices)
          indices = indices.reshape(indices_shape)
          updates = _AsType(np.random.randn(*(indices_shape + extra_shape)),
                            vtype)
          old = _AsType(np.random.randn(*((first_dim,) + extra_shape)), vtype)

          # Scatter via numpy
          new = old.copy()
          np_scatter(new, indices, updates)
          # Scatter via tensorflow
          ref = tf.Variable(old)
          ref.initializer.run()
          tf_scatter(ref, indices, updates).eval()
          # Compare
          self.assertAllClose(ref.eval(), new)

  def _VariableRankTests(self, np_scatter, tf_scatter):
    for vtype in (np.float32, np.float64):
      for use_gpu in (False, True):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, np.float32, use_gpu)

  def testVariableRankUpdate(self):
    def update(ref, indices, updates):
      ref[np.cast[np.int32](indices)] = updates
    self._VariableRankTests(update, floaty_ops.floaty_scatter_update)

  def testBooleanScatterUpdate(self):
    with self.test_session(use_gpu=False) as session:
      var = tf.Variable([True, False])
      update0 = floaty_ops.floaty_scatter_update(var, 1.0, True)
      update1 = floaty_ops.floaty_scatter_update(var, tf.constant(0, dtype=tf.float32), False)
      var.initializer.run()

      session.run([update0, update1])

      self.assertAllEqual([False, True], var.eval())

  def testScatterOutOfRangeCpu(self):
    params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    updates = np.array([-3, -4, -5]).astype(np.float32)
    with self.test_session(use_gpu=False):
      ref = tf.Variable(params)
      ref.initializer.run()

      # Indices all in range, no problem.
      indices = np.array([2, 0, 5], dtype=np.float32)
      floaty_ops.floaty_scatter_update(ref, indices, updates).eval()

      # Test some out of range errors.
      indices = np.array([-1, 0, 5], dtype=np.float32)
      with self.assertRaisesOpError(r'indices\[0\] = -1 is not in \[0, 6\)'):
        floaty_ops.floaty_scatter_update(ref, indices, updates).eval()

      indices = np.array([2, 0, 6], dtype=np.float32)
      with self.assertRaisesOpError(r'indices\[2\] = 6 is not in \[0, 6\)'):
        floaty_ops.floaty_scatter_update(ref, indices, updates).eval()


if __name__ == "__main__":
  tf.test.main()
