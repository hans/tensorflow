# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Positive definite Operator defined with diagonal covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import operator_pd
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


class OperatorPDSqrtDiag(operator_pd.OperatorPDBase):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}` defined by their their square root,
  `S`, such that `A = SS^T`.

  In this case, `S` is diagonal and is defined by a provided tensor `diag`,
  `S_{ii} = diag[i]`.  As a result, `A` is diagonal with `A_{ii} = diag[i]**2`.

  Determinants, solves, and storage are `O(k)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices designate a
  batch member.  For every batch member `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  a `k x k` matrix.

  For example,

  ```python
  distributions = tf.contrib.distributions
  diag = [1.0, 2.0]
  operator = OperatorPDSqrtDiag(diag)
  operator.det()  # ==> (1 * 2)**2

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = [1.0, 2.0]
  operator.inv_quadratic_form_on_vectors(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = [[1.0], [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class
  `MultivariateNormalDiag`.
  """

  def __init__(self, diag, verify_pd=True, name='OperatorPDSqrtDiag'):
    """Initialize an OperatorPDSqrtDiag.

    Args:
      diag:  Shape `[N1,...,Nn, k]` positive tensor with `n >= 0`, `k >= 1`.
      verify_pd: Whether to check `diag` is positive.
      name:  A name to prepend to all ops created by this class.
    """
    self._verify_pd = verify_pd
    self._name = name
    with ops.name_scope(name):
      with ops.op_scope([diag], 'init'):
        self._diag = self._check_diag(diag)

  def _check_diag(self, diag):
    """Verify that `diag` is positive."""
    diag = ops.convert_to_tensor(diag, name='diag')
    if not self.verify_pd:
      return diag
    deps = [check_ops.assert_positive(diag)]
    return control_flow_ops.with_dependencies(deps, diag)

  @property
  def name(self):
    """String name identifying this `Operator`."""
    return self._name

  @property
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    return self._verify_pd

  @property
  def dtype(self):
    """Data type of matrix elements of `A`."""
    return self._diag.dtype

  def _batch_log_det(self):
    return 2 * math_ops.reduce_sum(
        math_ops.log(self._diag), reduction_indices=[-1])

  @property
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    return [self._diag]

  def _inv_quadratic_form_on_vectors(self, x):
    # This Operator is defined in terms of diagonal entries of the sqrt.
    return self._iqfov_via_sqrt_solve(x)

  def get_shape(self):
    """`TensorShape` giving static shape."""
    d_shape = self._diag.get_shape()
    return d_shape.concatenate(d_shape[-1:])

  def _shape(self):
    d_shape = array_ops.shape(self._diag)
    k = array_ops.gather(d_shape, array_ops.size(d_shape) - 1)
    return array_ops.concat(0, (d_shape, [k]))

  def _batch_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.batch_matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return math_ops.square(diag_mat) * x

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.batch_matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return diag_mat * x

  def _batch_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / math_ops.square(diag_mat)

  def _batch_sqrt_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / diag_mat

  def _to_dense(self):
    return array_ops.batch_matrix_diag(math_ops.square(self._diag))

  def _sqrt_to_dense(self):
    return array_ops.batch_matrix_diag(self._diag)

  def _add_to_tensor(self, mat):
    mat_diag = array_ops.batch_matrix_diag_part(mat)
    new_diag = math_ops.square(self._diag) + mat_diag
    return array_ops.batch_matrix_set_diag(mat, new_diag)
