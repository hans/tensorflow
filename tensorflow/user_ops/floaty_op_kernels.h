/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_USER_OPS_FLOATY_OP_KERNELS_H_
#define TENSORFLOW_USER_OPS_FLOATY_OP_KERNELS_H_
// Functor definition for GatherOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

class OpKernelContext;

namespace floaty_op_kernels {

enum class UpdateOp { ASSIGN, ADD, SUB };

}  // namespace floaty_op_kernels

namespace functor {
template <typename Device, typename T, typename Index>
struct FloatyGather {
  // Performs gather op on (Tparams, Tindices), writing to Tout.
  // Returns an index to Tindices if the value at that index is out of range.
  // Returns -1 if all values of Tindices are in range.
  Index operator()(const Device& d, typename TTypes<T>::ConstMatrix Tparams,
                   typename TTypes<Index>::ConstFlat Tindices,
                   typename TTypes<T>::Matrix Tout);
};

// Functor used by ScatterOp to do the computations.
template <typename Device, typename T, typename Index, floaty_op_kernels::UpdateOp op>
struct FloatyScatterFunctor {
  // Returns -1 on success or a nonnegative i s.t. indices[i] is a bad index.
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USER_OPS_FLOATY_OP_KERNELS_H_
