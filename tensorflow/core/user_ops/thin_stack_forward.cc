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

#define EIGEN_USE_THREADS

#include "tensorflow/core/user_ops/thin_stack_forward.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/kernels/matmul_op.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("ThinStackForward")
    .Input("buffer: float")
    .Input("W_compose: float")
    .Output("representations: float")
    .Attr("num_timesteps: int")
    .Doc(R"doc(
Run a thin-stack feedforward recurrence.
)doc");

class ThinStackForwardOp : public OpKernel {
  public:
    explicit ThinStackForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_timesteps", &num_timesteps_));
    }

    void Compute(OpKernelContext* ctx) override {
      // num_timesteps * batch_size * emb_dim
      const Tensor& buffer = ctx->input(0);

      int num_timesteps = buffer.dim_size(0);
      int batch_size = buffer.dim_size(1);
      int model_dim = buffer.dim_size(2);

      int stack_size = num_timesteps;

      // Allocate recurrence-level auxiliary data structures.
      TensorShape queue_shape({batch_size, num_timesteps});
      TensorShape cursors_shape({batch_size});
      Tensor queue, cursors, buffer_cursors;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             queue_shape, &queue));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             cursors_shape, &cursors));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             cursors_shape, &buffer_cursors));

      // Allocate step-level auxiliary data structures.
      TensorShape stack_ptrs_shape({batch_size});
      TensorShape stack_row_shape({batch_size, model_dim});
      Tensor stack_1_ptrs, stack_2_ptrs, stack_1, stack_2;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<int>::value,
                                             stack_ptrs_shape, &stack_1_ptrs));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<int>::value,
                                             stack_ptrs_shape, &stack_2_ptrs));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             stack_row_shape, &stack_1));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             stack_row_shape, &stack_2));


      // DEV: just try a matmul here
      Tensor *outt;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::value,
                                             stack_row_shape, outt));
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = 1;
      dim_pair[0].second = 0;
      functor::MatMulFunctor<CPUDevice, float>()(ctx->eigen_device<CPUDevice>(),
                                                 outt->matrix<float>(), stack_1.matrix<float>(),
                                                 stack_2.matrix<float>(), dim_pair);

      // Allocate stack / output.
      // TODO: will probably need to be flat during rollout for efficiency
      // num_timesteps * batch_size * emb_dim
      TensorShape out_shape({num_timesteps, batch_size, model_dim});
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    }

  private:
    int num_timesteps_;
};

REGISTER_KERNEL_BUILDER(Name("ThinStackForward").Device(DEVICE_CPU), ThinStackForwardOp);
