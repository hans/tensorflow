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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/user_ops/floaty_op_kernels.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index>
__global__ void FloatyGatherOpKernel(const T* params, const Index* indices, T* out,
                                     int64 first_dim_size, int64 indices_size,
                                     int64 out_size) {
  const int32 slice_size = out_size / indices_size;
  CUDA_1D_KERNEL_LOOP(i, out_size) {
    Index indices_i = i / slice_size;
    Index indices_slice_i = i - indices_i * slice_size;
    Index params_first_index = ldg(indices + indices_i);
    if (!(params_first_index >= 0 && params_first_index < first_dim_size)) {
      // Set indices out of range to zero
      // TODO(fpmc): Log an error for transfer back to host.
      out[i] = T(0);
    } else {
      Index params_i_ = params_first_index * slice_size + indices_slice_i;
      int params_i = (int) params_i_;
      out[i] = ldg(params + params_i);
    }
  }
}

namespace functor {
template <typename T, typename Index>
struct FloatyGather<GPUDevice, T, Index> {
  Index operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix Tparams,
                   typename TTypes<Index>::ConstFlat Tindices,
                   typename TTypes<T>::Matrix Tout) {
    const int64 first_dim_size = Tparams.dimension(0);
    const int64 indices_size = Tindices.size();
    const int64 out_size = Tout.size();
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    // clang-format off
    GatherOpKernel<T, Index>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            Tparams.data(), Tindices.data(), Tout.data(), first_dim_size,
            indices_size, out_size);
    // clang-format on
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};


////////////////////////////////////////////////////////////////////////////////


template <typename T, typename Index, floaty_op_kernels::UpdateOp op>
__global__ void FloatyScatterOpCustomKernel(
    T* params, const T* updates, const Index* indices,
    int first_dim_size, int updates_size, int indices_size) {
  int update_block = updates_size / indices_size;
  CUDA_1D_KERNEL_LOOP(i, updates_size) {
    int indices_i = i / update_block;
    int updates_i = i;
    int param_first_index = (int) indices[indices_i];
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int params_i = param_first_index * update_block + (i % update_block);
    switch (op) {
      case scatter_op::UpdateOp::ASSIGN: {
        params[params_i] = ldg(updates + updates_i);
        break;
      }
      case scatter_op::UpdateOp::ADD: {
        CudaAtomicAdd(params + params_i, ldg(updates + updates_i));
        break;
      }
      case scatter_op::UpdateOp::SUB: {
        CudaAtomicSub(params + params_i, ldg(updates + updates_i));
        break;
      }
    }
  }
}

namespace functor {
// Specialization for a GPU device.
template <typename T, typename Index, floaty_op_kernels::UpdateOp op>
struct FloatyScatterFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO: Implement indices range check.  The hardest part is with returning
    // a value after the range check, as we do not want to do device to host
    // memcpy during a stream.
    int first_dim_size = params.dimension(0);
    int indices_size = indices.size();
    int updates_size = updates.size();
    CudaLaunchConfig config = GetCudaLaunchConfig(updates_size, d);
    ScatterOpCustomKernel<T,Index,op>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            params.data(), updates.data(), indices.data(),
            first_dim_size, updates_size, indices_size);
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_OP(T, Index, op)                               \
  template struct functor::ScatterFunctor<GPUDevice, T, Index, op>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)                        \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN);  \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);     \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);

#define DEFINE_GPU_SPECS(T)                     \
  DEFINE_GPU_SPECS_INDEX(T, float);

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
// TODO: The following fails to compile.
// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX
#undef DEFINE_GPU_SPECS_OP




}  // namespace functor

#define DEFINE_GPU_SPECS_INDEX(T, Index) \
  template struct functor::Gather<GPUDevice, T, Index>

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, float);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace tensorflow

#endif // GOOGLE_CUDA
