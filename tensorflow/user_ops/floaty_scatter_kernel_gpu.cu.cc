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

#include "tensorflow/user_ops/floaty_scatter_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, floaty_scatter_kernel::UpdateOp op>
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
      case floaty_scatter_kernel::UpdateOp::ASSIGN: {
        params[params_i] = ldg(updates + updates_i);
        break;
      }
    }
  }
}

namespace functor {
// Specialization for a GPU device.
template <typename T, typename Index, floaty_scatter_kernel::UpdateOp op>
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
    FloatyScatterOpCustomKernel<T,Index,op>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            params.data(), updates.data(), indices.data(),
            first_dim_size, updates_size, indices_size);
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_OP(T, Index, op)                               \
  template struct functor::FloatyScatterFunctor<GPUDevice, T, Index, op>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)                        \
  DEFINE_GPU_SPECS_OP(T, Index, floaty_scatter_kernel::UpdateOp::ASSIGN);

#define DEFINE_GPU_SPECS(T)                     \
  DEFINE_GPU_SPECS_INDEX(T, float);

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
// TODO: The following fails to compile.
// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX
#undef DEFINE_GPU_SPECS_OP

}  // namespace tensorflow

#endif // GOOGLE_CUDA
