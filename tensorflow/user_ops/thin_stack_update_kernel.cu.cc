#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <assert.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/user_ops/thin_stack_update_kernel.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;


__global__ void k_update(float* queue, float* cursors, const float* buffer_cursors,
                         float* buffer_cursors_out, const float* transitions,
                         int batch_size, float t) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size)
    return;

  float transition = transitions[idx];

  // cursors += transitions * -1 + (1 - transitions) * 1
  // ==== cursors += 1 - 2 * transitions
  float cursor = cursors[idx] + 1.0f - 2.0f * transition;
  cursors[idx] = cursor;

  // queue_idxs = max(0, cursors_next * batch_size + batch_range)
  // queue[queue_idxs] = t
  int queue_idx = ((int) cursor) * batch_size + idx;
  queue_idx = queue_idx < 0 ? 0 : queue_idx;
  queue[queue_idx] = t;

  // buffer_cursors += 1 - transitions
  buffer_cursors_out[idx] = buffer_cursors[idx] + 1.0f - transition;
}


namespace functor {
template <>
void ThinStackUpdate<GPUDevice>::operator()(
        OpKernelContext *c, const GPUDevice& d, int32 t,
        typename TTypes<float>::ConstMatrix input_val,
        typename TTypes<float>::ConstFlat transitions,
        typename TTypes<float>::ConstFlat buffer_cursors,
        typename TTypes<float>::Matrix stack_top,
        typename TTypes<float>::Flat queue,
        typename TTypes<float>::Flat cursors,
        typename TTypes<float>::Flat buffer_cursors_out) {

    const int32 batch_size = buffer_cursors.size();

    // Write in the new stack top.
    stack_top.device(d) = input_val;

    CudaLaunchConfig cfg = GetCudaLaunchConfig(batch_size, d);
    k_update<<<cfg.block_count, cfg.thread_per_block>>>(
        queue.data(), cursors.data(), buffer_cursors.data(),
        buffer_cursors_out.data(), transitions.data(),
        batch_size, (float) t);

}

} // namespace functor

template struct functor::ThinStackUpdate<GPUDevice>;

} // namespace tensorflow

#endif // GOOGLE_CUDA
