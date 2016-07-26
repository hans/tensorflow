#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/user_ops/thin_stack_lookup_grad_kernel.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;


// In vectorized form:
//
//     idxs = idxs + idx_scal_shift
//     idxs *= idx_scal_mul
//     idxs += idx_vec_shift_coeff * idx_vec_shift
//     idxs = min(0, idxs)
//     if idx_max > 0:
//        idxs = max(idx_max, idxs)
//     dst += src[idxs]
__global__ void k_scatter_add_shifted(const float* src, const float* idxs, float* dst,
                                      int N, int D, float idx_scal_shift,
                                      float idx_scal_mul, float idx_vec_shift_coeff,
                                      const float* idx_vec_shift, int idx_max) {
  for (int i0 = blockIdx.x; i0 < N; i0 += gridDim.x) {
    float fdst_idx = idxs[i0] + idx_scal_shift;
    fdst_idx *= idx_scal_mul;

    float shift = idx_vec_shift == NULL
      ? 0.0f : idx_vec_shift_coeff * idx_vec_shift[i0];

    int dst_idx = (int) (fdst_idx + shift);

    // idx_min = 0 always here
    dst_idx = dst_idx < 0 ? 0 : dst_idx;

    if (idx_max > 0)
      dst_idx = dst_idx < idx_max ? dst_idx : idx_max;

#if DEBUG
    printf("%d  %5f  %5f  %5f  %5f  %5f  %d  (+%d)\n", i0, idxs[i0], shift, idx_scal_mul, idx_scal_shift, fdst_idx, dst_idx, threadIdx.x);
#endif

    int dst_offset = dst_idx * D;
    int src_offset = i0 * D;
    for (int i1 = threadIdx.x; i1 < D; i1 += blockDim.x)
      CudaAtomicAdd(dst + dst_offset + i1, src[src_offset + i1]);
  }
}

static void scatter_add_shifted(const GPUDevice& d, const float* src,
                                const float* idxs, float* dst, int N, int D,
                                float idx_scal_shift, float idx_scal_mul,
                                float idx_vec_shift_coeff, const float* idx_vec_shift,
                                int idx_max) {
  int num_threads = std::min(D, d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor());
  int num_blocks = std::min(N, d.getNumCudaMultiProcessors());
  k_scatter_add_shifted<<<num_blocks, num_threads>>>(
      src, idxs, dst, N, D,
      idx_scal_shift, idx_scal_mul, idx_vec_shift_coeff, idx_vec_shift,
      idx_max);
}



static __global__ void k_fill_range(float *dst, float limit) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ((float) idx >= limit)
    return;
  dst[idx] = (float) idx;
}


namespace functor {
template <>
void ThinStackLookupGrad<GPUDevice>::operator()(
        OpKernelContext *c, const GPUDevice& d, int32 t,
        typename TTypes<float>::ConstMatrix stack2_grad,
        typename TTypes<float>::ConstMatrix buf_top_grad,
        typename TTypes<float>::ConstFlat stack2_ptrs,
        typename TTypes<float>::ConstFlat buffer_cursors,
        typename TTypes<float>::Matrix stack,
        typename TTypes<float>::Matrix buffer) {

  // Get useful shape constants from inputs.
  const int32 batch_size = buffer_cursors.size();
  const int32 num_timesteps = stack.dimension(0) / batch_size;
  const int32 buffer_size = buffer.dimension(0);
  const int32 model_dim = stack.dimension(1);
  const int32 embedding_dim = buffer.dimension(1);

  if (t == num_timesteps - 1) {
    cudaMemset(stack.data(), 0, stack.size());
    cudaMemset(buffer.data(), 0, buffer.size());
  }

  // Alloc helpers
  // TODO: do once?
  Tensor batch_range;
  OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                      TensorShape({batch_size}), &batch_range));
  TTypes<float>::Flat batch_range_d = batch_range.flat<float>();
  CudaLaunchConfig cfg = GetCudaLaunchConfig(batch_size, d);
  k_fill_range<<<cfg.block_count, cfg.thread_per_block>>>(
      batch_range_d.data(), batch_size);

  // stack[stack2_ptrs] += stack2_grad
  // TODO do we need to guard for invalid transitions?
  scatter_add_shifted(d, stack2_grad.data(), stack2_ptrs.data(), stack.data(),
                      batch_size, model_dim,
                      0.0f, (float) batch_size, 1.0f, batch_range_d.data(), -1);

  // buffer[min(buffer_cursors * batch_size + batch_range, buff_size - 1)] += buffer_top_grad
  int max_buff_idx = buffer_size - 1.0f;
  scatter_add_shifted(d, buf_top_grad.data(), buffer_cursors.data(), buffer.data(),
                      batch_size, embedding_dim,
                      0.0f, (float) batch_size, 1.0f, batch_range_d.data(),
                      max_buff_idx);

}

} // namespace functor

template struct functor::ThinStackLookupGrad<GPUDevice>;

} // namespace tensorflow

#endif // GOOGLE_CUDA
