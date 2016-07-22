#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/user_ops/thin_stack_lookup_kernel.h"

#include "tensorflow/user_ops/floaty_gather_kernel.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;


// In vectorized form:
//
//     idxs = idxs + idx_scal_shift
//     if idx_min >= 0:
//         idxs = min(idx_min, idxs)
//     else:
//         idxs = wraparound(idxs)
//     idxs *= idx_scal_mul
//     idxs += idx_vec_shift_coeff * idx_vec_shift
//     if idx_max >= 0:
//        idxs = max(idx_max, idxs)
//     out = src[idxs]
__global__ void k_gather_shifted(const float* src, const float* idxs, float* dst,
                                 int src_N, int N, int D, float idx_scal_shift,
                                 float idx_scal_mul, float idx_vec_shift_coeff,
                                 float* idx_vec_shift, int idx_min, int idx_max) {
  for (int i0 = blockIdx.x; i0 < N; i0 += gridDim.x) {
    float fsrc_idx = idxs[i0] + idx_scal_shift;
    fsrc_idx *= idx_scal_mul;

    float shift = idx_vec_shift == NULL
        ? 0.0f : idx_vec_shift_coeff * idx_vec_shift[i0];

    int src_idx = (int) (fsrc_idx + shift);

    if (idx_min >= 0) {
      src_idx = src_idx < idx_min ? idx_min : src_idx;
    } else if (src_idx < 0) {
      // Negative index. Read from other end of the source matrix.
      src_idx += src_N;
    }

    if (idx_max >= 0)
      src_idx = src_idx < idx_max ? src_idx : idx_max;

#if DEBUG
    printf("%d  %5f  %5f  %5f  %5f  %5f  %d\n", i0, idxs[i0], shift, idx_scal_mul, idx_scal_shift, fsrc_idx, src_idx);
#endif

    int src_offset = src_idx * D;
    int dst_offset = i0 * D;
    for (int i1 = threadIdx.x; i1 < D; i1 += blockDim.x)
      dst[dst_offset + i1] = src[src_offset + i1];
  }
}

static void gather_shifted(const GPUDevice& d,
                           const float* src, const float* idxs, float* dst,
                           int src_N, int N, int D, float idx_scal_shift,
                           float idx_scal_mul, float idx_vec_shift_coeff,
                           float* idx_vec_shift, int idx_min, int idx_max) {
  int num_threads = std::min(D, d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor());
  int num_blocks = std::min(N, d.getNumCudaMultiProcessors());
  // TODO use GetCudaLaunchConfig
  k_gather_shifted<<<num_blocks, num_threads>>>(src, idxs, dst, src_N, N, D,
                                                idx_scal_shift, idx_scal_mul,
                                                idx_vec_shift_coeff, idx_vec_shift,
                                                idx_min, idx_max);
}


__global__ void k_fill_range(float *dst, float limit) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ((float) idx >= limit)
    return;
  dst[idx] = (float) idx;
}


namespace functor {
template <>
void ThinStackLookup<GPUDevice>::operator()(
        OpKernelContext *c, const GPUDevice& d, int32 t,
        typename TTypes<float>::ConstMatrix stack,
        typename TTypes<float>::ConstMatrix buffer,
        typename TTypes<float>::ConstFlat queue,
        typename TTypes<float>::ConstFlat cursors,
        typename TTypes<float>::ConstFlat buffer_cursors,
        typename TTypes<float>::Matrix stack2,
        typename TTypes<float>::Matrix buffer_top,
        typename TTypes<float>::Flat stack2_ptrs) {

  // Get useful shape constants from inputs.
  const int32 batch_size = buffer_cursors.size();
  const int32 buffer_size = buffer.dimension(0);
  const int32 model_dim = stack.dimension(1);
  const int32 embedding_dim = buffer.dimension(1);

  // Alloc helpers
  // TODO: do once?
  Tensor batch_range, queue_ptrs, buffer_ptrs;
  OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                      TensorShape({batch_size}), &batch_range));
  OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                      TensorShape({batch_size}), &queue_ptrs));
  OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                      TensorShape({batch_size}), &buffer_ptrs));
  TTypes<float>::Flat batch_range_d = batch_range.flat<float>();
  TTypes<float>::Flat queue_ptrs_d = queue_ptrs.flat<float>();
  CudaLaunchConfig cfg = GetCudaLaunchConfig(batch_size, d);
  k_fill_range<<<cfg.block_count, cfg.thread_per_block>>>(
      batch_range_d.data(), batch_size);

  // queue_ptrs = (cursors - 1) * batch_size + batch_range
  // stack2_ptrs = gather(queue, queue_ptrs)
  // TODO could specialize kernel for this flat-gather case
  gather_shifted(d, queue.data(), cursors.data(), stack2_ptrs.data(),
                 queue.dimension(0), batch_size, 1,
                 -1.0f, (float) batch_size, 1.0f, batch_range_d.data(), -1, -1);

  // stack2_ptrs = max(0, stack2_ptrs) * batch_size + batch_range
  // stack2 = gather(stack2, stack2_ptrs)
  gather_shifted(d, stack.data(), stack2_ptrs.data(), stack2.data(),
                 stack.dimension(0), batch_size, model_dim,
                 0.0f, (float) batch_size, 1.0f, batch_range_d.data(), 0, -1);

  // buffer_ptrs = (buff_cursors * batch_size) + batch_range
  // buffer_ptrs = max(0, min(buffer_ptrs, buff_size - 1))
  // buffer_top = gather(buffer, buff_idxs)
  int max_buff_idx = buffer_size - 1.0;
  gather_shifted(d, buffer.data(), buffer_cursors.data(), buffer_top.data(),
                 buffer.dimension(0), batch_size, embedding_dim,
                 0.0f, (float) batch_size, 1.0f, batch_range_d.data(), 0, max_buff_idx);

}

} // namespace functor

template struct functor::ThinStackLookup<GPUDevice>;

} // namespace tensorflow

#endif // GOOGLE_CUDA
