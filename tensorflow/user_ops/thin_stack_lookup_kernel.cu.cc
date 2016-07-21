#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/user_ops/thin_stack_lookup_kernel.h"

#include "tensorflow/user_ops/floaty_gather_kernel.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <>
struct ThinStackLookup<GPUDevice> {

  void operator()(OpKernelContext *c, const GPUDevice& d, int32 t,
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
    for (int32 i = 0; i < batch_size; ++i)
      batch_range_d(i) = static_cast<float>(i);

    functor::FloatyGather<GPUDevice, float, float> gather_functor;

    // queue_ptrs = (cursors - 1) * batch_size + batch_range
    // stack2_ptrs = gather(queue, queue_ptrs)
    // stack2_ptrs = max(0, stack2_ptrs) * batch_size + batch_range
    queue_ptrs_d.device(d) = (cursors - 1.0f) * ((float) batch_size) + batch_range_d;
    for (int32 i = 0; i < batch_size; i++) {
      float queue_ptr = (cursors(i) - 1.0f) * ((float) batch_size) + ((float) i);
      float stack2_ptr_i = std::max(0.0f, queue(queue_ptr)) * ((float) batch_size) + ((float) i);
      stack2_ptrs(i) = stack2_ptr_i;
    }

    // stack2 = gather(stack, stack2_ptrs)
    TTypes<float>::ConstFlat stack2_ptrs_f(stack2_ptrs.data(), stack2_ptrs.dimensions());
    gather_functor(d, stack, stack2_ptrs_f, stack2);

    // buffer_ptrs = (buff_cursors * batch_size) + batch_range
    // buffer_ptrs = max(0, min(buffer_ptrs, buff_size - 1))
    // buffer_top = gather(buffer, buff_idxs)
    float max_buff_idx = buffer_size - 1.0f;
    buffer_ptrs.flat<float>().device(d) = (buffer_cursors * ((float) batch_size) + batch_range_d).cwiseMin(max_buff_idx).cwiseMax(0.0f);
    gather_functor(d, buffer, const_cast<const Tensor&>(buffer_ptrs).flat<float>(), buffer_top);

  }

};
} // namespace functor

template struct functor::ThinStackLookup<GPUDevice>;

} // namespace tensorflow

#endif // GOOGLE_CUDA
