#define EIGEN_USE_THREADS // CPU-only impl

#include "tensorflow/user_ops/thin_stack_lookup_kernel.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/user_ops/floaty_gather_kernel.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device>
class ThinStackLookupOp : public OpKernel {

  public:

    explicit ThinStackLookupOp(OpKernelConstruction* c) : OpKernel(c) {
      OP_REQUIRES_OK(c, c->GetAttr("timestep", &t));
    }

    void Compute(OpKernelContext *c) override {
      const Tensor& stack = c->input(0);
      const Tensor& buffer = c->input(1);
      const Tensor& queue = c->input(2);
      const Tensor& cursors = c->input(3);
      const Tensor& buffer_cursors = c->input(4);

      // Allocate outputs.
      const int32 batch_size = buffer_cursors.NumElements();
      const int32 model_dim = stack.dim_size(1);
      const int32 embedding_dim = buffer.dim_size(1);

      TensorShape stack_elm_shape = TensorShape({batch_size, model_dim});
      TensorShape buffer_elm_shape = TensorShape({batch_size, embedding_dim});
      Tensor *stack1_out, *stack2_out, *buf_top_out, *stack2_ptrs;
      OP_REQUIRES_OK(c, c->allocate_output(0, stack_elm_shape, &stack1_out));
      OP_REQUIRES_OK(c, c->allocate_output(1, stack_elm_shape, &stack2_out));
      OP_REQUIRES_OK(c, c->allocate_output(2, buffer_elm_shape, &buf_top_out));
      OP_REQUIRES_OK(c, c->allocate_output(3, cursors.shape(), &stack2_ptrs));

      functor::ThinStackLookup<Device> lookup_functor;
      lookup_functor(c, c->eigen_device<Device>(), t,
                     stack.matrix<float>(), buffer.matrix<float>(), queue.flat<float>(),
                     cursors.flat<float>(), buffer_cursors.flat<float>(),
                     stack1_out->matrix<float>(), stack2_out->matrix<float>(),
                     buf_top_out->matrix<float>(), stack2_ptrs->flat<float>());

    }

  private:

    int t;
};


namespace functor {

template <>
struct ThinStackLookup<CPUDevice> {

  void operator()(OpKernelContext *c, const CPUDevice& d,
                  int32 t,
                  typename TTypes<float>::ConstMatrix stack,
                  typename TTypes<float>::ConstMatrix buffer,
                  typename TTypes<float>::ConstFlat queue,
                  typename TTypes<float>::ConstFlat cursors,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack1,
                  typename TTypes<float>::Matrix stack2,
                  typename TTypes<float>::Matrix buffer_top,
                  typename TTypes<float>::Flat stack2_ptrs) {

    // Get useful shape constants from inputs.
    const int32 batch_size = buffer_cursors.size();
    const int32 model_dim = stack.dimension(1);
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

    // Copy over stack top.
    int32 start_row = std::max((t - 1) * batch_size, 0);
    Eigen::array<int, 2> offsets = {start_row, 0};
    Eigen::array<int, 2> extents = {batch_size, model_dim};
    stack1 = stack.slice(offsets, extents);

    functor::FloatyGather<CPUDevice, float, float> gather_functor;

    // queue_ptrs = (cursors - 1) * batch_size + batch_range
    // stack2_ptrs = gather(queue, queue_ptrs)
    queue_ptrs_d.device(d) = (cursors - 1.0f) * ((float) batch_size) + batch_range_d;
    for (int32 i = 0; i < batch_size; i++)
      stack2_ptrs(i) = queue(queue_ptrs_d(i));

    // stack2_ptrs_shift = max(0, stack2_ptrs) * batch_size + batch_range
    // stack2 = gather(stack, stack2_ptrs_shift)
    // HACK: reuse queue_ptrs for this purpose
    queue_ptrs_d = stack2_ptrs.cwiseMax(0.0f) * ((float) batch_size) + batch_range_d;
    TTypes<float>::ConstFlat stack2_ptrs_shift(queue_ptrs_d.data(), queue_ptrs_d.dimensions());
    gather_functor(d, stack, stack2_ptrs_shift, stack2);

    // buffer_ptrs = (buff_cursors * batch_size) + batch_range
    // buffer_ptrs = max(0, min(buffer_ptrs, buff_size - 1))
    // buffer_top = gather(buffer, buff_idxs)
    float max_buff_idx = buffer_size - 1.0f;
    buffer_ptrs.flat<float>().device(d) = (buffer_cursors * ((float) batch_size) + batch_range_d).cwiseMin(max_buff_idx).cwiseMax(0.0f);
    gather_functor(d, buffer, const_cast<const Tensor&>(buffer_ptrs).flat<float>(), buffer_top);

  }

};

} // namespace functor


REGISTER_KERNEL_BUILDER(Name("ThinStackLookup").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackLookupOp<CPUDevice>);


#if GOOGLE_CUDA
// Forward declare the GPU functor
namespace functor {
template <>
void ThinStackLookup<GPUDevice>::operator()(
    OpKernelContext *c, const GPUDevice& d, int32 t,
    typename TTypes<float>::ConstMatrix stack,
    typename TTypes<float>::ConstMatrix buffer,
    typename TTypes<float>::ConstFlat queue,
    typename TTypes<float>::ConstFlat cursors,
    typename TTypes<float>::ConstFlat buffer_cursors,
    typename TTypes<float>::Matrix stack1,
    typename TTypes<float>::Matrix stack2,
    typename TTypes<float>::Matrix buffer_top,
    typename TTypes<float>::Flat stack2_ptrs);
extern template struct ThinStackLookup<GPUDevice>;
} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackLookup").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        ThinStackLookupOp<GPUDevice>);
#endif // GOOGLE_CUDA


} // namespace tensorflow
