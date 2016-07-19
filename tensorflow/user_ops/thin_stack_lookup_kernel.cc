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

    explicit ThinStackLookupOp(OpKernelConstruction* c) : OpKernel(c) {  };

    void Compute(OpKernelContext *c) override {
      const Tensor& stack = c->input(0);
      const Tensor& buffer = c->input(1);
      const Tensor& queue = c->input(2);
      const Tensor& cursors = c->input(3);
      const Tensor& buffer_cursors = c->input(4);

      const Tensor& t_ = c->input(5);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(t_.shape()),
            errors::InvalidArgument("t should be a scalar, but got shape ",
                                    t_.shape().DebugString()));
      const int32 t = internal::SubtleMustCopy(t_.scalar<int32>()());

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
};


namespace functor {

template <>
struct ThinStackLookup<CPUDevice> {

  void operator()(OpKernelContext *c, const Device& d,
                  int32 t,
                  typename TTypes<float>::ConstMatrix stack,
                  typename TTypes<float>::ConstMatrix buffer,
                  typename TTypes<float>::ConstFlat queue,
                  typename TTypes<float>::ConstFlat cursors,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack1,
                  typename TTypes<float>::Matrix stack2,
                  typename TTypes<float>::Flat stack2_ptrs) {

    // Get useful shape constants from inputs.
    const int32 batch_size = buffer_cursors.NumElements();
    const int32 model_dim = stack.dim_size(1);
    const int32 embedding_dim = buffer.dim_size(1);
    const int32 buffer_size = buffer.dim_size(0);

    // Alloc helpers
    // TODO: do once?
    Tensor batch_range;
    OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                        cursors.shape(), &batch_range));
    TTypes<float>::Flat batch_range_d = batch_range.flat<float>();
    for (int32 i = 0; i < batch_size; ++i)
      batch_range_d(i) = static_cast<float>(i);

    // stack1 can share data with stack (no destructive writes on either)
    int32 start_row = (static_cast<int32>(t) - 1) * batch_size;
    OP_REQUIRES(c, stack1_out->CopyFrom(stack.Slice(start_row, batch_size),
                                        stack1_out->shape()),
                errors::Unknown("Derp"));

    functor::FloatyGather<Device, float, float> gather_functor;

    // queue_ptrs = (cursors - 1) * batch_size + batch_range
    // stack2_ptrs = gather(queue, queue_ptrs)
    TTypes<float>::ConstFlat queue_ptrs = cursors.flat<float>() - 1.0f;//(cursors.flat<float>() - 1.0f) * ((float) batch_size) + batch_range_d;
    gather_functor(device, queue.matrix<float>(), cursors.flat<float>(), stack2_ptrs->matrix<float>());

    // stack2_ptrs = max(0, stack2_ptrs) * batch_size + batch_range
    stack2_ptrs->flat<float>().device(device) = stack2_ptrs->flat<float>().cwiseMax(0.0f) * ((float) batch_size) + batch_range_d;

    // stack2 = gather(stack, stack2_ptrs)
    gather_functor(device, stack.matrix<float>(), (const_cast<const Tensor&>(*stack2_ptrs)).flat<float>(), stack2_out->matrix<float>());

    // buff_idxs = (buff_cursors * batch_size) + batch_range
    // buff_idxs = max(0, min(buff_idxs, (buff_size * batch_size) - 1))
    // buffer_top = gather(buffer, buff_idxs)
    float max_buff_idx = buffer_size - 1.0f;
    gather_functor(device, buffer.matrix<float>(), (buffer_cursors.flat<float>() * ((float) batch_size) + batch_range_d).cwiseMin(max_buff_idx).cwiseMax(0.0f), buf_top_out->matrix<float>());

  }

};

} // namespace functor


REGISTER_KERNEL_BUILDER(Name("ThinStackLookup").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackLookupOp<CPUDevice>);


} // namespace tensorflow
