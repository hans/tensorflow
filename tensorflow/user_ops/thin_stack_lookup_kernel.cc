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

      // Get useful shape constants from inputs.
      const int32 batch_size = buffer_cursors.NumElements();
      const int32 model_dim = stack.dim_size(1);
      const int32 embedding_dim = buffer.dim_size(1);

      // Allocate outputs.
      TensorShape stack_elm_shape = TensorShape({batch_size, model_dim});
      TensorShape buffer_elm_shape = TensorShape({batch_size, embedding_dim});
      Tensor *stack1_out, *stack2_out, *buf_top_out;
      OP_REQUIRES_OK(c, c->allocate_output(0, stack_elm_shape, &stack1_out));
      OP_REQUIRES_OK(c, c->allocate_output(1, stack_elm_shape, &stack2_out));
      OP_REQUIRES_OK(c, c->allocate_output(2, buffer_elm_shape, &buf_top_out));

      // Prepare lookup indices.
      Tensor stack2_ptrs, buf_ptrs;
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &stack2_ptrs));
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &buf_ptrs));

      // Alloc helpers
      // TODO: do once?
      Tensor batch_range, batch_ones;
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &batch_range));
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &batch_ones));
      TTypes<float>::Flat batch_range_d = batch_range.flat<float>();
      for (int32 i = 0; i < batch_size; ++i)
        batch_range_d(i) = static_cast<float>(i);

      auto batch_ones_d = batch_ones.flat<float>();
      // TODO why is this breaking?
      // batch_ones_d.device(c->eigen_device<Device>()) = batch_ones_d.constant(1.0);

      // Copy over stack1 data directly -- it's contiguous
      // TODO can we just return a pointer onto stack?
      // See e.g. CopyFromInternal usage in framework/tensor.h
      int32 start_row = (static_cast<int32>(t) - 1) * batch_size;
      stack1_out->UnsafeCopyFromInternal(stack.Slice(start_row, batch_size),
                                         stack1_out->shape());

      // gather stack2 pointers from queue
      // queue_ptrs = (cursors - 1) * batch_size + batch_range
      functor::FloatyGather<Device, float, float> gather_functor;
      // TODO run shift
      gather_functor(c->eigen_device<Device>(), queue.matrix<float>(), cursors.flat<float>(), stack2_ptrs.matrix<float>());
      // TODO max(0, stack2_ptrs) * batch_size + batch_range
      // gather stack2 data
      // TODO run shift
      gather_functor(c->eigen_device<Device>(), stack.matrix<float>(), (const_cast<const Tensor&>(stack2_ptrs)).flat<float>(), stack2_out->matrix<float>());

      // Run buffer lookup
      // TODO run shift
      gather_functor(c->eigen_device<Device>(), buffer.matrix<float>(), buffer_cursors.flat<float>(), buf_top_out->matrix<float>());
    }

};


REGISTER_KERNEL_BUILDER(Name("ThinStackLookup").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackLookupOp<CPUDevice>);


} // namespace tensorflow
