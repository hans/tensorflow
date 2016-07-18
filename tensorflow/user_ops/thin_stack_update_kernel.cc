#include "tensorflow/user_ops/thin_stack_update_kernel.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "tensorflow/user_ops/floaty_gather_kernel.h"
#include "tensorflow/user_ops/floaty_scatter_kernel.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device>
class ThinStackUpdateOp : public OpKernel {

  public:

    explicit ThinStackUpdateOp(OpKernelConstruction* c) : OpKernel(c) {  };

    void Compute(OpKernelContext *c) override {
      const Tensor& shift_val = c->input(0);
      const Tensor& reduce_val = c->input(1);
      const Tensor& transitions = c->input(2);

      // NB: acquires write lock on stack
      Tensor stack = c->mutable_input(3, false);
      const Tensor& queue = c->input(4);
      const Tensor& cursors = c->input(5);
      const Tensor& buffer_cursors = c->input(6);

      const Tensor& t_ = c->input(7);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(t_.shape()),
          errors::InvalidArgument("t should be a scalar, but got shape ",
                                  t_.shape().DebugString()));
      const int32 t = internal::SubtleMustCopy(t_.scalar<int32>()());

      // Useful shape constants.
      const int32 batch_size = buffer_cursors.NumElements();

      // Allocate outputs.
      c->forward_ref_input_to_ref_output(0, 0);
      Tensor *queue_out, *cursors_out, *buffer_cursors_out;
      OP_REQUIRES_OK(c, c->allocate_output(1, queue.shape(), &queue_out));
      OP_REQUIRES_OK(c, c->allocate_output(2, cursors.shape(), &cursors_out));
      OP_REQUIRES_OK(c, c->allocate_output(3, buffer_cursors.shape(), &buffer_cursors_out));

      // Allocate temp storage: masked shift_value, reduce_value
      Tensor stack_top;
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         shift_val.shape(), &stack_top));

      // TODO: perform masking

      Tensor stack_target_row = stack.Slice(t * batch_size, batch_size);
      OP_REQUIRES(c, stack_target_row.CopyFrom(stack_top, stack_top.shape()),
          errors::Internal("Something is very broken"));

      // TODO: update cursors
    }

};


REGISTER_KERNEL_BUILDER(Name("ThinStackUpdate").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackUpdateOp<CPUDevice>);

} // namespace tensorflow
