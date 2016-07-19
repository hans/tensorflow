#define EIGEN_USE_THREADS
// this prevents build errors; don't know why?

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
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
      Tensor stack_top, queue_idxs, batch_range;
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         shift_val.shape(), &stack_top));
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &queue_idxs));
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         cursors.shape(), &batch_range));
      TTypes<float>::Flat batch_range_d = batch_range.flat<float>();
      for (int32 i = 0; i < batch_size; ++i)
        batch_range_d(i) = static_cast<float>(i);

      // TODO: perform masking

      Tensor stack_target_row = stack.Slice(t * batch_size, batch_size);
      OP_REQUIRES(c, stack_target_row.CopyFrom(stack_top, stack_top.shape()),
          errors::Internal("Something is very broken"));

      const Device& device = c->eigen_device<Device>();

      // Update auxiliary data.

      // cursors = cursors + (transitions * -1 + (1 - transitions) * 1)
      // === cursors = cursors + 1 - 2 * transitions
      cursors_out->flat<float>().device(device) = cursors.flat<float>() + 1.0f - 2.0f * transitions.flat<float>();

      // queue_idxs = max(0, cursors_next * batch_size + batch_range)
      queue_idxs.flat<float>().device(device) = cursors_out->flat<float>() * ((float) batch_size) + batch_range_d;

      // Build broadcasted matrix from scalar `t`
      Eigen::IndexList<Eigen::type2index<1>, int> broadcast;
      broadcast.set(0, batch_size);

      // TODO: copy over queue into queue_out

      // scatter_update(queue, queue_idxs, t)
      // TODO: either shoe this into scatter_update or write a custom functor / kernel
      TTypes<float>::Flat queue_d = queue_out->flat<float>();
      const TTypes<float>::Flat& queue_idxs_d = queue_idxs.flat<float>();
      for (int32 i = 0; i < batch_size; i++)
        queue_d(queue_idxs_d(i)) = static_cast<float>(t);

      /* functor::FloatyScatterFunctor<Device, float, float, floaty_scatter_kernel::UpdateOp::ASSIGN> f_scatter; */
      /* auto t_broadcast = batch_range_d.constant((float) t).broadcast(broadcast); */
      /* f_scatter(c, c->eigen_device<Device>(), queue.matrix<float>(), t_broadcast, queue_idxs.flat<float>); */

      // TODO buffer_cursors_out
    }

};


REGISTER_KERNEL_BUILDER(Name("ThinStackUpdate").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackUpdateOp<CPUDevice>);

} // namespace tensorflow
