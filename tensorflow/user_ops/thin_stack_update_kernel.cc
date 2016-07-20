#define EIGEN_USE_THREADS // CPU-only impl

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

      // NB: acquires write lock on data -- probably not necessary
      Tensor stack = c->mutable_input(3, false);
      Tensor queue = c->mutable_input(4, true);
      Tensor cursors = c->mutable_input(5, true);
      Tensor buffer_cursors = c->mutable_input(6, true);

      const Tensor& t_ = c->input(7);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(t_.shape()),
          errors::InvalidArgument("t should be a scalar, but got shape ",
                                  t_.shape().DebugString()));
      const int32 t = internal::SubtleMustCopy(t_.scalar<int32>()());

      const int32 batch_size = buffer_cursors.NumElements();

      // Forward Ref outputs.
      c->forward_ref_input_to_ref_output(3, 0);
      c->forward_ref_input_to_ref_output(4, 1);
      c->forward_ref_input_to_ref_output(5, 2);
      c->forward_ref_input_to_ref_output(6, 3);

      // Fetch address of memory into which next stack top will be written.
      Tensor stack_top = stack.Slice(t * batch_size, batch_size);

      functor::ThinStackUpdate<Device> update_functor;
      update_functor(c, c->eigen_device<Device>(), t,
                     shift_val.matrix<float>(), reduce_val.matrix<float>(),
                     transitions.flat<float>(),
                     stack_top.matrix<float>(), queue.flat<float>(), cursors.flat<float>(),
                     buffer_cursors.flat<float>());

    }

};


namespace functor {

template <>
struct ThinStackUpdate<CPUDevice> {

  void operator()(OpKernelContext *c, const CPUDevice& d, int32 t,
                  typename TTypes<float>::ConstMatrix shift_val,
                  typename TTypes<float>::ConstMatrix reduce_val,
                  typename TTypes<float>::ConstFlat transitions,
                  typename TTypes<float>::Matrix stack_top,
                  typename TTypes<float>::Flat queue,
                  typename TTypes<float>::Flat cursors,
                  typename TTypes<float>::Flat buffer_cursors) {

    const int32 batch_size = buffer_cursors.size();

    // Allocate temp storage
    /* Tensor queue_idxs, batch_range; */
    /* OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value, */
    /*                                     cursors.shape(), &queue_idxs)); */
    /* OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value, */
    /*                                     cursors.shape(), &batch_range)); */
    /* TTypes<float>::Flat batch_range_d = batch_range.flat<float>(); */
    /* for (int32 i = 0; i < batch_size; ++i) */
    /*   batch_range_d(i) = static_cast<float>(i); */

    // Mask shift_val and reduce_val, writing into the stack top
    // representation
    stack_top.device(d) = transitions * reduce_val + (1.0f - transitions) * shift_val;
    // TODO: check if tf.select would support this
    // TODO: bring over something like MaskedCAReduce to do this efficiently


    // Update auxiliary data.

    // cursors = cursors + (transitions * -1 + (1 - transitions) * 1)
    // === cursors = cursors + 1 - 2 * transitions
    cursors.device(d) += 1.0f - 2.0f * transitions;

    /* // Build broadcasted matrix from scalar `t` */
    /* Eigen::IndexList<Eigen::type2index<1>, int> broadcast; */
    /* broadcast.set(0, batch_size); */

    // queue_idxs = max(0, cursors_next * batch_size + batch_range)
    // scatter_update(queue, queue_idxs, t)
    // TODO: either shoe this into scatter_update or write a custom functor / kernel
    for (int32 i = 0; i < batch_size; i++) {
      int32 cursor_i = (int32) cursors(i);
      int32 idx = std::max(0, cursor_i * batch_size + i);
      queue(idx) = static_cast<float>(t);
    }

    /* functor::FloatyScatterFunctor<Device, float, float, floaty_scatter_kernel::UpdateOp::ASSIGN> f_scatter; */
    /* auto t_broadcast = batch_range_d.constant((float) t).broadcast(broadcast); */
    /* f_scatter(c, c->eigen_device<Device>(), queue.matrix<float>(), t_broadcast, queue_idxs.flat<float>); */

    buffer_cursors.device(d) += 1.0f - transitions;
  }

};

} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackUpdate").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackUpdateOp<CPUDevice>);

} // namespace tensorflow
