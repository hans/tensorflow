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

    explicit ThinStackUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
      OP_REQUIRES_OK(c, c->GetAttr("timestep", &t));
    }

    void Compute(OpKernelContext *c) override {
      const Tensor& input_val = c->input(0);
      const Tensor& transitions = c->input(1);
      const Tensor& buffer_cursors = c->input(5);

      // NB: acquires write lock on data -- probably not necessary
      Tensor stack = c->mutable_input(2, false);
      Tensor queue = c->mutable_input(3, true);
      Tensor cursors = c->mutable_input(4, true);

      const int32 batch_size = buffer_cursors.NumElements();

      // Forward Ref outputs.
      c->forward_ref_input_to_ref_output(2, 0);
      c->forward_ref_input_to_ref_output(3, 1);
      c->forward_ref_input_to_ref_output(4, 2);

      // Allocate other outputs.
      Tensor *buffer_cursors_out;
      OP_REQUIRES_OK(c, c->allocate_output(3, buffer_cursors.shape(), &buffer_cursors_out));

      const Device& d = c->eigen_device<Device>();

      Tensor stack_top = stack.Slice(t * batch_size, t * batch_size + batch_size);

      functor::ThinStackUpdate<Device> update_functor;
      update_functor(c, d, t, input_val.matrix<float>(), transitions.flat<float>(),
                     buffer_cursors.flat<float>(),
                     stack_top.matrix<float>(), queue.flat<float>(), cursors.flat<float>(),
                     buffer_cursors_out->flat<float>());

    }

  private:

    int t;

};


namespace functor {

template <>
struct ThinStackUpdate<CPUDevice> {

  void operator()(OpKernelContext *c, const CPUDevice& d, int32 t,
                  typename TTypes<float>::ConstMatrix input_val,
                  typename TTypes<float>::ConstFlat transitions,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack_top,
                  typename TTypes<float>::Flat queue,
                  typename TTypes<float>::Flat cursors,
                  typename TTypes<float>::Flat buffer_cursors_out) {

    const int32 batch_size = buffer_cursors.size();

    // Write in stack top.
    stack_top.device(d) = input_val;

    // cursors = cursors + (transitions * -1 + (1 - transitions) * 1)
    // === cursors = cursors + 1 - 2 * transitions
    cursors.device(d) += 1.0f - 2.0f * transitions;

    // queue_idxs = max(0, cursors_next * batch_size + batch_range)
    // scatter_update(queue, queue_idxs, t)
    // TODO: either shoe this into scatter_update or write a custom functor / kernel
    for (int32 i = 0; i < batch_size; i++) {
      int32 cursor_i = (int32) cursors(i);
      int32 idx = std::max(0, cursor_i * batch_size + i);
      queue(idx) = static_cast<float>(t);
    }

    buffer_cursors_out.device(d) = buffer_cursors + 1.0f - transitions;

  }

};

} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackUpdate").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackUpdateOp<CPUDevice>);


#if GOOGLE_CUDA
// Forward declare the GPU functor
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
    typename TTypes<float>::Flat buffer_cursors_out);
extern template struct ThinStackUpdate<GPUDevice>;
} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackUpdate").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        ThinStackUpdateOp<GPUDevice>);
#endif // GOOGLE_CUDA

} // namespace tensorflow
