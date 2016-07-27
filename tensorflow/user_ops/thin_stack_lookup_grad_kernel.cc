#define EIGEN_USE_THREADS // CPU-only impl

#include <iostream>
#include "tensorflow/user_ops/thin_stack_lookup_grad_kernel.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/user_ops/floaty_scatter_kernel.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device>
class ThinStackLookupGradOp : public OpKernel {

  public:

    explicit ThinStackLookupGradOp(OpKernelConstruction* c) : OpKernel(c) {
      OP_REQUIRES_OK(c, c->GetAttr("timestep", &t));
    }

    void Compute(OpKernelContext *c) override {
      const Tensor& stack2_ptrs = c->input(2);
      const Tensor& stack1_grad = c->input(4);
      const Tensor& stack2_grad = c->input(5);
      const Tensor& buf_top_grad = c->input(6);
      const Tensor& transitions = c->input(7);

      // NB: not acquiring lock here; it's okay; we're running backprop
      // sequentially anyway
      Tensor stack = c->mutable_input(0, false);
      Tensor buffer = c->mutable_input(1, true);
      Tensor buffer_cursors = c->mutable_input(3, false);

      // Forward Ref outputs.
      c->forward_ref_input_to_ref_output(0, 0);
      c->forward_ref_input_to_ref_output(1, 1);
      c->forward_ref_input_to_ref_output(3, 2);

      // Allocate outputs.
      const int32 batch_size = buffer_cursors.NumElements();

      // Pass stack1 gradient back onto stack. Simple copy.
      if (t >= 1) {
        int32 start_row = (t - 1) * batch_size;
        stack.Slice(start_row, start_row + batch_size).CopyFrom(stack1_grad, stack1_grad.shape());
      }

      functor::ThinStackLookupGrad<Device> lookup_functor;
      lookup_functor(c, c->eigen_device<Device>(), t,
                     stack2_grad.matrix<float>(), buf_top_grad.matrix<float>(),
                     stack2_ptrs.flat<float>(), transitions.flat<float>(),
                     buffer_cursors.flat<float>(), stack.matrix<float>(),
                     buffer.matrix<float>());

    }

  private:

    int t;
};


namespace functor {

template <>
struct ThinStackLookupGrad<CPUDevice> {

  void operator()(OpKernelContext *c, const CPUDevice& d,
                  int32 t,
                  typename TTypes<float>::ConstMatrix stack2_grad,
                  typename TTypes<float>::ConstMatrix buf_top_grad,
                  typename TTypes<float>::ConstFlat stack2_ptrs,
                  typename TTypes<float>::ConstFlat transitions,
                  typename TTypes<float>::Flat buffer_cursors,
                  typename TTypes<float>::Matrix stack,
                  typename TTypes<float>::Matrix buffer) {

    // Zero out the stack and buffer if this is our last timestep.
    int32 batch_size = buffer_cursors.size();
    int32 num_timesteps = stack.dimension(0) / batch_size;
    int32 buffer_size = buffer.dimension(0);
    if (t == num_timesteps - 1) {
      stack.setZero();
      buffer.setZero();
    }

    // Rewind buffer cursors based on transitions.
    buffer_cursors += -1.0f + transitions;

    for (int32 i = 0; i < batch_size; i++) {
      float stack2_ptr = stack2_ptrs(i) * batch_size + i;
      stack.chip(stack2_ptr, 0) += stack2_grad.chip(i, 0);

      float buffer_ptr = buffer_cursors(i) * batch_size + i;
      if (buffer_ptr >= buffer_size)
        continue;
      std::cout << "buffer_ptr at (" << t << ", " << i << "): " << buffer_ptr << std::endl;
      buffer.chip(buffer_ptr, 0) += buf_top_grad.chip(i, 0);
    }

  }

};

} // namespace functor


REGISTER_KERNEL_BUILDER(Name("ThinStackLookupGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ThinStackLookupGradOp<CPUDevice>);


#if GOOGLE_CUDA
// Forward declare the GPU functor
namespace functor {
template <>
void ThinStackLookupGrad<GPUDevice>::operator()(
    OpKernelContext *c, const GPUDevice& d, int32 t,
    typename TTypes<float>::ConstMatrix stack2_grad,
    typename TTypes<float>::ConstMatrix buf_top_grad,
    typename TTypes<float>::ConstFlat stack2_ptrs,
    typename TTypes<float>::ConstFlat transitions,
    typename TTypes<float>::Flat buffer_cursors,
    typename TTypes<float>::Matrix stack,
    typename TTypes<float>::Matrix buffer);
extern template struct ThinStackLookupGrad<GPUDevice>;
} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackLookupGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        ThinStackLookupGradOp<GPUDevice>);
#endif // GOOGLE_CUDA


} // namespace tensorflow
