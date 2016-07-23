#define EIGEN_USE_THREADS // CPU-only impl

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
      const Tensor& stack1_grad = c->input(0);
      const Tensor& stack2_grad = c->input(1);
      const Tensor& buf_top_grad = c->input(2);

      const Tensor& stack2_ptrs = c->input(3);
      const Tensor& buffer_cursors = c->input(4);

      // NB: not acquiring lock here; it's okay; we're running backprop
      // sequentially anyway
      Tensor stack = c->mutable_input(5, true);
      Tensor buffer = c->mutable_input(6, true);

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

      // Pass stack1 gradient back onto stack. Simple copy.
      int32 start_row = (t - 1) * batch_size;
      stack.Slice(start_row, start_row + batch_size)->CopyFrom(stack1_grad, stack1_grad->shape());

      functor::ThinStackLookupGrad<Device> lookup_functor;
      lookup_functor(c, c->eigen_device<Device>(), t,
                     stack2_grad.matrix<float>(), buf_top_grad.matrix<float>(),
                     stack2_ptrs.flat<float>(), buffer_cursors.flat<float>(),
                     stack.matrix<float>(), buffer.matrix<float>());

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
                  typename TTypes<float>::ConstMatrix stack2_ptrs,
                  typename TTypes<float>::ConstMatrix buffer_cursors,
                  typename TTypes<float>::Matrix stack,
                  typename TTypes<float>::Matrix buffer) {

    // Zero out the stack and buffer if this is our last timestep.
    int32 batch_size = buffer_cursors.size();
    int32 num_timesteps = stack.dimension(0) / batch_size;
    if (t == num_timesteps - 1) {
      stack.setZero();
      buffer.setZero();
    }

    for (int32 i = 0; i < batch_size; i++) {
      float stack2_ptr = stack2_ptrs(i) * batch_size + i;
      stack(stack2_ptr) = stack2_grad(i);

      float buffer_ptr = buffer_cursors(i) * batch_size + i;
      buffer(buffer_ptr) = buf_top_grad(i);
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
    typename TTypes<float>::ConstMatrix stack2_ptrs,
    typename TTypes<float>::ConstMatrix buffer_cursors,
    typename TTypes<float>::Matrix stack,
    typename TTypes<float>::Matrix buffer);
extern template struct ThinStackLookupGrad<GPUDevice>;
} // namespace functor

REGISTER_KERNEL_BUILDER(Name("ThinStackLookupGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        ThinStackLookupGradOp<GPUDevice>);
#endif // GOOGLE_CUDA


} // namespace tensorflow
