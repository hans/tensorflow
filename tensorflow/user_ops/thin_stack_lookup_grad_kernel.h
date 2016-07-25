#ifndef TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_GRAD_KERNEL_H_
#define TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_GRAD_KERNEL_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {
template <typename Device>
struct ThinStackLookupGrad {
  void operator()(OpKernelContext *c, const Device& d, int32 t,
                  typename TTypes<float>::ConstMatrix stack2_grad,
                  typename TTypes<float>::ConstMatrix buf_top_grad,
                  typename TTypes<float>::ConstFlat stack2_ptrs,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack,
                  typename TTypes<float>::Matrix buffer);
};
} // namespace functor

} // namespace tensorflow

#endif // TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_GRAD_KERNEL_H_
