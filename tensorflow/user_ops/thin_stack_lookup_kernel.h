#ifndef TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_KERNEL_H_
#define TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_KERNEL_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {
template <typename Device>
struct ThinStackLookup {
  void operator()(OpKernelContext *c, const Device& d, int32 t,
                  typename TTypes<float>::ConstMatrix stack,
                  typename TTypes<float>::ConstMatrix buffer,
                  typename TTypes<float>::ConstFlat queue,
                  typename TTypes<float>::ConstFlat cursors,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack1,
                  typename TTypes<float>::Matrix stack2,
                  typename TTypes<float>::Matrix buffer_top,
                  typename TTypes<float>::Flat stack2_ptrs);
};
} // namespace functor

} // namespace tensorflow

#endif // TENSORFLOW_USER_OPS_THIN_STACK_LOOKUP_KERNEL_H_
