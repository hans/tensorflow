#ifndef TENSORFLOW_USER_OPS_THIN_STACK_UPDATE_KERNEL_H
#define TENSORFLOW_USER_OPS_THIN_STACK_UPDATE_KERNEL_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {
template <typename Device>
struct ThinStackUpdate {
  void operator()(OpKernelContext *c, const Device& d, int32 t,
                  typename TTypes<float>::ConstMatrix input_val,
                  typename TTypes<float>::ConstFlat transitions,
                  typename TTypes<float>::ConstFlat buffer_cursors,
                  typename TTypes<float>::Matrix stack_top,
                  typename TTypes<float>::Flat queue,
                  typename TTypes<float>::Flat cursors,
                  typename TTypes<float>::Flat buffer_cursors_out);
};

} // namespace functor

} // namespace tensorflow

#endif // TENSORFLOW_USER_OPS_THIN_STACK_UPDATE_KERNEL_H_
