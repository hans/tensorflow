

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("ThinStackPush")
    .Input("stack: float")
    .Input("push_value: float")
    .Input("t: int")
    .Doc(R("doc(
Run an inplace push on a stack representation.
)doc"));

class ThinStackPushOp : public OpKernel {
  public:
    explicit ThinStackPushOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

    void Compute(OpKernelContext* ctx) override {

    }
}
