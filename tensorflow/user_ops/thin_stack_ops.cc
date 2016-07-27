#include "tensorflow/core/framework/op.h"


namespace tensorflow {


REGISTER_OP("ThinStackLookup")
    .Input("stack: T")
    .Input("buffer: T")
    .Input("queue: float")
    .Input("cursors: float")
    .Input("buffer_cursors: float")
    .Input("transitions: float")
    .Output("stack1: T")
    .Output("stack2: T")
    .Output("buf_top: T")
    .Output("stack2_ptrs: float")
    .Attr("timestep: int")
    .Attr("T: {float}")
    .Doc(R"doc(
Perform thin-stack lookup operation.
)doc");


REGISTER_OP("ThinStackLookupGrad")
    .Input("stack: Ref(T)")
    .Input("buffer: Ref(T)")
    .Input("stack2_ptrs: float")
    .Input("buffer_cursors: float")
    .Input("stack1_grad: float")
    .Input("stack2_grad: float")
    .Input("buf_top_grad: float")
    .Input("transitions: float")
    .Output("stack_out: Ref(T)")
    .Output("buffer_out: Ref(T)")
    .Attr("timestep: int")
    .Attr("T: {float}")
    .Doc(R"doc(
Backward pass of ThinStackLookup op. Writes inplace onto a stack, which will be
automatically zeroed out if we are at the last timestep.
)doc");


REGISTER_OP("ThinStackUpdate")
    .Input("input_val: T")
    .Input("transitions: float")
    .Input("stack: Ref(T)")
    .Input("queue: Ref(float)")
    .Input("cursors: Ref(float)")
    .Input("buffer_cursors: float")
    .Output("stack_out: Ref(T)")
    .Output("queue_out: Ref(float)")
    .Output("cursors_out: Ref(float)")
    .Output("buffer_cursors_out: float")
    .Attr("timestep: int")
    .Attr("T: {float}")
    .Doc(R"doc(
Perform inplace update on thin-stack representation.
)doc");


}
