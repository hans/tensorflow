#include "tensorflow/core/framework/op.h"


namespace tensorflow {


REGISTER_OP("ThinStackLookup")
    .Input("stack: T")
    .Input("buffer: T")
    .Input("queue: float")
    .Input("cursors: float")
    .Input("buffer_cursors: float")
    .Input("t: int32")
    .Output("stack1: T")
    .Output("stack2: T")
    .Output("buf_top: T")
    .Output("stack2_ptrs: float")
    .Attr("T: {float}")
    .Doc(R"doc(
Perform thin-stack lookup operation.
)doc");


REGISTER_OP("ThinStackUpdate")
    .Input("shift_val: T")
    .Input("reduce_val: T")
    .Input("transitions: float")
    .Input("stack: Ref(T)")
    .Input("queue: Ref(float)")
    .Input("cursors: Ref(float)")
    .Input("buffer_cursors: Ref(float)")
    .Input("t: int32")
    .Output("stack_out: Ref(T)")
    .Output("queue: Ref(float)")
    .Output("cursors: Ref(float)")
    .Output("buffer_cursors: Ref(float)")
    .Attr("T: {float}")
    .Doc(R"doc(
Perform inplace update on thin-stack representation.
)doc");


}
