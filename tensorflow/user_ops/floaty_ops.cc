/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {


REGISTER_OP("FloatyGather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {float}")
    .Doc(R"doc(
Gather slices from `params` according to `indices`.
This is just like Gather except that `indices` may be floats.
)doc");


REGISTER_OP("UnsafeFloatyGather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Input("grad_container: Tparams")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {float}")
    .Doc(R"doc(
FloatyGather op specialized for repeated application to a dense matrix.

Exactly like FloatyGather during forward prop.
During backward prop, gradients are accumulated on a dense `grad` matrix instead
of being repeatedly generated with `IndexedSlices`.

Implemented as a simple subclass of FloatyGather, since the C++ implementations are
exactly the same.
)doc");


REGISTER_OP("FloatyScatterUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {float}")
    .Attr("use_locking: bool = true")
    .Doc(R"doc(
Applies sparse updates to a variable reference.
This is just like ScatterUpdate except that `indices` are floats.
)doc");


}  // namespace tensorflow
