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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

typedef shape_inference::InferenceContext InferenceContext;
typedef shape_inference::Shape Shape;

// --------------------------------------------------------------------------
REGISTER_OP("FloatyGather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Input("grad_container: Tparams")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {float}")
    .SetShapeFn(OpShapeInferenceFn([](InferenceContext* c) {
      const Shape* params_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &params_subshape));
      const Shape* indices_shape = c->input(1);
      const Shape* out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    }))
    .Doc(R"doc(
Gather slices from `params` according to `indices`.
This is just like Gather except that `indices` may be floats.
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
