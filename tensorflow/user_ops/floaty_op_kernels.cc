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

// See docs in floaty_ops.cc

#include "tensorflow/user_ops/floaty_op_kernels.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class FloatyGatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit FloatyGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
    // We used to grab the validate_indices attribute here, but now we
    // always validate indices since the speed difference was only 1.5%.
    // TODO(irving): Remove the validate_indices attribute once we have
    // support for removing attrs in a backwards compatible way.
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // Check that we have enough index space
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is indices.shape + params.shape[1:].
    TensorShape result_shape = indices.shape();
    for (int i = 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0) {
      auto params_flat = params.flat_outer_dims<T>();
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 2>({N, out->NumElements() / N});

      functor::FloatyGather<Device, T, Index> functor;
      int64 bad_i = functor(c->eigen_device<Device>(), params_flat,
                            indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopies(typename TTypes<T>::ConstMatrix params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T>::Matrix out) {
  const SliceIndex first_dim_size =
      static_cast<SliceIndex>(indices.dimension(0));
  int limit = static_cast<int>(params.dimension(0));
  T* out_base = &out(0, 0);
  const T* params_base = &params(0, 0);
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    CHECK_EQ(static_slice_elems, slice_elems);
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  for (SliceIndex i = 0; i < first_dim_size; i++) {
    const SliceIndex j = i + 1;
    if (j < first_dim_size) {
      port::prefetch<port::PREFETCH_HINT_T0>(&params(indices(j), 0));
      port::prefetch<port::PREFETCH_HINT_T0>(&out(j, 0));
    }
    // Grab the index and check its validity.  An earlier version of the
    // code checked it and then grabbed it from memory a second time, which
    // was a security risk since it could have changed in between.
    int index = static_cast<int>(indices(i));
    if (!FastBoundsCheck(index, limit)) return i;
    // Copy using memcpy if possible, otherwise an Eigen loop
    if (Allocator::is_simple<T>::value) {
      memcpy(out_base + i * slice_elems, params_base + index * slice_elems,
             slice_bytes);
    } else {
      out.template chip<0>(i) = params.template chip<0>(index);
    }
  }
  return -1;
}

// Specialization gather functor for CPU.
template <typename T, typename Index>
struct FloatyGather<CPUDevice, T, Index> {
  int64 operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out) {
    const int64 N = indices.size();
    const int64 slice_size = out.size() / N;
    int64 bad_i;

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      N > std::numeric_limits<int32>::max());
#define CALL(elems)                                                   \
  do {                                                                \
    if (use_large) {                                                  \
      bad_i = HandleCopies<T, Index, int64, elems>(params, indices,   \
                                                   slice_size, out);  \
    } else {                                                          \
      const int32 small_slice = static_cast<int32>(slice_size);       \
      bad_i = HandleCopies<T, Index, int32, elems>(params, indices,   \
                                                   small_slice, out); \
    }                                                                 \
  } while (0)

    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};
}  // namespace functor

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("FloatyGather")                         \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          FloatyGatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, float);

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX(T, Index)                          \
  template <>                                                      \
  Index FloatyGather<GPUDevice, T, Index>::operator()(             \
      const GPUDevice& d, typename TTypes<T>::ConstMatrix Tparams, \
      typename TTypes<Index>::ConstFlat Tindices,                  \
      typename TTypes<T>::Matrix Tout);                            \
  extern template struct FloatyGather<GPUDevice, T, Index>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, float);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL


///////////////////////////////////////////////////////


namespace {

template <floaty_op_kernels::UpdateOp Op>
struct Assign {};
template <>
struct Assign<floaty_op_kernels::UpdateOp::ASSIGN> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = u;
  }
};

}  // namespace

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidShapes(const Tensor& params, const Tensor& updates,
                        const Tensor& indices) {
  if (updates.dims() != indices.dims() + params.dims() - 1) return false;
  for (int d = 0; d < indices.dims(); d++) {
    if (updates.dim_size(d) != indices.dim_size(d)) {
      return false;
    }
  }
  for (int d = 1; d < params.dims(); d++) {
    if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
      return false;
    }
  }
  return true;
}

static void DoValidationChecking(OpKernelContext* c, const Tensor& params,
                                 const Tensor& indices, const Tensor& updates) {
  OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
  OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1-D, got shape ",
                                      params.shape().DebugString()));
  OP_REQUIRES(
      c, ValidShapes(params, updates, indices),
      errors::InvalidArgument(
          "Must have updates.shape = indices.shape + params.shape[1:], got ",
          "updates.shape ", updates.shape().DebugString(), ", indices.shape ",
          indices.shape().DebugString(), ", params.shape ",
          params.shape().DebugString()));
}

template <typename Device, typename T, typename Index, floaty_op_kernels::UpdateOp op>
class FloatyScatterUpdateOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here.  Should we have the framework do some sort of
  //   integer promotion automatically, or should that be something
  //   that users have to do explicitly with a conversion operator
  //   in the graph?
  explicit FloatyScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    DoValidationChecking(c, params, indices, updates);
    if (!c->status().ok()) return;

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(c, N_big <= std::numeric_limits<Index>::max(),
                errors::InvalidArgument(
                    "indices has too many elements for ",
                    DataTypeString(DataTypeToEnum<Index>::v()), " indexing: ",
                    N_big, " > ", std::numeric_limits<Index>::max()));
    int N = static_cast<int>(indices.NumElements());
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params.flat_outer_dims<T>();
      auto updates_flat = updates.shaped<T, 2>({N, updates.NumElements() / N});

      functor::FloatyScatterFunctor<Device, T, Index, op> functor;
      const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                  params_flat, updates_flat, indices_flat);
      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

namespace functor {
// Implementation of update functor for CPU.
template <typename T, typename Index, floaty_op_kernels::UpdateOp op>
struct FloatyScatterFunctor<CPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    int limit = static_cast<int>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  An earlier version of the
      // code checked it and then grabbed it from memory a second time, which
      // was a security risk since it could have changed in between.
      int index = static_cast<int>(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      Assign<op>::Run(params.template chip<0>(index),
                      updates.template chip<0>(i));
    }
    return -1;
  }
};
}  // namespace functor

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          FloatyScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, float, dev, name, op);

#define REGISTER_SCATTER_UPDATE(type, dev)            \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterUpdate", \
                          floaty_op_kernels::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_UPDATE_CPU(type) REGISTER_SCATTER_UPDATE(type, CPU);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_UPDATE_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_SCATTER_UPDATE_GPU(type) REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_UPDATE_GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTER_UPDATE
#undef REGISTER_SCATTER_UPDATE_CPU
#undef REGISTER_SCATTER_UPDATE_GPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPECS_OP(T, Index, op)                   \
  template <>                                                \
  Index FloatyScatterFunctor<GPUDevice, T, Index, op>::operator()( \
      OpKernelContext* c, const GPUDevice& d,                \
      typename TTypes<T>::Matrix params,                     \
      typename TTypes<T>::ConstMatrix updates,               \
      typename TTypes<Index>::ConstFlat indices);            \
  extern template struct FloatyScatterFunctor<GPUDevice, T, Index, op>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)                       \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, float);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_OP

}  // namespace functor
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
