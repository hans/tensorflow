# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/serving/...",
        "//platforms/techila/...",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "ackermann_op.so",
    srcs = ["ackermann_op.cc"],
)

py_tests(
    name = "ackermann_test",
    size = "small",
    srcs = ["ackermann_test.py"],
    data = [":ackermann_op.so"],
)

tf_custom_op_library(
    name = "duplicate_op.so",
    srcs = ["duplicate_op.cc"],
)

py_tests(
    name = "duplicate_op_test",
    size = "small",
    srcs = ["duplicate_op_test.py"],
    data = [":duplicate_op.so"],
)

tf_custom_op_library(
    name = "invalid_op.so",
    srcs = ["invalid_op.cc"],
)

py_tests(
    name = "invalid_op_test",
    size = "small",
    srcs = ["invalid_op_test.py"],
    data = [":invalid_op.so"],
)


tf_custom_op_library(
    name = "floaty_ops.so",
    srcs = [
        "floaty_ops.cc",

        "floaty_gather_kernel.cc",
        "floaty_scatter_kernel.cc",

        # HACK: include header files in src list, because this Bazel
        # directive doesn't have a separate "headers" field
        "floaty_gather_kernel.h",
        "floaty_scatter_kernel.h"
    ],
    gpu_srcs = [
        "floaty_gather_kernel_gpu.cu.cc",
        "floaty_scatter_kernel_gpu.cu.cc",

        # same hack
        "floaty_gather_kernel.h",
        "floaty_scatter_kernel.h"
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
