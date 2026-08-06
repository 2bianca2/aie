// RUN: iree-compile --compile-to=flow --iree-hal-target-device=npu=amdxdna \
// RUN:   --iree-hal-target-device=cpu=local --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-hal-default-device=npu --iree-amdaie-target-device=npu4 %s | FileCheck %s

// The plugin injects its heterogeneous-placement passes through
// `extendFlowTransformPassPipeline`, a hook the iree fork adds (upstream has one
// for preprocessing only). The hook is invoked from the IREEVM pipeline, so it
// does not fire under `iree-opt` -- the pipeline registered there is built with
// default hooks that carry no plugin extensions. That makes the pass-level tests
// in Transforms/test/assign_device_affinities.mlir unable to notice if the hook
// stops being called; this compiles far enough to see the result of it firing.
//
// Two independent roots: a matmul, which amd-aie can codegen, and a transpose,
// which it cannot. If the Flow hook ran, each dispatch carries the affinity the
// placement pass chose for it.

// CHECK: flow.dispatch @{{.*}}_dispatch_0::@{{.*}}matmul
// CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu>}
// CHECK: flow.dispatch @{{.*}}_dispatch_1::@{{.*}}transpose
// CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}

func.func @matmul_and_transpose(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
                                %c: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                      outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %t = linalg.transpose ins(%c : tensor<64x64xf32>)
                        outs(%empty : tensor<64x64xf32>) permutation = [1, 0]
  return %mm, %t : tensor<64x64xf32>, tensor<64x64xf32>
}
