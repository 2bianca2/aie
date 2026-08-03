// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-pad-contraction-dispatches)' %s | FileCheck %s

// A matmul dispatch pinned to the amd-aie (npu4) device whose reduction dim
// K=27 is not a multiple of the L2 reduction tile (32) has its operands padded
// to K=32: the executable bindings/loads/matmul grow, and each host operand is
// zero-padded by a separate padding dispatch placed on the host (CPU). The
// output (K is a reduction dim) is unchanged.

// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x32xbf16>, tensor<32x64xbf16>) outs(%{{.+}} : tensor<256x64xf32>)

// The padding dispatches zero-pad each operand to K=32 (strided insert via
// tensor.pad inside a dispatch).
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad

// CHECK-LABEL: util.func public @main
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<256x27xbf16>) -> tensor<256x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         flow.dispatch @dispatch_2::@matmul(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<256x32xbf16>, tensor<32x64xbf16>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_2 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>> -> tensor<256x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<256x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<256x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_2::@matmul(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x27xbf16>, tensor<27x64xbf16>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}

// -----

// A matmul whose K=64 is already a multiple of 32 is left untouched (no padding
// dispatch, no shape change) -- the pass is a no-op for divisible dispatches.

// CHECK-NOT:   @pad_executable
// CHECK-LABEL: func.func @matmul_ok
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>
// CHECK-LABEL: util.func public @main_ok
// CHECK:         flow.dispatch @dispatch_ok::@matmul_ok(%arg0, %arg1)
// CHECK-SAME:      : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_ok {
    flow.executable.export public @matmul_ok workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_ok(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>> -> tensor<256x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = tensor.empty() : tensor<256x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x64xbf16>, tensor<64x64xbf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main_ok(%arg0: tensor<256x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_ok::@matmul_ok(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}
