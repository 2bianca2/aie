// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities)' %s | FileCheck %s

// Heterogeneous placement: a contraction/conv dispatch is pinned to the amd-aie
// (NPU) device, everything else (here an elementwise dispatch) to the llvm-cpu
// device, and the device topology is injected so cross-device buffers resolve.

// CHECK: module attributes {
// CHECK-SAME: stream.topology = #hal.device.topology<links = [(@npu -> @cpu = {transparent_access = true}), (@cpu -> @npu = {transparent_access = true})]>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @elementwise workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elementwise(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @hetero(%arg0: tensor<16384xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_0::@elementwise
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %0 = flow.dispatch @dispatch_0::@elementwise(%arg0) : (tensor<16384xf32>) -> tensor<16384xf32>
    %1 = flow.tensor.reshape %0 : tensor<16384xf32> -> tensor<128x128xf32>
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu>}
    %2 = flow.dispatch @dispatch_1::@matmul(%1, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}

// -----

// Multiple accelerators: the front policy uses only the first accelerator
// (@npu0), so the matmul is pinned to @npu0 and the topology meshes just
// @npu0 <-> @cpu -- the extra accelerator @npu1 is not referenced (DCE'd).

// CHECK: module attributes {
// CHECK-SAME: stream.topology = #hal.device.topology<links = [(@npu0 -> @cpu = {transparent_access = true}), (@cpu -> @npu0 = {transparent_access = true})]>
module attributes {stream.affinity.default = #hal.device.affinity<@npu0>} {
  util.global private @npu0 = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @npu1 = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @elementwise workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elementwise(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @multi_accel(%arg0: tensor<16384xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_0::@elementwise
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %0 = flow.dispatch @dispatch_0::@elementwise(%arg0) : (tensor<16384xf32>) -> tensor<16384xf32>
    %1 = flow.tensor.reshape %0 : tensor<16384xf32> -> tensor<128x128xf32>
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu0>}
    %2 = flow.dispatch @dispatch_1::@matmul(%1, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}

// -----

// Host-only (no accelerator declared): the contraction also runs on the host,
// and no topology is injected (a single device needs no cross-device links).

// CHECK-NOT: stream.topology
// CHECK-LABEL: util.func public @host_only
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @host_only(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %2 = flow.dispatch @dispatch_1::@matmul(%arg0, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}
