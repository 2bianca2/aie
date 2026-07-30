// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Heterogeneous CPU+NPU device placement.
//
// amd-aie only codegens contraction/convolution dispatches; everything else
// (layout transposes, f32->bf16 casts introduced by demotion, other
// elementwise leftovers) must run on a host CPU device. This pass, run at the
// end of the Flow phase (after dispatch outlining, before the Stream affinity
// solver), pins each `flow.dispatch` to a device via a `stream.affinity`
// attribute: contraction/conv dispatches -> the amd-aie (NPU) device,
// everything else -> the llvm-cpu device. It also injects the required
// `stream.topology` module attribute so cross-device buffers can be resolved.
//
// It is a no-op unless the module declares both an amd-aie and an llvm-cpu
// device (i.e. only meaningful for heterogeneous multi-device compiles).

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#define DEBUG_TYPE "iree-amdaie-assign-device-affinities"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Returns the backend of the device global's target attr (e.g. "amd-aie",
/// "llvm-cpu"), or empty if the global is not a single device target.
static StringRef getDeviceBackend(IREE::Util::GlobalOpInterface global) {
  auto targetAttr = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!targetAttr) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      targetAttr.getExecutableTargets();
  if (targets.empty()) return {};
  return targets.front().getBackend().getValue();
}

/// Returns true if `exe` contains a contraction/convolution op (the ops
/// amd-aie can codegen), inspected structurally on the linalg IR.
static bool executableIsContractionOrConv(IREE::Flow::ExecutableOp exe) {
  ModuleOp innerModule = exe.getInnerModule();
  if (!innerModule) return false;
  bool found = false;
  innerModule.walk([&](linalg::LinalgOp linalgOp) {
    if (linalg::isaContractionOpInterface(linalgOp) ||
        linalg::isaConvolutionOpInterface(linalgOp)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

class AMDAIEAssignDeviceAffinitiesPass
    : public impl::AMDAIEAssignDeviceAffinitiesBase<
          AMDAIEAssignDeviceAffinitiesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEAssignDeviceAffinitiesPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Classify device globals by backend capability.
  FlatSymbolRefAttr npuRef, cpuRef;
  for (auto global : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    StringRef backend = getDeviceBackend(global);
    if (backend == "amd-aie") {
      npuRef = FlatSymbolRefAttr::get(ctx, global.getGlobalName());
    } else if (backend == "llvm-cpu") {
      cpuRef = FlatSymbolRefAttr::get(ctx, global.getGlobalName());
    }
  }
  // Only act for heterogeneous (NPU + CPU) compiles.
  if (!npuRef || !cpuRef) return;

  // Classify each executable: does it contain a contraction/conv op?
  DenseMap<StringRef, bool> executableIsNpu;
  for (auto exe : moduleOp.getOps<IREE::Flow::ExecutableOp>()) {
    executableIsNpu[exe.getSymName()] = executableIsContractionOrConv(exe);
  }

  auto npuAffinity = IREE::HAL::DeviceAffinityAttr::get(ctx, npuRef,
                                                        /*queue_mask=*/-1ll);
  auto cpuAffinity = IREE::HAL::DeviceAffinityAttr::get(ctx, cpuRef,
                                                        /*queue_mask=*/-1ll);

  // Pin each dispatch: contraction/conv -> NPU, everything else -> CPU. Both
  // are set explicitly so the affinity solver does not propagate CPU onto NPU
  // work (it minimizes transfers otherwise).
  moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    bool onNpu = false;
    for (SymbolRefAttr entryPoint : dispatchOp.getEntryPointRefs()) {
      auto it = executableIsNpu.find(entryPoint.getRootReference().getValue());
      if (it != executableIsNpu.end() && it->second) onNpu = true;
    }
    dispatchOp->setAttr("stream.affinity", onNpu ? npuAffinity : cpuAffinity);
  });

  // Inject the device topology (both directions, transparent access) so
  // cross-device boundary buffers can be resolved by ResolveTopologyQueries.
  if (!moduleOp->hasAttr("stream.topology")) {
    auto link = [&](FlatSymbolRefAttr from, FlatSymbolRefAttr to) {
      return IREE::HAL::DeviceLinkAttr::get(ctx, from, to,
                                            /*unified_memory=*/false,
                                            /*transparent_access=*/true,
                                            /*extra_properties=*/DictionaryAttr());
    };
    SmallVector<IREE::HAL::DeviceLinkAttr> links{link(npuRef, cpuRef),
                                                 link(cpuRef, npuRef)};
    moduleOp->setAttr("stream.topology",
                      IREE::HAL::DeviceTopologyAttr::get(ctx, links));
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignDeviceAffinitiesPass() {
  return std::make_unique<AMDAIEAssignDeviceAffinitiesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
