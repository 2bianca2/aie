// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Heterogeneous accelerator + host device placement.
//
// This pass offloads the ops amd-aie can codegen (contraction/convolution) to
// an accelerator device and runs everything else (layout transposes, f32->bf16
// casts introduced by demotion, other elementwise leftovers) on a host device.
// Devices are declared via `--iree-hal-target-device` (util.global with a
// DeviceTargetAttr); a single device-capability table (below) maps each backend
// to a role (Host/Accelerator) and each backend pair to its topology link
// capability -- the two seams (classification and topology) both read it.
//
// Run at the end of the Flow phase (after dispatch outlining, before the Stream
// affinity solver), it pins each `flow.dispatch` via `stream.affinity`
// (contraction/conv -> accelerator, else -> host) and injects a `stream.topology`
// module attribute over the devices it places work on (unless one is already
// provided) so cross-device buffers can be resolved.
//
// Requires at least one host device (accelerators are optional). Choosing among
// multiple same-role devices currently uses the first one (see policy TODO).

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

//===----------------------------------------------------------------------===//
// Device-capability table (single source of truth)
//
// Maps a device backend to its role and a backend pair to its topology link
// capability. Both the classification and the topology seams read this table;
// supporting new hardware means adding entries here (and nothing else).
//===----------------------------------------------------------------------===//

enum class DeviceRole { Host, Accelerator, Unknown };

/// Role of a device backend: Accelerator = a backend this pass offloads
/// contraction/conv to; Host = the CPU fallback for everything else.
static DeviceRole deviceRole(StringRef backend) {
  if (backend == "llvm-cpu") return DeviceRole::Host;
  if (backend == "amd-aie") return DeviceRole::Accelerator;
  return DeviceRole::Unknown;  // not managed by this pass
}

struct LinkCapability {
  bool unifiedMemory = false;
  bool transparentAccess = false;
};

/// Directed link capability between two device backends. IREE does not derive
/// the topology (it only consumes it), and `transparent_access` is a per-pair
/// hardware property, so this defaults to staging (no transparent access) and
/// only known-good pairs are marked true.
static LinkCapability linkFlags(StringRef srcBackend, StringRef dstBackend) {
  // npu4 (amd-aie) and the CPU share system memory on the integrated APU, so
  // either can access the other's buffers without staging (validated e2e).
  bool amdAieCpuPair =
      (srcBackend == "amd-aie" && dstBackend == "llvm-cpu") ||
      (srcBackend == "llvm-cpu" && dstBackend == "amd-aie");
  if (amdAieCpuPair)
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  return {};  // conservative default: stage (copy) across the boundary.
}

/// A classified device global: its symbol reference and backend.
struct DeviceInfo {
  FlatSymbolRefAttr ref;
  StringRef backend;
};

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

  // Classify device globals by role via the capability table.
  SmallVector<DeviceInfo> accelerators, hosts;
  for (auto global : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    StringRef backend = getDeviceBackend(global);
    DeviceInfo info{FlatSymbolRefAttr::get(ctx, global.getGlobalName()),
                    backend};
    switch (deviceRole(backend)) {
      case DeviceRole::Accelerator:
        accelerators.push_back(info);
        break;
      case DeviceRole::Host:
        hosts.push_back(info);
        break;
      case DeviceRole::Unknown:
        break;  // ignored: not placed, not in topology -> DCE
    }
  }
  // A host device is required; accelerators are optional.
  if (hosts.empty()) return;

  // Placement policy (currently trivial "front"). A performance/config-aware
  // policy would replace this selection (and then also need the two-phase
  // topology keep-alive noted below).
  // TODO: choose among multiple accelerators/hosts by device performance/config.
  DeviceInfo hostTarget = hosts.front();
  DeviceInfo accelTarget =
      accelerators.empty() ? hostTarget : accelerators.front();

  auto affinityFor = [&](FlatSymbolRefAttr ref) {
    return IREE::HAL::DeviceAffinityAttr::get(ctx, ref, /*queue_mask=*/-1ll);
  };

  // Pin each dispatch: contraction/conv -> accelerator, everything else -> host.
  // Both are set explicitly so the affinity solver does not propagate the host
  // affinity onto accelerator work (it minimizes transfers otherwise).
  DenseMap<StringRef, bool> executableIsAccel;
  for (auto exe : moduleOp.getOps<IREE::Flow::ExecutableOp>())
    executableIsAccel[exe.getSymName()] = executableIsContractionOrConv(exe);
  moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    bool onAccel = false;
    for (SymbolRefAttr entryPoint : dispatchOp.getEntryPointRefs()) {
      auto it = executableIsAccel.find(entryPoint.getRootReference().getValue());
      if (it != executableIsAccel.end() && it->second) onAccel = true;
    }
    dispatchOp->setAttr("stream.affinity",
                        affinityFor((onAccel ? accelTarget : hostTarget).ref));
  });

  // Inject the device topology over the devices we place work on so cross-device
  // boundary buffers resolve (ResolveTopologyQueries). Respect a user-provided
  // topology if present. Link flags come from the capability table.
  // NOTE: the "used" set is derivable here (before dispatches) only because the
  // front policy is dispatch-independent; a per-dispatch policy would instead
  // keep all candidates alive at preprocessing and narrow at the Flow call.
  if (!moduleOp->hasAttr("stream.topology")) {
    SmallVector<DeviceInfo> used;
    if (accelTarget.ref != hostTarget.ref) used.push_back(accelTarget);
    used.push_back(hostTarget);
    if (used.size() >= 2) {
      SmallVector<IREE::HAL::DeviceLinkAttr> links;
      for (const DeviceInfo &from : used)
        for (const DeviceInfo &to : used) {
          if (from.ref == to.ref) continue;
          LinkCapability cap = linkFlags(from.backend, to.backend);
          links.push_back(IREE::HAL::DeviceLinkAttr::get(
              ctx, from.ref, to.ref, cap.unifiedMemory, cap.transparentAccess,
              /*extra_properties=*/DictionaryAttr()));
        }
      moduleOp->setAttr("stream.topology",
                        IREE::HAL::DeviceTopologyAttr::get(ctx, links));
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignDeviceAffinitiesPass() {
  return std::make_unique<AMDAIEAssignDeviceAffinitiesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
