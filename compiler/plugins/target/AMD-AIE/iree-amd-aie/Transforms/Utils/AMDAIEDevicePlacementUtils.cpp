// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"

namespace mlir::iree_compiler::AMDAIE {

StringRef getDeviceBackend(IREE::Util::GlobalOpInterface global) {
  auto targetAttr = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!targetAttr) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      targetAttr.getExecutableTargets();
  if (targets.empty()) return {};
  return targets.front().getBackend().getValue();
}

DeviceRole deviceRole(StringRef backend) {
  if (backend == "llvm-cpu") return DeviceRole::Host;
  if (backend == "amd-aie") return DeviceRole::Accelerator;
  return DeviceRole::Unknown;  // not managed by this plugin
}

LinkCapability linkFlags(StringRef srcBackend, StringRef dstBackend) {
  // npu4 (amd-aie) and the CPU share system memory on the integrated APU, so
  // either can access the other's buffers without staging (validated e2e).
  bool amdAieCpuPair =
      (srcBackend == "amd-aie" && dstBackend == "llvm-cpu") ||
      (srcBackend == "llvm-cpu" && dstBackend == "amd-aie");
  if (amdAieCpuPair)
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  return {};  // conservative default: stage (copy) across the boundary.
}

IREE::HAL::DeviceAffinityAttr getFirstDeviceWithRole(ModuleOp module,
                                                     DeviceRole role) {
  MLIRContext *ctx = module.getContext();
  for (auto global : module.getOps<IREE::Util::GlobalOpInterface>()) {
    if (deviceRole(getDeviceBackend(global)) != role) continue;
    return IREE::HAL::DeviceAffinityAttr::get(
        ctx, FlatSymbolRefAttr::get(ctx, global.getGlobalName()),
        /*queue_mask=*/-1ll);
  }
  return {};
}

}  // namespace mlir::iree_compiler::AMDAIE
