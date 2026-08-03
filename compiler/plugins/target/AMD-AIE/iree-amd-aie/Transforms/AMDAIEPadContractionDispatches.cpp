// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pads the operands of NPU contraction dispatches up to the target's pack-peel
// tile multiples, at the Flow dispatch boundary.
//
// Runs right after `AMDAIEAssignDeviceAffinities`, when every `flow.dispatch`
// already carries a `stream.affinity` and dispatch regions are outlined. For a
// contraction dispatch pinned to an amd-aie device whose reduction (K) dim is
// not a multiple of the target's L2 reduction tile, this pass:
//   * grows the matmul operands to the padded shape *outside* the dispatch
//     (`flow.tensor.splat` zero + `flow.tensor.update` of the valid data), and
//   * rewrites the executable so its bindings/loads/matmul see the padded shape.
// The padded reduction elements are zero, so the extra products are zero and the
// result is unchanged -- no cropping is needed for a reduction dim. Because the
// dispatch shape entering codegen is now divisible, the validated pack-peel path
// works without any partial-tile handling in the DMA lowering.
//
// The padding multiples are read from each dispatch's own device affinity
// (affinity -> device global -> executable target attr), never hardcoded, so the
// logic is portable if device placement ever moves to a later (Stream) stage.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-amdaie-pad-contraction-dispatches"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static int64_t roundUpToMultiple(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

/// Tile multiples a matmul's M, N, K must be padded up to on a given target.
struct PaddingMultiples {
  int64_t m, n, k;
};

/// Resolves the amd-aie executable target a dispatch is pinned to via its
/// `stream.affinity` (affinity -> device global -> executable target attr).
/// Returns null if the dispatch is not pinned to an amd-aie device.
static IREE::HAL::ExecutableTargetAttr getDispatchTarget(
    IREE::Flow::DispatchOp dispatch, ModuleOp module) {
  auto affinity =
      dispatch->getAttrOfType<IREE::HAL::DeviceAffinityAttr>("stream.affinity");
  if (!affinity) return {};
  Operation *globalOp =
      SymbolTable::lookupNearestSymbolFrom(module, affinity.getDevice());
  auto global = dyn_cast_or_null<IREE::Util::GlobalOpInterface>(globalOp);
  if (!global) return {};
  auto deviceTarget = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!deviceTarget) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      deviceTarget.getExecutableTargets();
  if (targets.empty()) return {};
  IREE::HAL::ExecutableTargetAttr exec = targets.front();
  if (exec.getBackend().getValue() != "amd-aie") return {};
  return exec;
}

/// Computes the M/N/K pad multiples for `target` and the matmul element types,
/// entirely from the device info tables (no hardcoded geometry): M/N from the
/// core-array shape (num_rows/num_cols) times the vector instruction size, K
/// from the shared pack-peel reduction tile. Returns nullopt if `target` is not
/// a recognized amd-aie target.
static std::optional<PaddingMultiples> getPaddingMultiples(
    IREE::HAL::ExecutableTargetAttr target, Type lhsElemType, Type rhsElemType,
    Type accElemType) {
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(target);
  std::optional<int64_t> numRows = getConfigNumRows(target);
  std::optional<int64_t> numCols = getConfigNumColumns(target);
  if (!device || !numRows || !numCols) return std::nullopt;
  AMDAIEDeviceModel deviceModel = getDeviceModel(*device);
  FailureOr<std::array<uint32_t, 3>> instr =
      deviceModel.getAIEMatmulInstructionSize(lhsElemType, rhsElemType,
                                              accElemType);
  if (failed(instr)) return std::nullopt;
  // The default scalar pack-peel pipeline uses 2-level tiling (kPackScaleL1 = 1).
  return PaddingMultiples{/*m=*/*numRows * (*instr)[0],
                          /*n=*/*numCols * (*instr)[1],
                          /*k=*/getPackPeelReductionTile(/*kPackScaleL1=*/1)};
}

/// Where the reduction (K) dim of a matmul lives, per operand.
struct MatmulKInfo {
  linalg::MatmulOp matmul;
  IREE::TensorExt::DispatchTensorLoadOp lhsLoad, rhsLoad;
  BlockArgument lhsArg, rhsArg;  // executable bindings feeding LHS/RHS
  int64_t k;                     // current reduction size
};

/// Returns the func of a dispatch's single entry point, or null.
static func::FuncOp getDispatchFunc(IREE::Flow::DispatchOp dispatch,
                                    ModuleOp module) {
  SmallVector<SymbolRefAttr> entryPoints =
      llvm::to_vector(dispatch.getEntryPointRefs());
  if (entryPoints.size() != 1) return nullptr;
  SymbolRefAttr entryPoint = entryPoints.front();
  auto exe = module.lookupSymbol<IREE::Flow::ExecutableOp>(
      entryPoint.getRootReference());
  if (!exe) return nullptr;
  ModuleOp inner = exe.getInnerModule();
  if (!inner) return nullptr;
  return inner.lookupSymbol<func::FuncOp>(entryPoint.getLeafReference());
}

/// Traces a matmul input value to the executable binding (BlockArgument) it is
/// loaded from via a full-tensor `dispatch.tensor.load`.
static IREE::TensorExt::DispatchTensorLoadOp getFullTensorLoad(Value operand) {
  auto load = operand.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!load) return nullptr;
  // Only a plain full-tensor load (unit strides, zero offsets) is supported.
  for (OpFoldResult o : load.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return nullptr;
  for (OpFoldResult s : load.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return nullptr;
  return load;
}

/// Inspects the func for a single paddeable matmul. K is LHS dim 1 / RHS dim 0.
static std::optional<MatmulKInfo> getMatmulKInfo(func::FuncOp func) {
  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1) return std::nullopt;
  linalg::MatmulOp matmul = matmuls.front();
  auto lhsLoad = getFullTensorLoad(matmul.getInputs()[0]);
  auto rhsLoad = getFullTensorLoad(matmul.getInputs()[1]);
  if (!lhsLoad || !rhsLoad) return std::nullopt;
  auto lhsArg = dyn_cast<BlockArgument>(lhsLoad.getSource());
  auto rhsArg = dyn_cast<BlockArgument>(rhsLoad.getSource());
  if (!lhsArg || !rhsArg) return std::nullopt;
  int64_t k =
      cast<RankedTensorType>(lhsLoad.getType()).getShape()[1];
  return MatmulKInfo{matmul, lhsLoad, rhsLoad, lhsArg, rhsArg, k};
}

/// Grows executable binding `arg` and its full-tensor `load` at `dim` to
/// `newSize` (the matmul revalidates from the loaded operand types).
static void padBinding(IRRewriter &rewriter, BlockArgument arg,
                       IREE::TensorExt::DispatchTensorLoadOp load, int64_t dim,
                       int64_t newSize) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(arg.getType());
  auto tensorType = dtt.asRankedTensorType();
  SmallVector<int64_t> shape(tensorType.getShape());
  shape[dim] = newSize;
  auto newTensorType =
      RankedTensorType::get(shape, tensorType.getElementType());
  arg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                       newTensorType));
  rewriter.setInsertionPoint(load);
  auto newLoad = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
      load.getLoc(), newTensorType, arg, /*sourceDynamicDims=*/ValueRange{});
  rewriter.replaceOp(load, newLoad.getResult());
}

/// Returns a device affinity pinned to a host (non-amd-aie) device global, or
/// null if none is declared. The padding dispatch runs here (the CPU), where the
/// strided insert is trivially codegen-able.
static IREE::HAL::DeviceAffinityAttr getHostAffinity(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  for (auto global : module.getOps<IREE::Util::GlobalOpInterface>()) {
    auto deviceTarget = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
        global.getGlobalInitialValue());
    if (!deviceTarget) continue;
    ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
        deviceTarget.getExecutableTargets();
    if (targets.empty() || targets.front().getBackend().getValue() == "amd-aie")
      continue;
    return IREE::HAL::DeviceAffinityAttr::get(
        ctx, FlatSymbolRefAttr::get(ctx, global.getGlobalName()),
        /*queue_mask=*/-1ll);
  }
  return {};
}

/// Creates a `flow.executable` + `flow.dispatch` that zero-pads `v` at `dim` up
/// to `newSize` (empty + fill 0 + insert_slice), placed on `hostAffinity`.
/// Returns the padded value. A dispatch (unlike `flow.tensor.update`, which only
/// copies contiguous outer-dim sub-ranges) performs the strided placement
/// required to pad an inner dimension. `counter` uniquifies the symbol names.
static Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module,
                                   Value v, int64_t dim, int64_t newSize,
                                   IREE::HAL::DeviceAffinityAttr hostAffinity,
                                   int &counter) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> dstShape(srcType.getShape());
  dstShape[dim] = newSize;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load full input, fill a padded tensor with 0, insert the valid
  // data, store.
  auto funcOp = func::FuncOp::create(
      loc, "pad_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value zero = fb.create<arith::ConstantOp>(
      loc, fb.getZeroAttr(srcType.getElementType()));
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  Value padded = tensor::createPadHighOp(dstType, loaded, zero,
                                         /*nofold=*/false, loc, fb);
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, padded, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "pad_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy}, /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host dispatch, pinned to the host device.
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, /*arguments=*/ValueRange{v},
      /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

class AMDAIEPadContractionDispatchesPass
    : public impl::AMDAIEPadContractionDispatchesBase<
          AMDAIEPadContractionDispatchesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEPadContractionDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  module.walk([&](IREE::Flow::DispatchOp d) { dispatches.push_back(d); });

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device to place the padding dispatch
  int counter = 0;

  // Executables padded already (shared by multiple dispatches): pad the
  // executable once, but pad each caller's operands.
  DenseSet<Operation *> paddedExecutables;

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    IREE::HAL::ExecutableTargetAttr target = getDispatchTarget(dispatch, module);
    if (!target) continue;

    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<MatmulKInfo> info = getMatmulKInfo(func);
    if (!info) continue;

    auto elemType = [](Value v) {
      return cast<ShapedType>(v.getType()).getElementType();
    };
    std::optional<PaddingMultiples> mult = getPaddingMultiples(
        target, elemType(info->matmul.getInputs()[0]),
        elemType(info->matmul.getInputs()[1]),
        elemType(info->matmul.getResults()[0]));
    if (!mult) continue;

    int64_t kPad = roundUpToMultiple(info->k, mult->k);
    if (kPad == info->k) continue;

    int64_t lhsArgNo = info->lhsArg.getArgNumber();
    int64_t rhsArgNo = info->rhsArg.getArgNumber();

    // Rewrite the executable bindings/loads to the padded K (matmul revalidates
    // from the loaded operand types). Once per shared executable.
    if (paddedExecutables.insert(func.getOperation()).second) {
      padBinding(rewriter, info->lhsArg, info->lhsLoad, /*dim=*/1, kPad);
      padBinding(rewriter, info->rhsArg, info->rhsLoad, /*dim=*/0, kPad);
      SmallVector<Type> argTypes(func.getArgumentTypes());
      argTypes[lhsArgNo] = func.getArgument(lhsArgNo).getType();
      argTypes[rhsArgNo] = func.getArgument(rhsArgNo).getType();
      func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
    }

    // Pad the host operands to the padded K via padding dispatches (correct
    // strided placement for the LHS inner dim, which flow.tensor.update cannot
    // express), then rewire the matmul dispatch to the padded operands.
    rewriter.setInsertionPoint(dispatch);
    Value lhsPadded = createPaddingDispatch(
        rewriter, module, dispatch.getArguments()[lhsArgNo], /*dim=*/1, kPad,
        hostAffinity, counter);
    Value rhsPadded = createPaddingDispatch(
        rewriter, module, dispatch.getArguments()[rhsArgNo], /*dim=*/0, kPad,
        hostAffinity, counter);
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsPadded);
    dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(rhsPadded);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPadContractionDispatchesPass() {
  return std::make_unique<AMDAIEPadContractionDispatchesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
