// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-convert-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Applies dma transposition on the side that has lower number of dimensions,
/// which means the source side for pack ops and the destination side for unpack
/// ops.
template <typename PackOrUnpackOp>
LogicalResult dmaTransposeOnLowerNumDims(PackOrUnpackOp packOrUnpackOp,
                                         SmallVector<OpFoldResult> &offsets,
                                         SmallVector<OpFoldResult> &sizes,
                                         SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOrUnpackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOrUnpackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOrUnpackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  ArrayRef<int64_t> innerDimsPos = packOrUnpackOp.getInnerDimsPos();

  for (int i = 0; i < innerTiles.size(); i++) {
    // Calculate new sizes.
    innerSizes.push_back(getAsIndexOpFoldResult(ctx, innerTiles[i]));
    std::optional<int64_t> size = getConstantIntValue(sizes[innerDimsPos[i]]);
    assert(size.has_value() &&
           "expect constant index here in sizes vector of pack op");
    // Fail if tile doesnt perfectly divide the corresponding outer dim as we
    // do not support the padding semantics yet.
    if (size.value() % innerTiles[i] != 0) {
      auto message = llvm::formatv(
          "in dimension {0}, the tile size {1} does not divide the tensor size "
          "{2}. Imperfect/partial tiling is currently not supported.",
          i, innerTiles[i], size.value());
      return packOrUnpackOp->emitOpError(message);
    }

    sizes[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, size.value() / innerTiles[i]);
    // The tiled dim inherits the stride from the corresponding outer dim and
    // the outer dims stride gets multiplied by the size of the tile.
    innerStrides.push_back(strides[innerDimsPos[i]]);
    std::optional<int64_t> stride =
        getConstantIntValue(strides[innerDimsPos[i]]);
    assert(stride.has_value() &&
           "expect constant index in stride vector of pack op");
    strides[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, stride.value() * innerTiles[i]);
    // The tiled dim inherits the offset from the corresponding outer dim and
    // the outer dim offset is set to zero.
    innerOffsets.push_back(offsets[innerDimsPos[i]]);
    offsets[innerDimsPos[i]] = getAsIndexOpFoldResult(ctx, 0);
  }

  // Apply permutations to the outer dims if provided.
  if (!permutation.empty()) {
    applyPermutationToVector(strides, permutation);
    applyPermutationToVector(sizes, permutation);
    applyPermutationToVector(offsets, permutation);
  }

  // Merge the dims.
  sizes.insert(sizes.end(), innerSizes.begin(), innerSizes.end());
  strides.insert(strides.end(), innerStrides.begin(), innerStrides.end());
  offsets.insert(offsets.end(), innerOffsets.begin(), innerOffsets.end());
  return success();
}

/// Applies dma transposition on the side which has higher number of dimensions,
/// which means the destination side for pack ops and the source side for unpack
/// ops.
template <typename PackOrUnpackOp>
LogicalResult dmaTransposeOnHigherNumDims(PackOrUnpackOp packOrUnpackOp,
                                          SmallVector<OpFoldResult> &offsets,
                                          SmallVector<OpFoldResult> &sizes,
                                          SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOrUnpackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOrUnpackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOrUnpackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;
  ArrayRef<int64_t> innerDimsPos = packOrUnpackOp.getInnerDimsPos();

  int numOuterDims = sizes.size() - innerTiles.size();
  SmallVector<OpFoldResult> outerOffsets = SmallVector<OpFoldResult>(
      offsets.begin(), offsets.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerStrides = SmallVector<OpFoldResult>(
      strides.begin(), strides.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerSizes =
      SmallVector<OpFoldResult>(sizes.begin(), sizes.begin() + numOuterDims);

  // Apply inverse permutation to the outer dims if permutation provided (if
  // permutation not provided, it is identity, and therefore so is the inverse).
  if (!permutation.empty()) {
    SmallVector<int64_t> inversePermutation =
        invertPermutationVector(permutation);
    applyPermutationToVector(outerStrides, inversePermutation);
    applyPermutationToVector(outerSizes, inversePermutation);
    applyPermutationToVector(outerOffsets, inversePermutation);
  }

  // Initialize the indexing of each outer dim.
  llvm::SmallDenseMap<int64_t, int64_t> outerDimsIndexMap;
  for (int i = 0; i < numOuterDims; i++) {
    outerDimsIndexMap[i] = i;
  }

  // Update outer dim sizes/strides/offsts.
  for (int i = 0; i < innerTiles.size(); i++) {
    // Insert inner dims adjacent to their corresponding outer dims.
    int insertionIndex = outerDimsIndexMap[innerDimsPos[i]] + 1;
    outerSizes.insert(outerSizes.begin() + insertionIndex,
                      getAsIndexOpFoldResult(ctx, innerTiles[i]));
    outerStrides.insert(outerStrides.begin() + insertionIndex,
                        strides[numOuterDims + i]);
    outerOffsets.insert(outerOffsets.begin() + insertionIndex,
                        offsets[numOuterDims + i]);
    // Update the map as all the dimensions inner to the innerDimsPos[i] are now
    // shifted by 1.
    for (int j = innerDimsPos[i] + 1; j < numOuterDims; j++) {
      outerDimsIndexMap[j]++;
    }
  }

  // Make the outer dims as the final returned dims
  offsets = outerOffsets;
  strides = outerStrides;
  sizes = outerSizes;
  return success();
}

/// Recovers the constant base offset (in elements) of a memref whose offset is
/// typed dynamic but originates from a static `byte_offset` on the
/// `hal.interface.binding.subspan` that backs it -- e.g. a weight packed at a
/// non-zero offset in a shared constant pool. Walks the view-like defining-op
/// chain to the subspan. Returns failure if no static offset can be recovered
/// (no backing subspan, or a non-constant byte offset).
static FailureOr<int64_t> recoverSubspanElementOffset(Value source) {
  IREE::HAL::InterfaceBindingSubspanOp subspanOp;
  Value cur = source;
  while (Operation *def = cur.getDefiningOp()) {
    if (auto subspan = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(def)) {
      subspanOp = subspan;
      break;
    }
    if (auto assumeOp = dyn_cast<memref::AssumeAlignmentOp>(def)) {
      cur = assumeOp.getViewSource();
    } else if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
      cur = reinterpretOp.getSource();
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
      cur = subviewOp.getSource();
    } else {
      break;
    }
  }
  if (!subspanOp) return failure();
  Value byteOffset = subspanOp.getByteOffset();
  if (!byteOffset) return int64_t(0);
  std::optional<int64_t> maybeByteOffset = getConstantIntValue(byteOffset);
  if (!maybeByteOffset) return failure();
  unsigned elemBitWidth =
      cast<MemRefType>(source.getType()).getElementTypeBitWidth();
  if (elemBitWidth == 0 || elemBitWidth % 8 != 0) return failure();
  int64_t elemByteWidth = elemBitWidth / 8;
  if (*maybeByteOffset % elemByteWidth != 0) return failure();
  return *maybeByteOffset / elemByteWidth;
}

/// Handles a memref `source` whose non-zero base offset cannot be carried in the
/// memref type (downstream `memref.global` shim buffers cannot have an offset).
/// Recovers the static base offset, delinearizes it across the source `strides`
/// into `perDim` contributions, and rebases the source to an offset-0 memref
/// (returned via `rebasedOp`). The caller folds `perDim` into the DMA per-dim
/// offsets so the offset lives in the DMA access pattern instead.
static LogicalResult rebaseSourceToZeroOffset(IRRewriter &rewriter, Value source,
                                              ArrayRef<int64_t> strides,
                                              SmallVector<int64_t> &perDim,
                                              Operation *&rebasedOp) {
  if (llvm::any_of(strides,
                   [](int64_t s) { return ShapedType::isDynamic(s); })) {
    return failure();
  }
  FailureOr<int64_t> maybeOffset = recoverSubspanElementOffset(source);
  if (failed(maybeOffset)) return failure();
  int64_t remaining = *maybeOffset;
  perDim.assign(strides.size(), 0);
  for (unsigned i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0) continue;
    perDim[i] = remaining / strides[i];
    remaining = remaining % strides[i];
  }
  MLIRContext *ctx = source.getContext();
  Location loc = source.getLoc();
  ArrayRef<int64_t> srcShape = cast<MemRefType>(source.getType()).getShape();
  SmallVector<int64_t> rebaseShape(srcShape.size());
  for (unsigned i = 0; i < srcShape.size(); ++i)
    rebaseShape[i] = perDim[i] + srcShape[i];
  // Rebase to offset 0 by reinterpreting the SAME subspan-backed source (not its
  // extracted base buffer): this keeps the def-use chain to the
  // `hal.interface.binding.subspan` intact so downstream passes can still
  // recover the binding ordinal. reinterpret_cast uses the source's aligned
  // (allocation) base pointer, so offset 0 points at the buffer base; the
  // delinearized offset is carried by the DMA per-dim offsets instead.
  auto rebased = rewriter.create<memref::ReinterpretCastOp>(
      loc, source,
      /*offset=*/getAsIndexOpFoldResult(ctx, int64_t(0)),
      getAsIndexOpFoldResult(ctx, rebaseShape),
      getAsIndexOpFoldResult(ctx, SmallVector<int64_t>(strides)));
  rebasedOp = rebased.getOperation();
  return success();
}

/// Examines an input/output of a pack/unpack op and provides the
/// corresponding offsets, sizes and strides required by the dma op.
LogicalResult setDmaInputs(IRRewriter &rewriter, Operation *&operandOp,
                           SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = operandOp->getContext();
  if (isa<memref::AllocOp>(operandOp) ||
      isa<memref::AssumeAlignmentOp>(operandOp)) {
    MemRefType memRefType = cast<MemRefType>(operandOp->getResult(0).getType());
    auto [stridesI64, baseOffset] = memRefType.getStridesAndOffset();
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    auto sizesI64 = memRefType.getShape();
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return operandOp->emitOpError(
          "with dynamic shape is not supported by dma op.");
    }
    sizes = getAsIndexOpFoldResult(ctx, sizesI64);
    if (baseOffset != 0) {
      // The whole buffer is DMA'd from a non-zero base offset (e.g. a buffer
      // packed in a shared pool). Rebase to offset 0 and move the base offset
      // into the DMA per-dim offsets.
      Value source = operandOp->getResult(0);
      SmallVector<int64_t> perDim;
      Operation *rebasedOp = nullptr;
      if (failed(rebaseSourceToZeroOffset(rewriter, source, stridesI64, perDim,
                                          rebasedOp))) {
        return operandOp->emitOpError(llvm::formatv(
            "has a non-zero base offset {0} that could not be recovered from a "
            "backing subspan; not supported by this pass.",
            baseOffset));
      }
      operandOp = rebasedOp;
      for (int64_t p : perDim) offsets.push_back(getAsIndexOpFoldResult(ctx, p));
      return success();
    }
    // Alloc Op has no offsets.
    for (int i = 0; i < sizes.size(); i++) {
      offsets.push_back(getAsIndexOpFoldResult(ctx, 0));
    }
    return success();
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(operandOp)) {
    auto mixedStrides = subviewOp.getMixedStrides();
    if (llvm::any_of(mixedStrides, [](OpFoldResult ofr) {
          return !isConstantIntValue(ofr, 1);
        })) {
      auto message = llvm::formatv(
          "has non-unit mixed strides that are not currently supported by this "
          "pass.");
      return subviewOp->emitOpError(message);
    }
    offsets = subviewOp.getMixedOffsets();
    MemRefType subviewType = subviewOp.getSource().getType();
    auto [stridesI64, baseOffset] = subviewType.getStridesAndOffset();
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    operandOp = subviewOp.getSource().getDefiningOp();
    sizes = subviewOp.getMixedSizes();

    // A non-zero/dynamic base offset on the subview source (e.g. a buffer packed
    // at a non-zero offset in a shared constant pool) cannot be carried in the
    // memref type -- downstream `memref.global` shim buffers cannot have an
    // offset. Rebase the source to an offset-0 memref and fold the base offset
    // into the per-dimension DMA offsets, so the offset lives in the DMA access
    // pattern instead. Only a statically-recoverable offset is handled.
    if (baseOffset != 0) {
      SmallVector<int64_t> perDim;
      Operation *rebasedOp = nullptr;
      if (failed(rebaseSourceToZeroOffset(rewriter, subviewOp.getSource(),
                                          stridesI64, perDim, rebasedOp))) {
        return subviewOp->emitOpError(
            "has a non-zero base offset that could not be recovered from a "
            "backing subspan; not supported by this pass.");
      }
      operandOp = rebasedOp;
      // Fold the delinearized base offset into the subview offsets. A static
      // subview offset folds into a constant; a dynamic one (e.g. a tiling
      // induction variable) gets an arith.addi of the static contribution.
      Location loc = subviewOp.getLoc();
      for (unsigned i = 0; i < offsets.size() && i < perDim.size(); ++i) {
        if (perDim[i] == 0) continue;
        std::optional<int64_t> cur = getConstantIntValue(offsets[i]);
        if (cur) {
          offsets[i] = getAsIndexOpFoldResult(ctx, *cur + perDim[i]);
        } else {
          Value dynOffset = cast<Value>(offsets[i]);
          Value contribution =
              rewriter.create<arith::ConstantIndexOp>(loc, perDim[i]);
          offsets[i] =
              rewriter.create<arith::AddIOp>(loc, dynOffset, contribution)
                  .getResult();
        }
      }
    }
    if (llvm::any_of(sizes, [](OpFoldResult fr) {
          return !getConstantIntValue(fr).has_value();
        })) {
      return subviewOp->emitOpError(
          " has dynamic shape that is not supported by the target dma op.");
    }

    assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
           "mismatch in the number of offsets, sizes and strides");

    // Handle the case where some dimensions are dropped in the subview:
    llvm::SmallBitVector droppedDims = subviewOp.getDroppedDims();
    uint64_t insertionIndex{0};
    for (uint64_t extractionIndex = 0; extractionIndex < offsets.size();
         ++extractionIndex) {
      if (!droppedDims[extractionIndex]) {
        offsets[insertionIndex] = offsets[extractionIndex];
        sizes[insertionIndex] = sizes[extractionIndex];
        strides[insertionIndex] = strides[extractionIndex];
        insertionIndex++;
      }
    }
    offsets.resize(insertionIndex);
    sizes.resize(insertionIndex);
    strides.resize(insertionIndex);
    return success();
  }
  return operandOp->emitOpError(
      "is an unsupported operation. This pass currently only supports "
      "memref.assume_alignment, memref.alloc and memref.subview as "
      "inputs.");
}

/// Rewrite the pack/unpack op 'op' as a DMA operation. The function arguments
/// 'input', 'output', and 'innerTiles' are the input, output, and inner tile
/// of 'op'. If 'op' is not a pack/unpack op, or if it determined to not
/// currently be lowerable to a DMA operation, failure is returned.
///
/// Design note: arguments 'input', 'output', and 'innerTiles' could be
/// obtained from 'op' inside this function if it were templatized, but
/// I've factorized out that logic to reduce the total amount of templatized
/// code.
template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(IRRewriter &rewriter, PackOrUnpackOp op, Value input,
                           Value output, llvm::ArrayRef<int64_t> innerTiles,
                           bool transposeOnSource) {
  if (llvm::any_of(innerTiles,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    op->emitError("has a non-static shape: not yet supported by this pass.");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Operation *sourceOp = input.getDefiningOp();
  Operation *dstOp = output.getDefiningOp();

  // Prepare source DMA inputs.
  SmallVector<OpFoldResult> srcOffsets;
  SmallVector<OpFoldResult> srcStrides;
  SmallVector<OpFoldResult> srcShape;
  if (failed(setDmaInputs(rewriter, sourceOp, srcOffsets, srcShape,
                          srcStrides))) {
    return failure();
  }

  // Prepare destination DMA inputs.
  SmallVector<OpFoldResult> dstOffsets;
  SmallVector<OpFoldResult> dstStrides;
  SmallVector<OpFoldResult> dstShape;
  if (failed(setDmaInputs(rewriter, dstOp, dstOffsets, dstShape, dstStrides))) {
    return failure();
  }

  // Update dma source or destination addressing based on the side for dma
  // transposition.
  {
    SmallVector<OpFoldResult> &offsets =
        transposeOnSource ? srcOffsets : dstOffsets;

    SmallVector<OpFoldResult> &shape = transposeOnSource ? srcShape : dstShape;

    SmallVector<OpFoldResult> &strides =
        transposeOnSource ? srcStrides : dstStrides;

    bool sourceIsHigherDim = dstStrides.size() <= srcStrides.size();

    if (sourceIsHigherDim == transposeOnSource) {
      if (failed(dmaTransposeOnHigherNumDims(op, offsets, shape, strides))) {
        return failure();
      }
    } else {
      if (failed(dmaTransposeOnLowerNumDims(op, offsets, shape, strides))) {
        return failure();
      }
    }
  }

  // Create logical objectFifos from source and destination memrefs.
  Value srcVal = sourceOp->getResult(0);
  Value dstVal = dstOp->getResult(0);
  auto srcType = cast<MemRefType>(srcVal.getType());
  auto dstType = cast<MemRefType>(dstVal.getType());

  rewriter.setInsertionPointAfter(srcVal.getDefiningOp());
  auto src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
      rewriter.getUnknownLoc(), LogicalObjectFifoType::get(srcType), srcVal);

  rewriter.setInsertionPointAfter(dstVal.getDefiningOp());
  auto dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
      rewriter.getUnknownLoc(), LogicalObjectFifoType::get(dstType), dstVal);

  rewriter.setInsertionPoint(op);
  rewriter.create<AMDAIE::DmaCpyNdOp>(op->getLoc(), dst, dstOffsets, dstShape,
                                      dstStrides, src, srcOffsets, srcShape,
                                      srcStrides);
  rewriter.eraseOp(op);
  return success();
}

template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(PackOrUnpackOp op, IRRewriter &rewriter,
                           bool tranposeOnSource) {
  Value input = op.getSource();
  Value output = op.getDest();
  llvm::ArrayRef<int64_t> innerTiles = op.getStaticInnerTiles();
  return rewriteAsDma(rewriter, op, input, output, innerTiles,
                      tranposeOnSource);
}

/// Convert a linalg.copy operation on 2 memrefs to an equivalent pack/unpack
/// operation. If the linalg.copy operation is to a memory closer to the
/// core it is converted to a pack operation, otherwise an unpack operation.
///
/// Note: we could convert all copies to packs, but it would be potentially
/// confusing to have packs ops moving data away from cores.
LogicalResult copyToPack(IRRewriter &rewriter, linalg::CopyOp copyOp) {
  if (copyOp.getNumOperands() != 2 || copyOp.getNumResults() != 0) {
    copyOp.emitOpError()
        << "has " << copyOp.getNumOperands() << " operands and "
        << copyOp.getNumResults()
        << " results. It must have 2 operands and 0 results to convert "
           "to a linalg pack/unpack operation";
    return failure();
  }
  Value src = copyOp.getOperand(0);
  Value dst = copyOp.getOperand(1);

  // MemRefTypes with no memory space attribute return 0 here, so this is safe.
  uint32_t srcMemspace = cast<MemRefType>(src.getType()).getMemorySpaceAsInt();
  uint32_t dstMemspace = cast<MemRefType>(dst.getType()).getMemorySpaceAsInt();
  const bool towardsCore = srcMemspace <= dstMemspace;

  rewriter.setInsertionPoint(copyOp);
  if (towardsCore) {
    // Use the ODS-generated builder with null result type (Type{}) for buffer
    // semantics. The custom PackOp::build passes dest.getType() as result type
    // unconditionally, which is invalid for memref operands (buffer semantics
    // requires 0 results).
    rewriter.replaceOpWithNewOp<linalg::PackOp>(
        copyOp, /*result=*/Type{}, src, dst, /*padding_value=*/Value{},
        /*outer_dims_perm=*/ArrayRef<int64_t>{},
        /*inner_dims_pos=*/ArrayRef<int64_t>{},
        /*inner_tiles=*/ValueRange{},
        /*static_inner_tiles=*/ArrayRef<int64_t>{});
  } else {
    // Same fix for UnPackOp: pass null result type for buffer semantics.
    rewriter.replaceOpWithNewOp<linalg::UnPackOp>(
        copyOp, /*result=*/Type{}, src, dst,
        /*outer_dims_perm=*/ArrayRef<int64_t>{},
        /*inner_dims_pos=*/ArrayRef<int64_t>{},
        /*inner_tiles=*/ValueRange{},
        /*static_inner_tiles=*/ArrayRef<int64_t>{});
  }

  return success();
}

};  // namespace

class AMDAIEConvertToDmaPass
    : public impl::AMDAIEConvertToDmaBase<AMDAIEConvertToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tensor::TensorDialect, linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEConvertToDmaPass() = default;
  AMDAIEConvertToDmaPass(const AMDAIEConvertToDmaPass &pass){};
  AMDAIEConvertToDmaPass(const AMDAIEConvertToDmaOptions &options)
      : AMDAIEConvertToDmaBase(options) {}

  void runOnOperation() override;
};

void AMDAIEConvertToDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Convert all linalg.copy to iree_linalg_ext.pack/unpack ops. We then
  // bootstrap the work done for lowering the pack/unpack op to dmas as the next
  // step. This is easy to implement, but not the most direct lowering, so
  // we might want to revisit this.
  WalkResult convertCopiesWalkResult =
      getOperation()->walk([&](linalg::CopyOp copyOp) {
        if (failed(copyToPack(rewriter, copyOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (convertCopiesWalkResult.wasInterrupted()) return signalPassFailure();

  WalkResult walkResult = getOperation()->walk([&](linalg::PackOp packOp) {
    if (failed(rewriteAsDma(packOp, rewriter, packTransposeOnSource))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) signalPassFailure();

  walkResult = getOperation()->walk([&](linalg::UnPackOp unpackOp) {
    if (failed(rewriteAsDma(unpackOp, rewriter, unpackTransposeOnSource))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) signalPassFailure();
}

std::unique_ptr<Pass> createAMDAIEConvertToDmaPass(
    AMDAIEConvertToDmaOptions options) {
  return std::make_unique<AMDAIEConvertToDmaPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
