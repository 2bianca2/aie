// RUN: iree-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {

// CHECK-LABEL: @matmulbf16bf16f32
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xbf16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xf32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xbf16> to vector<32xbf16>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x4xbf16> to vector<32xbf16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x4xf32> to vector<16xf32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK:      %[[BCACC:.*]] = llvm.bitcast %[[FC]] : vector<16xf32> to vector<8xi64>
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(
// CHECK-SAME:         %[[FA]], %[[FB]], %[[BCACC]], %[[CONF]]) :
// CHECK-SAME:         (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32)
// CHECK-SAME:         -> vector<8xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<8xi64> to vector<16xf32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<16xf32> to vector<4x4xf32>
// CHECK:      return %[[R]] : vector<4x4xf32>
func.func @matmulbf16bf16f32(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                  %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: @matmuli8i8i32
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>
// CHECK-SAME: %[[B:.*]]: vector<8x8xi8>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xi8> to vector<32xi8>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x8xi8> to vector<64xi8>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x8xi32> to vector<32xi32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(776 : i32) : i32
// CHECK:      %[[C0I32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[IFA2512b:.*]] = llvm.bitcast %[[FA]] : vector<32xi8> to vector<8xi32>
// CHECK:      %[[IFA:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[IFA2512b]],
// CHECK-SAME:               %[[C0I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK:      %[[BCA:.*]] = llvm.bitcast %[[IFA]] : vector<16xi32> to vector<64xi8>
// CHECK:      %[[BCB:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK:      %[[BCC:.*]] = llvm.bitcast %[[FC]] : vector<32xi32> to vector<16xi64>
// CHECK:      %[[RACC:.*]] =
// CHECK-SAME:         "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:           %[[BCA]], %[[BCB]], %[[BCC]], %[[CONF]]) :
// CHECK-SAME:           (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32)
// CHECK-SAME:           -> vector<16xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<16xi64> to vector<32xi32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<32xi32> to vector<4x8xi32>
// CHECK:      return %[[R]] : vector<4x8xi32>
func.func @matmuli8i8i32(%A : vector<4x8xi8>, %B : vector<8x8xi8>,
                  %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi8>, vector<8x8xi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}
}

// -----

// strix matmul.
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu4"}>
module attributes {hal.executable.target = #foo} {
func.func @matmuli8i8i32npu4(%A : vector<8x8xi8>, %B : vector<8x8xi8>,
                  %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<8x8xi8>, vector<8x8xi8>
                                  into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}
}

// CHECK-LABEL: @matmuli8i8i32npu4
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>,  %[[B:.*]]: vector<8x8xi8>,
// CHECK-SAME: %[[C:.*]]: vector<8x8xi32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<8x8xi8> to vector<64xi8>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x8xi8> to vector<64xi8>
// CHECK:      %[[FC:.+]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<8x8xi32> to vector<64xi32>
// CHECK-DAG:  %[[CONF:.*]] = llvm.mlir.constant(776 : i32) : i32
// CHECK:      %[[BCA:.*]] = llvm.bitcast %[[FA]] : vector<64xi8> to vector<16xi32>
// CHECK:      %[[BCB:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<32xi16>
// CHECK:      %[[BCC:.*]] = llvm.bitcast %[[FC]] : vector<64xi32> to vector<32xi64>
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"(
// CHECK-SAME:         %[[BCA]], %[[BCB]], %[[BCC]], %[[CONF]]) :
// CHECK-SAME:         (vector<16xi32>, vector<32xi16>, vector<32xi64>, i32)
// CHECK-SAME:         -> vector<32xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<32xi64> to vector<64xi32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<64xi32> to vector<8x8xi32>
// CHECK:      return %[[R]] : vector<8x8xi32>

// -----

// [wmkim] added (experimental, test-only): AIE2P bf16 matmul via
// already-packed bfp16 operands. %A/%B are vector<72xi8> (64 mantissa
// bytes + 8 shared-exponent bytes - the `to_v64bfp16ebs8`/BFP16EBS8
// layout), and the op carries the `already_bfp16` unit attribute. This
// should make MatMulOpConversion (AIEVecToLLVM.cpp) skip
// emitBf16ToBfp16Ebs8 entirely (no v32bf16.to.v32accfloat /
// v64accfloat.to.v64bfp16ebs8 calls) and just split each packed operand
// into its mantissa/exponent halves before the MAC. Not reachable from the
// normal Flow/Stream/HAL pipeline - only from hand-written .mlir like this,
// fed straight to `iree-opt -convert-aievec-to-llvm`. Exists to test
// feedback item 2 (does the hardware/MAC accept an already-packed bfp16
// input directly, without a per-call in-kernel bf16->bfp16 conversion).
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu4"}>
module attributes {hal.executable.target = #foo} {
func.func @matmulbfp16packedf32npu4(%A : vector<72xi8>, %B : vector<72xi8>,
                  %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul %A, %B, %C {already_bfp16} : vector<72xi8>, vector<72xi8>
                                  into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
}

// CHECK-LABEL: @matmulbfp16packedf32npu4
// The whole point of `already_bfp16` is to skip the conversion - these two
// intrinsics must NOT appear anywhere in the lowered function.
// CHECK-NOT:  v32bf16.to.v32accfloat
// CHECK-NOT:  v64accfloat.to.v64bfp16ebs8
// Each packed operand (A, B) is split into mantissa + exponent via a plain
// llvm.shufflevector extract - 4 total, no conversion in between.
// CHECK-COUNT-4: llvm.shufflevector
// CHECK:      "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
// CHECK-SAME:         -> vector<64xi32>
// CHECK:      return {{.*}} : vector<8x8xf32>
