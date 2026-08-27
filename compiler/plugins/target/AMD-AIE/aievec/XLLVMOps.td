//===- XLLVMOps.td - XLLVM AIE intrinsic op defs. ----*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines external LLVM (XLLVM) intrinsic operations for AIE devices.
//
// These are a subset of the intrinsics defined for AIE, currently in the
// files:
//
// llvm/include/llvm/IR/IntrinsicsAIE2.td for the AIE2 architecture, and
// llvm/include/llvm/IR/IntrinsicsAIE2P.td for the AIE2P architecture.
//
// These files are in the llvm-aie fork of LLVM, currently at
//
// https://github.com/Xilinx/llvm-aie
//
// The ops defined in this file are a 1:1 mapping from the intrinsics
// defined in the files above.
//
//===----------------------------------------------------------------------===//


#ifndef AIE_DIALECT_XLLVM_IR_XLLVMINTROPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMINTROPS_TD

include "XLLVM.td"
include "XLLVMTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

////////////////////////////////////
// Intrinsics for AIE2P ('strix') //
////////////////////////////////////

class AIEVec2P_IntrOp<string mnemonic,
                      list<Trait> traits = [],
                      int numResults = 1> :
     ExtIntrOpBase</*opName =*/"intr.aie2p." # mnemonic,
                   /*enumName =*/"aie2p." # mnemonic,
                   traits, numResults>;

// ----- MAC -----

def AIEVec2PMacConfAcc64IntrOp :
    AIEVec2P_IntrOp<"I512.I512.ACC2048.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[32], [I16]>:$rhs,
                   VectorOfLengthAndType<[32], [I64]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({16}, IntegerType::get(context, 32)),
            VectorType::get({32}, IntegerType::get(context, 16)),
            VectorType::get({32}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({32}, IntegerType::get(context, 64));
    }
}];
}

// [wmkim] added: AIE2P bf16 matmul MAC intrinsics (were missing before this).
// [wmkim] Declaring the 2 shapes whose M/N/K interpretation we could verify
// [wmkim] cleanly against IntrinsicsAIE2P.td (lhs elems = M*K, rhs elems =
// [wmkim] K*N, acc elems = M*N, matching a single matmul tile exactly).
// [wmkim] peano also exposes I512.I512.ACC1024.bf.mac.conf and
// [wmkim] I512.I512.ACC2048.bf.mac.conf, but those pair 32-lane (I512)
// [wmkim] inputs with a 32/64-lane accumulator, which cannot correspond to a
// [wmkim] single M*K=32, K*N=32 tile (that forces M*N=16, i.e. ACC512) -
// [wmkim] they are presumably a batched/replicated mode controlled by the
// [wmkim] conf bits, not confirmed from the docs we have, and not needed for
// [wmkim] either target shape below, so left undeclared for now.
// [wmkim] All of AIE2P's bf16 acc/result types are genuine f32 vectors,
// [wmkim] unlike the AIE2 bf16 MAC below which bitcasts acc to i64.

// [wmkim] lhs/rhs: 32 bf16 lanes (512 bits, "I512"), acc/res: 16 f32 lanes
// [wmkim] (512 bits, "ACC512"). e.g. matches a 4x8 @ 8x4 -> 4x4 tile.
// [wmkim] Mirrors int_aie2p_I512_I512_ACC512_bf_mac_conf:
// [wmkim] DefaultAttrsIntrinsic<[llvm_v16f32_ty],
// [wmkim] [llvm_v32bf16_ty, llvm_v32bf16_ty, llvm_v16f32_ty, llvm_i32_ty],
// [wmkim] [IntrReadMem, IntrInaccessibleMemOnly]>.
def AIEVec2PMacConfAcc512BF16IntrOp :
    AIEVec2P_IntrOp<"I512.I512.ACC512.bf.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [F32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   VectorOfLengthAndType<[16], [F32]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({16}, Float32Type::get(context)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({16}, Float32Type::get(context));
    }
}];
}

// [wmkim] lhs/rhs: 64 bf16 lanes (1024 bits, "I1024"), acc/res: 64 f32 lanes
// [wmkim] (2048 bits, "ACC2048"). e.g. matches an 8x8 @ 8x8 -> 8x8 tile,
// [wmkim] i.e. the same M=N=K=8 shape already used for AIE2P's integer matmul
// [wmkim] in getSuportedAie2PTypes.
// [wmkim] Mirrors int_aie2p_I1024_I1024_ACC2048_bf_mac_conf:
// [wmkim] DefaultAttrsIntrinsic<[llvm_v64f32_ty],
// [wmkim] [llvm_v64bf16_ty, llvm_v64bf16_ty, llvm_v64f32_ty, llvm_i32_ty],
// [wmkim] [IntrReadMem, IntrInaccessibleMemOnly]>.
def AIEVec2PMacConfAcc2048BF16IntrOp :
    AIEVec2P_IntrOp<"I1024.I1024.ACC2048.bf.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[64], [F32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [BF16]>:$lhs,
                   VectorOfLengthAndType<[64], [BF16]>:$rhs,
                   VectorOfLengthAndType<[64], [F32]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({64}, BFloat16Type::get(context)),
            VectorType::get({64}, BFloat16Type::get(context)),
            VectorType::get({64}, Float32Type::get(context)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({64}, Float32Type::get(context));
    }
}];
}

// [wmkim] added: the REAL AIE2P bf16 matmul pipeline. AIEVec2PMacConfAcc2048BF16IntrOp
// [wmkim] above (raw v64bf16 x v64bf16 -> v64f32) turned out to not correspond to
// [wmkim] any real hardware calling convention: it silently produced garbage on
// [wmkim] real hardware (all-zero after the cMode fix below became a
// [wmkim] non-zero-but-wrong result). AIE2P's bf16 MAC datapath actually
// [wmkim] operates on "BFP" (block floating point, one shared 8-bit exponent per
// [wmkim] 8-lane group) operands, not raw bf16 - confirmed against the
// [wmkim] validated reference in uKernels/npu4/peano/matmul.cc
// [wmkim] (matmul_vectorized_bf16_f32 / load_v64bf16_as_bfp16(_T) / mac_8x8_8x8T),
// [wmkim] and by compiling the aie2p_ups.h/aie2p_srs.h/aie2p_vmult.h peano
// [wmkim] intrinsic headers with `clang --target=aie2p-none-unknown-elf
// [wmkim] -S -emit-llvm` and inspecting the emitted LLVM IR for the true
// [wmkim] intrinsic names/signatures (the upstream .td intrinsic sources aren't
// [wmkim] shipped with this peano release, only the C++ intrinsic headers and a
// [wmkim] prebuilt clang, so this was the only way to recover them).
// [wmkim] The 3 ops below implement: ups (widen bf16->f32 accumulator),
// [wmkim] pack (f32 accumulator -> block-floating-point mantissa/exponent
// [wmkim] pair), and the real mac. See MatMulOpConversion in AIEVecToLLVM.cpp
// [wmkim] for how they + the vshuffle op above are combined to reproduce
// [wmkim] matmul_vectorized_bf16_f32's data flow.

// ups: widen 32 bf16 lanes to a 32-lane f32 accumulator.
// Mirrors int_aie2p_v32bf16_to_v32accfloat (name recovered from codegen, see
// [wmkim] comment above; not independently confirmed against a .td source).
def AIEVec2PUpsV32BF16IntrOp :
    AIEVec2P_IntrOp<"v32bf16.to.v32accfloat",
        [TypeIs<"res", VectorOfLengthAndType<[32], [F32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$src)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {VectorType::get({32}, BFloat16Type::get(context))};
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({32}, Float32Type::get(context));
    }
}];
}

// Pack a 64-lane f32 accumulator into block-floating-point form: a struct of
// {64x i8 mantissa, 8x i8 exponent} (one shared exponent per 8-lane group,
// hence 64/8 = 8 exponents). Mirrors int_aie2p_v64accfloat_to_v64bfp16ebs8.
def AIEVec2PToBfp16Ebs8IntrOp :
    AIEVec2P_IntrOp<"v64accfloat.to.v64bfp16ebs8", []>,
    Arguments<(ins VectorOfLengthAndType<[64], [F32]>:$src)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {VectorType::get({64}, Float32Type::get(context))};
    }
    static Type expectedResultType(MLIRContext *context) {
        return LLVM::LLVMStructType::getLiteral(
            context, {VectorType::get({64}, IntegerType::get(context, 8)),
                      VectorType::get({8}, IntegerType::get(context, 8))});
    }
}];
}

// The real 8x8x8 bf16 MAC: takes mantissa/exponent pairs for both operands
// (the "BFP576" name = 64x8b mantissa + 8x8b exponent = 512+64 = 576 bits),
// an i32-bitcast f32 accumulator, and a config word. Mirrors
// int_aie2p_BFP576_BFP576_ACC2048_mac_conf / mac_8x8_8x8T's underlying
// builtin. Unlike AIEVec2PMacConfAcc2048BF16IntrOp above, this one is backed
// by real hardware behavior confirmed via the peano ukernel reference.
def AIEVec2PBfp576MacConfIntrOp :
    AIEVec2P_IntrOp<"BFP576.BFP576.ACC2048.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[64], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$aMantissa,
                   VectorOfLengthAndType<[8], [I8]>:$aExponent,
                   VectorOfLengthAndType<[64], [I8]>:$bMantissa,
                   VectorOfLengthAndType<[8], [I8]>:$bExponent,
                   VectorOfLengthAndType<[64], [I32]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({64}, IntegerType::get(context, 8)),
            VectorType::get({8}, IntegerType::get(context, 8)),
            VectorType::get({64}, IntegerType::get(context, 8)),
            VectorType::get({8}, IntegerType::get(context, 8)),
            VectorType::get({64}, IntegerType::get(context, 32)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({64}, IntegerType::get(context, 32));
    }
}];
}

// [wmkim] added: AIE2P vshuffle intrinsic (needed to lower a 1024-bit/16-bit
// [wmkim] element/8x8 vector.transpose, e.g. an 8x8 bf16 tile, since AIE2P's
// [wmkim] vshuffle hardware unit - like AIE2's - is fixed at 512 bits. A
// [wmkim] genuinely 1024-bit transpose is synthesized as two 512-bit
// [wmkim] vshuffle calls using the T16_8X8_LO/T16_8X8_HI modes (see
// [wmkim] AIEVecAttributes.td), whose results are concatenated back together
// [wmkim] in VectorToAIEVecConversions.cpp. Same signature shape as AIE2's
// [wmkim] AIEVec2VectorShuffleIntrOp below (v16i32, v16i32, i32 mode ->
// [wmkim] v16i32); confirmed via strings on libLLVM-19.so
// [wmkim] (llvm.aie2p.vshuffle) and against IntrinsicsAIE2P.td:
// [wmkim] class AIE2PVShuffle : DefaultAttrsIntrinsic<[llvm_v16i32_ty],
// [wmkim] [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty], [IntrNoMem]>;
// [wmkim] def int_aie2p_vshuffle : ClangBuiltin<"__builtin_aie2p_vshuffle">,
// [wmkim] AIE2PVShuffle;
def AIEVec2PVectorShuffleIntrOp :
    AIEVec2P_IntrOp<"vshuffle",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$mode)>;
// [wmkim] end of added block.

/////////////////////////////////////
// Intrinsics for AIE2 ('phoenix') //
/////////////////////////////////////

class AIEVec2_IntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    ExtIntrOpBase</*opName =*/"intr.aie2." # mnemonic,
                  /*enumName =*/"aie2." # mnemonic,
                  traits,
                  numResults>;


// ----- MAC -----

def AIEVec2MacConfAcc32IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc32.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({64}, IntegerType::get(context, 8)),
            VectorType::get({16}, IntegerType::get(context, 32)),
            VectorType::get({16}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({16}, IntegerType::get(context, 64));
    }
}];
}


def AIEVec2MacConfAcc64IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc64.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>;


def AIEVec2MacConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.mac16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   VectorOfLengthAndType<[8], [I64]>:$acc,
                   I32:$conf)> {
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({8}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({8}, IntegerType::get(context, 64));
    }
}];
}

// ----- SET -----

def AIEVec2VectorSetI512I128IntrOp :
    AIEVec2_IntrOp<"set.I512.I128",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[4], [I32]>:$src)>;

def AIEVec2VectorSetI512I256IntrOp :
    AIEVec2_IntrOp<"set.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$pos)>;

// ----- SRS -----

def AIEVec2I256V16Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I8]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V8Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v8.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I512V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I512.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I512V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I512.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Vector16AccFloatToV16BF16IntrOp :
    AIEVec2_IntrOp<"v16accfloat.to.v16bf16",
        [TypeIs<"res", VectorOfLengthAndType<[16], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src)>;

// ----- EXT -----

def AIEVec2ExtI256I512IntrOp :
    AIEVec2_IntrOp<"ext.I256.I512",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI512I1024IntrOp :
    AIEVec2_IntrOp<"ext.I512.I1024",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI256I1024IntrOp :
    AIEVec2_IntrOp<"ext.I256.I1024",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI128I512IntrOp :
    AIEVec2_IntrOp<"extract.I128.I512",
        [TypeIs<"res", VectorOfLengthAndType<[4], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src)>;

// ----- CONCAT -----

def AIEVec2ConcatI512I256IntrOp :
    AIEVec2_IntrOp<"concat.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$lhs,
                   VectorOfLengthAndType<[8], [I32]>:$rhs)>;

def AIEVec2ConcatI1024I512IntrOp :
    AIEVec2_IntrOp<"concat.I1024.I512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs)>;

// ----- SHUFFLE -----

def AIEVec2VectorShuffleIntrOp :
    AIEVec2_IntrOp<"vshuffle",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$mode)>;

// ----- UNDEF -----

def AIEVec2UndefV16I32IntrOp :
    AIEVec2_IntrOp<"v16int32",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>;

// ----- UPS -----

def AIEVec2Acc32V16I256UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v16.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc32V32I256UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v32.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I8]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc32V32I512UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v32.I512.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V16I256UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v16.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V16I512UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v16.I512.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V8I256UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v8.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Vector16BF16ToV16AccFloatIntrOp :
    AIEVec2_IntrOp<"v16bf16.to.v16accfloat",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [BF16]>:$src)>;

// ----- SHIFT -----

def AIEVec2VectorShiftI512I512IntrOp :
    AIEVec2_IntrOp<"vshift.I512.I512",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$step,
                   I32:$shift)>;

def AIEVec2VectorShiftBF512BF512IntrOp :
    AIEVec2_IntrOp<"vshift.bf512.bf512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   I32:$step,
                   I32:$shift)>;


#endif
