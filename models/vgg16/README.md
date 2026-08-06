# VGG-16 (ONNX) end-to-end on npu4

Full `vgg16-12.onnx` — 13 conv + 13 relu + 5 maxpool + flatten + 3 FC,
input `[1,3,224,224]` -> `[1,1000]` logits — compiled and run heterogeneously:
conv/matmul on the NPU, pooling/flatten/layout ops on the CPU.

This is the largest model the amd-aie plugin is known to run end to end, so the
recipe below doubles as the reference for which compile flags a real f32 CNN
currently needs. Every flag is listed with the reason it is required; none of
them is optional today.

## 1. Get the model

Not committed (528 MB). From the ONNX Model Zoo:

```bash
cd models/vgg16
curl -fSLO https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-12.onnx
```

## 2. Host prerequisites

Two host-side settings are required; without them the run fails at runtime, not
at compile time.

- **NPU dispatch timeout.** The default 2 s TDR aborts VGG's large dispatches
  with `ert state 6`. Raise it once (survives reboot) — see
  [`USER_GUIDE.md` 1-3](../../docs/2026-07-06_env_setup/USER_GUIDE.md).
- **`RLIMIT_MEMLOCK`.** amdxdna `HOST_ONLY` BOs pin memory; the default 8 MB
  container cap makes VGG's im2col/activation BOs fail with `errno 11`. The
  `scripts/docker/run-*.sh` wrappers already pass `--ulimit memlock=-1`.

## 3. Compile and run

Inside the dev container (`./scripts/docker/run-dev.sh`), from `/workspace`:

```bash
python3 -m iree.compiler.tools.import_onnx models/vgg16/vgg16-12.onnx -o /tmp/vgg.mlir

build/tools/iree-compile /tmp/vgg.mlir -o /tmp/vgg.vmfb \
  --iree-hal-target-device=npu=amdxdna \
  --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-global-opt-use-im2col-for-convs \
  --iree-amdaie-demote-contraction-inputs-to-bf16 \
  --iree-amdaie-enable-vectorization-passes=false \
  --iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-convert-conv-to-channels-last) \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots \
  --iree-global-opt-detach-elementwise-through-reshape \
  --iree-flow-enable-executable-deduplication=false

build/tools/iree-run-module --device=amdxdna --device=local-task \
  --module=/tmp/vgg.vmfb --function=mxnet_converted_model \
  --input=@input.npy --output=@out.npy
```

The entry function name comes from the ONNX graph name; read it back with
`grep -oE "func.func @[A-Za-z0-9_]+" /tmp/vgg.mlir | head -1`.

Measured on npu4: compile ~130 s, run ~75 s, output `[1,1000]` correlates
1.00000 with a torch reference and the argmax matches.

## 4. Why each flag

| Flag | Why |
| --- | --- |
| `--iree-hal-target-device=npu=amdxdna` + `cpu=local` + `--iree-hal-local-target-device-backends=llvm-cpu` + `--iree-hal-default-device=npu` | Declares both devices. Without a CPU device the heterogeneous placement pass is a no-op (`AMDAIEAssignDeviceAffinities` requires a host device), so pooling/flatten have nowhere to go and the model does not compile. |
| `--iree-amdaie-target-device=npu4` | Strix. Phoenix would be `npu1_4col`. |
| `--iree-amd-aie-peano-install-dir` | `PEANO_INSTALL_DIR` alone is not read by `iree-compile`. |
| `--iree-global-opt-use-im2col-for-convs` | The only conv path that reaches the NPU: convs are lowered to matmuls. Standard NCHW conv codegen is not supported. |
| `--iree-amdaie-demote-contraction-inputs-to-bf16` | npu4 has no f32 vector path. f32 conv aborts and f32 matmul only runs scalar; demoting inputs (accumulation stays f32) makes them run as bf16. This is also the precondition for turning vectorization on later — npu4's matmul intrinsics are bf16 — so it stays even though today, paired with vectorization off, it only costs precision. |
| `--iree-amdaie-enable-vectorization-passes=false` | aievec currently only handles i8 `vector.contract`; bf16 fails to vectorize, so the vectorization passes must be off. |
| `--iree-preprocessing-pass-pipeline=...convert-conv-to-channels-last` | ONNX convs are NCHW; the im2col path expects NHWC. |
| `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` | Works around a codegen gap: amd-aie cannot tile a contraction dispatch that has elementwise ops fused into it, so fusion across contraction/conv group boundaries is switched off. The real fix is on the codegen side, not here. |
| `--iree-global-opt-detach-elementwise-through-reshape` | Detaches the bias add from the conv init through the im2col reshape, so the contraction dispatch has a plain `linalg.fill` init. |
| `--iree-flow-enable-executable-deduplication=false` | Deduplicating two convs that differ only in constant-weight offset turns that offset into a runtime push constant, which `amdaie.npu.address_patch` cannot carry (it is static-only). The N-split pass also relies on its clones not being merged. |

The last three are flags of the `ace-knu/iree` fork (see `.gitmodules`); they do
not exist in upstream IREE.

## 5. Known limitations

- Batch size is fixed at 1 by the ONNX file; other batch sizes are untested.
- The flag list is not a stable interface, and passing it by hand is a stopgap.
  Three of the flags exist only in the `ace-knu/iree` fork, so nothing in
  `--help` or in upstream documentation leads a reader to this combination —
  this file is the only place it is written down. Two more work around current
  gaps (aievec has no bf16 path; `no-fuse` above) and should disappear as those
  are fixed.
- Omitting a flag does not produce a diagnostic naming it; the failure is a
  compile crash deep in the pipeline or an `ert state 6` at runtime. Follow the
  recipe exactly rather than bisecting it.
- The flags encode facts about npu4, not user preferences. A second NPU target
  would need its own combination, and this document would have to fork with it.
  Moving these decisions behind a per-target capability query is tracked as
  future work, not done here.
