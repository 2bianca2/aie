#!/usr/bin/env python3
"""Dump per-stage MLIR + artifacts for AMD-AIE compilation debugging.

Runs inside the dev container. Given an ONNX model (or an already-imported
.mlir), the default (full) mode dumps every compilation phase IR in one shot
via --dump-compilation-phases-to, dumps the backend artifacts, compiles to a
vmfb, runs it on the NPU with .npy inputs, and optionally compares the output
against a numpy reference. Every command is recorded in MANIFEST.txt so a stage
can be reproduced by hand.

For pass research there is a resume mode: --from-phase <phase> re-runs only a
segment (--compile-from) starting from a previously dumped phase IR of the same
--label, optionally extracting a specific pass's before/after IR with --pass.
This skips the front-end so an edited pass can be re-checked quickly. Resume
mode is IR-only (no NPU run) and does not wipe the baseline artifacts.

Only ONNX (and raw .mlir) inputs are supported by design; the target model
is ONNX. Inputs/outputs use .npy and are passed to iree-run-module WITHOUT a
shape prefix (`--input=@x.npy`): a prefix makes iree read the file as raw
bytes and swallow the 128-byte npy header, yielding garbage.

Typical use is through scripts/docker/run-debug.sh, which sets up the venv
and PYTHONPATH inside the container.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Valid IREEVM compilation phases (see third_party/iree/.../Pipelines/Pipelines.h,
# enumerateIREEVMPipelinePhases). --dump-compilation-phases-to writes one
# N.<phase>.mlir file per phase that runs (N = enum ordinal), so a full compile
# yields the whole set below. These names are also the accepted values for
# --from-phase / --to-phase. `start`/`end` are omitted (not useful cut-points for
# a resumable IR file). AIE codegen lives in executable-configurations (strategy
# selected) -> executable-targets (amdaie->aie translation).
VALID_PHASES = [
    "input",
    "abi",
    "preprocessing",
    "global-optimization",
    "dispatch-creation",
    "flow",
    "stream",
    "executable-sources",
    "executable-configurations",
    "executable-targets",
    "hal",
    "vm",
]

# Device selection passed to iree-compile when --device-flag is not given. A
# single amd-aie device is enough for a matmul-only model; a heterogeneous
# CPU+NPU model (anything with pooling, or a conv lowered through im2col) needs
# both devices declared, which is what --device-flag replaces this with.
DEFAULT_DEVICE_FLAGS = ["--iree-hal-target-backends=amd-aie"]


def tool_path(env_name, default):
    return os.environ.get(env_name, default)


class Manifest:
    """Accumulates every command run, with rc and duration, into MANIFEST.txt."""

    def __init__(self, path: Path):
        self.path = path
        self.lines = []

    def record(self, title, cmd, rc, dur, log=None):
        self.lines.append(f"## {title}")
        self.lines.append(f"# rc={rc}  duration={dur:.2f}s" + (f"  log={log}" if log else ""))
        self.lines.append(" \\\n  ".join(cmd))
        self.lines.append("")

    def note(self, text):
        self.lines.append(text)
        self.lines.append("")

    def flush(self):
        self.path.write_text("\n".join(self.lines) + "\n")


def run(cmd, manifest, title, log_path=None, check=True):
    """Run a subprocess, tee stderr to log_path if given, record in manifest."""
    t0 = time.time()
    if log_path is not None:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=logf, text=True)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dur = time.time() - t0
    rel_log = str(log_path.relative_to(manifest.path.parent)) if log_path else None
    manifest.record(title, cmd, proc.returncode, dur, rel_log)
    if proc.returncode != 0:
        sys.stderr.write(f"[FAIL] {title} (rc={proc.returncode})\n")
        if log_path is not None:
            sys.stderr.write(f"       see {log_path}\n")
        elif proc.stderr:
            sys.stderr.write(proc.stderr)
        if check:
            manifest.flush()
            sys.exit(proc.returncode)
    else:
        sys.stderr.write(f"[ok]   {title} ({dur:.1f}s)\n")
    return proc


def find_phase_file(phases_dir, phase):
    """Locate the dumped IR for a phase (--dump-compilation-phases-to writes
    <ordinal>.<phase>.mlir). Returns the Path or None."""
    matches = sorted(phases_dir.glob(f"*.{phase}.mlir"))
    return matches[0] if matches else None


def run_resume(args, model, run_dir, common):
    """Resume a segment from a prior baseline dump (IR-only, no NPU run).

    Reads phases/<N>.<from-phase>.mlir from the same --label run_dir and runs
    iree-compile --compile-from=<from> --compile-to=<to> on it, so the front-end
    is skipped. Baseline artifacts are left untouched; outputs go under resume/.
    """
    phases_dir = run_dir / "phases"
    if not phases_dir.is_dir():
        sys.exit(f"no baseline dump at {phases_dir}; run without --from-phase "
                 f"first (same --label) to produce phase IRs")
    src = find_phase_file(phases_dir, args.from_phase)
    if src is None:
        avail = ", ".join(sorted(p.name for p in phases_dir.glob("*.mlir")))
        sys.exit(f"phase '{args.from_phase}' not found in {phases_dir} "
                 f"(available: {avail or 'none'}); run a baseline dump first")

    resume_dir = run_dir / "resume"
    resume_dir.mkdir(exist_ok=True)
    manifest = Manifest(resume_dir / "MANIFEST.txt")
    manifest.note(f"# resume {args.from_phase} -> {args.to_phase} from "
                  f"phases/{src.name} (model={model.name})")

    tag = f"{args.from_phase}_to_{args.to_phase}"
    out = resume_dir / f"{tag}.mlir"
    cmd = [args.iree_compile, str(src), *common,
           f"--compile-from={args.from_phase}", f"--compile-to={args.to_phase}"]
    if args.passes:
        passes_dir = resume_dir / "passes"
        passes_dir.mkdir(exist_ok=True)
        names = ",".join(args.passes)
        cmd += [f"--mlir-print-ir-before={names}", f"--mlir-print-ir-after={names}",
                f"--mlir-print-ir-tree-dir={passes_dir}"]
    cmd += ["-o", str(out)]
    run(cmd, manifest, f"resume {args.from_phase} -> {args.to_phase}",
        log_path=resume_dir / f"{tag}.log")
    if args.passes:
        manifest.note(f"# requested passes: {','.join(args.passes)} "
                      f"(before/after IR under resume/passes/)")
    manifest.flush()
    sys.stderr.write(f"\nDone (resume). Artifacts in: {resume_dir}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help=".onnx or already-imported .mlir")
    ap.add_argument("--function", required=True, help="entry function name (ONNX graph name)")
    ap.add_argument("--input", action="append", default=[], dest="inputs",
                    help=".npy input file, repeat in func arg order")
    ap.add_argument("--expected", help=".npy reference output to compare out0 against")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--num-outputs", type=int, default=1, help="number of result tensors")
    ap.add_argument("--target-device", default="npu4", help="npu4 (Strix) | npu1_4col (Phoenix)")
    ap.add_argument("--device-flag", action="append", default=[], dest="device_flags",
                    help="device selection flag for iree-compile, repeatable. If "
                         "given at all, REPLACES the default (%s). Use to debug a "
                         "heterogeneous CPU+NPU compile." % " ".join(DEFAULT_DEVICE_FLAGS))
    ap.add_argument("--compile-flag", action="append", default=[], dest="compile_flags",
                    help="extra flag appended to every iree-compile invocation, "
                         "repeatable (e.g. the demote/im2col flags a conv model needs)")
    ap.add_argument("--run-device", action="append", default=[], dest="run_devices",
                    help="--device for iree-run-module, repeatable; default amdxdna. "
                         "A heterogeneous vmfb needs amdxdna and local-task both")
    ap.add_argument("--stack-size", type=int, default=2048)
    ap.add_argument("--elide", type=int, default=64,
                    help="--mlir-elide-elementsattrs-if-larger for readable dumps")
    ap.add_argument("--pass", action="append", default=[], dest="passes",
                    help="MLIR pass ARG name (e.g. iree-amdaie-lower-to-aie, not the C++ "
                         "class) to extract before/after IR for; repeatable")
    ap.add_argument("--from-phase", choices=VALID_PHASES, default=None, dest="from_phase",
                    help="resume mode: re-run only from this phase's dumped IR "
                         "(--compile-from), reusing a prior baseline dump of the same "
                         "--label; skips the front-end. IR-only (no NPU run)")
    ap.add_argument("--to-phase", choices=VALID_PHASES, default="executable-targets",
                    dest="to_phase", help="resume mode: stop at this phase "
                         "(--compile-to); default executable-targets (AIE codegen)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--outdir", default="debug_out")
    ap.add_argument("--iree-compile", default=tool_path(
        "IREE_COMPILE", "/workspace/build/tools/iree-compile"))
    ap.add_argument("--iree-run-module", default=tool_path(
        "IREE_RUN_MODULE", "/workspace/build/tools/iree-run-module"))
    ap.add_argument("--peano", default=tool_path("PEANO_INSTALL_DIR", "/workspace/llvm-aie"))
    args = ap.parse_args()

    model = Path(args.model).resolve()
    if not model.exists():
        sys.exit(f"model not found: {model}")
    in_paths = [Path(p).resolve() for p in args.inputs]
    for p in in_paths:
        if not p.exists():
            sys.exit(f"input not found: {p}")
        if p.suffix != ".npy":
            sys.exit(f"inputs must be .npy (no shape prefix is used): {p}")

    run_dir = Path(args.outdir).resolve() / f"{model.stem}_{args.label}"

    common = [
        *(args.device_flags or DEFAULT_DEVICE_FLAGS),
        f"--iree-amdaie-target-device={args.target_device}",
        f"--iree-amd-aie-peano-install-dir={args.peano}",
        f"--iree-amdaie-stack-size={args.stack_size}",
        f"--mlir-elide-elementsattrs-if-larger={args.elide}",
        # Last, so a repeated flag here wins over the defaults above.
        *args.compile_flags,
    ]

    # Resume mode: re-run only a segment from a prior baseline dump (IR-only).
    if args.from_phase is not None:
        run_resume(args, model, run_dir, common)
        return

    # ---- full mode: import -> dump every phase + vmfb -> run -> compare ----
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "phases").mkdir(parents=True)
    (run_dir / "backend").mkdir()
    (run_dir / "run").mkdir()
    manifest = Manifest(run_dir / "MANIFEST.txt")
    manifest.note(f"# pipeline_dump for {model.name} (function={args.function}, "
                  f"target={args.target_device})")

    # 1. import (onnx path) -> 00_source.mlir
    source = run_dir / "00_source.mlir"
    if model.suffix == ".mlir":
        shutil.copy(model, source)
        manifest.note(f"## import\n# copied {model} -> 00_source.mlir")
    else:
        run([sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(model), "-o", str(source)], manifest, "import (import_onnx)")

    # 2. full compile -> model.vmfb, dumping every phase IR in one shot
    #    (--dump-compilation-phases-to writes phases/N.<phase>.mlir per phase).
    vmfb = run_dir / "model.vmfb"
    run([args.iree_compile, str(source), *common,
         f"--dump-compilation-phases-to={run_dir / 'phases'}",
         f"--iree-hal-dump-executable-files-to={run_dir / 'backend' / 'dump_files'}",
         f"--iree-hal-dump-executable-intermediates-to={run_dir / 'backend' / 'dump_intermediates'}",
         # aie2xclbin IR printing requires single-threaded pass managers.
         "--aie2xclbin-print-ir-after-all", "--aie2xclbin-print-ir-module-scope",
         "--mlir-disable-threading",
         "-o", str(vmfb)],
        manifest, "full compile (-> vmfb, all-phase dump + backend dumps)",
        log_path=run_dir / "backend" / "aie2xclbin.log")
    # --dump-compilation-phases-to names files "<src-stem>.<N>.<phase>.mlir";
    # drop the redundant source-stem prefix so they read as N.<phase>.mlir.
    prefix = f"{source.stem}."
    for f in (run_dir / "phases").glob(f"{source.stem}.*.mlir"):
        f.rename(f.with_name(f.name[len(prefix):]))
    manifest.note("# phase IRs -> phases/N.<phase>.mlir (--dump-compilation-phases-to)")

    # 3. (optional) per-pass before/after extraction into passes/ tree.
    #    (For fast iteration on an edited pass, prefer resume mode: --from-phase.)
    if args.passes:
        passes_dir = run_dir / "passes"
        passes_dir.mkdir()
        names = ",".join(args.passes)
        run([args.iree_compile, str(source), *common,
             "--compile-to=executable-targets",
             f"--mlir-print-ir-before={names}", f"--mlir-print-ir-after={names}",
             f"--mlir-print-ir-tree-dir={passes_dir}", "-o", os.devnull],
            manifest, f"pass extraction: {names}",
            log_path=passes_dir / "_ir_print.log")
        manifest.note(f"# requested passes: {names} (files under passes/ tree)")

    # 4. run on NPU
    out_paths = [run_dir / "run" / f"out{k}.npy" for k in range(args.num_outputs)]
    run_cmd = [args.iree_run_module]
    run_cmd += [f"--device={d}" for d in (args.run_devices or ["amdxdna"])]
    run_cmd += [f"--module={vmfb}", f"--function={args.function}"]
    for p in in_paths:
        run_cmd.append(f"--input=@{p}")          # NO shape prefix (npy header read)
    for p in out_paths:
        run_cmd.append(f"--output=@{p}")
    run(run_cmd, manifest, "run on NPU (iree-run-module)",
        log_path=run_dir / "run" / "run.log")

    # 5. (optional) numpy compare
    if args.expected:
        import numpy as np
        exp = np.load(Path(args.expected).resolve())
        got = np.load(out_paths[0])
        diff = float(np.max(np.abs(exp - got))) if exp.shape == got.shape else float("nan")
        report = [
            f"expected: {args.expected} shape={exp.shape} dtype={exp.dtype}",
            f"got:      {out_paths[0].name} shape={got.shape} dtype={got.dtype}",
            f"max abs diff: {diff}",
            f"rtol={args.rtol} atol={args.atol}",
        ]
        try:
            np.testing.assert_allclose(got, exp, rtol=args.rtol, atol=args.atol)
            report.append("RESULT: PASS (allclose)")
        except AssertionError as e:
            report.append("RESULT: FAIL")
            report.append(str(e))
        (run_dir / "run" / "compare.txt").write_text("\n".join(report) + "\n")
        result_line = next(l for l in report if l.startswith("RESULT"))
        manifest.note(f"## compare\n# max abs diff={diff} | {result_line}")
        sys.stderr.write(f"[compare] max abs diff={diff} | {result_line}\n")

    manifest.flush()
    sys.stderr.write(f"\nDone. Artifacts in: {run_dir}\n")


if __name__ == "__main__":
    main()
