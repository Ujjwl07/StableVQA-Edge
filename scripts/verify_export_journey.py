#!/usr/bin/env python3
"""Record what actually happens when each StableVQA branch is exported to ONNX.

The paper claims a specific export journey: the direct export fails on fused
attention, three alternatives also fail, and an operator-level decomposition
works. This script runs that journey for real and writes a log, so that every
sentence in the paper can be traced to a captured traceback instead of to
recollection.

    python3 verify_export_journey.py --out export_journey.log

For each target it attempts, in order:

  1. classic      torch.onnx.export, TorchScript tracer, default settings.
  2. math-backend the same, but with attention pinned to the math backend.
                  THIS ONE MATTERS MOST: it is the workaround a reviewer will
                  say you should have used. If it succeeds, the paper must not
                  claim the model "cannot" be exported directly.
  3. dynamo       torch.onnx.export(dynamo=True), or dynamo_export on <2.5.
  4. script       torch.jit.script, then export.
  5. decomposed   your patched model with attention written out explicitly.

Every attempt is logged with its full traceback. Every export that succeeds is
then run under onnxruntime and compared against PyTorch on the same input, so
the numerical-agreement figure in the paper is measured rather than asserted.

WHAT YOU MUST EDIT: only build_targets(). Return a dict of
    {name: (module, example_args_tuple)}
Everything else is generic. Add a "<branch>-decomposed" entry for each patched
module so attempt 5 has something to run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import traceback

import torch

OPSET = 17          # match whatever the deployed exports actually used
TOL_REPORT = True   # run onnxruntime and report max |PyTorch - ONNX|


# ---------------------------------------------------------------------------
# EDIT THIS ONLY
# ---------------------------------------------------------------------------
def build_targets():
    """Return {name: (module_in_eval_mode, example_args_tuple)}.

    Populate from the StableVQA export code. Shapes below are the ones the
    paper's appendix specifies, so keep them unless the real code differs.

    Example:

        from stablevqa.models import SwinBackbone, Stripformer, NeuFlowV2
        return {
            "swin":       (SwinBackbone().eval(),  (torch.randn(32, 3, 224, 224),)),
            "stripformer":(Stripformer().eval(),   (torch.randn( 8, 3, 224, 224),)),
            "neuflow":    (NeuFlowV2().eval(),     (torch.randn(32, 6, 224, 224),)),
            "motion":     (Motion3DResNet().eval(),(torch.randn(1, 2, 32, 224, 224),)),
            "head":       (QualityHead().eval(),   (torch.randn(1, 27648),)),
            "full":       (StableVQA().eval(),     (torch.randn(1, 3, 32, 224, 224),)),
            # and the patched versions, for attempt 5:
            "swin-decomposed": (SwinBackboneDecomposed().eval(),
                                (torch.randn(32, 3, 224, 224),)),
        }
    """
    raise SystemExit(
        "build_targets() is empty. Fill it in with the real modules from the "
        "StableVQA export code, then re-run. See the docstring for the shape "
        "of the return value."
    )


# ---------------------------------------------------------------------------
# machinery below: no edits needed
# ---------------------------------------------------------------------------
LOG = []


def say(line=""):
    print(line)
    LOG.append(line)


def env_report():
    say("=" * 78)
    say("ENVIRONMENT  (paste this into the paper; the export claim is "
        "version-specific)")
    say("=" * 78)
    say(f"python           {sys.version.split()[0]}")
    say(f"torch            {torch.__version__}")
    for mod in ("onnx", "onnxruntime"):
        try:
            say(f"{mod:<16} {__import__(mod).__version__}")
        except Exception:
            say(f"{mod:<16} not installed")
    say(f"target opset     {OPSET}")
    say(f"cuda available   {torch.cuda.is_available()}")
    say()


@contextlib.contextmanager
def math_attention():
    """Force scaled_dot_product_attention onto its traceable math backend."""
    try:                                        # torch >= 2.3
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except Exception:
        pass
    try:                                        # torch 2.0 - 2.2
        with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False,
                enable_math=True):
            yield
        return
    except Exception:
        pass
    yield                                       # no control available


def numeric_agreement(path, module, args):
    """max |PyTorch - onnxruntime| on the same input, or a reason it was skipped."""
    if not TOL_REPORT:
        return "skipped"
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        return f"skipped ({e})"
    try:
        with torch.no_grad():
            ref = module(*args)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feed = {i.name: a.detach().cpu().numpy()
                for i, a in zip(sess.get_inputs(), args)}
        got = sess.run(None, feed)[0]
        return f"max |delta| = {np.abs(ref.detach().cpu().numpy() - got).max():.3e}"
    except Exception as e:
        return f"comparison failed: {type(e).__name__}: {e}"


def attempt(name, target, module, args, fn):
    """Run one export strategy; log success or the complete traceback."""
    say("-" * 78)
    say(f"[{target}] attempt: {name}")
    say("-" * 78)
    path = f"_export_{target}_{name}.onnx"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            fn(path)
    except BaseException:
        say("RESULT: FAILED")
        say("--- traceback ---")
        say(traceback.format_exc().rstrip())
        warn = buf.getvalue().strip()
        if warn:
            say("--- stderr ---")
            say(warn[:4000])
        say()
        return False
    ok = os.path.isfile(path)
    say(f"RESULT: {'SUCCEEDED' if ok else 'returned without writing a file'}")
    if ok:
        say(f"numeric check: {numeric_agreement(path, module, args)}")
        say(f"file size: {os.path.getsize(path) / 1e6:.1f} MB")
    warn = buf.getvalue().strip()
    if warn:
        say("--- warnings ---")
        say(warn[:4000])
    say()
    return ok


def run_target(target, module, args):
    say("#" * 78)
    say(f"# TARGET: {target}")
    say("#" * 78)
    say()

    attempt("1-classic", target, module, args,
            lambda p: torch.onnx.export(module, args, p, opset_version=OPSET))

    def math_export(p):
        with math_attention():
            torch.onnx.export(module, args, p, opset_version=OPSET)
    attempt("2-math-backend", target, module, args, math_export)

    def dynamo_export(p):
        try:
            torch.onnx.export(module, args, p, dynamo=True)   # torch >= 2.5
        except TypeError:
            torch.onnx.dynamo_export(module, *args).save(p)   # torch 2.1-2.4
    attempt("3-dynamo", target, module, args, dynamo_export)

    def script_export(p):
        torch.onnx.export(torch.jit.script(module), args, p, opset_version=OPSET)
    attempt("4-script", target, module, args, script_export)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="export_journey.log")
    ap.add_argument("--only", help="run a single target by name")
    args_cli = ap.parse_args()

    env_report()
    targets = build_targets()
    if args_cli.only:
        targets = {args_cli.only: targets[args_cli.only]}
    for name, (module, args) in targets.items():
        run_target(name, module.eval(), args)

    with open(args_cli.out, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    say(f"log written to {args_cli.out}")


if __name__ == "__main__":
    main()
