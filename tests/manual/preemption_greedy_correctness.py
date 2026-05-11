from __future__ import annotations

"""
Manual run:
  eval "$(conda shell.bash hook)" && conda activate minisgl
  PYTHONPATH=python python tests/manual/preemption_greedy_correctness.py \
    --model-path Qwen/Qwen3-0.6B \
    --enable-overlap-preemption
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare greedy outputs with and without preemption.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Recommended deferred-path pressure:
  --baseline-num-pages 4096 --preempt-num-pages 384 --num-prompts 12
  --prompt-repeat 8 --max-running-req 12 --max-tokens 96
  --max-extend-tokens 384 --require-deferred-preemption
""",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--baseline-num-pages", type=int, default=4096)
    parser.add_argument("--preempt-num-pages", type=int, default=384)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--max-running-req", type=int, default=12)
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--prompt-repeat", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-extend-tokens", type=int, default=384)
    parser.add_argument("--preempt-min-free-pages", type=int, default=1)
    parser.add_argument(
        "--enable-overlap-preemption",
        action="store_true",
        help="Run the preempt case with experimental overlap-safe preemption enabled.",
    )
    parser.add_argument(
        "--require-deferred-preemption",
        action="store_true",
        help="Require the preempt case to exercise deferred preemption.",
    )
    parser.add_argument("--worker-case", choices=["baseline", "preempt"])
    parser.add_argument("--output-json")
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_prompts(num_prompts: int, repeat: int) -> list[str]:
    prompt = (
        "Continue this exact repeated-word pattern without explanation:\n"
        + ("preemption " * repeat)
    )
    return [prompt for _ in range(num_prompts)]


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_case is not None
    assert args.output_json is not None
    if args.worker_case == "preempt":
        assert args.enable_overlap_preemption, (
            "Preempt worker must run with --enable-overlap-preemption"
        )
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "0"

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    enable_preemption = args.worker_case == "preempt"
    llm = LLM(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        num_page_override=(
            args.preempt_num_pages if enable_preemption else args.baseline_num_pages
        ),
        page_size=args.page_size,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        cuda_graph_max_bs=0,
        enable_preemption=enable_preemption,
        enable_overlap_preemption=enable_preemption and args.enable_overlap_preemption,
        dynamic_kv_allocation=True,
        decode_first=True,
        preempt_min_free_pages=args.preempt_min_free_pages,
    )
    try:
        results = llm.generate(
            make_prompts(args.num_prompts, args.prompt_repeat),
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=args.max_tokens,
            ),
        )
        payload = {
            "case": args.worker_case,
            "num_preemptions": llm.num_preemptions,
            "num_deferred_preemptions": llm.num_deferred_preemptions,
            "num_preemption_stalls": llm.num_preemption_stalls,
            "num_prefill_fit_failures": llm.num_prefill_fit_failures,
            "num_resumed_preempted_reqs": llm.num_resumed_preempted_reqs,
            "output_lengths": [len(result["token_ids"]) for result in results],
            "token_ids": [result["token_ids"] for result in results],
        }
        pathlib.Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        llm.shutdown()


def run_case(case: str, args: argparse.Namespace, output_json: pathlib.Path) -> dict:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "0"
    python_path = str(repo_root / "python")
    env["PYTHONPATH"] = (
        python_path if "PYTHONPATH" not in env else f"{python_path}:{env['PYTHONPATH']}"
    )
    cmd = [
        sys.executable,
        __file__,
        "--worker-case",
        case,
        "--output-json",
        str(output_json),
        "--model-path",
        args.model_path,
        "--dtype",
        args.dtype,
        "--baseline-num-pages",
        str(args.baseline_num_pages),
        "--preempt-num-pages",
        str(args.preempt_num_pages),
        "--page-size",
        str(args.page_size),
        "--max-running-req",
        str(args.max_running_req),
        "--num-prompts",
        str(args.num_prompts),
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--max-tokens",
        str(args.max_tokens),
        "--max-extend-tokens",
        str(args.max_extend_tokens),
        "--preempt-min-free-pages",
        str(args.preempt_min_free_pages),
    ]
    if args.enable_overlap_preemption:
        cmd.append("--enable-overlap-preemption")
    if args.require_deferred_preemption:
        cmd.append("--require-deferred-preemption")
    subprocess.run(cmd, env=env, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.worker_case is not None:
        run_worker(args)
        return
    assert args.enable_overlap_preemption, (
        "Pass --enable-overlap-preemption to validate overlap-safe preemption"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        baseline = run_case("baseline", args, tmp / "baseline.json")
        preempt = run_case("preempt", args, tmp / "preempt.json")

    assert preempt["num_preemptions"] > 0, "Expected preempt run to preempt at least once"
    if args.require_deferred_preemption:
        assert preempt["num_deferred_preemptions"] > 0, (
            "Expected preempt run to exercise deferred preemption"
        )
    assert baseline["token_ids"] == preempt["token_ids"], (
        "Greedy token ids differ between baseline and preemption runs"
    )
    print(
        json.dumps(
            {
                "baseline_num_preemptions": baseline["num_preemptions"],
                "baseline_output_lengths": baseline["output_lengths"],
                "preempt_num_deferred_preemptions": preempt["num_deferred_preemptions"],
                "preempt_num_prefill_fit_failures": preempt["num_prefill_fit_failures"],
                "preempt_num_preemptions": preempt["num_preemptions"],
                "preempt_num_preemption_stalls": preempt["num_preemption_stalls"],
                "preempt_num_resumed_preempted_reqs": preempt[
                    "num_resumed_preempted_reqs"
                ],
                "preempt_output_lengths": preempt["output_lengths"],
                "num_prompts": len(baseline["token_ids"]),
                "tokens_per_prompt": [len(ids) for ids in baseline["token_ids"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
