from __future__ import annotations

"""
Manual run:
  eval "$(conda shell.bash hook)" && conda activate minisgl
  PYTHONPATH=python python tests/manual/preemption_overlap_benchmark.py \
    --model-path Qwen/Qwen3-0.6B
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare normal preemption and overlap preemption throughput."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-pages", type=int, default=384)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--max-running-req", type=int, default=6)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--prompt-repeat", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-extend-tokens", type=int, default=256)
    parser.add_argument("--preempt-min-free-pages", type=int, default=1)
    parser.add_argument("--worker-case", choices=["normal", "overlap"])
    parser.add_argument("--output-json")
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_prompts(num_requests: int, repeat: int) -> list[str]:
    return [
        (
            f"Request {idx}: explain overlap-safe recompute preemption and dynamic KV allocation. "
            * repeat
        )
        for idx in range(num_requests)
    ]


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_case is not None
    assert args.output_json is not None
    overlap = args.worker_case == "overlap"
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "0" if overlap else "1"

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    class BenchmarkLLM(LLM):
        def generate(self, prompts, sampling_params):
            self._benchmark_start = time.perf_counter()
            self._first_token_times = {}
            self._finish_times = {}
            return super().generate(prompts, sampling_params)

        def offline_send_result(self, reply):
            now = time.perf_counter()
            for msg in reply:
                self._first_token_times.setdefault(msg.uid, now)
                if msg.finished:
                    self._finish_times[msg.uid] = now
            return super().offline_send_result(reply)

    llm = BenchmarkLLM(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        num_page_override=args.num_pages,
        page_size=args.page_size,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        cuda_graph_max_bs=0,
        enable_preemption=True,
        enable_overlap_preemption=overlap,
        dynamic_kv_allocation=True,
        decode_first=True,
        preempt_min_free_pages=args.preempt_min_free_pages,
    )
    try:
        results = llm.generate(
            make_prompts(args.num_requests, args.prompt_repeat),
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=args.max_tokens,
            ),
        )
        end_time = time.perf_counter()
        output_lengths = [len(result["token_ids"]) for result in results]
        total_tokens = sum(output_lengths)
        finish_times = [
            llm._finish_times.get(uid, end_time) for uid in range(len(output_lengths))
        ]
        payload = {
            "case": args.worker_case,
            "wall_time_s": end_time - llm._benchmark_start,
            "output_tokens": total_tokens,
            "tokens_per_s": total_tokens / (end_time - llm._benchmark_start),
            "request_latency_s": [
                finish_time - llm._benchmark_start for finish_time in finish_times
            ],
            "first_token_latency_s": [
                llm._first_token_times.get(uid, end_time) - llm._benchmark_start
                for uid in range(len(output_lengths))
            ],
            "num_preemptions": llm.num_preemptions,
            "num_deferred_preemptions": llm.num_deferred_preemptions,
            "num_preemption_stalls": llm.num_preemption_stalls,
            "num_resumed_preempted_reqs": llm.num_resumed_preempted_reqs,
            "output_lengths": output_lengths,
        }
        pathlib.Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        llm.shutdown()


def run_case(case: str, args: argparse.Namespace, output_json: pathlib.Path) -> dict:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
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
        "--num-pages",
        str(args.num_pages),
        "--page-size",
        str(args.page_size),
        "--max-running-req",
        str(args.max_running_req),
        "--num-requests",
        str(args.num_requests),
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--max-tokens",
        str(args.max_tokens),
        "--max-extend-tokens",
        str(args.max_extend_tokens),
        "--preempt-min-free-pages",
        str(args.preempt_min_free_pages),
    ]
    subprocess.run(cmd, env=env, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.worker_case is not None:
        run_worker(args)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        normal = run_case("normal", args, tmp / "normal.json")
        overlap = run_case("overlap", args, tmp / "overlap.json")

    print(json.dumps({"normal": normal, "overlap": overlap}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
