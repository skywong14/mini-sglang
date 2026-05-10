from __future__ import annotations

"""
Manual run:
  eval "$(conda shell.bash hook)" && conda activate minisgl
  PYTHONPATH=python python tests/manual/preemption_smoke.py \
    --model-path Qwen/Qwen3-0.6B \
    --enable-overlap-preemption
"""

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual preemption smoke test.")
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
    parser.add_argument(
        "--enable-overlap-preemption",
        action="store_true",
        help="Run with overlap scheduling and experimental overlap-safe preemption.",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = (
        "0" if args.enable_overlap_preemption else "1"
    )

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    prompts = [
        (
            f"Request {idx}: explain dynamic KV allocation and recompute preemption. "
            * args.prompt_repeat
        )
        for idx in range(args.num_requests)
    ]

    llm = LLM(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        num_page_override=args.num_pages,
        page_size=args.page_size,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        cuda_graph_max_bs=0,
        enable_preemption=True,
        enable_overlap_preemption=args.enable_overlap_preemption,
        dynamic_kv_allocation=True,
        decode_first=True,
        preempt_min_free_pages=args.preempt_min_free_pages,
    )
    try:
        results = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=args.max_tokens,
            ),
        )
        assert llm.num_preemptions > 0, "Expected at least one decode preemption"
        payload = {
            "enable_overlap_preemption": args.enable_overlap_preemption,
            "num_deferred_preemptions": llm.num_deferred_preemptions,
            "num_preemptions": llm.num_preemptions,
            "num_resumed_preempted_reqs": llm.num_resumed_preempted_reqs,
            "last_preempted_uids": llm.last_preempted_uids,
            "output_lengths": [len(result["token_ids"]) for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
