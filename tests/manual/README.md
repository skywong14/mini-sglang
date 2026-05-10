# Manual Preemption Validation

Run these from the repository root after activating the test environment:

```bash
eval "$(conda shell.bash hook)"
conda activate minisgl
```

## Overlap Greedy Correctness

Compares:

- baseline: large KV cache, preemption disabled, overlap scheduling enabled
- preempt: small KV cache, preemption enabled, overlap preemption enabled

The script uses greedy deterministic sampling with `temperature=0.0`, `top_k=1`,
`top_p=1.0`, `ignore_eos=True`, and `cuda_graph_max_bs=0`. It asserts that the
preempt run performs at least one preemption and that generated token ids match
the baseline exactly.

```bash
PYTHONPATH=python python tests/manual/preemption_greedy_correctness.py \
  --model-path Qwen/Qwen3-0.6B \
  --enable-overlap-preemption
```

Useful knobs when forcing more pressure:

```bash
PYTHONPATH=python python tests/manual/preemption_greedy_correctness.py \
  --model-path Qwen/Qwen3-0.6B \
  --enable-overlap-preemption \
  --baseline-num-pages 4096 \
  --preempt-num-pages 384 \
  --num-prompts 4 \
  --prompt-repeat 20 \
  --max-tokens 64
```

## Overlap Smoke

Runs one small-cache preemption workload and prints:

- `num_preemptions`
- `num_deferred_preemptions`
- `num_resumed_preempted_reqs`
- `output_lengths`

```bash
PYTHONPATH=python python tests/manual/preemption_smoke.py \
  --model-path Qwen/Qwen3-0.6B \
  --enable-overlap-preemption
```
