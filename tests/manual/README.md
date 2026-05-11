# Manual Preemption Validation

Run these from the repository root after activating the test environment:

```bash
eval "$(conda shell.bash hook)"
conda activate minisgl
```

## Non-Overlap Smoke

Runs one small-cache preemption workload with overlap scheduling disabled and
prints the preemption counters and output lengths.

```bash
PYTHONPATH=python python tests/manual/preemption_smoke.py \
  --model-path Qwen/Qwen3-0.6B
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
  --require-deferred-preemption \
  --baseline-num-pages 4096 \
  --preempt-num-pages 256 \
  --num-prompts 6 \
  --prompt-repeat 28 \
  --max-running-req 6 \
  --max-tokens 96
```

The `--require-deferred-preemption` variant should be used when validating the
overlap-safe path specifically. It fails unless `num_deferred_preemptions > 0`.

## Overlap Smoke

Runs one small-cache preemption workload and prints:

- `num_preemptions`
- `num_deferred_preemptions`
- `num_preemption_stalls`
- `num_prefill_fit_failures`
- `num_resumed_preempted_reqs`
- `output_lengths`

```bash
PYTHONPATH=python python tests/manual/preemption_smoke.py \
  --model-path Qwen/Qwen3-0.6B \
  --enable-overlap-preemption
```

## Overlap Benchmark

Compares normal preemption (`MINISGL_DISABLE_OVERLAP_SCHEDULING=1`) against
overlap preemption (`MINISGL_DISABLE_OVERLAP_SCHEDULING=0`) with the same
small-cache workload. It prints wall time, output tokens/s, per-request latency,
first-token latency, and preemption/deferred/resume counters.

```bash
PYTHONPATH=python python tests/manual/preemption_overlap_benchmark.py \
  --model-path Qwen/Qwen3-0.6B \
  --warmup-runs 1 \
  --repeats 3
```

## Counter Expectations

- `num_preemptions > 0`: the small-cache run actually exercised decode
  preemption.
- `num_deferred_preemptions > 0`: required only when validating the protected
  overlap path with `--require-deferred-preemption`.
- `num_prefill_fit_failures`: should usually be `0`; nonzero means prefill
  admission and exact page fit disagreed and the scheduler rolled the batch
  back.
- `num_preemption_stalls`: counts steps where the scheduler could not find a
  currently safe victim. A small nonzero value can occur under overlap when all
  candidates are protected or already deferred.
