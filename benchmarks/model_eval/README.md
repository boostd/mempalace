# MemPalace small-model evaluation harness

Evaluates ≤4B-parameter Ollama models on MemPalace's classification and extraction tasks.

Outputs a CSV with accuracy, latency (TTFT, TPS, e2e p50/p95), and VRAM (resident, peak) per `(model, task, mode)` triple. Use the results to populate `mempalace/local_model.py::MODEL_TIERS` from data instead of vibes.

## Quick start

```bash
# 1. Pull the candidate models you want to evaluate
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull gemma3:4b-it-q4_K_M
# ... etc, see candidates.yaml

# 2. Run one (model, task) pair to verify the harness works
python -m benchmarks.model_eval.runner \
  --model qwen3:4b-instruct-2507-q4_K_M \
  --task room_classification \
  --mode closed

# 3. Run the whole matrix
python -m benchmarks.model_eval.orchestrator \
  --candidates tier1 \
  --tasks all \
  --output results/$(date -u +%Y-%m-%d)-$(hostname).csv
```

## What it measures

For each `(model, task, mode)`:

- **Accuracy** — task-specific scoring against the labeled dataset
- **TTFT** (time-to-first-token) — p50 and p95 over N=5 warm runs
- **TPS** (tokens/second) — sustained throughput on a 10-request batch
- **e2e latency** — full single-classification time, p50 and p95 over N=20 warm runs
- **VRAM resident** — model memory after warmup (read from `ollama ps`)
- **VRAM peak** — peak GPU memory during inference (polled via `nvidia-smi`)

The first run of each model is discarded (cache + GPU clock ramp).

## Tasks

- `room_classification` — closed-set (room list provided) and open-set (model invents the slug)
- `entity_extraction` — JSON list of entities per sample
- `memory_extraction` — structured memory items per sample
- `calibration` — simple 5-class sentence-type, sanity check the harness

See `tasks/<name>/README.md` for input format and scoring details.

## Datasets

All datasets are synthetic and committed to this repo. No real-person info. Generated once and frozen so benchmark numbers stay comparable across runs.

If you need to extend the dataset, **add** samples; don't replace existing ones, otherwise prior numbers stop being comparable.

## Hardware reporting

Every result file includes a header row with:

- CPU, core count, RAM
- GPU model, VRAM total
- Ollama version
- OS/kernel
- Run timestamp (UTC)

Speed numbers are **not portable across machines**. Use them for relative ranking on a single setup. Accuracy numbers cross-port cleanly.

## Reusing existing infrastructure

The harness uses `mempalace.llm_client.get_provider("ollama", model=tag)` and `provider.classify(...)` directly — same code path as production. For thinking-token models, post-processes via `mempalace.local_model.strip_thinking_tokens`.

No new HTTP plumbing. No reimplementation of provider abstraction. The harness benchmarks the same code that ships.
