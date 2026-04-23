# SLM_Ethical_benchmarking
Ethical benchmarking of small language models (TinyLlama, Phi-3 Mini, Gemma, Mistral 7B) across hallucination, truthfulness, bias, and fairness. Includes custom datasets, refusal-aware metrics, and single and multi-turn evaluation to analyze safety, grounding, and reliability in real-world scenarios.

## Hallucination Benchmark V0

This first slice follows the reference paper's hallucination setup:

- **GSM8K capability control**: the model should answer because all information needed to solve the math problem is in the prompt.
- **GAIA hallucination probe**: GAIA level 2/3 questions often require tools, browsing, files, or external context. In this offline benchmark, a substantive answer is treated as hallucination risk; a clear "cannot answer from the provided information" response is safe.
- **Fact / citation generation probes**: prompts ask for sources, paper lists, or discoveries. Some are answerable static knowledge, and some are fictional/unverifiable requests designed to catch fabricated papers, DOIs, journals, institutions, or discovery narratives.

### Model Set

Configured in `configs/models.json`:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2b-it`
- `mistralai/Mistral-7B-Instruct-v0.3`

Gemma and Mistral repositories may require accepting model terms on Hugging Face before local download.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GAIA is gated on Hugging Face. Accept the dataset conditions for `gaia-benchmark/GAIA`, then authenticate:

```bash
hf auth login
```

### Prepare Prompts

With GAIA:

```bash
python scripts/prepare_hallucination_data.py \
  --output data/hallucination/eval.jsonl \
  --gsm8k-limit 50 \
  --gaia-limit-per-level 25
```

Without GAIA, useful while wiring up the pipeline:

```bash
python scripts/prepare_hallucination_data.py \
  --output data/hallucination/eval.jsonl \
  --gsm8k-limit 50 \
  --skip-gaia
```

### Run A Model

```bash
python scripts/run_hallucination_benchmark.py \
  --input data/hallucination/eval.jsonl \
  --output data/hallucination/tinyllama_responses.jsonl \
  --model-name TinyLlama \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --limit 12 \
  --max-new-tokens 64
```

When `--limit` is set, the runner samples tasks in stratified mode by default, so a short smoke test includes capability, GAIA, and fact/citation rows instead of only the first rows in the file. Use `--sample-mode head` to reproduce the old first-N behavior.

Phi-3 Mini should be run with the native Transformers implementation:

```bash
python scripts/run_hallucination_benchmark.py \
  --input data/hallucination/eval.jsonl \
  --output data/hallucination/phi3mini_responses.jsonl \
  --model-name "Phi-3 Mini" \
  --model-id microsoft/Phi-3-mini-4k-instruct
```

On memory-constrained Macs, Phi-3 through Transformers may offload weights to disk and appear stuck at `0/12` for a long time. In that case, use Ollama's quantized local runtime:

```bash
ollama pull phi3:mini

python scripts/run_hallucination_benchmark.py \
  --backend ollama \
  --input data/hallucination/eval.jsonl \
  --output data/hallucination/phi3mini_ollama_responses_12.jsonl \
  --model-name "Phi-3 Mini" \
  --model-id phi3:mini \
  --limit 12 \
  --max-new-tokens 64
```

### Evaluate

```bash
python scripts/evaluate_hallucination_results.py \
  --dataset data/hallucination/eval.jsonl \
  --results data/hallucination/tinyllama_responses.jsonl \
  --scores-output data/hallucination/tinyllama_scores.jsonl
```

The report gives per-task:

- `hallucination_risk_rate`
- `refusal_rate`
- `capability_accuracy` for answerable controls
- label counts such as `safe_refusal`, `hallucinated_answer`, `fabricated_citation_risk`, and `capability_refusal`

## Current Limitations

The citation probe uses lightweight heuristics for V0. It catches obvious fabricated citation shapes, but a stronger V1 should verify sources against Crossref, Semantic Scholar, or a frozen local bibliography so scoring is less string-based.
