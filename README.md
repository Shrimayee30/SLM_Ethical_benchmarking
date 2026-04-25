# SLM_Ethical_benchmarking

Ethical benchmarking of small language models (SLMs) across three safety dimensions:

- hallucination and unsupported factual generation
- toxicity handling and refusal behavior
- bias and fairness across social attributes

The project focuses on four small language models:

- TinyLlama
- Phi-3 Mini
- Gemma
- Mistral 7B

The repository combines lightweight scripts, benchmark data preparation, and notebook-based evaluation so results can be inspected directly during experimentation.

## Paper

The paper accompanying this repository is included here:

- [SLM_Benchmarking_paper.pdf](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/SLM_Benchmarking_paper.pdf)

## Overview

This project studies how small language models behave when they are asked to:

- answer questions that may be solvable, unsolvable, or unverifiable
- transform or continue toxic text while preserving safe behavior
- make choices in socially sensitive settings involving stereotype-related content

Rather than measuring only raw task accuracy, the benchmarks are designed to capture behavioral properties such as:

- when a model answers correctly
- when it should refuse
- when it hallucinates unsupported information
- when it over-refuses safe content
- when it exhibits stereotypical or biased preferences

## Benchmarks Included

### 1. Hallucination Benchmark

The hallucination benchmark evaluates whether a model produces unsupported or fabricated information, especially when the prompt does not provide enough evidence.

It currently combines:

- **GSM8K** as a capability control
  - checks whether the model can still answer ordinary reasoning questions
- **GAIA** as an offline/unanswerable benchmark
  - checks whether the model refuses when the question requires tools, browsing, or missing external context
- **Fact / citation prompts**
  - checks whether the model fabricates papers, authors, journals, institutions, or source-like details

Main files:

- [notebooks/hallucination_benchmark.ipynb](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/notebooks/hallucination_benchmark.ipynb)
- [scripts/prepare_hallucination_data.py](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/scripts/prepare_hallucination_data.py)
- [scripts/run_hallucination_benchmark.py](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/scripts/run_hallucination_benchmark.py)
- [scripts/evaluate_hallucination_results.py](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/scripts/evaluate_hallucination_results.py)
- [data/hallucination/fact_citation_prompts.jsonl](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/data/hallucination/fact_citation_prompts.jsonl)

Main metrics:

- `hallucination_risk_rate`
- `refusal_rate`
- `capability_accuracy`

### 2. Toxicity Benchmark

The toxicity benchmark evaluates whether a model refuses harmful or toxic transformations while still completing harmless tasks correctly.

It uses:

- **RealToxicityPrompts** to construct toxic and safe prompt sets
- task-based transformations such as:
  - paraphrase
  - translation
  - grammar correction
- **borderline cases** such as sarcasm, idioms, and hyperbole
- **multi-turn pressure prompts** to test whether a model initially refuses and then later gives in

Main files:

- [Slm toxicity benchmark/toxicity_benchmark_final (2).ipynb](</Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/Slm toxicity benchmark/toxicity_benchmark_final (2).ipynb>)
- [Slm toxicity benchmark/benchmark_results.json](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/Slm toxicity benchmark/benchmark_results.json)

Main metrics:

- `refusal_accuracy`
- `toxic_leakage_rate`
- `task_completion`
- `over_refusal_rate`
- `borderline_refusal`
- `multiturn_failure`

### 3. Bias and Fairness Benchmark

The bias benchmark evaluates whether models display stereotypical preferences or fairness gaps across socially sensitive contexts.

It uses:

- **StereoSet**
  - stereotype vs anti-stereotype sentence preferences
- **BBQ**
  - biased answering in ambiguous vs disambiguated QA settings
- **WinoBias**
  - gender bias in pronoun/coreference resolution

Main file:

- [slm_bias_benchmarking.ipynb](/Users/shrimayee/sdeshpan/projects/SLM_Ethical_benchmarking/slm_bias_benchmarking.ipynb)

Main metrics:

- StereoSet `SS`, `LMS`, `iCAT`
- BBQ `bias_score_ambig`, `accuracy_disambig`
- WinoBias `pro_stereo_accuracy`, `anti_stereo_accuracy`, `delta`

## Repository Structure

```text
SLM_Ethical_benchmarking/
├── notebooks/
│   └── hallucination_benchmark.ipynb
├── scripts/
│   ├── prepare_hallucination_data.py
│   ├── run_hallucination_benchmark.py
│   └── evaluate_hallucination_results.py
├── src/
│   └── ethical_benchmarking/
│       └── hallucination.py
├── data/
│   └── hallucination/
├── configs/
│   ├── hallucination.json
│   └── models.json
├── Slm toxicity benchmark/
│   ├── toxicity_benchmark_final (2).ipynb
│   └── benchmark_results.json
└── slm_bias_benchmarking.ipynb
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use gated Hugging Face datasets such as GAIA, authenticate first:

```bash
hf auth login
```

## Hallucination Benchmark V0

The current hallucination slice follows a refusal-aware setup inspired by recent factuality and hallucination benchmarking work.

### Prepare Prompts

With GAIA:

```bash
python scripts/prepare_hallucination_data.py \
  --output data/hallucination/eval.jsonl \
  --gsm8k-limit 50 \
  --gaia-limit-per-level 25
```

Without GAIA:

```bash
python scripts/prepare_hallucination_data.py \
  --output data/hallucination/eval.jsonl \
  --gsm8k-limit 50 \
  --skip-gaia
```

### Run A Model

Example with TinyLlama:

```bash
python scripts/run_hallucination_benchmark.py \
  --input data/hallucination/eval.jsonl \
  --output data/hallucination/tinyllama_responses.jsonl \
  --model-name TinyLlama \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --limit 12 \
  --max-new-tokens 64
```

When `--limit` is set, the runner uses stratified sampling by default so a short smoke test includes GSM8K, GAIA, and fact/citation rows together.

Example with Phi-3 Mini:

```bash
python scripts/run_hallucination_benchmark.py \
  --input data/hallucination/eval.jsonl \
  --output data/hallucination/phi3mini_responses.jsonl \
  --model-name "Phi-3 Mini" \
  --model-id microsoft/Phi-3-mini-4k-instruct
```

### Ollama Option

On memory-constrained Macs, running larger models through Transformers can be slow because weights may be offloaded to disk. The runner also supports Ollama:

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

The evaluation report includes:

- `hallucination_risk_rate`
- `refusal_rate`
- `capability_accuracy`
- label counts such as `safe_refusal`, `hallucinated_answer`, and `fabricated_citation_risk`

## Notes and Limitations

- The hallucination fact/citation benchmark currently uses heuristic scoring and can be improved with stronger source verification.
- The toxicity and bias benchmarks are notebook-driven and were built for iterative experimentation rather than as polished standalone packages.
- Some datasets and models may require Hugging Face access approval before use.

## Appendix

The implementation of the ethical benchmarking framework, including code, notebooks, and supporting files, is available at:

[https://github.com/Shrimayee30/SLM_Ethical_benchmarking](https://github.com/Shrimayee30/SLM_Ethical_benchmarking)
