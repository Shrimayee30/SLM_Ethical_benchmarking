#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethical_benchmarking.hallucination import read_jsonl, write_jsonl


def limited_rows(rows: list[dict], limit: int | None, sample_mode: str, seed: int) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    if sample_mode == "head":
        return rows[:limit]
    if sample_mode == "random":
        rng = random.Random(seed)
        return rng.sample(rows, limit)
    if sample_mode != "stratified":
        raise ValueError(f"Unknown sample mode: {sample_mode}")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)

    task_types = sorted(by_task)
    base = limit // len(task_types)
    remainder = limit % len(task_types)
    selected = []
    for idx, task_type in enumerate(task_types):
        quota = base + int(idx < remainder)
        selected.extend(by_task[task_type][:quota])

    if len(selected) < limit:
        selected_ids = {row["id"] for row in selected}
        for row in rows:
            if row["id"] not in selected_ids:
                selected.append(row)
                if len(selected) == limit:
                    break
    return selected


def load_transformers_model(model_id: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "Phi-3" in model_id or "phi-3" in model_id.lower():
        if trust_remote_code:
            print(
                "Ignoring --trust-remote-code for Phi-3; current Transformers versions load it natively "
                "and the cached remote implementation can fail on rope_scaling.",
                file=sys.stderr,
            )
        trust_remote_code = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return tokenizer, model


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_transformers(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch

    rendered = build_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_ollama(model: str, prompt: str, max_new_tokens: int, temperature: float, host: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Install Ollama, run `ollama serve`, and pull the model "
            "with a command such as `ollama pull phi3:mini`."
        ) from exc
    return body.get("message", {}).get("content", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local model on hallucination prompts.")
    parser.add_argument("--input", default="data/hallucination/eval.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["transformers", "ollama"], default="transformers")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or Ollama model name.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        choices=["stratified", "head", "random"],
        default="stratified",
        help="How to choose rows when --limit is set.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    rows = read_jsonl(ROOT / args.input)
    rows = limited_rows(rows, args.limit, args.sample_mode, args.seed)
    task_counts = defaultdict(int)
    for row in rows:
        task_counts[row["task_type"]] += 1
    print(json.dumps({"selected": len(rows), "task_counts": dict(sorted(task_counts.items()))}, indent=2))

    tokenizer = model = None
    if args.backend == "transformers":
        tokenizer, model = load_transformers_model(args.model_id, args.trust_remote_code)
    results = []
    for item in tqdm(rows, desc=args.model_name or args.model_id):
        if args.backend == "ollama":
            response = generate_ollama(
                args.model_id,
                item["prompt"],
                args.max_new_tokens,
                args.temperature,
                args.ollama_host,
            )
        else:
            response = generate_transformers(tokenizer, model, item["prompt"], args.max_new_tokens, args.temperature)
        results.append(
            {
                "model": args.model_name or args.model_id,
                "model_id": args.model_id,
                "backend": args.backend,
                "item_id": item["id"],
                "task_type": item["task_type"],
                "prompt": item["prompt"],
                "expected_answer": item.get("expected_answer"),
                "response": response,
            }
        )

    write_jsonl(ROOT / args.output, results)
    print(json.dumps({"output": args.output, "n": len(results)}, indent=2))


if __name__ == "__main__":
    main()
