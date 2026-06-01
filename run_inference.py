#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, re, sys, signal
import nest_asyncio
from pathlib import Path
from collections import Counter

nest_asyncio.apply()
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SAVE_EVAL   = False
MODEL_NAME  = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH   = "data/public.jsonl" if SAVE_EVAL else "data/private.jsonl"
OUTPUT_PATH = "results/vllm_inference_predictions.jsonl"

MAX_MODEL_LEN          = 12288
MAX_TOKENS_TO_GENERATE = 8192
SAMPLE_LIMIT           = None

# HF_TOKEN = None # put HF_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN

print("Config loaded.")


# In[2]:


from vllm import LLM, SamplingParams

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician competing in a math olympiad.\n"
    "For each problem:\n"
    "1. Identify the problem type and the relevant theorem or method\n"
    "2. Solve step by step, showing all work\n"
    "3. Verify your answer by checking edge cases or substituting back\n"
    "You MUST end with \\boxed{your answer} and nothing after it. "
    "For multiple parts use \\boxed{a, b, c}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician.\n"
    "For each multiple choice problem:\n"
    "1. Solve the problem independently before looking at the options\n"
    "2. Match your answer to the closest option\n"
    "3. Eliminate obviously wrong options to confirm your choice\n"
    "You MUST end with \\boxed{X} where X is the letter of your answer. "
    "Nothing after the \\boxed{}."
)

SYSTEM_PROMPT_RETRY_MATH = (
    "You are an expert mathematician. "
    "Your previous attempt did not produce a correctly formatted answer.\n"
    "Solve the problem step by step. "
    "The LAST thing you write MUST be \\boxed{your answer} and nothing after it. "
    "For multiple parts use \\boxed{a, b, c}."
)

SYSTEM_PROMPT_RETRY_MCQ = (
    "You are an expert mathematician. "
    "Your previous attempt did not produce a correctly formatted answer.\n"
    "Select the correct option. "
    "The LAST thing you write MUST be \\boxed{X} where X is the option letter. "
    "Nothing after the \\boxed{}."
)

print("Prompts ready.")


# In[ ]:


import torch

print(f"Loading {MODEL_NAME}...")

llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=1,
    quantization="bitsandbytes",
    dtype=torch.bfloat16,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    hf_token=HF_TOKEN,
    structured_outputs_config={"reasoning_parser": "qwen3"}
)

tokenizer = llm.get_tokenizer()
print("Model ready.")


# In[4]:


def build_prompt(item, system_math, system_mcq, tokenizer):
    options = item.get("options")
    is_mcq  = bool(options)

    if is_mcq:
        labels       = [chr(65 + i) for i in range(len(options))]
        opts_text    = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content = (
            f"{item['question']}\n\nOptions:\n{opts_text}"
            f"\n\nSelect one letter from A\u2013{labels[-1]}."
        )
        system_content = system_mcq
    else:
        user_content   = item["question"]
        system_content = system_math

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

data          = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
if SAMPLE_LIMIT:
    data = data[:SAMPLE_LIMIT]

private_by_id = {item["id"]: item for item in data}
prompts       = [build_prompt(item, SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, tokenizer)
                 for item in data]
metadata      = [
    {
        "id":        item["id"],
        "is_mcq":    bool(item.get("options")),
        "n_options": len(item["options"]) if item.get("options") else None,
        "gold":      item.get("answer")
    }
    for item in data
]

print(f"Loaded {len(prompts)} prompts.")
print(f"MCQ: {sum(m['is_mcq'] for m in metadata)}, "
      f"Free-form: {sum(not m['is_mcq'] for m in metadata)}")
print(f"Sample prompt tokens: {len(tokenizer.encode(prompts[0]))} "
      f"(budget: {MAX_MODEL_LEN - MAX_TOKENS_TO_GENERATE})")


# In[5]:


sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

def has_boxed(response: str) -> bool:
    """Check if response contains at least one complete \\boxed{} block."""
    return bool(re.search(r'\\boxed\{[^}]+\}', response))

def extract_letter(text: str, n_options: int = 10) -> str:
    valid = set(chr(65 + i) for i in range(n_options))
    m = re.search(r'\\boxed\{([A-Za-z])\}', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(
        r'(?:the answer is|answer:|correct answer is|option|choice|select)\s+([A-J])',
        text, re.IGNORECASE
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    matches = re.findall(r'\b([A-J])\b', text.upper())
    for letter in reversed(matches):
        if letter in valid:
            return letter
    return ""

def score_mcq(response, gold_letter, n_options=10):
    return extract_letter(response, n_options) == gold_letter.strip().upper()

def post_process(response):
    response = response.replace("\\\\boxed", "\\boxed")
    response = re.sub(r'\\boxed\s*\{', r'\\boxed{', response)
    last_end, start = -1, 0
    while True:
        idx = response.find("\\boxed{", start)
        if idx < 0:
            break
        depth, i = 1, idx + 7
        while i < len(response) and depth > 0:
            if response[i] == '{':   depth += 1
            elif response[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            last_end = i
        start = idx + 1
    if last_end > 0:
        response = response[:last_end]
    return response.strip()

def extract_answer(response: str, is_mcq: bool, n_options: int = 10) -> str:
    """
    Two-pass extraction:
    Pass 1 — if \\boxed{} present: use extract_letter (MCQ) or judger.extract_ans (free-form)
    Pass 2 — fallback: heuristic letter scan (MCQ) or judger last-number/LaTeX (free-form)
    """
    if is_mcq:
        if has_boxed(response):
            letter = extract_letter(response, n_options)
            if letter:
                return letter
        return extract_letter(response, n_options)
    else:
        if has_boxed(response):
            ans = judger.extract_ans(response)
            if ans and ans.strip():
                return ans.strip()
        ans = judger.extract_ans(response)
        return ans.strip() if ans else ""

print("Extractors ready.")


# In[6]:


# ── ROUND 1 ───────────────────────────────────────────────────────────
sampling_r1 = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.05,
    max_tokens=MAX_TOKENS_TO_GENERATE,
    n=1,
    stop=["<|im_end|>", "<|endoftext|>"]
)

print(f"Round 1: Running {len(prompts)} prompts...")
outputs_r1   = llm.generate(prompts, sampling_r1)
responses_r1 = [post_process(out.outputs[0].text.strip()) for out in outputs_r1]

no_box_indices = [i for i, r in enumerate(responses_r1) if not has_boxed(r)]
print(f"Round 1 complete. Missing \\boxed{{}}: {len(no_box_indices)} / {len(responses_r1)}")


# In[7]:


# ── ROUND 2: REGENERATE MISSING BOXED ────────────────────────────────
responses_r2_map = {}

if no_box_indices:
    retry_prompts = [
        build_prompt(
            private_by_id[metadata[i]["id"]],
            SYSTEM_PROMPT_RETRY_MATH,
            SYSTEM_PROMPT_RETRY_MCQ,
            tokenizer
        )
        for i in no_box_indices
    ]

    sampling_r2 = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=MAX_TOKENS_TO_GENERATE,
        n=1,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    print(f"Round 2: Regenerating {len(no_box_indices)} responses...")
    outputs_r2 = llm.generate(retry_prompts, sampling_r2)

    for i, out in zip(no_box_indices, outputs_r2):
        responses_r2_map[i] = post_process(out.outputs[0].text.strip())

    still_no_box = [i for i in no_box_indices if not has_boxed(responses_r2_map[i])]
    print(f"Round 2 complete. Still missing \\boxed{{}}: {len(still_no_box)} / {len(no_box_indices)}")
    print(f"  {len(no_box_indices) - len(still_no_box)} recovered, {len(still_no_box)} using judger fallback")
else:
    print("All responses have \\boxed{} \u2014 skipping round 2.")

# ── MERGE ─────────────────────────────────────────────────────────────
# Priority: r1 with boxed > r2 with boxed > longer of r1/r2 (judger fallback)
responses = []
for i, r1 in enumerate(responses_r1):
    if has_boxed(r1):
        responses.append(r1)
    elif i in responses_r2_map and has_boxed(responses_r2_map[i]):
        responses.append(responses_r2_map[i])
    elif i in responses_r2_map:
        r2 = responses_r2_map[i]
        responses.append(r2 if len(r2) > len(r1) else r1)
    else:
        responses.append(r1)

boxed_count = sum(1 for r in responses if has_boxed(r))
print(f"\nFinal: {boxed_count} / {len(responses)} responses have \\boxed{{}}")
print(f"Remaining {len(responses) - boxed_count} will use judger fallback extraction")


# In[8]:


from tqdm import tqdm

results = []
for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
    is_mcq    = bool(item.get("options"))
    n_options = len(item["options"]) if is_mcq else 10
    correct   = False
    gold      = None

    if SAVE_EVAL:
        gold = item.get("answer")
        if is_mcq:
            letter  = extract_answer(response, is_mcq=True, n_options=n_options)
            correct = (letter == str(gold).strip().upper())
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            extracted = extract_answer(response, is_mcq=False)
            # If no boxed in response, wrap extracted value so judger can parse it
            pred_for_judger = (
                f"\\boxed{{{extracted}}}"
                if extracted and not has_boxed(response)
                else response
            )
            try:
                signal.alarm(3)
                correct = judger.auto_judge(
                    pred=pred_for_judger,
                    gold=gold_list,
                    options=[[]] * len(gold_list)
                )
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
                correct = False

    record = {"id": item["id"], "is_mcq": is_mcq, "response": response}
    if SAVE_EVAL:
        record.update({"gold": gold, "correct": correct})
    results.append(record)

if SAVE_EVAL:
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]
    def acc(s): return sum(r["correct"] for r in s) / len(s) * 100 if s else 0
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)


# In[9]:


out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

existing = {}
if out_path.exists():
    for line in open(out_path):
        r = json.loads(line)
        existing[r["id"]] = r

for r in results:
    existing[r["id"]] = r

with open(out_path, "w") as f:
    for record in sorted(existing.values(), key=lambda x: x["id"]):
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(existing)} records to {out_path}")


# In[ ]:


import sys
import json
import csv
 
 
def jsonl_to_csv(input_path: str, output_path: str) -> None:
    rows = []
 
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_num} — JSON parse error: {e}", file=sys.stderr)
                continue
 
            record_id = record.get("id", "")
            response = record.get("response", "")
            rows.append({"id": record_id, "response": response})
 
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"Done. Wrote {len(rows)} rows to {output_path}")
 
 
 
jsonl_to_csv("./results/vllm_inference_predictions.jsonl","./lastsubpredictions.csv")


# In[ ]:




