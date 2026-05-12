

import json, os, re, random, argparse, time, warnings, shutil, gc
import numpy as np
import pandas as pd
from collections import defaultdict
import torch

warnings.filterwarnings("ignore")

PROBE_LAYER = 23
MAX_NEW     = 200
MAX_INPUT   = 768





def safe_load_model(name, no_4bit=False, max_retries=3):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    cache_path = f"/tmp/hf_cache/hub/models--{name.replace('/', '--')}"

    def clean_corrupted():
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)

    for attempt in range(max_retries):
        try:
            print(f"Loading model (attempt {attempt+1}/{max_retries})...")

            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            if no_4bit:
                mdl = AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True)
            else:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4")
                mdl = AutoModelForCausalLM.from_pretrained(
                    name, quantization_config=bnb,
                    device_map="auto", trust_remote_code=True)

            mdl.eval()
            print("✅ Model loaded")
            return tok, mdl

        except OSError as e:
            print(f"❌ Load failed: {e}")
            clean_corrupted()
            time.sleep(5)

    raise RuntimeError("Model failed after retries")


def warmup_model(model, tokenizer):
    prompt = "2+2="
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=2)





def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["A","B","C"])
    p.add_argument("--dataset", default="contrastive_math_dataset.json")
    p.add_argument("--probe_weights", default="results_phase4/probe_weights.npz")
    p.add_argument("--threshold", default="results_phase4/probe_threshold.json")
    p.add_argument("--output_dir", default="results_exp/")
    p.add_argument("--n_sample", type=int, default=300)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--inst_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--seed", type=int, default=99)
    return p.parse_args()


class Probe:
    def __init__(self, weights_path, threshold_path):
        d = np.load(weights_path)
        self.W = d["W"]; self.b = float(d["b"])
        self.mu = d["scaler_mean"]; self.sc = d["scaler_scale"]
        self.layer = int(d["layer"])
        with open(threshold_path) as f:
            t = json.load(f)
        self.tau = float(t["threshold"])
        self.tau_cons = float(t["conservative_threshold"])
        self.tau_loose = self.tau + (1 - self.tau) * 0.3

    def score(self, h):
        x = (h - self.mu) / self.sc
        return float(1/(1+np.exp(-(np.dot(x, self.W)+self.b))))

    def fires(self, s, mode="youden"):
        th = {"conservative": self.tau_cons,
              "youden": self.tau,
              "loose": self.tau_loose}
        return s >= th[mode]


def get_device(model):
    return next(model.parameters()).device


def probe_score_at_prompt(model, tokenizer, prompt, probe):
    enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT, truncation=True)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        acts = out.hidden_states[probe.layer][0, -1, :].float().cpu().numpy()
    torch.cuda.empty_cache()
    return probe.score(acts)


def greedy_generate(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT, truncation=True)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(ids[0, in_len:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return text.split("\n")[0].strip()


def check_correct(generated, expected):
    nums = re.findall(r"[-+]?\d*\.?\d+", str(expected))
    if not nums: return None
    target = float(nums[-1])
    candidates = re.findall(r"[-+]?\d*\.?\d+", generated)
    for c in reversed(candidates):
        try:
            v = float(c)
            if abs(v-target)<1e-3: return True
        except: pass
    return False






def run_exp_a(args, dataset, probe):
    print("\n=== EXP A (ROBUST) ===")

    tok, mdl = safe_load_model(args.base_model, args.no_4bit)
    warmup_model(mdl, tok)

    by_base = defaultdict(list)
    for t in dataset["traces"]:
        by_base[t["base_problem_id"]].append(t)

    all_ids = list(by_base.keys())
    random.shuffle(all_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "intervention_exp_a.csv")

    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        done_ids = set(existing_df["base_id"].unique())
        print(f"Resuming: {len(done_ids)} done")
    else:
        existing_df = pd.DataFrame()
        done_ids = set()

    remaining_ids = [x for x in all_ids if x not in done_ids]
    sampled = remaining_ids[:args.n_sample]

    records = []

    for i, base_id in enumerate(sampled):
        try:
            variants = by_base[base_id]
            clean = next((t for t in variants if t["variant"]=="clean"), None)
            error = next((t for t in variants if t["variant"]=="error_at_1"), None)
            if clean is None or error is None: continue

            prompt = error["hops"][0]["prompt"]
            text = greedy_generate(mdl, tok, prompt)

            records.append({
                "base_id": base_id,
                "final_correct": check_correct(text, clean["correct_final_answer"])
            })

        except Exception as e:
            print(f"Trace failed: {e}")
            continue

        if (i+1)%5==0:
            df = pd.concat([existing_df, pd.DataFrame(records)])
            df.to_csv(out_path, index=False)
            print("checkpoint saved")

        gc.collect()
        torch.cuda.empty_cache()

    df = pd.concat([existing_df, pd.DataFrame(records)])
    df.to_csv(out_path, index=False)
    print("DONE")






if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed)

    with open(args.dataset) as f:
        dataset = json.load(f)

    probe = Probe(args.probe_weights, args.threshold)

    if args.exp == "A":
        run_exp_a(args, dataset, probe)