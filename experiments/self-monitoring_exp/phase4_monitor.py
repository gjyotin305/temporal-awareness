

import json, os, re, random, warnings, argparse, time, traceback
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional


warnings.filterwarnings('ignore')





PROBE_LAYER     = 23     
THRESHOLD_MODE  = 'youden'  
BRANCH_K        = 4      
MAX_INTERVENT   = 3      
ABSURDITY_PATTERNS = [
    r'-\$?\d', r'\bnegative\b', r'minus\s+\d',
    r'\bnot\s+possible\b', r'\bimpossible\b', r'\bcannot\s+be\b',
    r"doesn'?t\s+make\s+sense", r'\bnonsensical\b', r'\binvalid\b',
]





def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_probe',      action='store_true')
    p.add_argument('--run_intervention', action='store_true')
    p.add_argument('--plot_only',        action='store_true')
    p.add_argument('--phase3_dir',  default='results_phase3_v3')
    p.add_argument('--output_dir',  default='results_phase4')
    p.add_argument('--dataset',     default='contrastive_math_dataset.json')
    p.add_argument('--probe_weights',default=None)
    p.add_argument('--threshold',   default=None)
    p.add_argument('--model',       default='Qwen/Qwen2.5-3B')
    p.add_argument('--n_sample',    type=int, default=150)
    p.add_argument('--max_tokens',  type=int, default=180)
    p.add_argument('--max_input',   type=int, default=768)
    p.add_argument('--branch_k',    type=int, default=BRANCH_K)
    p.add_argument('--no_4bit',     action='store_true')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--probe_layer', type=int, default=PROBE_LAYER)
    return p.parse_args()







def train_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

    print('═' * 72)
    print('  PHASE 4a: Training probe from Phase 3 activations')
    print('═' * 72)

    act_arr_path = os.path.join(args.phase3_dir, 'act_arr.npy')
    albl_path    = os.path.join(args.phase3_dir, 'act_labels.npy')
    if not os.path.exists(act_arr_path):
        raise FileNotFoundError(
            f'{act_arr_path} not found. Run Phase 3 first or set --phase3_dir.')

    act_arr    = np.load(act_arr_path)
    act_labels = np.load(albl_path)
    L          = args.probe_layer
    print(f'  Loaded act_arr shape={act_arr.shape}, labels shape={act_labels.shape}')
    print(f'  Training probe at layer L{L}')

    X = act_arr[:, L, :]
    y = act_labels.astype(int)

    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_predict(
        _make_pipeline(args.seed), X, y, cv=kf, method='predict_proba')[:, 1]
    cv_auroc = roc_auc_score(y, cv_scores)
    cv_acc   = accuracy_score(y, (cv_scores > 0.5).astype(int))
    print(f'  CV AUROC: {cv_auroc:.4f}   CV accuracy: {cv_acc:.4f}')

    
    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X)
    clf    = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf.fit(Xs, y)

    
    W = clf.coef_[0]
    b = float(clf.intercept_[0])
    out_path = os.path.join(args.output_dir, 'probe_weights.npz')
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(out_path,
             W=W, b=b,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             layer=L, cv_auroc=cv_auroc, cv_acc=cv_acc)
    print(f'  Saved probe → {out_path}')

    
    fpr, tpr, thr = roc_curve(y, cv_scores)
    youden    = tpr - fpr
    best_idx  = int(np.argmax(youden))
    threshold = float(thr[best_idx])
    tpr_at    = float(tpr[best_idx])
    fpr_at    = float(fpr[best_idx])

    
    clean_scores  = cv_scores[y == 0]
    cons_thresh   = float(np.percentile(clean_scores, 95))

    threshold_data = {
        'mode':              THRESHOLD_MODE,
        'threshold':         threshold,
        'youden_J':          float(youden[best_idx]),
        'tpr_at_threshold':  tpr_at,
        'fpr_at_threshold':  fpr_at,
        'conservative_threshold': cons_thresh,
        'cv_auroc':          cv_auroc,
        'layer':             L,
    }
    thresh_path = os.path.join(args.output_dir, 'probe_threshold.json')
    with open(thresh_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f'  Saved threshold → {thresh_path}')
    print(f'  Operating point: threshold={threshold:.3f}, TPR={tpr_at:.3f}, FPR={fpr_at:.3f}')
    print('═' * 72)
    return out_path, thresh_path


def _make_pipeline(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('s', StandardScaler()),
        ('c', LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    ])





class RuntimeProbe:
    

    def __init__(self, weights_path: str, threshold_path: str):
        d = np.load(weights_path)
        self.W            = d['W']
        self.b            = float(d['b'])
        self.scaler_mean  = d['scaler_mean']
        self.scaler_scale = d['scaler_scale']
        self.layer        = int(d['layer'])
        with open(threshold_path) as f:
            t = json.load(f)
        self.threshold       = float(t['threshold'])
        self.cons_threshold  = float(t['conservative_threshold'])

    def score(self, activation_at_layer: np.ndarray) -> float:
        
        x = (activation_at_layer - self.scaler_mean) / self.scaler_scale
        z = float(np.dot(x, self.W) + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def fires(self, score: float, conservative: bool = False) -> bool:
        thr = self.cons_threshold if conservative else self.threshold
        return score >= thr





def build_chain_prompt(trace, target_hop_index: int,
                        prior_overrides: Optional[dict] = None):
    
    parts = []
    for i, hop in enumerate(trace['hops']):
        if i < target_hop_index:
            if prior_overrides and i in prior_overrides:
                resp = prior_overrides[i]
            elif hop['is_injected'] and hop['injected_response'] is not None:
                resp = hop['injected_response']
            else:
                resp = hop['correct_response']
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer: {resp}\n')
        else:
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer:')
            break
    return ''.join(parts)


def build_reprompt_for_hop(trace, hop_idx: int,
                            prior_overrides: Optional[dict] = None):
    
    parts = []
    parts.append(
        'Carefully reconsider the following math problem. Some prior steps '
        'may contain mistakes. Recompute each step from scratch using only '
        'the original question. Show your work clearly.\n\n'
    )
    for i, hop in enumerate(trace['hops']):
        if i < hop_idx:
            if prior_overrides and i in prior_overrides:
                resp = prior_overrides[i]
            elif hop['is_injected'] and hop['injected_response'] is not None:
                resp = hop['injected_response']
            else:
                resp = hop['correct_response']
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer: {resp}\n')
        else:
            parts.append(
                f'Step {i+1} question: {hop["prompt"]}\n'
                f'Reconsider this step carefully. '
                f'Step {i+1} answer:')
            break
    return ''.join(parts)


def build_recompute_prior_prompt(trace, hop_idx_to_fix: int):
    
    target_hop = trace['hops'][hop_idx_to_fix]
    prompt = (
        f'Solve this single math problem carefully. Show only the answer.\n\n'
        f'Question: {target_hop["prompt"]}\n'
        f'Answer:')
    return prompt





def check_answer_correct(generated_text: str, correct_answer) -> Optional[bool]:
    
    correct_nums = re.findall(r'[-+]?\d*\.?\d+', str(correct_answer))
    if not correct_nums:
        return None
    try:
        target = float(correct_nums[-1])
    except ValueError:
        return None

    markers = [
        r'(?:final\s+answer|the\s+answer|answer\s+is|=\s*)\s*\$?([-+]?\d*\.?\d+)',
        r'\$?([-+]?\d*\.?\d+)\s*(?:dollars?|miles?|kg|grams?|cm|m\b|m²|km|hours?|minutes?|seconds?)',
    ]
    candidates = []
    for m in markers:
        candidates.extend(re.findall(m, generated_text.lower()))
    if not candidates:
        all_nums = re.findall(r'[-+]?\d*\.?\d+', generated_text)
        if all_nums:
            candidates = [all_nums[-1]]
    if not candidates:
        return False
    for c in candidates:
        try:
            v = float(c)
            if abs(v - target) < 1e-3 or (target != 0 and abs(v - target)/abs(target) < 0.01):
                return True
        except ValueError:
            continue
    return False


def is_absurd(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in ABSURDITY_PATTERNS)


def extract_answer(text: str) -> Optional[str]:
    
    
    markers = [
        r'(?:final\s+answer|the\s+answer|answer\s+is|=\s*)\s*\$?([-+]?\d*\.?\d+)',
    ]
    for m in markers:
        match = re.search(m, text.lower())
        if match:
            return match.group(1)
    
    all_nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if all_nums:
        return all_nums[-1]
    return None





def generate_with_probe(model, tokenizer, prompt, device, probe: RuntimeProbe,
                         max_new_tokens=180, max_input=768,
                         stop_at_newline=True, return_acts=False):
    
    import torch

    inputs    = tokenizer(prompt, return_tensors='pt',
                          max_length=max_input, truncation=True).to(device)
    input_ids = inputs['input_ids'].clone()
    S         = input_ids.shape[1]

    with torch.no_grad():
        out_s = model(**inputs, output_hidden_states=True)
        hs    = torch.stack(out_s.hidden_states, dim=0)
        
        acts_at_layer = hs[probe.layer, 0, S-1, :].float().cpu().numpy()
        del out_s, hs
        torch.cuda.empty_cache()

    probe_score = probe.score(acts_at_layer)

    confs, ents, gen_ids = [], [], []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out    = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]
            probs  = torch.softmax(logits, dim=-1)
            confs.append(probs.max().item())
            ents.append(-(probs * torch.log(probs + 1e-10)).sum().item())
            nxt = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            gen_ids.append(nxt.item())
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if stop_at_newline and len(gen_ids) > 5 and '\n' in tokenizer.decode([nxt.item()]):
                del out
                break
            del out
        torch.cuda.empty_cache()

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        'text':         text,
        'confidences':  confs,
        'entropies':    ents,
        'probe_score':  probe_score,
        'acts':         acts_at_layer if return_acts else None,
    }





def run_baseline(trace, model, tokenizer, device, probe: RuntimeProbe, args):
    
    hop_depth = len(trace['hops'])
    trajectory = []
    final_text = ''
    for hop_idx in range(hop_depth):
        prompt = build_chain_prompt(trace, hop_idx)
        result = generate_with_probe(
            model, tokenizer, prompt, device, probe,
            max_new_tokens=args.max_tokens, max_input=args.max_input)
        trajectory.append({
            'hop_idx':     hop_idx,
            'probe_score': result['probe_score'],
            'fired':       probe.fires(result['probe_score']),
            'text':        result['text'],
            'mean_conf':   float(np.mean(result['confidences'])) if result['confidences'] else 0.0,
        })
        if hop_idx == hop_depth - 1:
            final_text = result['text']
    return {'condition': 'baseline', 'trajectory': trajectory,
            'final_text': final_text, 'n_interventions': 0}


def run_reprompt(trace, model, tokenizer, device, probe: RuntimeProbe, args):
    
    hop_depth = len(trace['hops'])
    trajectory = []
    final_text = ''
    n_interv   = 0
    for hop_idx in range(hop_depth):
        prompt = build_chain_prompt(trace, hop_idx)
        result = generate_with_probe(
            model, tokenizer, prompt, device, probe,
            max_new_tokens=args.max_tokens, max_input=args.max_input)
        intervened = False
        if probe.fires(result['probe_score']) and n_interv < MAX_INTERVENT:
            
            reprompt   = build_reprompt_for_hop(trace, hop_idx)
            reprompt_r = generate_with_probe(
                model, tokenizer, reprompt, device, probe,
                max_new_tokens=args.max_tokens * 2, max_input=args.max_input,
                stop_at_newline=True)
            
            result['text']        = reprompt_r['text']
            
            result['probe_score_after_reprompt'] = reprompt_r['probe_score']
            n_interv  += 1
            intervened = True
        trajectory.append({
            'hop_idx':     hop_idx,
            'probe_score': result['probe_score'],
            'fired':       probe.fires(result['probe_score']),
            'intervened':  intervened,
            'text':        result['text'],
        })
        if hop_idx == hop_depth - 1:
            final_text = result['text']
    return {'condition': 'reprompt', 'trajectory': trajectory,
            'final_text': final_text, 'n_interventions': n_interv}


def run_replace_prior(trace, model, tokenizer, device, probe: RuntimeProbe, args):
    
    hop_depth = len(trace['hops'])
    trajectory = []
    final_text = ''
    n_interv   = 0
    prior_overrides: dict = {}

    for hop_idx in range(hop_depth):
        prompt = build_chain_prompt(trace, hop_idx, prior_overrides)
        result = generate_with_probe(
            model, tokenizer, prompt, device, probe,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        intervened = False
        if probe.fires(result['probe_score']) and hop_idx > 0 and n_interv < MAX_INTERVENT:
            
            recomp_prompt = build_recompute_prior_prompt(trace, hop_idx - 1)
            recomp_r = generate_with_probe(
                model, tokenizer, recomp_prompt, device, probe,
                max_new_tokens=args.max_tokens, max_input=args.max_input)
            new_prior_answer = extract_answer(recomp_r['text']) or recomp_r['text'].strip()
            prior_overrides[hop_idx - 1] = new_prior_answer
            n_interv  += 1
            intervened = True
            
            prompt2 = build_chain_prompt(trace, hop_idx, prior_overrides)
            result2 = generate_with_probe(
                model, tokenizer, prompt2, device, probe,
                max_new_tokens=args.max_tokens, max_input=args.max_input)
            result = result2

        trajectory.append({
            'hop_idx':     hop_idx,
            'probe_score': result['probe_score'],
            'fired':       probe.fires(result['probe_score']),
            'intervened':  intervened,
            'text':        result['text'],
            'prior_overrides': dict(prior_overrides),
        })
        if hop_idx == hop_depth - 1:
            final_text = result['text']
    return {'condition': 'replace_prior', 'trajectory': trajectory,
            'final_text': final_text, 'n_interventions': n_interv}


def run_branch_and_pick(trace, model, tokenizer, device, probe: RuntimeProbe, args):
    
    import torch
    hop_depth = len(trace['hops'])
    trajectory = []
    final_text = ''
    n_interv   = 0

    for hop_idx in range(hop_depth):
        prompt = build_chain_prompt(trace, hop_idx)
        result = generate_with_probe(
            model, tokenizer, prompt, device, probe,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        intervened = False
        branches = None
        if probe.fires(result['probe_score']) and n_interv < MAX_INTERVENT:
            
            branches = []
            for k in range(args.branch_k):
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors='pt',
                                        max_length=args.max_input,
                                        truncation=True).to(device)
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        temperature=0.8 + 0.1 * k,    
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    branch_text = tokenizer.decode(
                        gen[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True)
                    
                    if '\n' in branch_text:
                        branch_text = branch_text.split('\n')[0]

                
                
                rescore_prompt = prompt + branch_text
                rescore_inputs = tokenizer(rescore_prompt, return_tensors='pt',
                                            max_length=args.max_input,
                                            truncation=True).to(device)
                with torch.no_grad():
                    rescore_out = model(**rescore_inputs, output_hidden_states=True)
                    hs = torch.stack(rescore_out.hidden_states, dim=0)
                    branch_acts = hs[probe.layer, 0, -1, :].float().cpu().numpy()
                    del rescore_out, hs
                branch_score = probe.score(branch_acts)
                branches.append({'text': branch_text, 'probe_score': branch_score})

            
            best = min(branches, key=lambda b: b['probe_score'])
            result['text']        = best['text']
            result['probe_score'] = best['probe_score']
            n_interv   += 1
            intervened  = True

        trajectory.append({
            'hop_idx':     hop_idx,
            'probe_score': result['probe_score'],
            'fired':       probe.fires(result['probe_score']),
            'intervened':  intervened,
            'text':        result['text'],
            'branches':    branches,
        })
        if hop_idx == hop_depth - 1:
            final_text = result['text']
    return {'condition': 'branch_and_pick', 'trajectory': trajectory,
            'final_text': final_text, 'n_interventions': n_interv}





def run_intervention(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print('═' * 72)
    print('  PHASE 4b: Intervention Experiment')
    print('═' * 72)

    weights_path = args.probe_weights or os.path.join(args.output_dir, 'probe_weights.npz')
    thresh_path  = args.threshold     or os.path.join(args.output_dir, 'probe_threshold.json')
    if not os.path.exists(weights_path) or not os.path.exists(thresh_path):
        raise FileNotFoundError(f'Run --train_probe first.')

    probe = RuntimeProbe(weights_path, thresh_path)
    print(f'  Probe loaded: layer L{probe.layer}, threshold={probe.threshold:.3f}')

    print('  Loading dataset...')
    with open(args.dataset) as f:
        dataset = json.load(f)
    by_base = defaultdict(list)
    for t in dataset['traces']:
        by_base[t['base_problem_id']].append(t)
    all_ids = list(by_base.keys())

    print('  Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb,
            device_map='auto', trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
            device_map='auto', trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    print(f'  Device: {device}')

    sampled = random.sample(all_ids, min(args.n_sample, len(all_ids)))
    records       = []
    trace_log     = {}
    t0            = time.time()
    n_done        = 0

    print(f'\n  Running {len(sampled)} traces × 4 conditions = {4*len(sampled)} generations\n')

    for idx, base_id in enumerate(sampled):
        variants  = by_base[base_id]
        hop_depth = variants[0]['hop_depth']
        if hop_depth < 2:
            continue
        clean = next((t for t in variants if t['variant'] == 'clean'),      None)
        error = next((t for t in variants if t['variant'] == 'error_at_1'), None)
        if clean is None or error is None:
            continue

        elapsed = time.time() - t0
        eta     = (elapsed / max(idx, 1)) * (len(sampled) - idx)
        print(f'  [{idx+1:3d}/{len(sampled)}] {base_id} ({hop_depth}-hop) '
              f'elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m', flush=True)

        
        condition_results = {}
        for fn, cond_name in [
            (run_baseline,        'baseline'),
            (run_reprompt,        'reprompt'),
            (run_replace_prior,   'replace_prior'),
            (run_branch_and_pick, 'branch_and_pick'),
        ]:
            try:
                cr = fn(error, model, tokenizer, device, probe, args)
                condition_results[cond_name] = cr
            except Exception as e:
                print(f'    ! {cond_name} failed: {e}')
                traceback.print_exc()
                condition_results[cond_name] = {
                    'condition': cond_name, 'trajectory': [],
                    'final_text': '', 'n_interventions': 0,
                    'error': str(e)
                }

        
        correct_answer = clean['correct_final_answer']
        for cond_name, cr in condition_results.items():
            ft        = cr.get('final_text', '')
            correct   = check_answer_correct(ft, correct_answer)
            absurd    = is_absurd(ft)
            traj      = cr.get('trajectory', [])
            records.append({
                'base_id':         base_id,
                'condition':       cond_name,
                'hop_depth':       hop_depth,
                'subcategory':     clean.get('subcategory', 'unknown'),
                'error_type':      error['injected_error_type'],
                'final_correct':   correct,
                'final_absurd':    absurd,
                'final_text':      ft,
                'n_interventions': cr.get('n_interventions', 0),
                'max_probe_score': max([t['probe_score'] for t in traj], default=0.0),
                'mean_probe_score':float(np.mean([t['probe_score'] for t in traj])) if traj else 0.0,
                'fired_at_least_once': any(t.get('fired', False) for t in traj),
                'n_hops':          len(traj),
            })

        
        trace_log[base_id] = {
            'base_id':   base_id,
            'hop_depth': hop_depth,
            'error_type': error['injected_error_type'],
            'subcategory': clean.get('subcategory', 'unknown'),
            'correct_answer': correct_answer,
            'hops': [
                {'idx': i, 'question': h['prompt'],
                 'correct_response': h['correct_response'],
                 'injected_response': h.get('injected_response'),
                 'is_injected': h.get('is_injected', False)}
                for i, h in enumerate(error['hops'])
            ],
            'conditions': {
                k: {
                    'trajectory':      cr.get('trajectory', []),
                    'final_text':      cr.get('final_text', ''),
                    'n_interventions': cr.get('n_interventions', 0),
                }
                for k, cr in condition_results.items()
            }
        }
        n_done += 1

        
        if (idx + 1) % 10 == 0:
            _save_records(records, trace_log, args.output_dir)
            print(f'    [checkpoint at {idx+1}]')

    _save_records(records, trace_log, args.output_dir)
    print(f'\n  Completed {n_done} traces × 4 conditions = {len(records)} records')
    print(f'  Total time: {(time.time()-t0)/60:.1f} min')
    print('═' * 72)


def _save_records(records, trace_log, output_dir):
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'intervention_records.csv'), index=False)
    with open(os.path.join(output_dir, 'trace_log.json'), 'w') as f:
        json.dump(_jsonable(trace_log), f, indent=1)


def _jsonable(obj):
    
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj





def plot_only(args):
    
    import subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, 'phase4_analysis.py'),
           '--output_dir', args.output_dir]
    subprocess.run(cmd, check=True)





def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_probe:
        train_probe(args)
    if args.run_intervention:
        run_intervention(args)
    if args.plot_only:
        plot_only(args)
    if not (args.train_probe or args.run_intervention or args.plot_only):
        print('No action specified. Use --train_probe, --run_intervention, or --plot_only.')
        print('Recommended sequence:')
        print('  1. python phase4_monitor.py --train_probe \\')
        print('         --phase3_dir results_phase3_v3/ \\')
        print('         --output_dir results_phase4/')
        print('  2. python phase4_monitor.py --run_intervention \\')
        print('         --output_dir results_phase4/ --n_sample 150')
        print('  3. python phase4_analysis.py --output_dir results_phase4/')


if __name__ == '__main__':
    main()