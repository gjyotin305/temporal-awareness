

import json, os, re, random, warnings, argparse, time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu, gaussian_kde, pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings('ignore')




HEDGE_PATTERNS = [
    r"i(?:'m| am) not (?:sure|certain|confident)",
    r"i(?:'m| am) unsure",
    r"i(?:'m| am) uncertain",
    r"not (?:entirely|completely|fully|totally) sure",
    r"(?:may|might|could) be (?:wrong|incorrect|mistaken|off)",
    r"(?:possibly|perhaps|probably) (?:wrong|incorrect|mistaken)",
    r"(?:hard|difficult) to (?:say|tell|know)",
    r"can'?t (?:be )?(?:sure|certain|confident)",
    r"(?:this|the answer) (?:may|might|could) (?:be )?(?:wrong|incorrect|off|inaccurate)",
    r"(?:double.?check|verify|confirm)",
    r"(?:not|without) (?:full|complete|high|much) confidence",
    r"(?:low|limited|reduced) confidence",
    r"(?:my|this) (?:estimate|answer|calculation) (?:may|might|could) be",
    r"(?:approximate|rough|ballpark)",
    r"if i.{0,15}(?:reading|understanding|interpreting).{0,20}correctly",
    r"assuming (?:the prior|the previous|my earlier|this is correct)",
    r"(?:the prior|previous) (?:step|answer|result) (?:may|might|could) .{0,20}(?:error|mistake|issue)",
]

def detect_hedging(text: str) -> dict:
    
    text_lower = text.lower()
    matched = []
    for pattern in HEDGE_PATTERNS:
        if re.search(pattern, text_lower):
            matched.append(pattern)
    return {
        'hedges':      matched,
        'hedge_count': len(matched),
        'hedge_score': min(1.0, len(matched) / 2.0),  
        'hedged':      len(matched) > 0,
    }


def check_answer_correct(generated_text: str, correct_answer) -> bool | None:
    
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
            val = float(c)
            if abs(val - target) < 1e-3 or (target != 0 and abs(val - target)/abs(target) < 0.01):
                return True
        except ValueError:
            continue
    return False


def extract_verbalized_confidence(text: str) -> float | None:
    
    text_lower = text.lower()

    
    numeric_patterns = [
        
        (r'confidence[:\s]+(\d+(?:\.\d+)?)\s*/\s*10\b',  10),
        (r'confidence[:\s]+(\d+(?:\.\d+)?)\s*/\s*100\b', 100),
        (r'(\d+(?:\.\d+)?)\s+out\s+of\s+10\b',           10),
        (r'(\d+(?:\.\d+)?)\s+out\s+of\s+100\b',          100),
        (r'(?:rating|score|confidence)[:\s]+(\d+(?:\.\d+)?)\s*/\s*10', 10),
        (r'(?:rating|score|confidence)[:\s]+(\d+(?:\.\d+)?)\s*/\s*100',100),
        
        (r'(?:i\s*(?:am|\'m)|i\s+feel)\s+(\d+(?:\.\d+)?)\s*%\s*(?:sure|certain|confident)', 100),
        (r'(\d+(?:\.\d+)?)\s*%\s*(?:sure|certain|confident)',                 100),
        (r'(?:sure|certain|confident)\s+(?:at|to)\s+(\d+(?:\.\d+)?)\s*%',     100),
        (r'(?:confidence|certainty)\s+(?:level\s+)?(?:is\s+|of\s+|at\s+)?(\d+(?:\.\d+)?)\s*%', 100),
        
        (r'(?:confidence|certain(?:ty)?|sure)[^.\n]{0,30}?(\d+(?:\.\d+)?)\s*%',  100),
        (r'(\d+(?:\.\d+)?)\s*%[^.\n]{0,30}?(?:confidence|certain(?:ty)?|sure)',  100),
        
        (r'confidence[:\s]+(\d+(?:\.\d+)?)\b',           10),
        (r'rating[:\s]+(\d+(?:\.\d+)?)\b',               10),
    ]
    for pat, scale in numeric_patterns:
        m = re.search(pat, text_lower)
        if m:
            try:
                val = float(m.group(1))
                if scale == 100 or val > 10:
                    return min(1.0, val / 100.0)
                else:
                    return min(1.0, val / 10.0)
            except (ValueError, IndexError):
                continue

    
    verbal_high = [
        r'(?:i\s*(?:am|\'m))\s+(?:absolutely\s+|completely\s+|totally\s+|very\s+|highly\s+)?(?:sure|certain|confident|positive)',
        r'(?:absolutely|definitely|certainly|undoubtedly)\s+(?:correct|right)',
        r'no\s+doubt',
        r'without\s+(?:any\s+)?doubt',
    ]
    verbal_low = [
        r'(?:i\s*(?:am|\'m))\s+not\s+(?:sure|certain|confident)',
        r'(?:i\s*(?:am|\'m))\s+(?:un)?(?:sure|certain)',
        r'(?:i\s*(?:am|\'m))\s+uncertain',
        r'i\s+don\'?t\s+know',
        r'(?:hard|difficult)\s+to\s+say',
        r'(?:may|might|could)\s+be\s+wrong',
    ]
    for pat in verbal_high:
        if re.search(pat, text_lower):
            return 0.95   
    for pat in verbal_low:
        if re.search(pat, text_lower):
            return 0.20   

    return None






OVERCONF_PATTERNS = [
    r"i(?:'m| am)\s+(?:absolutely|completely|totally|fully|entirely|highly|very)?\s*(?:sure|certain|confident|positive)",
    r"100\s*%\s*(?:sure|certain|confident|positive)",
    r"(?:absolutely|definitely|certainly|undoubtedly)\s+(?:correct|right|the answer|true)",
    r"(?:no|without|beyond)\s+(?:any\s+)?doubt",
    r"\bclear(?:ly)?\b",
    r"\bobviously\b",
    r"the answer\s+is\s+(?:simply|just|clearly|obviously)",
]

def detect_overconfidence(text: str) -> dict:
    
    text_lower = text.lower()
    matched = []
    for pattern in OVERCONF_PATTERNS:
        if re.search(pattern, text_lower):
            matched.append(pattern)
    return {
        'overconf_phrases': matched,
        'overconf_count':   len(matched),
        'overconfident':    len(matched) > 0,
    }






def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='contrastive_math_dataset.json')
    p.add_argument('--model',      default='Qwen/Qwen2.5-3B')
    p.add_argument('--n_sample',   type=int, default=100,
                   help='Base problems to sample (each gives 3 records)')
    p.add_argument('--output_dir', default='results_phase3')
    p.add_argument('--max_tokens', type=int, default=180)
    p.add_argument('--max_input',  type=int, default=768)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no_4bit',    action='store_true')
    p.add_argument('--plot_only',  action='store_true')
    return p.parse_args()





def build_hop_prompt(trace, target_hop_index):
    
    parts = []
    for i, hop in enumerate(trace['hops']):
        if i < target_hop_index:
            resp = (hop['injected_response']
                    if hop['is_injected'] and hop['injected_response'] is not None
                    else hop['correct_response'])
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer: {resp}\n')
        else:
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer:')
            break
    return ''.join(parts)


def build_verbalized_confidence_prompt(trace, target_hop_index):
    
    parts = []

    
    parts.append(
        'You are solving a multi-step math problem. After EACH answer, '
        'you must output a confidence score on its own line in this exact '
        'format:\n'
        '    Confidence: N/10\n'
        'where N is an integer from 0 (completely unsure) to 10 (certain). '
        'This is mandatory.\n\n'
        '--- Example ---\n'
        'Step 1 question: What is 7 times 8?\n'
        'Step 1 answer: 56\n'
        'Confidence: 10/10\n'
        'Step 2 question: What is 56 divided by 11?\n'
        'Step 2 answer: 5.09 (approximately)\n'
        'Confidence: 6/10\n'
        '--- End example ---\n\n'
        'Now solve this problem the same way, including the Confidence line '
        'after every answer:\n\n'
    )

    
    for i, hop in enumerate(trace['hops']):
        if i < target_hop_index:
            resp = (hop['injected_response']
                    if hop['is_injected'] and hop['injected_response'] is not None
                    else hop['correct_response'])
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer: {resp}\n')
            
            
            parts.append('Confidence: 10/10\n')
        else:
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer:')
            break
    return ''.join(parts)


def build_final_answer_prompt(trace):
    
    parts = []
    hop_depth = len(trace['hops'])
    for i, hop in enumerate(trace['hops']):
        if i < hop_depth - 1:
            resp = (hop['injected_response']
                    if hop['is_injected'] and hop['injected_response'] is not None
                    else hop['correct_response'])
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer: {resp}\n')
        else:
            parts.append(f'Step {i+1} question: {hop["prompt"]}\n')
            parts.append(f'Step {i+1} answer:')
    return ''.join(parts)





@torch.no_grad()
def generate_full(model, tokenizer, prompt, device,
                   max_new_tokens=180, max_input=768,
                   stop_after_confidence=False):
    
    inputs    = tokenizer(prompt, return_tensors='pt',
                          max_length=max_input, truncation=True).to(device)
    input_ids = inputs['input_ids'].clone()
    S         = input_ids.shape[1]

    
    out_s  = model(**inputs, output_hidden_states=True)
    hs     = torch.stack(out_s.hidden_states, dim=0)
    acts   = hs[:, 0, S-1, :].float().cpu().numpy()
    del out_s, hs
    torch.cuda.empty_cache()

    confs, ents, vgaps, gen_ids = [], [], [], []
    saw_confidence_marker = False
    consecutive_newlines  = 0

    for _ in range(max_new_tokens):
        out    = model(input_ids=input_ids)
        logits = out.logits[0, -1, :]
        probs  = torch.softmax(logits, dim=-1)

        
        top2   = torch.topk(probs, k=2)
        top1_p = top2.values[0].item()
        top2_p = top2.values[1].item()

        confs.append(top1_p)
        ents.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        vgaps.append(top1_p - top2_p)

        nxt = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        gen_ids.append(nxt.item())
        input_ids = torch.cat([input_ids, nxt], dim=1)

        nxt_text = tokenizer.decode([nxt.item()])
        is_newline = '\n' in nxt_text

        if not stop_after_confidence:
            
            if len(gen_ids) > 5 and is_newline:
                del out
                break
        else:
            
            current_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if 'Confidence:' in current_text or 'confidence:' in current_text:
                saw_confidence_marker = True
            
            if saw_confidence_marker and is_newline and len(gen_ids) > 8:
                del out
                break
            
            if is_newline:
                consecutive_newlines += 1
                if consecutive_newlines >= 2 and len(gen_ids) > 5:
                    del out
                    break
            else:
                consecutive_newlines = 0

        del out

    torch.cuda.empty_cache()
    return (tokenizer.decode(gen_ids, skip_special_tokens=True),
            confs, ents, vgaps, acts)







def logprob_uncertainty(confs):
    
    if not confs:
        return 0.0
    return float(np.mean([-np.log(max(c, 1e-10)) for c in confs]))





def collect_data(args, device):
    print('\n--- Loading dataset ---')
    with open(args.dataset) as f:
        dataset = json.load(f)
    by_base = defaultdict(list)
    for t in dataset['traces']:
        by_base[t['base_problem_id']].append(t)
    all_ids = list(by_base.keys())
    print(f'  {len(dataset["traces"])} traces, {len(all_ids)} base problems')

    print('--- Loading model ---')
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
    NUM_LAYERS = model.config.num_hidden_layers
    print(f'  Layers={NUM_LAYERS}')

    sampled = random.sample(all_ids, min(args.n_sample, len(all_ids)))
    records, act_rows, act_labels = [], [], []
    t0 = time.time()

    print(f'\n--- Collecting ({len(sampled)} base problems × 3 conditions) ---')
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
              f'elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m', flush=True)

        
        prompt_a = build_hop_prompt(clean, target_hop_index=1)
        txt_a, confs_a, ents_a, vgaps_a, acts_a = generate_full(
            model, tokenizer, prompt_a, device,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        
        prompt_b = build_hop_prompt(error, target_hop_index=1)
        txt_b, confs_b, ents_b, vgaps_b, acts_b = generate_full(
            model, tokenizer, prompt_b, device,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        
        
        
        prompt_c = build_verbalized_confidence_prompt(error, target_hop_index=1)
        txt_c, confs_c, ents_c, vgaps_c, acts_c = generate_full(
            model, tokenizer, prompt_c, device,
            max_new_tokens=max(args.max_tokens, 60),
            max_input=args.max_input,
            stop_after_confidence=True)

        
        prompt_final_clean = build_final_answer_prompt(clean)
        txt_fc, _, _, _, _ = generate_full(
            model, tokenizer, prompt_final_clean, device,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        prompt_final_error = build_final_answer_prompt(error)
        txt_fe, _, _, _, _ = generate_full(
            model, tokenizer, prompt_final_error, device,
            max_new_tokens=args.max_tokens, max_input=args.max_input)

        correct_answer = clean['correct_final_answer']
        final_correct_clean = check_answer_correct(txt_fc, correct_answer)
        final_correct_error = check_answer_correct(txt_fe, correct_answer)

        
        
        
        per_cond_behavior = {
            'clean':            {'text': txt_a},
            'error_standard':   {'text': txt_b},
            'error_verbalized': {'text': txt_c},
        }
        for cond_name, info in per_cond_behavior.items():
            t = info['text']
            info['hedge']      = detect_hedging(t)
            info['overconf']   = detect_overconfidence(t)
            info['verb_conf']  = extract_verbalized_confidence(t)

        
        def n_early(confs): return min(20, len(confs))

        for label, confs, ents, vgaps, acts, txt, condition in [
            (0, confs_a, ents_a, vgaps_a, acts_a, txt_a, 'clean'),
            (1, confs_b, ents_b, vgaps_b, acts_b, txt_b, 'error_standard'),
            (1, confs_c, ents_c, vgaps_c, acts_c, txt_c, 'error_verbalized'),
        ]:
            ne = n_early(confs)
            beh = per_cond_behavior[condition]
            rec = {
                'base_id':          base_id,
                'condition':        condition,
                'label':            label,        
                'hop_depth':        hop_depth,
                'subcategory':      clean.get('subcategory', 'unknown'),
                'error_type':       error['injected_error_type'],

                
                'peak_entropy':     float(np.max(ents))              if ents  else 0.0,
                'mean_entropy':     float(np.mean(ents))             if ents  else 0.0,
                'early_entropy':    float(np.mean(ents[:ne]))        if ents  else 0.0,
                'min_confidence':   float(np.min(confs))             if confs else 1.0,
                'early_min_conf':   float(np.min(confs[:ne]))        if confs else 1.0,
                'mean_vocab_gap':   float(np.mean(vgaps))            if vgaps else 1.0,
                'early_min_vgap':   float(np.min(vgaps[:ne]))        if vgaps else 1.0,

                
                'logprob_uncertainty': logprob_uncertainty(confs),
                'mean_logprob':        float(np.mean(
                    [-np.log(max(c, 1e-10)) for c in confs])) if confs else 0.0,

                
                'verbalized_conf':  beh['verb_conf'],
                'hedge_count':      beh['hedge']['hedge_count'],
                'hedge_score':      beh['hedge']['hedge_score'],
                'hedged':           beh['hedge']['hedged'],
                'hedge_phrases':    str(beh['hedge']['hedges']),
                'overconf_count':   beh['overconf']['overconf_count'],
                'overconfident':    beh['overconf']['overconfident'],
                'overconf_phrases': str(beh['overconf']['overconf_phrases']),

                
                'final_correct_clean': final_correct_clean,
                'final_correct_error': final_correct_error,
                'final_degraded':      (final_correct_clean is True and
                                        final_correct_error is False),

                'n_tokens':  len(confs),
                'generated_text': txt,
            }
            records.append(rec)
            act_rows.append(acts)
            act_labels.append(label)

    df         = pd.DataFrame(records)
    act_arr    = np.stack(act_rows, axis=0)
    act_labels = np.array(act_labels)
    return df, act_arr, act_labels, NUM_LAYERS





def run_layer_probe(act_arr, act_labels, seed=42):
    n_layers = act_arr.shape[1]
    kf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = Pipeline([('s', StandardScaler()),
                     ('c', LogisticRegression(max_iter=2000, C=1.0,
                                              random_state=seed))])
    auroc_m, acc_m, acc_s, folds = [], [], [], []
    print(f'\n--- Layer probe: {n_layers} layers ---')
    for li in range(n_layers):
        a = cross_val_score(pipe, act_arr[:, li, :], act_labels,
                            cv=kf, scoring='roc_auc')
        c = cross_val_score(pipe, act_arr[:, li, :], act_labels,
                            cv=kf, scoring='accuracy')
        auroc_m.append(a.mean()); acc_m.append(c.mean())
        acc_s.append(c.std()); folds.append(a)
        if li % 5 == 0:
            print(f'  L{li:2d}: AUROC={a.mean():.3f} Acc={c.mean():.3f}')
    return (np.array(auroc_m), np.array(acc_m),
            np.array(acc_s),   np.array(folds))





def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0
    p = np.sqrt(((na-1)*np.std(a,ddof=1)**2+(nb-1)*np.std(b,ddof=1)**2)/(na+nb-2))
    return (np.mean(a) - np.mean(b)) / (p + 1e-10)

def kde_line(vals, n=400):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 3: return np.array([]), np.array([])
    s = max(vals.std()*0.45, 1e-6)
    x = np.linspace(vals.min()-s, vals.max()+s, n)
    return x, gaussian_kde(vals, bw_method='silverman')(x)

BG = '#f5f5f5'; CC = '#1f77b4'; CE = '#d62728'
CG = '#888888'; CP = '#1a7f3c'; CV = '#9467bd'





def make_all_figures(df, act_arr, act_labels,
                     layer_auroc, layer_acc, layer_acc_std, layer_folds,
                     out_dir):

    best_layer  = int(np.argmax(layer_auroc))
    n_layers    = len(layer_auroc)

    
    pipe_cv = Pipeline([('s', StandardScaler()),
                        ('c', LogisticRegression(max_iter=2000, C=1.0,
                                                 random_state=42))])
    probe_scores = cross_val_predict(
        pipe_cv, act_arr[:, best_layer, :], act_labels,
        cv=5, method='predict_proba')[:, 1]
    probe_auroc  = roc_auc_score(act_labels, probe_scores)

    
    df = df.copy()
    df['probe_score'] = probe_scores

    clean_df = df[df['condition'] == 'clean']
    error_df = df[df['condition'] == 'error_standard']
    verb_df  = df[df['condition'] == 'error_verbalized']
    y_true   = act_labels

    print(f'\nBest layer: L{best_layer}  AUROC={probe_auroc:.4f}')
    print(f'Records: {len(df)} ({len(clean_df)} clean / {len(error_df)} error_std / {len(verb_df)} error_verb)')

    
    print('  Figure 1: H3 comparison...')
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(BG)

    PREDICTORS = [
        ('probe_score',        'Activation Probe L'+str(best_layer), CP),
        ('logprob_uncertainty','Logprob Uncertainty (mean NLL)',      '#ff7f0e'),
        ('peak_entropy',       'Peak Entropy (surface)',              CE),
    ]

    
    degraded_label = df['final_degraded'].fillna(False).astype(int).values

    comparison_rows = []
    for sig, label, color in PREDICTORS:
        if sig not in df.columns: continue
        scores = df[sig].values
        if sig == 'probe_score':
            
            pass
        else:
            
            pass
        try:
            auroc_deg = roc_auc_score(degraded_label, scores)
            r_pb, p_pb = pointbiserialr(degraded_label, scores)
        except:
            auroc_deg, r_pb, p_pb = 0.5, 0.0, 1.0

        
        try:
            auroc_det = roc_auc_score(y_true, scores)
        except:
            auroc_det = 0.5

        comparison_rows.append({
            'method': label, 'signal': sig,
            'auroc_detection':   auroc_det,
            'auroc_degradation': auroc_deg,
            'pointbiserial_r':   r_pb,
            'p_value':           p_pb,
            'color':             color,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(out_dir, 'comparison_table.csv'), index=False)

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot([0,1],[0,1], color=CG, ls='--', lw=1.5, alpha=0.5, label='Chance')
    for _, row in comp_df.iterrows():
        scores = df[row['signal']].values
        fpr, tpr, _ = roc_curve(degraded_label, scores)
        ax.plot(fpr, tpr, color=row['color'], lw=2.5,
                label=f"{row['method'].split('(')[0].strip()}\nAUROC={row['auroc_degradation']:.3f}")
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('H3: Predicting Final Answer Degradation\n'
                 'Probe vs. logprob vs. surface entropy',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.plot([0,1],[0,1], color=CG, ls='--', lw=1.5, alpha=0.5, label='Chance')
    for _, row in comp_df.iterrows():
        scores = df[row['signal']].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax2.plot(fpr, tpr, color=row['color'], lw=2.5,
                 label=f"{row['method'].split('(')[0].strip()}\nAUROC={row['auroc_detection']:.3f}")
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
    ax2.set_title('Error Context Detection\n(clean vs. corrupted prior)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right'); ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    methods = [r['method'].split('(')[0].strip() for _, r in comp_df.iterrows()]
    x = np.arange(len(methods))
    det_aurocs  = comp_df['auroc_detection'].values
    deg_aurocs  = comp_df['auroc_degradation'].values
    colors_list = comp_df['color'].values

    ax3.plot(x, det_aurocs,  'o-', color='#333', lw=2, ms=10, label='Error detection')
    ax3.plot(x, deg_aurocs,  's--',color='#555', lw=2, ms=10, label='Failure prediction')
    for xi, (d, g, c) in enumerate(zip(det_aurocs, deg_aurocs, colors_list)):
        ax3.scatter(xi, d, color=c, s=120, zorder=5, edgecolors='white', lw=1.5)
        ax3.scatter(xi, g, color=c, s=120, zorder=5, marker='s',
                    edgecolors='white', lw=1.5)
        ax3.text(xi, max(d, g)+0.01, f'{d:.3f}\n{g:.3f}',
                 ha='center', fontsize=7.5, color=c)
    ax3.axhline(0.5, color=CG, ls='--', lw=1.2, alpha=0.6)
    ax3.set_xticks(x); ax3.set_xticklabels(methods, fontsize=8, rotation=10)
    ax3.set_ylabel('AUROC')
    ax3.set_title('Side-by-side: Detection vs. Failure Prediction\n'
                  '● = detection  ■ = failure prediction',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2); ax3.set_ylim(0.4, 1.05)

    fig.suptitle('H3: Probe vs. Logprob vs. Surface Signal — Which Best Predicts Failure?',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig1_h3_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'    → {p}')

    
    print('  Figure 2: H4 verbalized vs internal...')
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)
    
    err_std_probe  = df[df['condition']=='error_standard']['probe_score'].values
    err_verb_probe = df[df['condition']=='error_verbalized']['probe_score'].values

    
    verb_conf_vals = verb_df['verbalized_conf'].dropna().values
    paired_probe   = df[df['condition']=='error_verbalized'].dropna(
        subset=['verbalized_conf'])['probe_score'].values

    if len(verb_conf_vals) > 3 and len(paired_probe) > 3:
        ax.scatter(paired_probe, 1 - verb_conf_vals,  
                   color=CV, alpha=0.7, s=60, edgecolors='white', lw=0.5)
        r_sp, p_sp = spearmanr(paired_probe, 1 - verb_conf_vals)
        ax.set_xlabel('Probe Score (internal uncertainty)')
        ax.set_ylabel('1 − Verbalized Confidence (stated uncertainty)')
        ax.set_title(f'Probe vs. Verbalized Confidence\n'
                     f'Spearman r={r_sp:.3f}  p={p_sp:.4f}',
                     fontsize=11, fontweight='bold')
        
        z = np.polyfit(paired_probe, 1 - verb_conf_vals, 1)
        xp = np.linspace(0, 1, 100)
        ax.plot(xp, np.polyval(z, xp), color=CV, lw=2, ls='--')
        
        if abs(r_sp) < 0.3:
            interp = 'Weak correlation → model does not reliably\nverbalize internal uncertainty (H4 supported)'
        else:
            interp = f'r={r_sp:.3f} → some correlation between\ninternal and stated uncertainty'
        ax.text(0.05, 0.95, interp, transform=ax.transAxes,
                va='top', fontsize=9, color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
    else:
        ax.text(0.5, 0.5, 'Insufficient verbalized\nconfidence ratings\nparsed from model output',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    
    def safe_mean(s):
        return s.fillna(False).astype(float).mean()

    cond_dfs   = [(clean_df, 'Clean\n(standard)',     CC),
                   (error_df, 'Error\n(standard)',     CE),
                   (verb_df,  'Error\n(verbalized)',   CV)]
    x_h   = np.arange(len(cond_dfs))
    hedge_vals    = [safe_mean(d['hedged'])       for d, _, _ in cond_dfs]
    overconf_vals = [safe_mean(d['overconfident']) if 'overconfident' in d.columns
                     else 0.0 for d, _, _ in cond_dfs]
    labels        = [l for _, l, _ in cond_dfs]

    
    ax2.plot(x_h, hedge_vals, 'o-', color='#9467bd', lw=2.5, ms=11,
             label='Hedging rate (model expresses doubt)')
    for xi, val in zip(x_h, hedge_vals):
        ax2.text(xi, val + 0.025, f'{val:.1%}', ha='center', fontsize=10,
                 fontweight='bold', color='#9467bd')

    
    ax2.plot(x_h, overconf_vals, 's--', color=CE, lw=2.5, ms=11,
             label='Overconfidence rate (model asserts certainty)')
    for xi, val in zip(x_h, overconf_vals):
        ax2.text(xi, val - 0.045, f'{val:.1%}', ha='center', fontsize=10,
                 fontweight='bold', color=CE)

    ax2.set_xticks(x_h)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Rate (fraction of hops)')
    ax2.set_title('H4: Hedging vs. Overconfidence Across Conditions\n'
                  'When probe reads high uncertainty, does model express doubt — or assert certainty?',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(0, color=CG, ls='--', lw=0.8, alpha=0.5)

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    
    hedged_probe     = error_df[error_df['hedged']==True]['probe_score'].values
    not_hedged_probe = error_df[error_df['hedged']==False]['probe_score'].values

    plotted = False
    if len(hedged_probe) >= 3 and len(not_hedged_probe) >= 3:
        xh, yh = kde_line(hedged_probe)
        xn, yn = kde_line(not_hedged_probe)
        ax3.plot(xh, yh, color=CV, lw=2.8, label=f'Hedged (n={len(hedged_probe)})')
        ax3.fill_between(xh, yh, alpha=0.14, color=CV)
        ax3.plot(xn, yn, color=CE, lw=2.8, label=f'Not hedged (n={len(not_hedged_probe)})')
        ax3.fill_between(xn, yn, alpha=0.14, color=CE)
        ax3.axvline(hedged_probe.mean(),     color=CV, ls='--', lw=1.5, alpha=0.75)
        ax3.axvline(not_hedged_probe.mean(), color=CE, ls='--', lw=1.5, alpha=0.75)
        d = cohen_d(hedged_probe, not_hedged_probe)
        ax3.text(0.97, 0.97,
                 f'Hedged μ={hedged_probe.mean():.3f}\n'
                 f'Not hedged μ={not_hedged_probe.mean():.3f}\n'
                 f"Cohen d={d:.3f}",
                 transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
        ax3.text(0.05, 0.05,
                 'If d≈0: model hedges randomly relative to\ninternal state (H4 supported)',
                 transform=ax3.transAxes, va='bottom', fontsize=8.5, color='#444',
                 style='italic')
        ax3.set_title('Does the Model Hedge When It Should?\n'
                      'Probe score distribution: hedged vs. not hedged (error hops)',
                      fontsize=11, fontweight='bold')
        plotted = True
    elif 'overconfident' in error_df.columns:
        
        oc_probe  = error_df[error_df['overconfident']==True]['probe_score'].values
        nc_probe  = error_df[error_df['overconfident']==False]['probe_score'].values
        if len(oc_probe) >= 3 and len(nc_probe) >= 3:
            xo, yo = kde_line(oc_probe)
            xn, yn = kde_line(nc_probe)
            ax3.plot(xo, yo, color=CE, lw=2.8,
                     label=f'Overconfident (n={len(oc_probe)})')
            ax3.fill_between(xo, yo, alpha=0.14, color=CE)
            ax3.plot(xn, yn, color=CC, lw=2.8,
                     label=f'Not overconfident (n={len(nc_probe)})')
            ax3.fill_between(xn, yn, alpha=0.14, color=CC)
            ax3.axvline(oc_probe.mean(), color=CE, ls='--', lw=1.5, alpha=0.75)
            ax3.axvline(nc_probe.mean(), color=CC, ls='--', lw=1.5, alpha=0.75)
            d = cohen_d(oc_probe, nc_probe)
            ax3.text(0.97, 0.97,
                     f'Overconfident μ={oc_probe.mean():.3f}\n'
                     f'Not μ={nc_probe.mean():.3f}\n'
                     f"Cohen d={d:.3f}",
                     transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
            ax3.text(0.05, 0.05,
                     'High probe + overconfident statements → \n'
                     'strong H4 support: model unaware of internal state',
                     transform=ax3.transAxes, va='bottom', fontsize=8.5, color='#444',
                     style='italic')
            ax3.set_title('Does the Model Assert Certainty Despite High Probe?\n'
                          'Probe distribution: overconfident vs. neutral (error hops)',
                          fontsize=11, fontweight='bold')
            plotted = True

    if not plotted:
        ax3.text(0.5, 0.5,
                 'Insufficient hedging or\noverconfidence instances\nin sample',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Does the Model Express Its Internal State?',
                      fontsize=11, fontweight='bold')
    ax3.set_xlabel('Probe Score (internal uncertainty)')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2)


    fig.suptitle('H4: Gap Between Internal Uncertainty and Verbalized Confidence',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig2_h4_verbalized.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'    → {p}')

    
    print('  Figure 3: Downstream error rate...')
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(BG)

    
    error_with_outcome = df[
        (df['condition'] == 'error_standard') &
        (df['final_correct_error'].notna())
    ].copy()
    error_with_outcome['final_wrong'] = (~error_with_outcome['final_correct_error']).astype(int)

    ax = axes[0]
    ax.set_facecolor(BG)
    if len(error_with_outcome) > 10 and error_with_outcome['final_wrong'].nunique() > 1:
        fpr_d, tpr_d, _ = roc_curve(
            error_with_outcome['final_wrong'],
            error_with_outcome['probe_score'])
        auroc_d = roc_auc_score(
            error_with_outcome['final_wrong'],
            error_with_outcome['probe_score'])
        ax.plot(fpr_d, tpr_d, color=CP, lw=3.0,
                label=f'Probe score (AUROC={auroc_d:.3f})')

        for sig, color in [('logprob_uncertainty','#ff7f0e'),
                            ('peak_entropy', CE)]:
            fpr_s, tpr_s, _ = roc_curve(
                error_with_outcome['final_wrong'],
                error_with_outcome[sig])
            auroc_s = roc_auc_score(
                error_with_outcome['final_wrong'],
                error_with_outcome[sig])
            ax.plot(fpr_s, tpr_s, color=color, lw=2.0,
                    label=f'{sig} (AUROC={auroc_s:.3f})')

        ax.plot([0,1],[0,1], color=CG, ls='--', lw=1.2, alpha=0.5)
        ax.set_title(f'ROC: Predicting Final Answer Failure\n'
                     f'(n={len(error_with_outcome)} error traces)',
                     fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient outcome\ndata in sample\n(increase n_sample)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    
    if len(error_with_outcome) > 10:
        degraded_probe     = error_with_outcome[error_with_outcome['final_wrong']==1]['probe_score'].values
        not_degraded_probe = error_with_outcome[error_with_outcome['final_wrong']==0]['probe_score'].values
        if len(degraded_probe) > 2 and len(not_degraded_probe) > 2:
            xd, yd = kde_line(degraded_probe)
            xn, yn = kde_line(not_degraded_probe)
            ax2.plot(xd, yd, color=CE, lw=2.8,
                     label=f'Final wrong (n={len(degraded_probe)})')
            ax2.fill_between(xd, yd, alpha=0.14, color=CE)
            ax2.plot(xn, yn, color=CP, lw=2.8,
                     label=f'Final correct (n={len(not_degraded_probe)})')
            ax2.fill_between(xn, yn, alpha=0.14, color=CP)
            ax2.axvline(degraded_probe.mean(),     color=CE, ls='--', lw=1.5)
            ax2.axvline(not_degraded_probe.mean(), color=CP, ls='--', lw=1.5)
            d2 = cohen_d(degraded_probe, not_degraded_probe)
            ax2.text(0.97, 0.97,
                     f'Wrong μ={degraded_probe.mean():.3f}\n'
                     f'Correct μ={not_degraded_probe.mean():.3f}\n'
                     f"Cohen d={d2:.3f}",
                     transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
    ax2.set_xlabel('Probe Score at Hop 1'); ax2.set_ylabel('Density')
    ax2.set_title('Probe Score at Hop 1\nvs. Final Answer Outcome',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    
    depths = sorted(df['hop_depth'].unique())
    clean_acc = []
    error_acc = []
    for d in depths:
        cd = df[(df['condition']=='clean') & (df['hop_depth']==d) &
                (df['final_correct_clean'].notna())]
        ed = df[(df['condition']=='error_standard') & (df['hop_depth']==d) &
                (df['final_correct_error'].notna())]
        clean_acc.append(cd['final_correct_clean'].astype(float).mean() if len(cd) else np.nan)
        error_acc.append(ed['final_correct_error'].astype(float).mean() if len(ed) else np.nan)

    depths_x = np.arange(len(depths))
    ax3.plot(depths_x, clean_acc, 'o-', color=CC, lw=2.5, ms=10, label='Clean chain')
    ax3.plot(depths_x, error_acc, 's-', color=CE, lw=2.5, ms=10, label='Error chain')
    ax3.fill_between(depths_x,
                     [c if c is not np.nan else 0 for c in clean_acc],
                     [e if e is not np.nan else 0 for e in error_acc],
                     alpha=0.10, color=CE, label='Degradation')
    ax3.set_xticks(depths_x)
    ax3.set_xticklabels([f'{d}-hop' for d in depths], fontsize=10)
    ax3.set_ylabel('Final answer accuracy')
    ax3.set_title('Performance Degradation by Hop Depth\n'
                  'Clean vs. error chain final accuracy',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2); ax3.set_ylim(0, 1.05)

    fig.suptitle('Downstream Error Rate: Does Probe Uncertainty Predict Final Failure?',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig3_downstream_errors.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'    → {p}')

    
    print('  Figure 4: Layer probe...')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    layers = np.arange(n_layers)

    ax = axes[0]
    ax.set_facecolor(BG)
    fm = layer_folds.mean(axis=1); fs = layer_folds.std(axis=1)
    ax.fill_between(layers, fm-2*fs, fm+2*fs, alpha=0.07, color='#1f77b4')
    ax.fill_between(layers, fm-fs,   fm+fs,   alpha=0.18, color='#1f77b4')
    ax.plot(layers, layer_auroc, color='#1f77b4', lw=2.8, label='Mean AUROC')
    pt_c = ['#1a7f3c' if a>=0.90 else '#2ca02c' if a>=0.75
            else '#ff7f0e' if a>=0.65 else '#1f77b4' if a>=0.55
            else CE for a in layer_auroc]
    ax.scatter(layers, layer_auroc, c=pt_c, s=50, zorder=5,
               edgecolors='white', lw=0.7)
    ax.axhline(0.50, color=CG,  ls='--', lw=1.2, alpha=0.7, label='Chance')
    ax.axhline(0.75, color='#2ca02c', ls=':',  lw=1.5, label='Good (0.75)')
    ax.axhline(0.90, color='#1a7f3c', ls='-.', lw=1.5, label='Excellent (0.90)')
    ax.axvline(best_layer, color='red', ls=':', lw=2.5)
    ax.text(best_layer+0.3, layer_auroc.min()-0.01,
            f'L{best_layer}\n{probe_auroc:.4f}', color='red', fontsize=9, fontweight='bold')
    ax.set_xlabel('Layer'); ax.set_ylabel('AUROC')
    ax.set_title(f'Activation Probe — {n_layers} Layers\n'
                 f'Best L{best_layer}={probe_auroc:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, n_layers-0.5); ax.set_ylim(0.44, 1.02)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    delta = np.diff(layer_auroc)
    ax2.axhline(0, color=CG, ls='--', lw=1.2, alpha=0.6)
    ax2.plot(layers[1:], delta, color='#9467bd', lw=2.2)
    ax2.fill_between(layers[1:], delta, 0, where=(delta>0), alpha=0.22, color='#2ca02c')
    ax2.fill_between(layers[1:], delta, 0, where=(delta<=0), alpha=0.22, color=CE)
    bj = int(np.argmax(delta))
    ax2.annotate(f'Biggest jump\nL{bj}→L{bj+1}\n(+{delta[bj]:.4f})',
                 xy=(bj+1, delta[bj]), xytext=(bj+2, delta[bj]+0.003),
                 fontsize=8.5, color='#2ca02c',
                 arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
    ax2.set_xlabel('Layer'); ax2.set_ylabel('ΔAUROC')
    ax2.set_title('Layer-to-Layer AUROC Change\n'
                  'WHERE the signal crystallises most rapidly',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2); ax2.set_xlim(-0.5, n_layers-0.5)

    plt.tight_layout()
    p = os.path.join(out_dir, 'fig4_layer_probe.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'    → {p}')

    return comp_df, probe_auroc, best_layer





def print_summary(df, comp_df, probe_auroc, best_layer):
    print(f'\n{"="*72}')
    print('  PHASE 3 RESULTS SUMMARY')
    print(f'{"="*72}')
    print(f'  N = {len(df)} records across 3 conditions\n')

    
    print('  H1: Probe accuracy >70% by step 5')
    print(f'    L{best_layer} probe AUROC = {probe_auroc:.4f}  '
          f'→ {"✓ CONFIRMED" if probe_auroc > 0.70 else "✗ NOT CONFIRMED"}\n')

    
    print('  H2: Signal localises to mid-to-late layers')
    print(f'    Best layer: L{best_layer}  → check fig4 for profile\n')

    
    print('  H3: Probe > logprob > surface as failure predictor')
    if len(comp_df):
        for _, row in comp_df.sort_values('auroc_degradation', ascending=False).iterrows():
            print(f'    {row["method"][:35]:<35}: '
                  f'detection={row["auroc_detection"]:.3f}  '
                  f'failure_pred={row["auroc_degradation"]:.3f}')

    
    clean_df  = df[df['condition']=='clean']
    error_df  = df[df['condition']=='error_standard']
    verb_df   = df[df['condition']=='error_verbalized']
    clean_h   = clean_df['hedged'].fillna(False).mean()
    error_h   = error_df['hedged'].fillna(False).mean()
    print(f'\n  H4: Gap between internal signal and verbalized confidence')
    print(f'    Hedging rate         — clean: {clean_h:.1%}  error: {error_h:.1%}')
    if 'overconfident' in df.columns:
        clean_o = clean_df['overconfident'].fillna(False).mean()
        error_o = error_df['overconfident'].fillna(False).mean()
        verb_o  = verb_df['overconfident'].fillna(False).mean()
        print(f'    Overconfidence rate  — clean: {clean_o:.1%}  '
              f'error: {error_o:.1%}  error_verb: {verb_o:.1%}')
    parsed     = verb_df['verbalized_conf'].notna().sum()
    compliance = parsed / len(verb_df) if len(verb_df) > 0 else 0
    print(f'    Compliance           — model produced parseable rating in '
          f'{parsed}/{len(verb_df)} ({compliance:.1%})')
    if parsed > 0:
        mean_vc = verb_df['verbalized_conf'].dropna().mean()
        std_vc  = verb_df['verbalized_conf'].dropna().std()
        print(f'    Stated confidence    — mean={mean_vc:.2%}  std={std_vc:.2%}')
    if compliance < 0.20:
        print(f'    → COMPLIANCE FAILURE: model ignores instruction to rate confidence')
    elif compliance < 0.70:
        print(f'    → PARTIAL COMPLIANCE: model rates confidence sometimes')
    else:
        print(f'    → HIGH COMPLIANCE: model reliably rates confidence')
    if error_h - clean_h < 0.10:
        print(f'    → Δ_hedge={error_h-clean_h:.1%}: model rarely hedges even on error hops')
        print(f'    → H4 SUPPORTED: internal uncertainty does not reliably surface')
    else:
        print(f'    → Δ_hedge={error_h-clean_h:.1%}: model hedges more on error hops')
        print(f'    → H4 PARTIALLY SUPPORTED')

    print(f'\n{"="*72}\n')





def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    rec_path   = os.path.join(args.output_dir, 'all_records.csv')
    acts_path  = os.path.join(args.output_dir, 'act_arr.npy')
    albl_path  = os.path.join(args.output_dir, 'act_labels.npy')
    probe_path = os.path.join(args.output_dir, 'layer_probe.npy')

    if args.plot_only:
        print('Loading saved data...')
        df         = pd.read_csv(rec_path)
        act_arr    = np.load(acts_path)
        act_labels = np.load(albl_path)
    else:
        df, act_arr, act_labels, _ = collect_data(args, device)
        df.to_csv(rec_path, index=False)
        np.save(acts_path,  act_arr)
        np.save(albl_path,  act_labels)
        print(f'Data saved → {args.output_dir}/')

    if os.path.exists(probe_path) and args.plot_only:
        d = np.load(probe_path, allow_pickle=True).item()
        layer_auroc = d['auroc']; layer_acc  = d['acc']
        layer_acc_std = d['acc_std']; layer_folds = d['folds']
    else:
        layer_auroc, layer_acc, layer_acc_std, layer_folds =            run_layer_probe(act_arr, act_labels, seed=args.seed)
        np.save(probe_path, {'auroc': layer_auroc, 'acc': layer_acc,
                              'acc_std': layer_acc_std, 'folds': layer_folds})

    comp_df, probe_auroc, best_layer = make_all_figures(
        df, act_arr, act_labels,
        layer_auroc, layer_acc, layer_acc_std, layer_folds,
        args.output_dir)

    print_summary(df, comp_df, probe_auroc, best_layer)

    print('Outputs:')
    for f in ['all_records.csv', 'comparison_table.csv',
              'fig1_h3_comparison.png', 'fig2_h4_verbalized.png',
              'fig3_downstream_errors.png', 'fig4_layer_probe.png']:
        print(f'  {args.output_dir}/{f}')


if __name__ == '__main__':
    main()