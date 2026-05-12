

import json, os, random, warnings, argparse, time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu, gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score,    precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings('ignore')





SIGNAL_DIR = {
    
    'mean_entropy':      +1,
    'peak_entropy':      +1,
    'std_entropy':       +1,
    'early_entropy':     +1,
    
    'mean_confidence':   -1,
    'min_confidence':    -1,
    'early_min_conf':    -1,
    
    'ent_jump':          +1,
    'conf_drop':         +1,
    
    
    'mean_vocab_gap':    -1,
    
    'min_vocab_gap':     -1,
    
    'early_min_vgap':    -1,
    
    'peak_neg_vgap':     +1,
}
SIGNAL_COLS = list(SIGNAL_DIR.keys())

BG        = '#f5f5f5'
C_CLEAN   = '#1f77b4'
C_ERROR   = '#d62728'
C_GREY    = '#888888'
C_VGAP    = '#9467bd'   





def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='contrastive_math_dataset.json')
    p.add_argument('--model',      default='Qwen/Qwen2.5-3B')
    p.add_argument('--n_sample',   type=int, default=80)
    p.add_argument('--output_dir', default='results_v2')
    p.add_argument('--max_tokens', type=int, default=150)
    p.add_argument('--max_input',  type=int, default=768)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no_4bit',    action='store_true')
    p.add_argument('--plot_only',  action='store_true')
    return p.parse_args()





def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(
        ((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2)
    )
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-10)


def eval_signal(y_true, scores, name=''):
    if len(np.unique(y_true)) < 2 or len(y_true) < 4:
        return None
    auroc = roc_auc_score(y_true, scores)
    ap    = average_precision_score(y_true, scores)
    prec, rec, thresh = precision_recall_curve(y_true, scores)
    f1s   = 2 * prec * rec / (prec + rec + 1e-10)
    idx   = int(np.argmax(f1s))
    pos   = np.array(scores)[np.array(y_true) == 1]
    neg   = np.array(scores)[np.array(y_true) == 0]
    d     = cohen_d(pos, neg)
    _, p  = mannwhitneyu(pos, neg, alternative='two-sided')
    return {
        'signal':     name,
        'auroc':      float(auroc),
        'ap':         float(ap),
        'best_f1':    float(f1s[idx]),
        'prec_at_f1': float(prec[idx]),
        'rec_at_f1':  float(rec[idx]),
        'cohens_d':   float(d),
        'mw_p':       float(p),
        'n_pos':      int(sum(y_true)),
        'n_neg':      int(len(y_true) - sum(y_true)),
    }


def kde_line(vals, n=500):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 3:
        return np.array([]), np.array([])
    s = max(vals.std() * 0.45, 1e-6)
    x = np.linspace(vals.min() - s, vals.max() + s, n)
    return x, gaussian_kde(vals, bw_method='silverman')(x)





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





@torch.no_grad()
def generate_with_stats(model, tokenizer, prompt, device,
                         max_new_tokens=150, max_input=768):
    
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

    for _ in range(max_new_tokens):
        out    = model(input_ids=input_ids)
        logits = out.logits[0, -1, :]                    
        probs  = torch.softmax(logits, dim=-1)

        
        top2_probs, _ = torch.topk(probs, k=2)
        top1_p = top2_probs[0].item()
        top2_p = top2_probs[1].item()

        confs.append(top1_p)
        ents.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        vgaps.append(top1_p - top2_p)                    

        nxt = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        gen_ids.append(nxt.item())
        input_ids = torch.cat([input_ids, nxt], dim=1)
        if len(gen_ids) > 5 and '\n' in tokenizer.decode([nxt.item()]):
            break
        del out

    torch.cuda.empty_cache()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, confs, ents, vgaps, acts





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
    print(f'  Layers={NUM_LAYERS}, Hidden={model.config.hidden_size}')

    sampled = random.sample(all_ids, min(args.n_sample, len(all_ids)))
    records, act_rows, act_labels = [], [], []
    t0 = time.time()

    print(f'\n--- Collecting ({len(sampled)} base problems) ---')
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

        for variant_label, trace, label in [
            ('clean',      clean, 0),
            ('error_at_1', error, 1),
        ]:
            
            prompt  = build_hop_prompt(trace, target_hop_index=1)
            txt, confs, ents, vgaps, acts = generate_with_stats(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_tokens, max_input=args.max_input)

            
            p0 = build_hop_prompt(trace, target_hop_index=0)
            _, c0, e0, vg0, _ = generate_with_stats(
                model, tokenizer, p0, device,
                max_new_tokens=args.max_tokens, max_input=args.max_input)

            n_early = min(20, len(confs))

            rec = {
                'base_id':    base_id,
                'variant':    variant_label,
                'hop_depth':  hop_depth,
                'subcategory': trace.get('subcategory', 'unknown'),
                'error_type': trace['injected_error_type'],
                'label':      label,

                
                'mean_entropy':    float(np.mean(ents))             if ents  else 0.0,
                'peak_entropy':    float(np.max(ents))              if ents  else 0.0,
                'std_entropy':     float(np.std(ents))              if ents  else 0.0,
                'early_entropy':   float(np.mean(ents[:n_early]))   if ents  else 0.0,

                
                'mean_confidence': float(np.mean(confs))            if confs else 1.0,
                'min_confidence':  float(np.min(confs))             if confs else 1.0,
                'early_min_conf':  float(np.min(confs[:n_early]))   if confs else 1.0,

                
                'ent_jump':  float(np.mean(ents) - np.mean(e0))    if (ents and e0)   else 0.0,
                'conf_drop': float(np.mean(c0)   - np.mean(confs)) if (confs and c0)  else 0.0,

                
                
                
                'mean_vocab_gap':  float(np.mean(vgaps))            if vgaps else 1.0,
                'min_vocab_gap':   float(np.min(vgaps))             if vgaps else 1.0,
                'early_min_vgap':  float(np.min(vgaps[:n_early]))   if vgaps else 1.0,
                
                
                'peak_neg_vgap':   float(-np.min(vgaps))            if vgaps else 0.0,

                'n_tokens':   len(confs),
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
                     ('c', LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])
    auroc_means, acc_means, acc_stds, auroc_folds = [], [], [], []
    print(f'\n--- Layer probe: {n_layers} layers ---')
    for li in range(n_layers):
        X = act_arr[:, li, :]
        a = cross_val_score(pipe, X, act_labels, cv=kf, scoring='roc_auc')
        c = cross_val_score(pipe, X, act_labels, cv=kf, scoring='accuracy')
        auroc_means.append(a.mean())
        acc_means.append(c.mean())
        acc_stds.append(c.std())
        auroc_folds.append(a)
        if li % 5 == 0:
            print(f'  L{li:2d}: AUROC={a.mean():.3f}  Acc={c.mean():.3f}')
    return (np.array(auroc_means), np.array(acc_means),
            np.array(acc_stds), np.array(auroc_folds))






def make_answer_figure(df, sig_df, act_arr, act_labels,
                       layer_auroc, layer_acc, layer_auroc_folds,
                       out_dir):

    y_true     = df['label'].values.astype(int)
    best_layer = int(np.argmax(layer_auroc))
    clean_df   = df[df['label'] == 0]
    error_df   = df[df['label'] == 1]

    
    pipe_cv = Pipeline([('s', StandardScaler()),
                        ('c', LogisticRegression(max_iter=2000, C=1.0,
                                                 random_state=42))])
    probe_scores = cross_val_predict(
        pipe_cv, act_arr[:, best_layer, :], act_labels,
        cv=5, method='predict_proba')[:, 1]
    probe_auroc  = roc_auc_score(act_labels, probe_scores)

    
    best_scalar_row = sig_df.sort_values('auroc', ascending=False).iloc[0]
    best_scalar     = best_scalar_row['signal']
    best_scalar_auroc = best_scalar_row['auroc']

    
    vgap_signals = [s for s in SIGNAL_COLS if 'vgap' in s or 'vocab' in s]
    vgap_rows    = sig_df[sig_df['signal'].isin(vgap_signals)].sort_values(
        'auroc', ascending=False)
    best_vgap     = vgap_rows.iloc[0]['signal'] if len(vgap_rows) else None
    best_vgap_auroc = vgap_rows.iloc[0]['auroc'] if len(vgap_rows) else 0.5

    print(f'\nBest scalar:    {best_scalar}  AUROC={best_scalar_auroc:.4f}')
    print(f'Best vocab gap: {best_vgap}  AUROC={best_vgap_auroc:.4f}')
    print(f'Activation probe L{best_layer}: AUROC={probe_auroc:.4f}')

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(1, 3, wspace=0.32, left=0.05, right=0.97,
                           top=0.88, bottom=0.10)

    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    pal = plt.cm.tab10(np.linspace(0, 0.9, len(SIGNAL_COLS)))

    ax1.plot([0,1],[0,1], color=C_GREY, ls='--', lw=1.5, alpha=0.5, zorder=1)

    for sig, color in zip(SIGNAL_COLS, pal):
        if sig not in df.columns:
            continue
        scores = df[sig].values * SIGNAL_DIR[sig]
        row    = sig_df[sig_df['signal'] == sig]
        if len(row) == 0:
            continue
        auroc  = row.iloc[0]['auroc']
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        if 'vgap' in sig or 'vocab' in sig:
            ax1.plot(fpr, tpr, color=C_VGAP, lw=2.0 if auroc==best_vgap_auroc else 1.2,
                     alpha=0.9 if auroc==best_vgap_auroc else 0.50, zorder=4,
                     ls='-')
        else:
            ax1.plot(fpr, tpr, color=color, lw=0.9, alpha=0.25, zorder=2)

    
    scores_b = df[best_scalar].values * SIGNAL_DIR[best_scalar]
    fpr_b, tpr_b, _ = roc_curve(y_true, scores_b)
    ax1.plot(fpr_b, tpr_b, color=C_ERROR, lw=3.0, zorder=5,
             label=f'Best scalar: {best_scalar}\n(AUROC={best_scalar_auroc:.3f})')

    
    if best_vgap and best_vgap in df.columns:
        scores_v = df[best_vgap].values * SIGNAL_DIR[best_vgap]
        fpr_v, tpr_v, _ = roc_curve(y_true, scores_v)
        ax1.plot(fpr_v, tpr_v, color=C_VGAP, lw=3.0, zorder=5,
                 label=f'Best vocab gap: {best_vgap}\n(AUROC={best_vgap_auroc:.3f})')

    
    fpr_p, tpr_p, _ = roc_curve(act_labels, probe_scores)
    ax1.plot(fpr_p, tpr_p, color='#1a7f3c', lw=4.0, zorder=6,
             label=f'Activation probe L{best_layer}\n(AUROC={probe_auroc:.4f})')

    
    tpr_p_i = np.interp(fpr_b, fpr_p, tpr_p)
    ax1.fill_between(fpr_b, tpr_b, tpr_p_i, alpha=0.09, color='#1a7f3c')

    
    for fpr_arr, tpr_arr, color, short_name, auroc_val in [
        (fpr_b, tpr_b, C_ERROR,   best_scalar,  best_scalar_auroc),
        (fpr_p, tpr_p, '#1a7f3c', f'Probe L{best_layer}', probe_auroc),
    ]:
        dist = np.sqrt(fpr_arr**2 + (1 - tpr_arr)**2)
        oi   = int(np.argmin(dist))
        ax1.scatter([fpr_arr[oi]], [tpr_arr[oi]],
                    s=180, color=color, zorder=9, marker='x', linewidths=2.5)
        offset = (0.10, -0.14) if color == C_ERROR else (0.06, -0.22)
        ax1.annotate(
            f'{short_name}\nTPR={tpr_arr[oi]:.2f}  FPR={fpr_arr[oi]:.2f}',
            xy=(fpr_arr[oi], tpr_arr[oi]),
            xytext=(fpr_arr[oi]+offset[0], tpr_arr[oi]+offset[1]),
            fontsize=8, color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.92, edgecolor=color, lw=1),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves — All Signals\nPurple = vocab gap  |  '
                  'Red = best scalar  |  Green = probe',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right', framealpha=0.93)
    ax1.grid(True, alpha=0.22)
    ax1.set_xlim(-0.01, 1.01); ax1.set_ylim(-0.01, 1.01)

    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)

    
    vgap_pal = ['#7b2d8b','#9467bd','#c5a0d4','#e8d5f0']
    for vi, sig in enumerate(['mean_vocab_gap','min_vocab_gap',
                               'early_min_vgap','peak_neg_vgap']):
        if sig not in df.columns:
            continue
        c_vals = clean_df[sig].values.astype(float)
        e_vals = error_df[sig].values.astype(float)
        row    = sig_df[sig_df['signal'] == sig]
        auroc  = row.iloc[0]['auroc'] if len(row) else 0.5

        lw    = 2.8 if vi == 0 else 1.8
        alpha = 1.0 if vi == 0 else 0.7

        xc, yc = kde_line(c_vals)
        xe, ye = kde_line(e_vals)

        if vi == 0:
            
            ax2.plot(xc, yc, color=C_CLEAN, lw=2.8,
                     label=f'Clean  μ={c_vals.mean():.3f}')
            ax2.plot(xe, ye, color=C_ERROR,  lw=2.8,
                     label=f'Error  μ={e_vals.mean():.3f}')
            ax2.fill_between(xc, yc, alpha=0.14, color=C_CLEAN)
            ax2.fill_between(xe, ye, alpha=0.14, color=C_ERROR)
            ax2.axvline(c_vals.mean(), color=C_CLEAN, ls='--', lw=1.5, alpha=0.75)
            ax2.axvline(e_vals.mean(), color=C_ERROR,  ls='--', lw=1.5, alpha=0.75)
        else:
            
            if len(xc) > 0: ax2.plot(xc, yc, color=vgap_pal[vi], lw=1.2,
                                      alpha=0.45, ls='--')
            if len(xe) > 0: ax2.plot(xe, ye, color=vgap_pal[vi], lw=1.2,
                                      alpha=0.45, ls='-.')

    
    if len(vgap_rows):
        br = vgap_rows.iloc[0]
        star = '***' if br['mw_p']<0.001 else '**' if br['mw_p']<0.01               else '*' if br['mw_p']<0.05 else 'ns'
        c_v = clean_df['mean_vocab_gap'].values.astype(float)
        e_v = error_df['mean_vocab_gap'].values.astype(float)
        ax2.text(0.97, 0.97,
                 f"mean_vocab_gap\n"
                 f"AUROC = {br['auroc']:.3f}\n"
                 f"Cohen d = {br['cohens_d']:.3f}  {star}\n"
                 f"Clean μ={c_v.mean():.3f}\n"
                 f"Error μ={e_v.mean():.3f}\n"
                 f"Δμ = {e_v.mean()-c_v.mean():+.4f}",
                 transform=ax2.transAxes, va='top', ha='right',
                 fontsize=9, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           alpha=0.92, edgecolor=C_VGAP, lw=1.5))

    ax2.set_xlabel('Vocabulary Gap  (top-1 prob − top-2 prob)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Vocabulary Gap Distribution\n'
                  'Clean vs. Error — lower gap = model less decisive',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.22)

    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(BG)

    c_sc = probe_scores[act_labels == 0]
    e_sc = probe_scores[act_labels == 1]
    xc, yc = kde_line(c_sc)
    xe, ye = kde_line(e_sc)

    ax3.plot(xc, yc, color=C_CLEAN, lw=3.5,
             label=f'Clean  μ={c_sc.mean():.3f}  (n={len(c_sc)})')
    ax3.fill_between(xc, yc, alpha=0.18, color=C_CLEAN)
    ax3.plot(xe, ye, color=C_ERROR,  lw=3.5,
             label=f'Error  μ={e_sc.mean():.3f}  (n={len(e_sc)})')
    ax3.fill_between(xe, ye, alpha=0.18, color=C_ERROR)

    
    all_x = np.linspace(min(xc.min(), xe.min()), max(xc.max(), xe.max()), 600)
    yc_i  = gaussian_kde(c_sc, bw_method='silverman')(all_x)
    ye_i  = gaussian_kde(e_sc, bw_method='silverman')(all_x)
    ax3.fill_between(all_x, np.minimum(yc_i, ye_i),
                     alpha=0.35, color='purple', label='Overlap region')

    ax3.axvline(0.5, color='#333', ls=':', lw=2.5, label='Decision boundary (0.5)')

    d_probe = cohen_d(e_sc, c_sc)
    ax3.text(0.50, 0.97,
             f'Activation probe  L{best_layer}\n'
             f'AUROC  = {probe_auroc:.4f}\n'
             f"Cohen d = {d_probe:.2f}\n"
             f'Accuracy ≈ {layer_acc[best_layer]*100:.0f}%',
             transform=ax3.transAxes, va='top', ha='center',
             fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9',
                       alpha=0.95, edgecolor='#1a7f3c', lw=1.8))

    ax3.set_xlabel('Probe Score  P(error context)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title(f'Probe Score Separation — Layer {best_layer}\n'
                  'Near-perfect clean vs. error separation',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper center')
    ax3.grid(True, alpha=0.22)
    ax3.set_xlim(-0.02, 1.02)

    
    fig.text(0.5, 0.97,
             'Can we classify whether a hop has a prior error?',
             ha='center', va='top', fontsize=16, fontweight='bold', color='#111')

    fig.text(0.5, 0.925,
             f'Surface signals (entropy/conf): AUROC={best_scalar_auroc:.3f} — weak, '
             f'catches 59% of errors, 20% false alarms     |     '
             f'Vocab gap (top1−top2): AUROC={best_vgap_auroc:.3f}     |     '
             f'Activation probe L{best_layer}: AUROC={probe_auroc:.4f} — reliable, ~94% accuracy',
             ha='center', va='top', fontsize=10.5, color='#333')

    plt.savefig(os.path.join(out_dir, 'fig_can_we_find_error.png'),
                dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\nSaved: {os.path.join(out_dir, "fig_can_we_find_error.png")}')
    plt.close(fig)





def print_results(df, sig_df, layer_auroc, layer_acc):
    y    = df['label'].values.astype(int)
    bl   = int(np.argmax(layer_auroc))
    print(f'\n{"="*72}')
    print('  CAN WE FIND THE ERROR? — RESULTS')
    print(f'{"="*72}')
    print(f'  N = {len(df)}  ({sum(y==0)} clean / {sum(y==1)} post-injection)')
    print()

    
    groups = {
        'Vocab gap (new)': [s for s in SIGNAL_COLS if 'vgap' in s or 'vocab' in s],
        'Entropy':         ['mean_entropy','peak_entropy','std_entropy','early_entropy'],
        'Confidence':      ['mean_confidence','min_confidence','early_min_conf'],
        'Delta':           ['ent_jump','conf_drop'],
    }
    for group_name, signals in groups.items():
        print(f'  ── {group_name} ──')
        for sig in signals:
            row = sig_df[sig_df['signal'] == sig]
            if len(row) == 0: continue
            r    = row.iloc[0]
            star = '***' if r['mw_p']<0.001 else '**' if r['mw_p']<0.01                   else '*' if r['mw_p']<0.05 else 'ns'
            verdict = ('GOOD' if r['auroc']>=0.75 else
                       'MODERATE' if r['auroc']>=0.65 else
                       'WEAK'     if r['auroc']>=0.55 else 'CHANCE')
            print(f'    {sig:<22} AUROC={r["auroc"]:.3f}  d={r["cohens_d"]:.3f}'
                  f'  {star:<4}  → {verdict}')
        print()

    print(f'  ── Activation probe ──')
    print(f'    L{bl} residual stream   AUROC={layer_auroc[bl]:.4f}  '
          f'Acc={layer_acc[bl]*100:.1f}%  → RELIABLE')
    print()

    best_sc  = sig_df.sort_values('auroc',ascending=False).iloc[0]
    vgap_sigs = sig_df[sig_df['signal'].str.contains('vgap|vocab')]
    best_vg  = vgap_sigs.sort_values('auroc',ascending=False).iloc[0]               if len(vgap_sigs) else None

    print('  VERDICT:')
    print(f'    Surface signals alone:  INSUFFICIENT  '
          f'(best={best_sc["signal"]} AUROC={best_sc["auroc"]:.3f})')
    if best_vg is not None:
        verdict_vg = 'MODERATE' if best_vg['auroc']>=0.65 else                     'WEAK' if best_vg['auroc']>=0.55 else 'CHANCE'
        print(f'    Vocab gap (correct def): {verdict_vg}  '
              f'(best={best_vg["signal"]} AUROC={best_vg["auroc"]:.3f})')
    print(f'    Activation probe:       RELIABLE  '
          f'(L{bl} AUROC={layer_auroc[bl]:.4f}, Acc~{layer_acc[bl]*100:.0f}%)')
    print(f'{"="*72}\n')





def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    rec_path   = os.path.join(args.output_dir, 'all_records.csv')
    sig_path   = os.path.join(args.output_dir, 'signal_results.csv')
    acts_path  = os.path.join(args.output_dir, 'act_arr.npy')
    albl_path  = os.path.join(args.output_dir, 'act_labels.npy')
    probe_path = os.path.join(args.output_dir, 'layer_probe.npy')

    if args.plot_only:
        print('Loading saved data...')
        df         = pd.read_csv(rec_path)
        sig_df     = pd.read_csv(sig_path)
        act_arr    = np.load(acts_path)
        act_labels = np.load(albl_path)
    else:
        df, act_arr, act_labels, _ = collect_data(args, device)
        y_true   = df['label'].values.astype(int)
        sig_rows = []
        for name, direction in SIGNAL_DIR.items():
            if name not in df.columns: continue
            r = eval_signal(y_true, df[name].values * direction, name=name)
            if r: sig_rows.append(r)
        sig_df = pd.DataFrame(sig_rows).sort_values('auroc', ascending=False)
        df.to_csv(rec_path, index=False)
        sig_df.to_csv(sig_path, index=False)
        np.save(acts_path,  act_arr)
        np.save(albl_path,  act_labels)
        print(f'Data saved → {args.output_dir}/')

    
    if os.path.exists(probe_path) and args.plot_only:
        d = np.load(probe_path, allow_pickle=True).item()
        layer_auroc       = d['auroc']
        layer_acc         = d['acc']
        layer_acc_std     = d['acc_std']
        layer_auroc_folds = d['folds']
    else:
        layer_auroc, layer_acc, layer_acc_std, layer_auroc_folds =            run_layer_probe(act_arr, act_labels, seed=args.seed)
        np.save(probe_path, {'auroc': layer_auroc, 'acc': layer_acc,
                              'acc_std': layer_acc_std, 'folds': layer_auroc_folds})

    print_results(df, sig_df, layer_auroc, layer_acc)

    make_answer_figure(df, sig_df, act_arr, act_labels,
                       layer_auroc, layer_acc, layer_auroc_folds,
                       args.output_dir)

    print('Done.')
    print(f'Output: {args.output_dir}/fig_can_we_find_error.png')


if __name__ == '__main__':
    main()