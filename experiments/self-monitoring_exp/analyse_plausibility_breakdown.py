

import os, re, argparse, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')




ABSURDITY_PATTERNS = [
    
    r'-\$?\d',                              
    r'\bnegative\b',                        
    r'minus\s+\d',
    
    r'\bnot\s+possible\b',
    r'\bimpossible\b',
    r'\bcannot\s+be\b',
    r'check\s+the\s+problem',
    r'\berror\s+in\s+the\s+problem',
    r"doesn'?t\s+make\s+sense",
    r'\bnonsensical\b',
    r'\binvalid\b',
    
    r'=\s*0\s+(?:slices|hours|minutes|miles|km|kg|grams)',
]

def is_absurd_answer(text: str) -> bool:
    
    text_lower = text.lower()
    for pat in ABSURDITY_PATTERNS:
        if re.search(pat, text_lower):
            return True
    return False





def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--records',
                   default='results_phase3_v3/all_records.csv',
                   help='Path to all_records.csv from Phase 3 run')
    p.add_argument('--output_dir',
                   default='results_phase3_v3',
                   help='Where to save the analysis figure and CSV')
    return p.parse_args()





def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2: return 0.0
    p = np.sqrt(((len(a)-1)*np.std(a,ddof=1)**2+(len(b)-1)*np.std(b,ddof=1)**2)
                /(len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / (p + 1e-10)

def kde_line(vals, n=400):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3: return np.array([]), np.array([])
    s = max(vals.std() * 0.45, 1e-6)
    x = np.linspace(vals.min() - s, vals.max() + s, n)
    return x, gaussian_kde(vals, bw_method='silverman')(x)





BG = '#f5f5f5'
C_PLAUSIBLE = '#2ca02c'    
C_ABSURD    = '#d62728'    
C_VERB10    = '#1f77b4'    
C_VERB0     = '#ff7f0e'    
C_GREY      = '#888888'





def main():
    args = get_args()
    df = pd.read_csv(args.records)
    print(f'Loaded {len(df)} records from {args.records}')

    
    verb = df[df['condition'] == 'error_verbalized'].copy()
    print(f'  {len(verb)} error_verbalized traces')
    parsed = verb['verbalized_conf'].notna().sum()
    print(f'  {parsed} have parseable verbalized confidence')

    
    verb['is_absurd'] = verb['generated_text'].fillna('').apply(is_absurd_answer)
    n_absurd    = verb['is_absurd'].sum()
    n_plausible = (~verb['is_absurd']).sum()
    print(f'\nAbsurdity classification:')
    print(f'  Absurd (negative qty / impossible):  {n_absurd}/{len(verb)}')
    print(f'  Plausible-looking:                   {n_plausible}/{len(verb)}')

    
    print('\n=== H4 main result: verbalized confidence × answer plausibility ===')
    print()
    verb_complete = verb.dropna(subset=['verbalized_conf']).copy()
    if len(verb_complete) == 0:
        print('  No verbalized confidence ratings — nothing to analyse.')
        return

    crosstab = pd.crosstab(
        verb_complete['verbalized_conf'].apply(lambda x: f'{x*10:.0f}/10'),
        verb_complete['is_absurd'].map({True: 'Absurd', False: 'Plausible'}),
        margins=True
    )
    print(crosstab)
    print()

    
    if n_absurd > 0:
        absurd_says_low = verb_complete[
            (verb_complete['is_absurd']) & (verb_complete['verbalized_conf'] < 0.5)
        ]
        rate_absurd_low = len(absurd_says_low) / n_absurd
        print(f'  When answer IS absurd: model says <50% confidence in {len(absurd_says_low)}/{n_absurd} ({rate_absurd_low:.0%})')

    if n_plausible > 0:
        plausible_says_low = verb_complete[
            (~verb_complete['is_absurd']) & (verb_complete['verbalized_conf'] < 0.5)
        ]
        rate_plausible_low = len(plausible_says_low) / n_plausible
        print(f'  When answer is plausible-looking: model says <50% confidence in {len(plausible_says_low)}/{n_plausible} ({rate_plausible_low:.0%})')

    
    print('\n=== Correctness × confidence × plausibility ===')
    err_lookup = df[df['condition']=='error_standard'].set_index('base_id')[['final_correct_error']]
    verb_complete = verb_complete.set_index('base_id').join(err_lookup, rsuffix='_err').reset_index()
    correctness_col = ('final_correct_error_err'
                       if 'final_correct_error_err' in verb_complete.columns
                       else 'final_correct_error')

    for is_absurd_val in [True, False]:
        for conf_threshold_label, conf_filter in [
            ('low (<50%) ', verb_complete['verbalized_conf'] < 0.5),
            ('high (≥50%)', verb_complete['verbalized_conf'] >= 0.5),
        ]:
            sub = verb_complete[
                (verb_complete['is_absurd'] == is_absurd_val) & conf_filter
            ]
            if len(sub) == 0:
                continue
            n_correct = sub[correctness_col].fillna(False).astype(bool).sum()
            n_total   = sub[correctness_col].notna().sum()
            absurd_label = 'absurd' if is_absurd_val else 'plausible'
            if n_total > 0:
                print(f'  {absurd_label:9s}, conf {conf_threshold_label}: '
                      f'final correct {n_correct}/{n_total} ({n_correct/n_total:.0%})  n={len(sub)}')

    
    print('\n=== Surface signal differences: absurd vs plausible ===')
    for sig in ['logprob_uncertainty','peak_entropy','min_confidence','mean_vocab_gap']:
        if sig not in verb.columns:
            continue
        a = verb[verb['is_absurd']==True][sig].dropna().values
        p = verb[verb['is_absurd']==False][sig].dropna().values
        if len(a) < 3 or len(p) < 3:
            continue
        d = cohen_d(a, p)
        try:
            stat, pv = mannwhitneyu(a, p, alternative='two-sided')
        except ValueError:
            pv = 1.0
        star = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
        print(f'  {sig:<22s}: absurd μ={a.mean():.3f}  plausible μ={p.mean():.3f}  '
              f'd={d:+.3f}  p={pv:.4f} {star}')

    
    out_csv = os.path.join(args.output_dir, 'plausibility_breakdown.csv')
    verb[['base_id','hop_depth','error_type','verbalized_conf','is_absurd',
          'logprob_uncertainty','peak_entropy','min_confidence','mean_vocab_gap',
          'generated_text']].to_csv(out_csv, index=False)
    print(f'\nSaved tagged CSV → {out_csv}')

    
    
    
    print('\nGenerating figure...')
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor(BG)

    
    ax = axes[0]
    ax.set_facecolor(BG)
    
    abs_at_0  = ((verb_complete['is_absurd']) & (verb_complete['verbalized_conf']==0.0)).sum()
    abs_at_10 = ((verb_complete['is_absurd']) & (verb_complete['verbalized_conf']==1.0)).sum()
    pls_at_0  = ((~verb_complete['is_absurd']) & (verb_complete['verbalized_conf']==0.0)).sum()
    pls_at_10 = ((~verb_complete['is_absurd']) & (verb_complete['verbalized_conf']==1.0)).sum()

    x = np.array([0, 1])
    abs_counts = np.array([abs_at_0, abs_at_10])
    pls_counts = np.array([pls_at_0, pls_at_10])

    ax.plot(x, abs_counts, 'o-', color=C_ABSURD, lw=3, ms=15,
            label=f'Absurd answers (n={n_absurd})')
    ax.plot(x, pls_counts, 's-', color=C_PLAUSIBLE, lw=3, ms=15,
            label=f'Plausible answers (n={n_plausible})')
    for xi, va, vp in zip(x, abs_counts, pls_counts):
        ax.text(xi, va + 2, str(va), ha='center', fontsize=11,
                fontweight='bold', color=C_ABSURD)
        ax.text(xi, vp + 2, str(vp), ha='center', fontsize=11,
                fontweight='bold', color=C_PLAUSIBLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Confidence: 0/10\n("not sure")',
                        'Confidence: 10/10\n("sure")'],
                       fontsize=10)
    ax.set_ylabel('Number of error_verbalized traces', fontsize=11)
    ax.set_title('Verbalized Confidence × Answer Plausibility\n'
                 'Does model use 0/10 only on absurd outputs?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, max(abs_counts.max(), pls_counts.max()) * 1.18)

    
    if n_absurd > 0 and abs_at_0 / n_absurd >= 0.7:
        msg = ('When answer is absurd → model says 0/10\n'
               'When answer is plausible → model says 10/10\n'
               '→ verbalized confidence tracks PLAUSIBILITY,\n   not internal uncertainty')
        ax.text(0.5, 0.50, msg, transform=ax.transAxes, ha='center', va='center',
                fontsize=9.5, color='#222', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          alpha=0.9, edgecolor='#aaa'))

    
    ax = axes[1]
    ax.set_facecolor(BG)
    a_lp = verb[verb['is_absurd']==True]['logprob_uncertainty'].dropna().values
    p_lp = verb[verb['is_absurd']==False]['logprob_uncertainty'].dropna().values
    if len(a_lp) >= 3:
        xa, ya = kde_line(a_lp)
        ax.plot(xa, ya, color=C_ABSURD, lw=2.8,
                label=f'Absurd answer (n={len(a_lp)})')
        ax.fill_between(xa, ya, alpha=0.16, color=C_ABSURD)
        ax.axvline(a_lp.mean(), color=C_ABSURD, ls='--', lw=1.5, alpha=0.75)
    if len(p_lp) >= 3:
        xp, yp = kde_line(p_lp)
        ax.plot(xp, yp, color=C_PLAUSIBLE, lw=2.8,
                label=f'Plausible answer (n={len(p_lp)})')
        ax.fill_between(xp, yp, alpha=0.16, color=C_PLAUSIBLE)
        ax.axvline(p_lp.mean(), color=C_PLAUSIBLE, ls='--', lw=1.5, alpha=0.75)
    if len(a_lp) >= 3 and len(p_lp) >= 3:
        d = cohen_d(a_lp, p_lp)
        try:
            _, pv = mannwhitneyu(a_lp, p_lp, alternative='two-sided')
        except ValueError:
            pv = 1.0
        star = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
        ax.text(0.97, 0.97,
                f'Absurd μ={a_lp.mean():.3f}\n'
                f'Plausible μ={p_lp.mean():.3f}\n'
                f'Cohen d={d:+.2f}  {star}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.92, edgecolor='#aaa'))
    ax.set_xlabel('Logprob Uncertainty (mean NLL)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Internal Uncertainty by Plausibility\n'
                 'Are absurd-output traces also internally less certain?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.20)

    
    ax = axes[2]
    ax.set_facecolor(BG)
    err_outcome = df[df['condition']=='error_standard'].set_index('base_id')[['final_correct_error']]
    verb_with_outcome = verb.set_index('base_id').join(err_outcome, rsuffix='_err').reset_index()
    out_col = ('final_correct_error_err'
               if 'final_correct_error_err' in verb_with_outcome.columns
               else 'final_correct_error')

    groups = []
    for absurd_val, conf_label, conf_filter, color in [
        (True,  'Absurd\n+ said 0/10',
         verb_with_outcome['verbalized_conf']==0.0, C_ABSURD),
        (True,  'Absurd\n+ said 10/10',
         verb_with_outcome['verbalized_conf']==1.0, '#fc8d59'),
        (False, 'Plausible\n+ said 0/10',
         verb_with_outcome['verbalized_conf']==0.0, '#74c476'),
        (False, 'Plausible\n+ said 10/10',
         verb_with_outcome['verbalized_conf']==1.0, C_PLAUSIBLE),
    ]:
        sub = verb_with_outcome[
            (verb_with_outcome['is_absurd']==absurd_val) & conf_filter]
        n_total = sub[out_col].notna().sum()
        if n_total == 0:
            wrong_rate = np.nan
        else:
            wrong_rate = (~sub[out_col].fillna(True).astype(bool)).sum() / n_total
        groups.append({
            'label':   conf_label,
            'wrong':   wrong_rate,
            'n':       n_total,
            'color':   color,
        })

    valid = [g for g in groups if not np.isnan(g['wrong'])]
    if valid:
        x_g = np.arange(len(valid))
        y_g = [g['wrong'] for g in valid]
        labels = [g['label'] for g in valid]
        ax.plot(x_g, y_g, 'o-', color=C_GREY, lw=2, zorder=2)
        for xi, g in zip(x_g, valid):
            ax.scatter(xi, g['wrong'], color=g['color'], s=200, zorder=4,
                       edgecolors='white', linewidths=1.5)
            ax.text(xi, g['wrong']+0.04, f'{g["wrong"]:.0%}\nn={g["n"]}',
                    ha='center', fontsize=9.5, fontweight='bold', color=g['color'])
        ax.set_xticks(x_g)
        ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.5, color=C_GREY, ls='--', lw=1.0, alpha=0.5,
               label='Chance')
    ax.set_ylabel('Final answer wrong rate', fontsize=11)
    ax.set_title('Does Verbalized Confidence Predict Failure?\n'
                 'Final-wrong rate within each plausibility × confidence group',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.20)
    ax.legend(fontsize=9, loc='lower right')

    
    ax = axes[3]
    ax.set_facecolor(BG)
    err_types = sorted(verb_complete['error_type'].dropna().unique())
    n_per_type   = []
    rate_low_per_type = []
    rate_absurd_per_type = []
    for et in err_types:
        sub = verb_complete[verb_complete['error_type']==et]
        n_per_type.append(len(sub))
        rate_low_per_type.append(
            (sub['verbalized_conf']<0.5).mean() if len(sub) else 0)
        rate_absurd_per_type.append(
            sub['is_absurd'].mean() if len(sub) else 0)

    x_e = np.arange(len(err_types))
    ax.plot(x_e, rate_low_per_type, 'o-', color=C_VERB0, lw=2.5, ms=12,
            label='% saying 0/10')
    ax.plot(x_e, rate_absurd_per_type, 's--', color=C_ABSURD, lw=2.5, ms=12,
            label='% absurd output')
    for xi, rl, ra, n in zip(x_e, rate_low_per_type, rate_absurd_per_type, n_per_type):
        ax.text(xi, rl + 0.04, f'{rl:.0%}', ha='center', fontsize=8.5,
                color=C_VERB0, fontweight='bold')
        ax.text(xi, ra - 0.06, f'{ra:.0%}', ha='center', fontsize=8.5,
                color=C_ABSURD, fontweight='bold')
        ax.text(xi, -0.10, f'n={n}', ha='center', fontsize=7.5, color=C_GREY)
    ax.set_xticks(x_e)
    ax.set_xticklabels([e.replace('_','\n') for e in err_types],
                        fontsize=8.5, rotation=15, ha='right')
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('Behaviour by Error Type\n'
                 'Two lines tracking together = model only doubts when output is absurd',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(-0.18, 1.10)

    
    fig.suptitle('H4 Refined: Verbalized Confidence Tracks Output Plausibility, '
                 'Not Internal Uncertainty',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'fig5_plausibility.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved figure → {fig_path}')

    
    print(f'\n{"="*72}')
    print('  PLAUSIBILITY HYPOTHESIS TEST')
    print(f'{"="*72}')
    if n_absurd > 0 and n_plausible > 0:
        absurd_low_rate = ((verb_complete['is_absurd']) &
                            (verb_complete['verbalized_conf']<0.5)).sum() / n_absurd
        plausible_low_rate = ((~verb_complete['is_absurd']) &
                               (verb_complete['verbalized_conf']<0.5)).sum() / n_plausible
        print(f'  Among ABSURD outputs:    model says <50% confidence in {absurd_low_rate:.0%}')
        print(f'  Among PLAUSIBLE outputs: model says <50% confidence in {plausible_low_rate:.0%}')
        diff = absurd_low_rate - plausible_low_rate
        print(f'  Difference:              {diff:+.0%}')
        if diff > 0.50:
            print(f'  → STRONG SUPPORT: verbalized confidence tracks plausibility,')
            print(f'    not internal uncertainty. The model expresses doubt only')
            print(f'    when its own output is logically absurd.')
        elif diff > 0.20:
            print(f'  → PARTIAL SUPPORT: plausibility plays a role.')
        else:
            print(f'  → NOT SUPPORTED: confidence does not track plausibility.')
    print(f'{"="*72}\n')


if __name__ == '__main__':
    main()