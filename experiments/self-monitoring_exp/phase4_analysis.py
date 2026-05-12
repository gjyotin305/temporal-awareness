

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, mannwhitneyu, fisher_exact


BG = '#fafafa'
COLORS = {
    'baseline':         '#888888',
    'reprompt':         '#1f77b4',
    'replace_prior':    '#2ca02c',
    'branch_and_pick':  '#9467bd',
}
LABELS = {
    'baseline':         'Baseline\n(no intervention)',
    'reprompt':         'Reprompt\n(C1)',
    'replace_prior':    'Replace prior\n(C2)',
    'branch_and_pick':  'Branch + pick\n(C3)',
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='results_phase4')
    return p.parse_args()


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return 0.0
    p = np.sqrt(((len(a)-1)*np.std(a,ddof=1)**2 + (len(b)-1)*np.std(b,ddof=1)**2)
                / (len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b)) / (p+1e-10)


def kde_line(vals, n=400):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3: return np.array([]), np.array([])
    s = max(vals.std()*0.45, 1e-6)
    x = np.linspace(vals.min()-s, vals.max()+s, n)
    return x, gaussian_kde(vals, bw_method='silverman')(x)



def main():
    args = get_args()
    rec_path  = os.path.join(args.output_dir, 'intervention_records.csv')
    thr_path  = os.path.join(args.output_dir, 'probe_threshold.json')

    df = pd.read_csv(rec_path)
    with open(thr_path) as f:
        thr_data = json.load(f)

    print(f'Loaded {len(df)} records, {df["base_id"].nunique()} unique traces')
    print(f'Conditions: {df["condition"].unique().tolist()}')

    
    summary_rows = []
    baseline_acc = None
    for cond in ['baseline','reprompt','replace_prior','branch_and_pick']:
        sub = df[df['condition']==cond]
        if len(sub) == 0: continue
        n_total   = sub['final_correct'].notna().sum()
        n_correct = sub['final_correct'].fillna(False).astype(bool).sum()
        n_absurd  = sub['final_absurd'].fillna(False).astype(bool).sum()
        n_interv  = sub['n_interventions'].sum()
        acc       = n_correct / n_total if n_total > 0 else 0
        if cond == 'baseline':
            baseline_acc = acc
        delta = (acc - baseline_acc) if baseline_acc is not None else 0
        summary_rows.append({
            'condition':         cond,
            'n_traces':          int(n_total),
            'n_correct':         int(n_correct),
            'accuracy':          acc,
            'delta_vs_baseline': delta,
            'n_absurd':          int(n_absurd),
            'absurd_rate':       n_absurd / n_total if n_total > 0 else 0,
            'total_interventions': int(n_interv),
            'avg_interv_per_trace': float(n_interv / n_total) if n_total > 0 else 0,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, 'intervention_summary.csv'),
                       index=False)
    print('\n=== Summary table ===')
    print(summary_df.to_string(index=False))

    
    
    
    print('\nGenerating fig_intervention_main.png...')
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                            left=0.05, right=0.97, top=0.93, bottom=0.06)

    
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    conds = summary_df['condition'].tolist()
    accs  = summary_df['accuracy'].values
    x = np.arange(len(conds))
    ax.plot(x, accs, 'o-', color='#444', lw=2.5, ms=14, zorder=2)
    for xi, acc, cond in zip(x, accs, conds):
        ax.scatter(xi, acc, s=220, color=COLORS[cond], zorder=4,
                   edgecolors='white', lw=2)
        ax.text(xi, acc + 0.025, f'{acc:.1%}', ha='center', fontsize=11,
                fontweight='bold', color=COLORS[cond])
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color=COLORS['baseline'], ls=':', lw=1.5,
                   alpha=0.6, label=f'Baseline ({baseline_acc:.1%})')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], fontsize=9.5)
    ax.set_ylabel('Final answer accuracy', fontsize=12)
    ax.set_title('Headline result: Final accuracy by intervention\n'
                 'on traces with corrupted prior context',
                 fontsize=11.5, fontweight='bold')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, max(accs)*1.20 if max(accs)>0 else 0.5)
    ax.legend(fontsize=9, loc='lower right')

    
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    interv_only = summary_df[summary_df['condition'] != 'baseline']
    if len(interv_only):
        x = np.arange(len(interv_only))
        deltas = interv_only['delta_vs_baseline'].values
        ax.axhline(0, color='#666', ls='-', lw=1, alpha=0.7)
        ax.plot(x, deltas, 'o-', color='#444', lw=2.5, ms=14, zorder=2)
        for xi, d, cond in zip(x, deltas, interv_only['condition'].values):
            ax.scatter(xi, d, s=220, color=COLORS[cond], zorder=4,
                       edgecolors='white', lw=2)
            sign = '+' if d >= 0 else ''
            ax.text(xi, d + (0.012 if d >= 0 else -0.025),
                    f'{sign}{d*100:+.1f}pp',
                    ha='center', fontsize=11.5, fontweight='bold',
                    color=COLORS[cond])
        ax.fill_between(x, deltas, 0,
                         where=(deltas > 0), alpha=0.15, color='#2ca02c')
        ax.fill_between(x, deltas, 0,
                         where=(deltas < 0), alpha=0.15, color='#d62728')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in interv_only['condition']], fontsize=9.5)
        ax.set_ylabel('Δ accuracy vs. baseline (pp)', fontsize=12)
        ax.set_title('Intervention efficacy — gain over no-intervention\n'
                     'Positive = intervention helped',
                     fontsize=11.5, fontweight='bold')
        ax.grid(True, alpha=0.20)

    
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(BG)
    if len(interv_only):
        for _, r in interv_only.iterrows():
            ax.scatter(r['avg_interv_per_trace'], r['delta_vs_baseline']*100,
                        color=COLORS[r['condition']], s=400,
                        edgecolors='white', lw=2, zorder=4)
            ax.annotate(LABELS[r['condition']].replace('\n', ' '),
                         xy=(r['avg_interv_per_trace'], r['delta_vs_baseline']*100),
                         xytext=(8, 8), textcoords='offset points',
                         fontsize=9.5, fontweight='bold',
                         color=COLORS[r['condition']])
        ax.axhline(0, color='#666', ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('Average interventions per trace', fontsize=12)
        ax.set_ylabel('Δ accuracy (pp)', fontsize=12)
        ax.set_title('Cost-benefit tradeoff\n(more interventions = more compute)',
                     fontsize=11.5, fontweight='bold')
        ax.grid(True, alpha=0.20)

    
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(BG)
    base_df = df[df['condition']=='baseline']
    if len(base_df) > 5:
        wrong = base_df[base_df['final_correct']==False]['max_probe_score'].dropna().values
        right = base_df[base_df['final_correct']==True]['max_probe_score'].dropna().values
        if len(wrong) >= 3 and len(right) >= 3:
            xw, yw = kde_line(wrong)
            xr, yr = kde_line(right)
            ax.plot(xw, yw, color='#d62728', lw=2.8, label=f'Final wrong (n={len(wrong)})')
            ax.fill_between(xw, yw, alpha=0.15, color='#d62728')
            ax.plot(xr, yr, color='#2ca02c', lw=2.8, label=f'Final correct (n={len(right)})')
            ax.fill_between(xr, yr, alpha=0.15, color='#2ca02c')
            d = cohen_d(wrong, right)
            try:
                _, pv = mannwhitneyu(wrong, right, alternative='two-sided')
            except ValueError:
                pv = 1.0
            ax.axvline(thr_data['threshold'], color='#333', ls=':', lw=2,
                       label=f'Threshold ({thr_data["threshold"]:.2f})')
            star = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
            ax.text(0.97, 0.97,
                     f"Cohen's d = {d:.3f}\np = {pv:.4f} {star}",
                     transform=ax.transAxes, ha='right', va='top', fontsize=10,
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.92, edgecolor='#aaa'))
        ax.set_xlabel('Max probe score on baseline trace', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Probe predicts which baseline traces will fail\n'
                     '(probe score distribution split by final correctness)',
                     fontsize=11.5, fontweight='bold')
        ax.legend(fontsize=9.5)
        ax.grid(True, alpha=0.20)

    
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(BG)
    
    
    if 'baseline' in df['condition'].unique():
        base_wrong_ids = set(base_df[base_df['final_correct']==False]['base_id'])
        rescue_rates = []
        for cond in ['reprompt','replace_prior','branch_and_pick']:
            sub = df[(df['condition']==cond) & (df['base_id'].isin(base_wrong_ids))]
            if len(sub) == 0:
                rescue_rates.append({'condition': cond, 'rescue_rate': 0, 'n': 0})
                continue
            n_rescued = sub['final_correct'].fillna(False).astype(bool).sum()
            rescue_rates.append({'condition': cond,
                                  'rescue_rate': n_rescued / len(sub),
                                  'n_rescued': int(n_rescued),
                                  'n': len(sub)})
        x = np.arange(len(rescue_rates))
        ys = [r['rescue_rate'] for r in rescue_rates]
        ax.plot(x, ys, 'o-', color='#444', lw=2.5, ms=14, zorder=2)
        for xi, r in zip(x, rescue_rates):
            ax.scatter(xi, r['rescue_rate'], s=220, color=COLORS[r['condition']],
                       zorder=4, edgecolors='white', lw=2)
            ax.text(xi, r['rescue_rate']+0.025,
                     f'{r["rescue_rate"]:.1%}\n(n={r["n_rescued"]}/{r["n"]})',
                     ha='center', fontsize=10, fontweight='bold',
                     color=COLORS[r['condition']])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[r['condition']] for r in rescue_rates], fontsize=9.5)
        ax.set_ylabel('Rescue rate', fontsize=12)
        ax.set_title('Of baseline failures, how many does intervention rescue?\n'
                     '(among traces baseline got wrong)',
                     fontsize=11.5, fontweight='bold')
        ax.grid(True, alpha=0.20)
        ax.set_ylim(0, max(ys)*1.30 if max(ys)>0 else 0.5)

    
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(BG)
    depths = sorted(df['hop_depth'].unique())
    for cond in ['reprompt','replace_prior','branch_and_pick']:
        rates = []
        for d in depths:
            sub = df[(df['condition']==cond) & (df['hop_depth']==d)]
            rate = sub['fired_at_least_once'].fillna(False).astype(bool).mean()                   if len(sub) else 0
            rates.append(rate)
        ax.plot(depths, rates, 'o-', color=COLORS[cond], lw=2.5, ms=12,
                 label=LABELS[cond].replace('\n', ' '))
    ax.set_xticks(depths)
    ax.set_xticklabels([f'{d}-hop' for d in depths], fontsize=10)
    ax.set_ylabel('Fraction of traces where probe fired', fontsize=12)
    ax.set_title('Probe firing rate by hop depth\n(consistency check across difficulty levels)',
                 fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        'Phase 4: Probe-Based Real-Time Monitoring + Intervention Comparison',
        fontsize=14.5, fontweight='bold', y=0.985)
    plt.savefig(os.path.join(args.output_dir, 'fig_intervention_main.png'),
                 dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    
    
    
    print('Generating fig_intervention_breakdown.png...')
    err_types = sorted(df['error_type'].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)
    for cond in ['baseline','reprompt','replace_prior','branch_and_pick']:
        accs = []
        for et in err_types:
            sub = df[(df['condition']==cond) & (df['error_type']==et)]
            n = sub['final_correct'].notna().sum()
            c = sub['final_correct'].fillna(False).astype(bool).sum()
            accs.append(c/n if n > 0 else 0)
        ax.plot(range(len(err_types)), accs, 'o-', color=COLORS[cond],
                lw=2.5, ms=12, label=LABELS[cond].replace('\n', ' '))
        for xi, a in enumerate(accs):
            ax.text(xi, a+0.018, f'{a:.0%}', ha='center', fontsize=8,
                     color=COLORS[cond], fontweight='bold')
    ax.set_xticks(range(len(err_types)))
    ax.set_xticklabels([e.replace('_','\n') for e in err_types],
                        fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Final answer accuracy', fontsize=12)
    ax.set_title('Accuracy by error type × intervention\n'
                 'Which interventions help which error types?',
                 fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9.5, loc='best')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.set_facecolor(BG)
    for cond in ['baseline','reprompt','replace_prior','branch_and_pick']:
        accs = []
        depths = sorted(df['hop_depth'].unique())
        for d in depths:
            sub = df[(df['condition']==cond) & (df['hop_depth']==d)]
            n = sub['final_correct'].notna().sum()
            c = sub['final_correct'].fillna(False).astype(bool).sum()
            accs.append(c/n if n > 0 else 0)
        ax.plot(depths, accs, 'o-', color=COLORS[cond], lw=2.5, ms=14,
                label=LABELS[cond].replace('\n', ' '))
        for xi, a in zip(depths, accs):
            ax.text(xi, a+0.018, f'{a:.0%}', ha='center', fontsize=9,
                     color=COLORS[cond], fontweight='bold')
    ax.set_xticks(depths)
    ax.set_xticklabels([f'{d}-hop' for d in depths], fontsize=10)
    ax.set_ylabel('Final answer accuracy', fontsize=12)
    ax.set_title('Accuracy by hop depth × intervention\n'
                 'Does intervention scale with chain length?',
                 fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9.5, loc='best')
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, 1.05)

    fig.suptitle('Phase 4 Breakdown: Where do interventions help?',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig_intervention_breakdown.png'),
                 dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    
    
    
    print('Generating fig_intervention_roc.png...')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG)

    if len(base_df) > 0 and base_df['final_correct'].notna().any():
        ax = axes[0]
        ax.set_facecolor(BG)
        
        from sklearn.metrics import roc_curve, roc_auc_score
        valid = base_df.dropna(subset=['final_correct','max_probe_score'])
        y_wrong = (~valid['final_correct'].astype(bool)).astype(int).values
        scores  = valid['max_probe_score'].values
        if len(y_wrong) > 5 and y_wrong.sum() > 0 and y_wrong.sum() < len(y_wrong):
            fpr, tpr, _ = roc_curve(y_wrong, scores)
            auroc = roc_auc_score(y_wrong, scores)
            ax.plot([0,1],[0,1], color='#888', ls='--', lw=1.2, alpha=0.5)
            ax.plot(fpr, tpr, color='#1a7f3c', lw=3,
                    label=f'Probe predicts failure  AUROC={auroc:.3f}')
            ax.fill_between(fpr, tpr, alpha=0.10, color='#1a7f3c')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('Can probe predict baseline failures?\n'
                         '(used to decide WHEN to intervene)',
                         fontsize=11.5, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.20)

        
        ax2 = axes[1]
        ax2.set_facecolor(BG)
        thresholds = np.linspace(0.1, 0.9, 25)
        for cond in ['reprompt','replace_prior','branch_and_pick']:
            efficacies = []
            for thr in thresholds:
                
                cond_df = df[df['condition']==cond]
                
                fired = cond_df[cond_df['max_probe_score'] >= thr]
                if len(fired) < 3: efficacies.append(np.nan); continue
                
                base_match = base_df[base_df['base_id'].isin(fired['base_id'])]
                if len(base_match) < 3: efficacies.append(np.nan); continue
                acc_fired = fired['final_correct'].fillna(False).astype(bool).mean()
                acc_base  = base_match['final_correct'].fillna(False).astype(bool).mean()
                efficacies.append(acc_fired - acc_base)
            ax2.plot(thresholds, efficacies, 'o-', color=COLORS[cond],
                     lw=2, ms=6, label=LABELS[cond].replace('\n', ' '))
        ax2.axhline(0, color='#666', ls=':', lw=1, alpha=0.6)
        ax2.axvline(thr_data['threshold'], color='red', ls=':', lw=1.5,
                    alpha=0.6, label=f'Chosen threshold ({thr_data["threshold"]:.2f})')
        ax2.set_xlabel('Probe firing threshold'); ax2.set_ylabel('Δ accuracy vs baseline (rescued only)')
        ax2.set_title('Efficacy as a function of probe threshold\n'
                      'How does intervention benefit change with sensitivity?',
                      fontsize=11.5, fontweight='bold')
        ax2.legend(fontsize=9.5, loc='best')
        ax2.grid(True, alpha=0.20)

    fig.suptitle('Phase 4: Probe Operating Characteristics',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig_intervention_roc.png'),
                 dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f'\nSaved figures to {args.output_dir}/:')
    print('  fig_intervention_main.png       — headline 6-panel figure')
    print('  fig_intervention_breakdown.png  — per-error-type breakdown')
    print('  fig_intervention_roc.png        — threshold analysis')
    print('  intervention_summary.csv        — paper-ready table')


if __name__ == '__main__':
    main()