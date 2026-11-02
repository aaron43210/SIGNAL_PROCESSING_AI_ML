"""Figures and tables for the controlled (station-grouped) benchmark.

Reads results/dgx_rigorous/ (unpacked results_rigorous.zip from dgx_run/rigorous_benchmark.py) and
writes figs/r_*.png plus figure_sources/rigorous_*.csv. Nothing is trained here.

Run from the project root:  python figure_sources/make_rigorous_figures.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'dgx_rigorous'
OUT = ROOT / 'figs'
SRC = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC))
import make_paper_figures as base  # noqa: E402  (shared style and palette)

INK, INK2, MUTED, AXIS, GRID = base.INK, base.INK2, base.MUTED, base.AXIS, base.GRID
FAM = dict(base.FAM)
FAM['Neural'] = FAM['ANN']                      # dense / convolutional networks share the ANN colour
ABL = '#86b6ef'                                 # lighter step of the GAT hue for ablations

PRIMARY = ['Logistic reg.', 'XGBoost', '1D-CNN', 'MLP (28 features)', 'SVM (RBF)', 'GAT', 'Random forest',
           'ANN', 'KAN', 'XGBoost (pooled peaks)', 'Grad. boosting', 'GCN', 'k-NN']
FAMILY = {'GAT': 'GAT', 'GCN': 'GCN', 'KAN': 'KAN', 'ANN': 'Neural', 'MLP (28 features)': 'Neural',
          '1D-CNN': 'Neural'}
SHORT = {'MLP (28 features)': 'MLP (28 feat.)', 'XGBoost (pooled peaks)': 'XGBoost (peak feat.)',
         'Logistic reg.': 'Logistic reg.'}


def fam(model):
    if model.startswith('GAT'):
        return 'GAT'
    return FAMILY.get(model, 'ML')


def colour(model):
    if model.startswith('GAT') and model != 'GAT':
        return ABL
    return FAM[fam(model)]


def slug(name):
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_')


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print('wrote figs/' + name)


# ----------------------------------------------------------------------------- data
def load_runs():
    rows = []
    for p in (RES / 'runs').glob('*/*.json'):
        r = json.load(open(p))
        rows.append(dict(model=r['model'], fold=r['fold'], seed=r['seed'], val_ba=r.get('val_ba'),
                         best_epoch=r.get('best_epoch'), **{f't_{k}': v for k, v in r['test'].items()}))
    return pd.DataFrame(rows)


RUNS = load_runs()
SUMMARY = pd.read_csv(RES / 'tables' / 'summary_by_model.csv').set_index('model')
SPLITS = pd.read_csv(RES / 'tables' / 'splits.csv')
FOLD_ST = {r.fold: r.stations for r in SPLITS[(SPLITS.cohort == 'peak') & (SPLITS.part == 'test')].itertuples()}


def fold_ba(model):
    """Balanced accuracy per outer fold (held-out stations), averaged over seeds."""
    return RUNS[RUNS.model == model].groupby('fold').t_bal_acc.mean()


# ----------------------------------------------------------------------------- figures
def fig_main():
    """Pooled balanced accuracy per model with the five held-out-station folds as dots."""
    models = sorted(PRIMARY, key=lambda m: SUMMARY.loc[m, 'bal_acc'])
    fig, ax = plt.subplots(figsize=(3.45, 4.3))
    y = np.arange(len(models))
    for yy, m in zip(y, models):
        fb = fold_ba(m)
        ax.plot([fb.min(), fb.max()], [yy, yy], color=AXIS, lw=2.2, solid_capstyle='round', zorder=1)
        ax.scatter(fb.values, np.full(len(fb), yy), s=14, color=colour(m), alpha=0.55, edgecolor='none', zorder=2)
        v = SUMMARY.loc[m, 'bal_acc']
        ax.scatter([v], [yy], s=58, color=colour(m), edgecolor='white', linewidth=1.3, zorder=3)
        ax.text(0.727, yy, f'{v:.3f}', va='center', fontsize=6.3, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(0.501, len(models) - 0.35, 'chance', fontsize=6.6, color=INK2)
    ax.set_yticks(y, [SHORT.get(m, m) for m in models], fontsize=6.8)
    ax.set_xlim(0.40, 0.76)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel('Balanced accuracy on held-out stations', fontsize=7)
    ax.grid(axis='y', visible=False)
    handles = [plt.Line2D([], [], marker='o', ls='', color=FAM[f], ms=5) for f in ('GAT', 'Neural', 'GCN', 'KAN', 'ML')]
    handles += [plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, alpha=0.6)]
    ax.legend(handles, ['GAT', 'ANN / MLP / 1D-CNN', 'GCN', 'KAN', 'Classical ML', 'single fold'],
              frameon=False, loc='upper center', bbox_to_anchor=(0.38, -0.12), ncol=3, fontsize=6.0,
              handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout()
    save(fig, 'r_main_comparison.png')


def fig_fold_heatmap():
    """Balanced accuracy of every model on every held-out station group."""
    models = sorted(PRIMARY, key=lambda m: -SUMMARY.loc[m, 'bal_acc'])
    M = np.array([[fold_ba(m).get(f, np.nan) for f in range(5)] for m in models])
    cmap = LinearSegmentedColormap.from_list('div', ['#d03b3b', '#f3c7c0', '#f0efec', '#b7d3f6', '#2a78d6'])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(M, cmap=cmap, vmin=0.35, vmax=0.65, aspect='auto')
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                    color='white' if abs(M[i, j] - 0.5) > 0.12 else INK)
    ax.set_xticks(range(5), [f'fold {f}\nst. {FOLD_ST[f]}' for f in range(5)], fontsize=6.4)
    ax.set_yticks(range(len(models)), [SHORT.get(m, m) for m in models], fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('balanced accuracy (0.5 = chance)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, 'r_fold_heatmap.png')


def fig_ablation():
    """GAT graph ablations and the graph-free baseline on the same peak features."""
    rows = [('GAT (selected)', 'GAT'), ('random edges', 'GAT edges=random'), ('temporal edges only', 'GAT edges=temporal'),
            ('k-NN edges only', 'GAT edges=knn'), ('no edges', 'GAT edges=none'),
            ('other node scaling', 'GAT nodes=other scaling'), ('no graph: XGBoost on\npooled peak features', 'XGBoost (pooled peaks)')]
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, m) in zip(y, rows):
        fb = fold_ba(m)
        v = SUMMARY.loc[m, 'bal_acc']
        c = FAM['GAT'] if m == 'GAT' else (FAM['ML'] if m.startswith('XGB') else ABL)
        ax.barh(yy, v - 0.5, left=0.5, height=0.6, color=c)
        ax.scatter(fb.values, np.full(len(fb), yy), s=10, color=INK2, zorder=3, alpha=0.7, edgecolor='none')
        ax.text(max(v, fb.max()) + 0.006, yy, f'{v:.3f}', va='center', fontsize=6.6, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.8)
    ax.set_xlim(0.42, 0.7)
    ax.set_xlabel('Balanced accuracy (dots: folds)')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'r_graph_ablation.png')


def fig_val_vs_test():
    """Validation-station BA of each selected model against its test-station BA."""
    sub = RUNS[(RUNS.seed == 0) & RUNS.model.isin(PRIMARY)].dropna(subset=['val_ba'])
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.plot([0.3, 0.75], [0.3, 0.75], color=AXIS, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=':')
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=':')
    for f in ('ML', 'Neural', 'GAT', 'GCN', 'KAN'):
        s = sub[sub.model.map(fam) == f]
        ax.scatter(s.val_ba, s.t_bal_acc, s=16, color=FAM[f], edgecolor='white', linewidth=0.5, label=
                   {'ML': 'Classical ML', 'Neural': 'ANN / MLP / CNN'}.get(f, f))
    r = np.corrcoef(sub.val_ba, sub.t_bal_acc)[0, 1]
    ax.text(0.31, 0.72, f'r = {r:.2f}  (n = {len(sub)})', fontsize=6.8, color=INK2)
    ax.set_xlim(0.3, 0.75)
    ax.set_ylim(0.3, 0.75)
    ax.set_xlabel('Validation-station balanced accuracy')
    ax.set_ylabel('Test-station balanced accuracy')
    ax.legend(frameon=False, fontsize=6.2, loc='lower left', scatterpoints=1, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, 'r_val_vs_test.png')
    return r, len(sub)


def fig_detector():
    d = pd.read_csv(RES / 'tables' / 'detector_sweep.csv')
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.4))
    for dist, ls in ((10, '-'), (40, (0, (3, 1.5)))):
        s = d[d.distance_samples == dist].sort_values('prominence')
        a.plot(s.prominence, s.kept_elephant, ls=ls, marker='o', ms=4, color=FAM['GAT'], label=f'Elephantidae, {dist * 5} ms')
        a.plot(s.prominence, s.kept_giraffe, ls=ls, marker='s', ms=4, color=FAM['ANN'], label=f'Giraffidae, {dist * 5} ms')
        ok = s.dropna(subset=['xgb_pooled_bal_acc'])
        b.plot(ok.prominence, ok.xgb_pooled_bal_acc, ls=ls, marker='o', ms=4, color=FAM['ML'],
               label=f'min. separation {dist * 5} ms')
    a.set_xscale('log')
    a.set_xticks([0.05, 0.1, 0.25, 0.5], ['0.05', '0.1', '0.25', '0.5'])
    a.minorticks_off()
    a.set_xlabel('Peak prominence (window SD)')
    a.set_ylabel('Fraction of windows kept')
    a.legend(frameon=False, fontsize=6.2)
    b.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.set_xscale('log')
    b.set_xticks([0.05, 0.1, 0.25], ['0.05', '0.1', '0.25'])
    b.minorticks_off()
    b.set_ylim(0.45, 0.6)
    b.set_xlabel('Peak prominence (window SD)')
    b.set_ylabel('Balanced accuracy')
    b.legend(frameon=False, fontsize=6.2)
    a.set_title('(a) Windows with at least two peaks', fontsize=7.5)
    b.set_title('(b) XGBoost on pooled peak features', fontsize=7.5)
    fig.tight_layout(h_pad=1.0)
    save(fig, 'r_detector_sweep.png')


def fig_attention():
    a = pd.read_csv(RES / 'tables' / 'attention_faithfulness.csv')
    g = a.groupby(['fraction', 'method']).agg(flip=('flip_rate', 'mean'), flip_sd=('flip_rate', 'std'),
                                              ba=('bal_acc', 'mean'), intact=('bal_acc_intact', 'mean')).reset_index()
    fig, (p, q) = plt.subplots(2, 1, figsize=(3.45, 4.2), sharex=True)
    for meth, c, mk in (('top attention', FAM['GAT'], 'o'), ('random', MUTED, 's')):
        s = g[g.method == meth]
        p.errorbar(s.fraction, s.flip, yerr=s.flip_sd, color=c, marker=mk, ms=4, capsize=2, lw=1.4, label=meth)
        q.plot(s.fraction, s.ba, color=c, marker=mk, ms=4, label=meth)
    q.axhline(g.intact.mean(), color=INK2, lw=0.8, ls=(0, (3, 2)))
    q.text(0.11, g.intact.mean() + 0.002, 'intact graphs', fontsize=6.4, color=INK2)
    for ax in (p, q):
        ax.set_xticks([0.1, 0.25, 0.5], ['10%', '25%', '50%'])
        ax.set_xlabel('Fraction of nodes deleted')
        ax.legend(frameon=False, fontsize=6.4)
    p.set_ylabel('Predictions flipped')
    q.set_ylabel('Balanced accuracy')
    q.set_ylim(0.49, 0.56)
    p.set_xlabel('')
    p.set_title('(a) Prediction changes', fontsize=7.5)
    q.set_title('(b) Balanced accuracy after deletion', fontsize=7.5)
    fig.tight_layout(h_pad=1.0)
    save(fig, 'r_attention.png')


def fig_training():
    """Validation balanced accuracy and training loss per epoch (fold 0, seed 0)."""
    models = ['GAT', 'ANN', 'MLP (28 features)', '1D-CNN', 'KAN', 'GCN']
    fig, axs = plt.subplots(2, 6, figsize=(7.4, 3.0), sharex='col')
    for c, m in enumerate(models):
        r = json.load(open(RES / 'runs' / slug(m) / 'f0_s0.json'))
        h = pd.DataFrame(r['history'])
        col = colour(m)
        axs[0, c].plot(h.epoch, h.train_loss, color=col, ls=(0, (3, 1.5)), lw=1.2, label='train')
        axs[0, c].plot(h.epoch, h.val_loss, color=col, lw=1.2, label='validation')
        axs[1, c].plot(h.epoch, h.val_ba, color=col, lw=1.2)
        axs[1, c].axhline(0.5, color=MUTED, lw=0.6, ls=':')
        axs[1, c].axvline(r['best_epoch'], color=INK2, lw=0.7, ls=(0, (2, 2)))
        axs[0, c].set_title(SHORT.get(m, m), fontsize=7.5)
        axs[1, c].set_xlabel('epoch', fontsize=7)
        for ax in axs[:, c]:
            ax.tick_params(labelsize=6)
    axs[0, 0].set_ylabel('loss', fontsize=7)
    axs[1, 0].set_ylabel('val. BA', fontsize=7)
    fig.legend([plt.Line2D([], [], color=INK2, ls=(0, (3, 1.5))), plt.Line2D([], [], color=INK2),
                plt.Line2D([], [], color=INK2, lw=0.7, ls=(0, (2, 2)))],
               ['training loss', 'validation loss / BA', 'epoch kept (early stopping)'],
               frameon=False, fontsize=6.4, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0, 0.06, 1, 1))
    save(fig, 'r_training_curves.png')


def fig_control():
    """Same code, random window-level split versus station-held-out split (needs the control runs)."""
    pairs = [('XGBoost', 'XGBoost [all windows]', 'XGBoost [random split]'),
             ('Logistic reg.', 'Logistic reg.', 'Logistic reg. [random split]'),
             ('GAT', 'GAT', 'GAT [random split]')]
    have = [p for p in pairs if p[2] in SUMMARY.index]
    if not have:
        print('control runs not present yet: skipping r_control.png')
        return None
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    x = np.arange(len(have))
    w = 0.36
    for k, (lab, station, rnd) in enumerate(have):
        for off, m, c in ((-w / 2, rnd, MUTED), (w / 2, station, colour(station))):
            v = SUMMARY.loc[m, 'bal_acc']
            ax.bar(k + off, v - 0.5, bottom=0.5, width=w, color=c)
            ax.text(k + off, v + 0.008, f'{v:.2f}', ha='center', fontsize=6.6, color=INK2)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(x, [h[0] for h in have])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel('Balanced accuracy')
    ax.grid(axis='x', visible=False)
    ax.legend([Rectangle((0, 0), 1, 1, color=MUTED), Rectangle((0, 0), 1, 1, color=INK2)],
              ['random window split', 'station-held-out split'], frameon=False, fontsize=6.4, loc='upper right')
    fig.tight_layout()
    save(fig, 'r_control_split.png')
    return {h[0]: (SUMMARY.loc[h[2], 'bal_acc'], SUMMARY.loc[h[1], 'bal_acc']) for h in have}


def fig_edge():
    e = pd.read_csv(RES / 'tables' / 'edge_profile.csv').set_index('model')
    rows = []
    for m in ('GAT', 'ANN', 'MLP (28 features)', '1D-CNN', 'KAN', 'GCN'):
        r = e.loc[m]
        rows.append((m, r.fp32_bytes / 1024, r.int8_bytes / 1024, r.pre_ms + r.inf_ms))
    x = e.loc['XGBoost']
    rows.append(('XGBoost', x.xgb_c_array_bytes / 1024, x.xgb_c_array_bytes / 1024, x.pre_ms + x.inf_ms))
    lr = e.loc['Logistic reg.']
    rows.append(('Logistic reg.', lr.pickle_bytes / 1024, lr.pickle_bytes / 1024, np.nan))
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.9))
    y = np.arange(len(rows))[::-1]
    for yy, (m, fp, i8, ms) in zip(y, rows):
        c = colour(m)
        a.barh(yy + 0.17, fp, height=0.32, color=c)
        a.barh(yy - 0.17, i8, height=0.32, color=c, hatch='////', edgecolor='white', lw=0)
        a.text(max(fp, i8) * 1.25, yy, f'{i8:,.0f} KB' if i8 >= 1 else f'{i8:.1f} KB', va='center', fontsize=6.3,
               color=INK2, bbox=dict(facecolor='white', edgecolor='none', pad=0.6), zorder=4)
        if not np.isnan(ms):
            b.barh(yy, ms, height=0.5, color=c)
            b.text(ms + 0.08, yy, f'{ms:.2f} ms', va='center', fontsize=6.3, color=INK2)
        else:
            b.text(0.05, yy, '28 multiply-adds (not timed)', va='center', fontsize=6.0, color=MUTED)
    a.axvline(512, color=INK, lw=0.9, ls=(0, (3, 2)))
    a.text(560, y.max() + 0.5, '512 KB flash budget', fontsize=6.2, color=INK)
    a.set_xscale('log')
    a.set_xlim(0.5, 2e4)
    a.set_yticks(y, [SHORT.get(r[0], r[0]) for r in rows], fontsize=7)
    a.set_xlabel('Stored size (KB, log scale)')
    a.set_title('(a) Size: FP32 (solid), INT8 or C arrays (hatched)', fontsize=7.5)
    b.set_yticks(y, [SHORT.get(r[0], r[0]) for r in rows], fontsize=7)
    b.set_xlim(0, 3.6)
    b.set_xlabel('Host CPU time per window (ms, 1 thread)')
    b.set_title('(b) Preprocessing + inference', fontsize=7.5)
    for ax in (a, b):
        ax.grid(axis='y', visible=False)
    fig.tight_layout(h_pad=1.0)
    save(fig, 'r_edge_profile.png')


def fig_architectures():
    """The seven pipelines of the controlled benchmark; braces give the tuned alternatives."""
    from matplotlib.patches import FancyBboxPatch
    rows = [
        ('GAT', FAM['GAT'], ['raw window\n3 x 2048', 'peak graph: 8-D\nnodes, chain + 5-NN', '3 x GATConv\nH {32,64}, K {2,4}',
                             'mean || max\npooling', 'MLP\n2H-H-2']),
        ('ANN', FAM['ANN'], ['X or XYZ\n2048 or 6144', 'standardise\n(train stats)', 'Linear H {256,512}\nBN, ReLU, drop',
                             'Linear H/2\nBN, ReLU, drop', 'Linear\nH/2-2']),
        ('MLP', FAM['ANN'], ['raw window\n3 x 2048', '28 statistics\naxis / mag. / peaks', 'Linear H {64,128}\nBN, ReLU, drop',
                             'Linear H/2\nBN, ReLU, drop', 'Linear\nH/2-2']),
        ('1D-CNN', FAM['ANN'], ['raw window\n3 x 2048', 'standardise\nper component', '3 conv blocks\nc, 2c, 4c; c {16,32}',
                                'mean || max\npooling', 'Linear\n8c-2']),
        ('KAN', FAM['KAN'], ['raw window\n3 x 2048', 'peak descriptors\npooled: 24-D', 'KANLinear\n24-{32,64}',
                             'KANLinear\n-{16,32}', 'KANLinear\n-2']),
        ('GCN', FAM['GCN'], ['raw window\n3 x 2048', '64-segment chain\nraw or 21 stats', 'GCN layer 64\nReLU',
                             'GCN layer 32\nReLU, mean pool', 'Linear\n32-2']),
        ('ML', FAM['ML'], ['raw window\n3 x 2048', '28 statistics\n(or 24 peak feat.)', 'standardise\n(LR, SVM, k-NN)',
                           'LR, SVM, k-NN,\nRF, GB, XGBoost', 'class\nprobability']),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bw, gap, x_start = 1.58, 0.34, 1.12
    ax.set_xlim(0, x_start + 5 * bw + 4 * gap + 0.05)
    ax.set_ylim(-0.5, len(rows) - 0.15)
    ax.axis('off')
    for r, (title, col, steps) in enumerate(rows):
        yc = len(rows) - 1 - r + 0.3
        ax.text(0.0, yc, title, fontsize=7.6, fontweight='bold', color=INK, va='center')
        for s_i, label in enumerate(steps):
            x0 = x_start + s_i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x0, yc - 0.36), bw, 0.72, boxstyle='round,pad=0.01,rounding_size=0.07',
                                        fc=matplotlib.colors.to_rgba(col, 0.20 if s_i >= 2 else 0.0), ec=col, lw=1.1))
            ax.text(x0 + bw / 2, yc, label, ha='center', va='center', fontsize=6.0, color=INK, linespacing=1.15)
            if s_i:
                ax.annotate('', (x0 - 0.03, yc), (x0 - gap + 0.03, yc),
                            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.9))
    for s_i, lab in [(0, 'input'), (1, 'representation'), (3, 'classifier (shaded); {a,b}: tuned on validation stations')]:
        ax.text(x_start + s_i * (bw + gap) + bw / 2, -0.42, lab, ha='center', fontsize=6.4, color=MUTED)
    save(fig, 'r_architectures.png')


def fig_original_vs_controlled():
    """Balanced accuracy as originally reported (random split, test-set selection) against the
    station-held-out benchmark, for the models that exist in both."""
    import paper_data as D
    pairs = [('XGBoost', 'XGBoost'), ('Grad. boosting', 'Grad. boosting'), ('SVM (RBF)', 'SVM (RBF)'),
             ('7-NN', 'k-NN'), ('Random forest', 'Random forest'), ('Logistic reg.', 'Logistic reg.'),
             ('GAT', 'GAT'), ('ANN', 'ANN')]
    rows = [(old, new, D.binary_metrics(D.BINARY[old][1])['bal_acc'], SUMMARY.loc[new, 'bal_acc']) for old, new in pairs]
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    y = np.arange(len(rows))[::-1]
    for yy, (old, new, a, b) in zip(y, rows):
        c = colour(new)
        ax.plot([a, b], [yy, yy], color=AXIS, lw=2, zorder=1)
        ax.scatter([a], [yy], s=30, facecolor='white', edgecolor=c, linewidth=1.4, zorder=2)
        ax.scatter([b], [yy], s=30, color=c, zorder=3)
        ax.text(max(a, b) + 0.012, yy, f'{a:.2f} \u2192 {b:.2f}', va='center', fontsize=6.2, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y, [r[1] for r in rows], fontsize=7)
    ax.set_xlim(0.44, 1.02)
    ax.set_xlabel('Balanced accuracy')
    ax.grid(axis='y', visible=False)
    ax.legend([plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=5),
               plt.Line2D([], [], marker='o', ls='', color=INK2, ms=5)],
              ['original (random split,\ntest-set selection)', 'station-held-out'], frameon=False,
              fontsize=6.0, loc='lower right')
    fig.tight_layout()
    save(fig, 'r_original_vs_controlled.png')
    return rows


def write_tables():
    cols = ['bal_acc', 'bal_acc_sd_seeds', 'bal_acc_fold_mean', 'bal_acc_fold_sd', 'mcc', 'macro_f1', 'accuracy',
            'prec_g', 'rec_g', 'f1_g', 'auroc', 'auprc', 'seeds', 'windows']
    SUMMARY.loc[[m for m in SUMMARY.index], cols].round(4).to_csv(SRC / 'rigorous_summary.csv')
    fb = pd.DataFrame({m: fold_ba(m) for m in SUMMARY.index}).T.round(4)
    fb.columns = [f'fold{f} ({FOLD_ST[f]})' for f in fb.columns]
    fb.to_csv(SRC / 'rigorous_fold_ba.csv')
    print('wrote figure_sources/rigorous_summary.csv and rigorous_fold_ba.csv')


if __name__ == '__main__':
    fig_architectures()
    fig_original_vs_controlled()
    fig_main()
    fig_fold_heatmap()
    fig_ablation()
    r, n = fig_val_vs_test()
    fig_detector()
    fig_attention()
    fig_training()
    ctrl = fig_control()
    fig_edge()
    write_tables()
    print(f'validation/test correlation r = {r:.3f} over {n} selected models')
    if ctrl:
        print('control (random split vs station split):', {k: tuple(round(v, 3) for v in t) for k, t in ctrl.items()})
