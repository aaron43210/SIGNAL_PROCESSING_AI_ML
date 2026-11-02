"""Draw every new paper figure from logged notebook outputs (figure_sources/paper_data.py).

Run from the project root:  python figure_sources/make_paper_figures.py
Images are written to figs/ (the upload folder); derived CSVs stay in figure_sources/.
No model is trained or evaluated; edge figures use arithmetic on the logged
architectures and are labelled as estimates where they are not file sizes.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'figs')

# One fixed colour per model family (validated categorical slots 1-5, light mode)
FAM = {'GAT': '#2a78d6', 'ANN': '#eb6834', 'ML': '#1baf7a', 'GCN': '#eda100', 'KAN': '#e87ba4'}
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
       '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
    'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'lines.linewidth': 1.6,
})
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQ)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('wrote figs/' + name)


def bar_labels(ax, bars, fmt='{:.3f}', inside=False):
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.01, b.get_y() + b.get_height() / 2, fmt.format(w),
                va='center', ha='left', fontsize=6.8, color=INK2)


def gat_params(hid, heads, f_in=8, n_cls=2):
    """Trainable parameters of PeakGATRaw (GATConv concat=False, no lin bias)."""
    def conv(fi):
        return fi * heads * hid + 2 * heads * hid + hid
    head = (2 * hid) * hid + hid + 2 * hid + hid * n_cls + n_cls
    return conv(f_in) + 2 * conv(hid) + head


assert gat_params(64, 4) == 45058


# ==========================================================================
# 1. GAT grid search: heatmap + main effects
# ==========================================================================
def fig_gat_grid():
    rows = [(32, 2), (32, 4), (64, 2), (64, 4)]
    cols = [(0.3, 1e-3), (0.3, 5e-4), (0.4, 1e-3), (0.4, 5e-4)]
    M = np.zeros((4, 4))
    for h, k, d, lr, f in D.GAT_GRID:
        M[rows.index((h, k)), cols.index((d, lr))] = f
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.75), gridspec_kw={'width_ratios': [1.25, 1, 0.9]})
    im = a.imshow(M, cmap=SEQ_CMAP, vmin=0.535, vmax=0.60, aspect='auto')
    a.grid(False)
    a.set_xticks(range(4), [f'{d}\n{lr:g}' for d, lr in cols])
    a.set_yticks(range(4), [f'{h}/{k}' for h, k in rows])
    a.set_ylabel('hidden width / heads')
    a.set_xlabel('dropout / learning rate')
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            a.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=6.8,
                   color='white' if v > 0.575 else INK)
    i, j = rows.index(D.GAT_SELECTED[:2]), cols.index(D.GAT_SELECTED[2:])
    a.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.8))
    a.set_title('(a) Macro-F1, 16 configurations')
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    factors = [('width', 0, [32, 64]), ('heads', 1, [2, 4]),
               ('dropout', 2, [0.3, 0.4]), ('lr', 3, [1e-3, 5e-4])]
    y = 0
    ticks, labels = [], []
    for name, idx, levels in factors:
        for lv in levels:
            vals = [g[4] for g in D.GAT_GRID if g[idx] == lv]
            b.plot([min(vals), max(vals)], [y, y], color=AXIS, lw=2.2, solid_capstyle='round')
            b.plot(np.mean(vals), y, 'o', ms=6.5, color=FAM['GAT'], mec='white', mew=1.2)
            b.text(max(vals) + 0.0015, y, f'{np.mean(vals):.3f}', va='center', fontsize=6.6, color=INK2)
            ticks.append(y)
            labels.append(f'{name} {lv:g}')
            y += 1
        y += 0.6
    b.set_yticks(ticks, labels)
    b.invert_yaxis()
    b.set_xlim(0.535, 0.612)
    b.set_xlabel('Macro-F1 (mean, range)')
    b.set_title('(b) Main effects')
    b.grid(axis='y', visible=False)
    for h, k, d, lr, f in D.GAT_GRID:
        sel = (h, k, d, lr) == D.GAT_SELECTED
        c.plot(gat_params(h, k) / 1000, f, 'o', ms=7 if sel else 5,
               color=FAM['GAT'] if lr == 1e-3 else '#86b6ef', mec=INK if sel else 'white', mew=1.2)
    c.set_xscale('log')
    c.set_xticks([7, 12, 27, 45], ['7', '12', '27', '45'])
    c.minorticks_off()
    c.set_xlabel('Parameters (thousands)')
    c.set_title('(c) Accuracy vs size')
    handles = [plt.Line2D([], [], marker='o', ls='', color=FAM['GAT'], ms=5),
               plt.Line2D([], [], marker='o', ls='', color='#86b6ef', ms=5)]
    c.legend(handles, ['lr 0.001', 'lr 0.0005'], frameon=False, loc='lower right')
    fig.tight_layout()
    save(fig, 'gs_gat_grid_search.png')


# ==========================================================================
# 2. Model-selection evidence for every family (equal treatment)
# ==========================================================================
def fig_selection():
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.35), gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})
    # (a) classical estimators: logged test macro-F1 (selection partition)
    ml = [(k, D.binary_metrics(v[1])['macro_f1']) for k, v in D.BINARY.items() if v[0] == 'ML']
    ml.sort(key=lambda t: t[1])
    bars = axs[0].barh([k for k, _ in ml], [v for _, v in ml], color=FAM['ML'], height=0.62)
    bar_labels(axs[0], bars)
    axs[0].set_xlim(0, 0.95)
    axs[0].set_title('(a) ML: estimator search\nmacro-F1 (test)')
    # (b) ANN preprocessing ablation: balanced accuracy (robust to collapse)
    ann = [('no std.', D.binary_metrics(D.BINARY['ANN (no std.)'][1])['bal_acc']),
           ('std.', D.binary_metrics(D.BINARY['ANN'][1])['bal_acc'])]
    bars = axs[1].barh([k for k, _ in ann], [v for _, v in ann], color=FAM['ANN'], height=0.5)
    bar_labels(axs[1], bars)
    axs[1].axvline(0.5, color=INK2, lw=0.9, ls=(0, (3, 2)))
    axs[1].text(0.51, -0.42, 'chance', fontsize=6.4, color=INK2)
    axs[1].set_xlim(0, 0.75)
    axs[1].set_title('(b) ANN: input scaling\nbalanced acc. (test)')
    # (c) GCN variants: best validation accuracy
    gcn = [('features', max(D.GCN_FEAT['va_acc'])), ('raw', max(D.GCN_RAW['va_acc']))]
    bars = axs[2].barh([k for k, _ in gcn], [v for _, v in gcn], color=FAM['GCN'], height=0.5)
    bar_labels(axs[2], bars)
    axs[2].axvline(0.2, color=INK2, lw=0.9, ls=(0, (3, 2)))
    axs[2].text(0.21, -0.42, 'chance (5 classes)', fontsize=6.4, color=INK2)
    axs[2].set_xlim(0, 0.65)
    axs[2].set_title('(c) GCN: node encoding\nbest val. accuracy')
    # (d) KAN imbalance treatment: test macro-F1
    kan = [('weighted +\nundersampled', 0.142), ('unweighted', D.KAN_U_SUMMARY['macro_f1'])]
    bars = axs[3].barh([k for k, _ in kan], [v for _, v in kan], color=FAM['KAN'], height=0.5)
    bar_labels(axs[3], bars)
    axs[3].set_xlim(0, 0.25)
    axs[3].set_title('(d) KAN: imbalance\nmacro-F1 (test)')
    for ax in axs:
        ax.grid(axis='y', visible=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'gs_model_selection.png')


# ==========================================================================
# 3. Training dynamics (uniform format)
# ==========================================================================
def fig_training():
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex='col')
    panels = [
        ('ANN (std.)', 'ANN', range(1, 51), D.ANN_LOSS, None, [a / 100 for a in D.ANN_ACC], None),
        ('ANN (no std.)', 'ANN', range(1, 21), D.ANN_NOSTD_LOSS, None,
         [a / 100 for a in D.ANN_NOSTD_ACC], None),
        ('GCN (raw segments)', 'GCN', range(1, 31), D.GCN_RAW['tr_loss'], D.GCN_RAW['va_loss'],
         D.GCN_RAW['tr_acc'], D.GCN_RAW['va_acc']),
        ('KAN (weighted)', 'KAN', D.KAN_W['epoch'], D.KAN_W['tr_loss'], D.KAN_W['va_loss'],
         D.KAN_W['tr_acc'], D.KAN_W['va_acc']),
    ]
    for c, (title, fam, ep, tl, vl, ta, va) in enumerate(panels):
        col = FAM[fam]
        ep = list(ep)
        mk = 'o' if len(ep) < 15 else None
        axs[0, c].plot(ep, tl, color=col, ls=(0, (3, 1.5)), marker=mk, ms=3, label='train')
        if vl:
            axs[0, c].plot(ep, vl, color=col, marker=mk, ms=3, label='validation')
        axs[1, c].plot(ep, ta, color=col, ls=(0, (3, 1.5)), marker=mk, ms=3, label='train')
        if va:
            axs[1, c].plot(ep, va, color=col, marker=mk, ms=3, label='validation')
        axs[0, c].set_title(title)
        axs[1, c].set_xlabel('epoch')
        axs[0, c].legend(frameon=False, loc='upper right')
    axs[0, 0].set_ylabel('loss')
    axs[1, 0].set_ylabel('accuracy')
    axs[0, 1].axhline(np.log(2), color=INK2, lw=0.8, ls=':')
    axs[0, 1].text(20, np.log(2) + 0.0012, 'ln 2', ha='right', va='bottom', fontsize=6.8, color=INK2)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, 'training_dynamics.png')


# ==========================================================================
# 4. Binary comparison: small multiples of four metrics
# ==========================================================================
ORDER = ['GAT', 'ANN', 'ANN (no std.)', 'Logistic reg.', 'SVM (RBF)', '7-NN',
         'Random forest', 'Grad. boosting', 'XGBoost']


def fig_binary_metrics():
    m = {k: D.binary_metrics(D.BINARY[k][1]) for k in ORDER}
    metrics = [('acc', 'Accuracy'), ('bal_acc', 'Balanced accuracy'),
               ('macro_f1', 'Macro-F1'), ('f1_g', 'Giraffe F1 (minority)')]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (key, title) in zip(axs, metrics):
        cols = [FAM[D.BINARY[k][0]] for k in ORDER]
        bars = ax.barh(y, [m[k][key] for k in ORDER], color=cols, height=0.66)
        if key in ('acc', 'bal_acc'):
            for yy, k in zip(y, ORDER):
                cm = D.BINARY[k][1]
                if key == 'acc':
                    lo, hi = D.wilson(cm[0][0] + cm[1][1], m[k]['n'])
                else:
                    lo, hi = D.balanced_accuracy_ci(cm)
                ax.plot([lo, hi], [yy, yy], color=INK, lw=1.0, solid_capstyle='butt')
        bar_labels(ax, bars, fmt='{:.2f}')
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title)
        ax.grid(axis='y', visible=False)
        if key == 'bal_acc':
            ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    axs[0].set_yticks(y, ORDER)
    handles = [Rectangle((0, 0), 1, 1, color=FAM[f]) for f in ('GAT', 'ANN', 'ML')]
    fig.legend(handles, ['GAT (10,981 test graphs)', 'ANN (13,363 test records)',
                         'Classical ML (13,363 test records)'],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.4)
    save(fig, 'binary_metric_comparison.png')


def fig_accuracy_paradox():
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.plot([0.4, 1], [0.4, 1], color=AXIS, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=':')
    ax.text(0.995, 0.507, 'chance (balanced acc. 0.5)', fontsize=6.5, color=INK2, ha='right')
    offs = {'GAT': (6, -9), 'ANN': (6, -11), 'ANN (no std.)': (-6, -11), 'Logistic reg.': (-7, -4),
            'SVM (RBF)': (-7, 4), '7-NN': (-2, 7), 'Random forest': (-6, -11),
            'Grad. boosting': (6, 2), 'XGBoost': (-7, 3)}
    for k in ORDER:
        mm = D.binary_metrics(D.BINARY[k][1])
        ax.plot(mm['acc'], mm['bal_acc'], 'o', ms=7, color=FAM[D.BINARY[k][0]], mec='white', mew=1.4)
        dx, dy = offs[k]
        ax.annotate(k, (mm['acc'], mm['bal_acc']), xytext=(dx, dy), textcoords='offset points',
                    fontsize=6.6, color=INK2, ha='left' if dx > 0 else 'right')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Balanced accuracy')
    handles = [plt.Line2D([], [], marker='o', ls='', color=FAM[f], ms=6) for f in ('GAT', 'ANN', 'ML')]
    ax.legend(handles, ['GAT', 'ANN', 'Classical ML'], frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, 'accuracy_vs_balanced_accuracy.png')


def fig_binary_cms():
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.3))
    for ax, k in zip(axs.flat, ORDER):
        cm = np.array(D.BINARY[k][1], dtype=float)
        rn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rn, cmap=SEQ_CMAP, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{int(cm[i, j]):,}\n({rn[i, j]:.1%})', ha='center', va='center',
                        fontsize=7.4, color='white' if rn[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ['Elephant', 'Giraffe'])
        ax.set_yticks([0, 1], ['Elephant', 'Giraffe'])
        ax.set_title(k, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('predicted')
    for ax in axs[:, 0]:
        ax.set_ylabel('true')
    fig.tight_layout(h_pad=1.0)
    save(fig, 'binary_confusion_matrices.png')


# ==========================================================================
# 5. Multi-class models (GCN and KAN)
# ==========================================================================
def fig_multiclass():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [5, 8]})
    bars = a.bar(D.GCN_CLASSES, D.GCN_F1, color=FAM['GCN'], width=0.62)
    for r in bars:
        a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.024, f'{r.get_height():.2f}',
               ha='center', fontsize=6.8, color=INK2, bbox=dict(fc='white', ec='none', pad=0.5))
    a.axhline(0.4395, color=INK2, lw=0.8, ls=(0, (3, 2)))
    a.text(3.0, 0.452, 'macro-F1 0.44', ha='center', fontsize=6.6, color=INK2)
    a.set_ylim(0, 0.75)
    a.set_ylabel('F1')
    a.set_title('(a) GCN, five balanced classes (n = 338)')
    a.tick_params(axis='x', rotation=35)
    bars = b.bar(D.KAN_CLASSES, D.KAN_F1, color=FAM['KAN'], width=0.62)
    for r, s in zip(bars, D.KAN_SUPPORT):
        b.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.024, f'{r.get_height():.2f}',
               ha='center', fontsize=6.8, color=INK2, bbox=dict(fc='white', ec='none', pad=0.5))
    b.axhline(0.142, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.text(2.0, 0.152, 'macro-F1 0.14', ha='center', fontsize=6.6, color=INK2)
    b.set_ylim(0, 0.75)
    b.set_title('(b) KAN, eight imbalanced classes (n = 11,601)')
    b.tick_params(axis='x', rotation=35)
    for ax in (a, b):
        ax.grid(axis='x', visible=False)
        for t in ax.get_xticklabels():
            t.set_ha('right')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'multiclass_per_class_f1.png')


def _cm_panel(ax, cm, labels, title):
    cm = np.array(cm, dtype=float)
    rn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(rn, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    k = len(labels)
    fs = 7.2 if k <= 5 else 6.2
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{int(cm[i, j]):,}', ha='center', va='center', fontsize=fs,
                    color='white' if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(k), labels, rotation=40, ha='right')
    ax.set_yticks(range(k), labels)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_multiclass_cms():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={'width_ratios': [5, 8]})
    _cm_panel(a, D.GCN_CM, D.GCN_CLASSES, '(a) GCN, five classes (n = 338)')
    _cm_panel(b, D.KAN_CM, D.KAN_CLASSES, '(b) KAN, eight classes (n = 11,601)')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'multiclass_confusion_matrices.png')


def fig_xgb_importance():
    names = [D.FEATURE_LABELS[k] for k, _ in D.XGB_TOP8][::-1]
    vals = [v for _, v in D.XGB_TOP8][::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    bars = ax.barh(names, vals, color=FAM['ML'], height=0.62)
    for r in bars:
        ax.text(r.get_width() + 0.002, r.get_y() + r.get_height() / 2, f'{r.get_width():.3f}',
                va='center', fontsize=6.8, color=INK2)
    ax.set_xlim(0, 0.13)
    ax.set_xlabel('XGBoost importance (gain share)')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'xgb_feature_importance.png')


def fig_signal_peaks():
    """Re-draw the GAT notebook's example: the first Elephant/Giraffe record in file
    order (selected_chunks[0]). Reads one record; nothing is trained."""
    import netCDF4 as nc
    from scipy.signal import find_peaks
    path = os.path.join(os.path.dirname(HERE), 'dset_allspec_150_chunks_clean.nc')
    ds = nc.Dataset(path)
    labels = np.asarray(ds['class'][:]).astype(int)
    idx = int(np.flatnonzero(np.isin(labels, [1, 6]))[0])
    sig = np.asarray(ds['chunk'][idx], dtype=np.float64)
    dist = float(ds['distance'][idx])
    ds.close()
    mag = np.sqrt((sig ** 2).sum(axis=0))
    mag_n = (mag - mag.mean()) / (mag.std() + 1e-8)
    peaks, _ = find_peaks(mag_n, prominence=0.05, distance=10)
    t = np.arange(sig.shape[1]) / D.FS_HZ
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 4.4), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
    for ax, comp, name in zip(axs[:3], sig, ['X', 'Y', 'Z']):
        ax.plot(t, comp, color=MUTED, lw=0.6)
        ax.set_ylabel(name, rotation=0, ha='right', va='center')
        ax.tick_params(labelleft=False)
    axs[3].plot(t, mag_n, color=FAM['GAT'], lw=0.8, label='normalised magnitude')
    axs[3].plot(t[peaks], mag_n[peaks], 'o', ms=4, color=INK, label=f'{len(peaks)} detected peaks (nodes)')
    axs[3].set_ylabel('z-score')
    axs[3].set_xlabel('time (s, 200 Hz)')
    axs[3].legend(frameon=False, loc='upper right')
    axs[0].set_title(f'Elephantidae record, sensor distance {dist:.1f} m')
    fig.tight_layout(h_pad=0.3)
    save(fig, 'signal_and_peaks.png')
    print(f'  record {idx}: {len(peaks)} peaks')


def fig_gat_curves():
    """Notebook figure of the GAT training curves with its notebook title cropped."""
    from PIL import Image
    src = os.path.join(HERE, 'notebook_images', 'gat_training_curves.png')
    im = Image.open(src)
    im.crop((0, 40, im.width, im.height)).save(os.path.join(OUT, 'gat_training_curves.png'))
    print('wrote figs/gat_training_curves.png')


# ==========================================================================
# 6. Edge-readiness analysis (arithmetic on logged architectures)
# ==========================================================================
def gat_macs(n, e, f_in=8, hid=64, heads=4, n_cls=2):
    """Multiply-accumulates for the 3-layer GATConv (concat=False) + head.
    GATConv adds one self-loop per node, so E' = E + N."""
    ep = e + n
    total = 0
    for fi in (f_in, hid, hid):
        total += n * fi * heads * hid          # shared linear projection
        total += 2 * n * heads * hid           # source/target attention logits
        total += ep * heads * hid              # attention-weighted aggregation
        total += n * heads * hid               # head averaging
    total += 2 * n * hid                       # mean + max pooling
    total += 2 * hid * hid + hid * n_cls       # classifier 128->64->2
    return total


def gat_peak_activation_bytes(n, e, hid=64, heads=4):
    """Largest live FP32 buffer: per-head node features plus edge attention."""
    return 4 * (n * heads * hid + (e + n) * heads)


def gcn_macs(n=64, nnz=190 + 64, f_in=96, h1=64, h2=32, n_cls=5):
    return n * f_in * h1 + nnz * h1 + n * h1 * h2 + nnz * h2 + n * h2 + h2 * n_cls


def kan_macs(sizes=(24, 32, 16, 8), n_basis=8):
    return sum(a * b * (1 + n_basis) for a, b in zip(sizes[:-1], sizes[1:]))


# Representative graph sizes: the logged example graph (22 nodes, 152 edges,
# class1_and_class6_weighted.ipynb cell 15) and the logged dataset mean with the
# same detector settings (68.4 nodes, 475.7 edges; peak_based_GAT.ipynb).
EDGE = [
    # name, family, params, stored_kb, stored_kind, ops, ops_kind, act_kb
    ('GAT', 'GAT', 45058, 161.3, 'logged INT8 file', gat_macs(68.4, 475.7), 'MAC',
     gat_peak_activation_bytes(68.4, 475.7) / 1024),
    ('ANN', 'ANN', 1182466, 1182466 * 4 / 1024, 'FP32, 4 B/param', 2048 * 512 + 512 * 256 + 256 * 2,
     'MAC', 4 * (2048 + 512) / 1024),
    ('XGBoost (saved)', 'ML', None, 715931 / 1024, 'saved .joblib', 300 * 5, 'comparison', 28 * 4 / 1024),
    # 300 trees, 7,385 splits and 7,685 leaves in the saved model; 9 B per split, 4 B per leaf
    ('XGBoost (C arrays)', 'ML', None, (7385 * 9 + 7685 * 4) / 1024, 'saved-model node count', 300 * 5,
     'comparison', 28 * 4 / 1024),
    ('7-NN', 'ML', None, 53451 * 28 * 8 / 1024, 'stored train set', 53451 * 28, 'MAC',
     53451 * 8 / 1024),
    ('Logistic reg.', 'ML', 29, 29 * 4 / 1024, 'FP32, 4 B/param', 28, 'MAC', 28 * 4 / 1024),
    ('GCN', 'GCN', 8453, 8453 * 4 / 1024, 'FP32, 4 B/param', gcn_macs(), 'MAC',
     4 * (64 * 96 + 64 * 64) / 1024),
    ('KAN', 'KAN', 14080, 14080 * 4 / 1024, 'FP32, 4 B/param', kan_macs(), 'MAC',
     4 * (24 + 24 * 8 + 32 * 9) / 1024),
]
FLASH_BUDGET_KB, SRAM_BUDGET_KB = 512, 200   # study budgets, wildlife_esp32_comparative_study/PLAN.md


def fig_edge():
    names = [e[0] for e in EDGE]
    y = np.arange(len(EDGE))[::-1]
    cols = [FAM[e[1]] for e in EDGE]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    panels = [(3, 'Stored model size (KB)', FLASH_BUDGET_KB, 'flash budget 512 KB'),
              (7, 'Peak working memory (KB, est.)', SRAM_BUDGET_KB, 'SRAM budget 200 KB'),
              (5, 'Operations per inference (est.)', None, None)]
    for ax, (idx, title, budget, blabel) in zip(axs, panels):
        vals = [e[idx] for e in EDGE]
        hatch = ['' if (idx == 3 and e[4] in ('logged INT8 file', 'saved .joblib')) else '////'
                 for e in EDGE]
        bars = ax.barh(y, vals, color=cols, height=0.62, hatch=hatch, edgecolor='white', lw=0.0)
        ax.set_xscale('log')
        for b, v in zip(bars, vals):
            txt = f'{v:,.0f}' if v >= 10 else f'{v:.2g}'
            ax.text(v * 1.25, b.get_y() + b.get_height() / 2, txt, va='center', fontsize=6.6, color=INK2,
                    bbox=dict(fc='white', ec='none', pad=0.6))
        if budget:
            ax.axvline(budget, color=INK, lw=0.9, ls=(0, (3, 2)))
            ax.text(budget * 1.1, y.max() + 0.55, blabel, fontsize=6.4, color=INK)
        ax.set_title(title)
        ax.grid(axis='y', visible=False)
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo / 3, hi * 40)
    axs[0].set_yticks(y, names)
    handles = [Rectangle((0, 0), 1, 1, fc=MUTED, ec='white'),
               Rectangle((0, 0), 1, 1, fc=MUTED, ec='white', hatch='////')]
    fig.legend(handles, ['measured file size', 'computed, not measured'],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.5)
    save(fig, 'edge_footprint.png')


def fig_pareto():
    pts = [('GAT (INT8)', 'GAT', 161.3, 'GAT'), ('GAT (FP32)', 'GAT', 184.3, 'GAT'),
           ('ANN', 'ANN', 1182466 * 4 / 1024, 'ANN'), ('XGBoost', 'ML', 715931 / 1024, 'XGBoost'),
           ('7-NN', 'ML', 53451 * 28 * 8 / 1024, '7-NN'),
           ('Logistic reg.', 'ML', 29 * 4 / 1024, 'Logistic reg.')]
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.axvline(FLASH_BUDGET_KB, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(FLASH_BUDGET_KB * 1.12, 0.42, 'ESP32-S3\nflash budget', fontsize=6.4, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=':')
    for lab, fam, kb, key in pts:
        ba = D.binary_metrics(D.BINARY[key][1])['bal_acc']
        ax.plot(kb, ba, 'o', ms=7, color=FAM[fam], mec='white', mew=1.4)
        ax.annotate(lab, (kb, ba), xytext=(0, 7), textcoords='offset points', ha='center',
                    fontsize=6.6, color=INK2)
    ax.set_xscale('log')
    ax.set_xlim(0.05, 6e4)
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel('Stored model size (KB, log scale)')
    ax.set_ylabel('Balanced accuracy')
    fig.tight_layout()
    save(fig, 'pareto_size_vs_balanced_accuracy.png')


def fig_tradeoffs():
    """(a) accuracy vs balanced accuracy for all binary models; (b) stored size vs
    balanced accuracy for the binary models whose size is known."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a.plot([0.4, 1], [0.4, 1], color=AXIS, lw=0.9, ls=(0, (3, 2)))
    a.axhline(0.5, color=MUTED, lw=0.7, ls=':')
    a.text(0.62, 0.507, 'chance (constant predictor)', fontsize=6.3, color=INK2, ha='center')
    offs = {'GAT': (6, -9), 'ANN': (6, -11), 'ANN (no std.)': (-6, -11), 'Logistic reg.': (-7, -4),
            'SVM (RBF)': (-7, 4), '7-NN': (-2, 7), 'Random forest': (-6, -11),
            'Grad. boosting': (6, 2), 'XGBoost': (-7, 3)}
    for k in ORDER:
        mm = D.binary_metrics(D.BINARY[k][1])
        a.plot(mm['acc'], mm['bal_acc'], 'o', ms=6.5, color=FAM[D.BINARY[k][0]], mec='white', mew=1.3)
        dx, dy = offs[k]
        a.annotate(k, (mm['acc'], mm['bal_acc']), xytext=(dx, dy), textcoords='offset points',
                   fontsize=6.3, color=INK2, ha='left' if dx > 0 else 'right')
    a.set_xlim(0.4, 1.0)
    a.set_ylim(0.4, 0.95)
    a.set_xlabel('Accuracy')
    a.set_ylabel('Balanced accuracy')
    a.set_title('(a) Accuracy versus balanced accuracy')
    handles = [plt.Line2D([], [], marker='o', ls='', color=FAM[f], ms=6) for f in ('GAT', 'ANN', 'ML')]
    a.legend(handles, ['GAT', 'ANN', 'Classical ML'], frameon=False, loc='upper left')

    pts = [('GAT (INT8)', 'GAT', 161.3, 'GAT', (0, 7)), ('ANN', 'ANN', 1182466 * 4 / 1024, 'ANN', (0, 7)),
           ('XGBoost (saved)', 'ML', 715931 / 1024, 'XGBoost', (24, -12)),
           ('XGBoost (C arrays)', 'ML', (7385 * 9 + 7685 * 4) / 1024, 'XGBoost', (-34, -12)),
           ('7-NN', 'ML', 53451 * 28 * 8 / 1024, '7-NN', (0, 7)),
           ('Logistic reg.', 'ML', 29 * 4 / 1024, 'Logistic reg.', (0, 7))]
    b.axvline(FLASH_BUDGET_KB, color=INK, lw=0.9, ls=(0, (3, 2)))
    b.text(FLASH_BUDGET_KB * 1.15, 0.42, '512 KB\nflash budget', fontsize=6.3, color=INK)
    b.axhline(0.5, color=MUTED, lw=0.7, ls=':')
    for lab, fam, kb, key, off in pts:
        ba = D.binary_metrics(D.BINARY[key][1])['bal_acc']
        b.plot(kb, ba, 'o', ms=6.5, color=FAM[fam], mec='white', mew=1.3)
        b.annotate(lab, (kb, ba), xytext=off, textcoords='offset points', ha='center',
                   fontsize=6.3, color=INK2)
    b.set_xscale('log')
    b.set_xlim(0.05, 6e4)
    b.set_ylim(0.4, 0.95)
    b.set_xlabel('Stored model size (KB, log scale)')
    b.set_title('(b) Model size versus balanced accuracy')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'tradeoffs_accuracy_size.png')


# ==========================================================================
# 7. Architecture schematic for all five model families
# ==========================================================================
def fig_architectures():
    rows = [
        ('GAT', 'GAT', ['raw signal\n3 x 2048', 'peak graph: 8-D\nnodes, chain+5-NN',
                        '3 x GATConv\n64 wide, 4 heads', 'mean || max\npooling', 'MLP\n128-64-2']),
        ('ANN', 'ANN', ['X component\n1 x 2048', 'standardise\n(train stats)', 'Linear 512\nBN, ReLU, drop',
                        'Linear 256\nBN, ReLU, drop', 'Linear\n256-2']),
        ('ML', 'ML', ['raw signal\n3 x 2048', '28 statistics\naxis/mag./peaks',
                      'standardise\n(LR, SVM, kNN)', 'LR, SVM, 7-NN,\nRF, GB, XGB', 'class\nlabel']),
        ('GCN', 'GCN', ['raw signal\n3 x 2048', '64 windows\n= node chain',
                        'GCNConv 64\nReLU', 'GCNConv 32\nReLU, mean pool', 'Linear\n32-5']),
        ('KAN', 'KAN', ['raw signal\n3 x 2048', '8 descriptors\nper peak',
                        'mean/max/std\npool = 24-D', 'KANLinear\n24-32-16', 'KANLinear\n16-8']),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    bw, gap, x_start = 1.62, 0.36, 0.9
    ax.set_xlim(0, x_start + 5 * bw + 4 * gap + 0.05)
    ax.set_ylim(-0.45, len(rows) - 0.15)
    ax.axis('off')
    for r, (title, fam, steps) in enumerate(rows):
        yc = len(rows) - 1 - r + 0.3
        ax.text(0.0, yc, title, fontsize=8.5, fontweight='bold', color=INK, va='center')
        for s_i, label in enumerate(steps):
            x0 = x_start + s_i * (bw + gap)
            shaded = s_i >= 2
            ax.add_patch(FancyBboxPatch((x0, yc - 0.34), bw, 0.68,
                                        boxstyle='round,pad=0.01,rounding_size=0.07',
                                        fc=matplotlib.colors.to_rgba(FAM[fam], 0.20 if shaded else 0.0),
                                        ec=FAM[fam], lw=1.2))
            ax.text(x0 + bw / 2, yc, label, ha='center', va='center', fontsize=6.3, color=INK,
                    linespacing=1.15)
            if s_i:
                ax.annotate('', (x0 - 0.03, yc), (x0 - gap + 0.03, yc),
                            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.9))
    for s_i, lab in [(0, 'input'), (1, 'representation'), (3, 'classifier (shaded)')]:
        ax.text(x_start + s_i * (bw + gap) + bw / 2, -0.38, lab, ha='center', fontsize=6.6, color=MUTED)
    save(fig, 'model_architectures.png')


# ==========================================================================
# Derived-metric table for the manuscript
# ==========================================================================
def write_tables():
    with open(os.path.join(HERE, 'derived_binary_metrics.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'n', 'acc', 'acc_ci_lo', 'acc_ci_hi', 'bal_acc', 'macro_f1',
                    'weighted_f1', 'prec_giraffe', 'rec_giraffe', 'f1_giraffe', 'mcc', 'kappa', 'source'])
        for k in ORDER:
            fam, cm, src = D.BINARY[k]
            m = D.binary_metrics(cm)
            lo, hi = D.wilson(cm[0][0] + cm[1][1], m['n'])
            w.writerow([k, m['n'], f"{m['acc']:.4f}", f'{lo:.4f}', f'{hi:.4f}', f"{m['bal_acc']:.4f}",
                        f"{m['macro_f1']:.4f}", f"{m['weighted_f1']:.4f}", f"{m['prec_g']:.4f}",
                        f"{m['rec_g']:.4f}", f"{m['f1_g']:.4f}", f"{m['mcc']:.4f}", f"{m['kappa']:.4f}", src])
    print('wrote figure_sources/derived_binary_metrics.csv')
    with open(os.path.join(HERE, 'derived_edge_estimates.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'params', 'stored_kb', 'stored_basis', 'ops', 'ops_unit', 'peak_working_kb'])
        for e in EDGE:
            w.writerow([e[0], e[2] if e[2] else '', f'{e[3]:.1f}', e[4], f'{e[5]:.0f}', e[6], f'{e[7]:.1f}'])
    print('wrote figure_sources/derived_edge_estimates.csv')
    print('GAT MACs at the logged example graph (22 nodes, 152 edges):', gat_macs(22, 152))


if __name__ == '__main__':
    fig_gat_grid()
    fig_selection()
    fig_training()
    fig_binary_metrics()
    fig_tradeoffs()
    fig_binary_cms()
    fig_multiclass_cms()
    fig_xgb_importance()
    fig_signal_peaks()
    fig_edge()
    fig_architectures()
    write_tables()
