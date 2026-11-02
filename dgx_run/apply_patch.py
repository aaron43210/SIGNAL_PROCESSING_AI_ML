"""Upgrade rigorous_benchmark.py (1479 lines) in place: dataset auto-search + control phase."""
from pathlib import Path
p = Path(__file__).with_name("rigorous_benchmark.py")
s = p.read_text()
assert len(s.splitlines()) == 1479, "expected the 1479-line version"
R = [
    ("        path = args.data or next((p for p in ('dset_allspec_150_chunks_clean.nc',\n                                              'PROJECT/dset_allspec_150_chunks_clean.nc',\n                                              str(HERE.parent / 'dset_allspec_150_chunks_clean.nc'))\n                                  if os.path.exists(p)), None)\n",
     '        path = args.data or find_dataset()\n'),
    ('def prepare(args):',
     'def find_dataset(name=\'dset_allspec_150_chunks_clean.nc\'):\n    """Look next to the script, in the usual project folders, then up to 4 levels below home."""\n    home, user = Path.home(), os.environ.get(\'USER\', \'\')\n    roots = [Path.cwd(), HERE, HERE.parent, HERE.parent / \'PROJECT\', home, home / \'PROJECT\',\n             Path(\'/jupyter\') / user, Path(\'/jupyter\') / user / \'PROJECT\']\n    for r in roots:\n        if (r / name).exists():\n            return str(r / name)\n    for top in (home, Path(\'/jupyter\') / user):\n        if not top.exists():\n            continue\n        base = len(top.parts)\n        for dirpath, dirnames, filenames in os.walk(top):\n            if name in filenames:\n                return os.path.join(dirpath, name)\n            if len(Path(dirpath).parts) - base >= 4:\n                dirnames[:] = []                    # do not descend further\n            dirnames[:] = [d for d in dirnames if not d.startswith(\'.\') and d not in (\'site-packages\', \'node_modules\')]\n    return None\n\n\ndef prepare(args):'),
    ('def make_splits(',
     'def make_random_splits(idx, y, n_outer, n_inner, seed=42):\n    """Control only: window-level stratified splits that ignore stations, as in the original\n    notebooks. Used to show how much of a score comes from the split itself."""\n    from sklearn.model_selection import StratifiedKFold\n    folds = []\n    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)\n    for f, (tr_pos, te_pos) in enumerate(outer.split(idx, y[idx])):\n        tr, te = idx[tr_pos], idx[te_pos]\n        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + f)\n        itr, iva = next(inner.split(tr, y[tr]))\n        folds.append(dict(fold=f, train=np.sort(tr[itr]), val=np.sort(tr[iva]), test=np.sort(te)))\n    return folds\n\n\ndef make_splits('),
    ("    sp = dict(peak=make_splits(cohort, y, station, CFG['n_outer'], CFG['n_inner']),\n              all=make_splits(np.arange(len(y)), y, station, CFG['n_outer'], CFG['n_inner']))",
     "    sp = dict(peak=make_splits(cohort, y, station, CFG['n_outer'], CFG['n_inner']),\n              all=make_splits(np.arange(len(y)), y, station, CFG['n_outer'], CFG['n_inner']),\n              random_peak=make_random_splits(cohort, y, CFG['n_outer'], CFG['n_inner']),\n              random_all=make_random_splits(np.arange(len(y)), y, CFG['n_outer'], CFG['n_inner']))"),
    ("    cfg, gvba = best_config('GAT', fold, 'gat')\n    variant = j.get('variant', 'full')\n    if j.get('scale_flip'):\n        cfg = dict(cfg, scale='std' if cfg['scale'] == 'raw' else 'raw')\n    d, b = SPLITS['peak'][fold], CFG['budget']",
     "    if 'fixed_config' in j:                         # control: the original notebook's configuration\n        cfg, gvba = j['fixed_config'], None\n    else:\n        cfg, gvba = best_config('GAT', fold, 'gat')\n    variant = j.get('variant', 'full')\n    if j.get('scale_flip'):\n        cfg = dict(cfg, scale='std' if cfg['scale'] == 'raw' else 'raw')\n    d, b = SPLITS[j.get('cohort', 'peak')][fold], CFG['budget']"),
    ("PHASES = ['1_search', '2_final', '3_extras']",
     "PHASES = ['1_search', '2_final', '3_extras', '4_control']"),
    ("                     for p in CFG['detector_sweep']['prominence'] for d in CFG['detector_sweep']['distance']]\n    return jobs",
     "                     for p in CFG['detector_sweep']['prominence'] for d in CFG['detector_sweep']['distance']]\n    elif phase == '4_control' and r.get('control', True):\n        # Same code, but window-level random splits as in the original notebooks: isolates how much\n        # of the original scores came from the split (leakage between near-duplicate windows).\n        if r['gat']:\n            orig = dict(hidden=64, heads=4, lr=1e-3, scale='raw')\n            jobs += [dict(type='gat_final', label='GAT [random split]', cohort='random_peak', fixed_config=orig,\n                          fold=f, seed=s, prio=0) for f in folds for s in CFG['seeds']]\n        if r['ml']:\n            for est in ('XGBoost', 'Logistic reg.'):\n                jobs += [dict(type='ml', label=f'{est} [random split]', est=est, features='ml28',\n                              cohort='random_all', fold=f, prio=1) for f in folds]\n    return jobs"),
]
for a, b in R:
    assert s.count(a) == 1, "anchor not found: " + a[:50]
    s = s.replace(a, b)
p.with_suffix(".py.bak1479").write_text(p.read_text())
p.write_text(s)
print("patched:", len(s.splitlines()), "lines")
