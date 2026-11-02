"""Logged results used by the paper, transcribed from saved notebook outputs.

Nothing here is re-computed by training or evaluating a model. Every value is
copied from a printed notebook output (source notebook + cell noted beside it)
or derived arithmetically from those values (derived quantities are marked).

Label note: the notebooks call class 6 "Wildebeest". The class_dict group
embedded in dset_allspec_150_chunks_clean.nc maps 6 -> GIRAFFIDAE (and
5 -> BOVIDAE, 9 -> EQUIDAE), confirmed by PROJECT/class_verification. The
paper therefore reports class 6 as Giraffe; the numbers are unchanged.
"""

import math

# --------------------------------------------------------------------------
# Official family names (netCDF group `class_dict`)
# --------------------------------------------------------------------------
FAMILY = {0: 'Hominidae', 1: 'Elephantidae', 2: 'Suidae', 3: 'Felidae',
          4: 'Numididae', 5: 'Bovidae', 6: 'Giraffidae', 7: 'Hyaenidae',
          8: 'Leporidae', 9: 'Equidae', 10: 'Mustelidae', 11: 'Hippopotamidae'}

# --------------------------------------------------------------------------
# GAT grid search -- class1_and_class6_weighted.ipynb, cell 21
# 20 epochs, patience 5, score = best macro-F1 on the 10,981-graph test loader
# weight_decay fixed at 1e-4 for every configuration
# --------------------------------------------------------------------------
GAT_GRID = [  # (hidden_dim, heads, dropout, lr, best_macro_f1)
    (32, 2, 0.3, 1e-3, 0.5727), (32, 2, 0.3, 5e-4, 0.5488),
    (32, 2, 0.4, 1e-3, 0.5664), (32, 2, 0.4, 5e-4, 0.5548),
    (32, 4, 0.3, 1e-3, 0.5658), (32, 4, 0.3, 5e-4, 0.5633),
    (32, 4, 0.4, 1e-3, 0.5649), (32, 4, 0.4, 5e-4, 0.5530),
    (64, 2, 0.3, 1e-3, 0.5744), (64, 2, 0.3, 5e-4, 0.5407),
    (64, 2, 0.4, 1e-3, 0.5964), (64, 2, 0.4, 5e-4, 0.5675),
    (64, 4, 0.3, 1e-3, 0.5746), (64, 4, 0.3, 5e-4, 0.5731),
    (64, 4, 0.4, 1e-3, 0.5972), (64, 4, 0.4, 5e-4, 0.5596),
]
GAT_SELECTED = (64, 4, 0.4, 1e-3)   # cell 21: "BEST CONFIG ... Macro F1: 0.5972"

# --------------------------------------------------------------------------
# Binary Elephant (0) vs Giraffe (1) confusion matrices [[TN, FP], [FN, TP]]
# (positive = Giraffe). GAT/ANN/XGBoost read from the saved confusion-matrix
# figures; the other five estimators are back-solved (derived) from logged
# accuracy + Giraffe-F1 with known supports and verified against logged
# macro-F1 and weighted-F1 (each has exactly one integer solution).
# --------------------------------------------------------------------------
BINARY = {
    # name: (family, confusion matrix, source)
    'GAT':            ('GAT', [[9633, 650], [535, 163]],
                       'class1_and_class6_weighted.ipynb cell 27 (figure)'),
    'ANN':            ('ANN', [[5519, 6980], [408, 456]],
                       'ann_classifier_weighted.ipynb cell 13 (figure)'),
    'ANN (no std.)':  ('ANN', [[12499, 0], [864, 0]],
                       'ann_classifier_unweighted.ipynb cell 14 (figure)'),
    'Logistic reg.':  ('ML', [[8914, 3585], [349, 515]], 'ml_classifier.ipynb cell 16 (derived)'),
    'SVM (RBF)':      ('ML', [[9901, 2598], [321, 543]], 'ml_classifier.ipynb cell 16 (derived)'),
    '7-NN':           ('ML', [[12408, 91], [562, 302]], 'ml_classifier.ipynb cell 16 (derived)'),
    'Random forest':  ('ML', [[12054, 445], [552, 312]], 'ml_classifier.ipynb cell 16 (derived)'),
    'Grad. boosting': ('ML', [[9097, 3402], [208, 656]], 'ml_classifier.ipynb cell 16 (derived)'),
    'XGBoost':        ('ML', [[11628, 871], [134, 730]],
                       'ml_classifier.ipynb cell 20 (figure)'),
}


def binary_metrics(cm):
    """Standard metrics from a 2x2 confusion matrix (positive = Giraffe)."""
    (tn, fp), (fn, tp) = cm
    n = tn + fp + fn + tp
    rec_p = tp / (tp + fn)
    rec_n = tn / (tn + fp)
    prec_p = tp / (tp + fp) if tp + fp else 0.0
    prec_n = tn / (tn + fn) if tn + fn else 0.0
    f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if prec_p + rec_p else 0.0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n else 0.0
    acc = (tp + tn) / n
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kappa = (acc - pe) / (1 - pe) if pe != 1 else 0.0
    return dict(n=n, acc=acc, bal_acc=(rec_p + rec_n) / 2,
                macro_f1=(f1_p + f1_n) / 2,
                weighted_f1=((tp + fn) * f1_p + (tn + fp) * f1_n) / n,
                prec_g=prec_p, rec_g=rec_p, f1_g=f1_p, rec_e=rec_n,
                mcc=mcc, kappa=kappa, gmean=math.sqrt(rec_p * rec_n))


def wilson(k, n, z=1.959964):
    """Wilson score 95% interval for a binomial proportion k/n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# --------------------------------------------------------------------------
# Training histories
# --------------------------------------------------------------------------
# ann_classifier_weighted.ipynb cell 11 -- 50 epochs, train loss / train acc (%)
ANN_LOSS = [0.7242, 0.6793, 0.6691, 0.6532, 0.6450, 0.6339, 0.6228, 0.6128, 0.5958, 0.5853,
            0.5733, 0.5598, 0.5506, 0.5428, 0.5382, 0.5323, 0.5204, 0.5097, 0.4972, 0.5158,
            0.4950, 0.4832, 0.4889, 0.4784, 0.4751, 0.4775, 0.4681, 0.4659, 0.4599, 0.4598,
            0.4607, 0.4434, 0.4505, 0.4473, 0.4537, 0.4450, 0.4394, 0.4287, 0.4490, 0.4275,
            0.4325, 0.4341, 0.4249, 0.4156, 0.4274, 0.4211, 0.4131, 0.4144, 0.4193, 0.4094]
ANN_ACC = [66.51, 74.65, 75.22, 73.23, 72.21, 73.11, 72.63, 72.95, 73.75, 73.94,
           74.22, 74.57, 75.23, 75.63, 75.97, 76.16, 76.01, 77.23, 77.11, 76.43,
           77.49, 78.42, 77.64, 78.35, 77.86, 78.39, 78.27, 78.57, 78.57, 78.98,
           78.74, 79.63, 79.57, 79.62, 79.34, 79.70, 79.43, 80.07, 80.17, 79.96,
           80.19, 80.30, 80.36, 80.98, 80.14, 80.38, 80.85, 81.08, 80.42, 81.05]
# ann_classifier_unweighted.ipynb cell 12 -- 20 epochs, no input standardisation
ANN_NOSTD_LOSS = [0.7006, 0.6929, 0.6927, 0.6929, 0.6929, 0.6928, 0.6929, 0.6928, 0.6929, 0.6928,
                  0.6928, 0.6928, 0.6928, 0.6928, 0.6927, 0.6928, 0.6928, 0.6928, 0.6928, 0.6929]
ANN_NOSTD_ACC = [58.63, 91.14] + [93.53] * 18

# gcn(final).ipynb cell 33 -- raw-segment GCN, 30 epochs
GCN_RAW = dict(
    tr_loss=[1.5887, 1.5599, 1.5380, 1.5242, 1.5021, 1.4802, 1.4667, 1.4489, 1.4361, 1.4221,
             1.4005, 1.3902, 1.3774, 1.3564, 1.3431, 1.3273, 1.3126, 1.2976, 1.2820, 1.2703,
             1.2548, 1.2407, 1.2230, 1.2245, 1.2045, 1.1973, 1.1852, 1.1761, 1.1662, 1.1607],
    tr_acc=[0.2381, 0.2717, 0.2616, 0.2673, 0.2889, 0.2895, 0.3105, 0.3346, 0.3708, 0.3568,
            0.3994, 0.4013, 0.4083, 0.4114, 0.4254, 0.4197, 0.4406, 0.4425, 0.4533, 0.4527,
            0.4730, 0.4800, 0.4794, 0.4660, 0.4959, 0.5105, 0.5232, 0.4997, 0.5213, 0.5346],
    va_loss=[1.5644, 1.5516, 1.5341, 1.5189, 1.5055, 1.4967, 1.4895, 1.4856, 1.4707, 1.4657,
             1.4615, 1.4534, 1.4488, 1.4488, 1.4433, 1.4403, 1.4330, 1.4324, 1.4333, 1.4285,
             1.4244, 1.4184, 1.4186, 1.4218, 1.4222, 1.4301, 1.4105, 1.4133, 1.4240, 1.4265],
    va_acc=[0.2196, 0.2908, 0.2819, 0.2997, 0.2849, 0.3056, 0.2878, 0.3323, 0.3205, 0.3264,
            0.3828, 0.3739, 0.3353, 0.3976, 0.3501, 0.3620, 0.3591, 0.3739, 0.3739, 0.4392,
            0.4421, 0.4006, 0.4006, 0.4421, 0.4154, 0.4570, 0.3976, 0.4629, 0.4510, 0.4540])
# gcn(final).ipynb cell 45 -- feature-based GCN, 30 epochs (best val acc 0.3947)
GCN_FEAT = dict(
    tr_loss=[1.5597, 1.5119, 1.4919, 1.4746, 1.4776, 1.4608, 1.4656, 1.4448, 1.4409, 1.4392,
             1.4344, 1.4208, 1.4175, 1.4042, 1.4130, 1.4072, 1.3961, 1.3982, 1.3878, 1.3937,
             1.3779, 1.3860, 1.3982, 1.3808, 1.3733, 1.3808, 1.3570, 1.3599, 1.3715, 1.3640],
    tr_acc=[0.2648, 0.2775, 0.2832, 0.3041, 0.3130, 0.3079, 0.3232, 0.3283, 0.3537, 0.3517,
            0.3441, 0.3435, 0.3530, 0.3676, 0.3606, 0.3581, 0.3663, 0.3803, 0.3790, 0.3898,
            0.3860, 0.3917, 0.3721, 0.3937, 0.3867, 0.3733, 0.3937, 0.4013, 0.3898, 0.3994],
    va_loss=[1.5239, 1.4805, 1.4735, 1.4528, 1.4874, 1.4632, 1.5012, 1.4319, 1.4789, 1.4600,
             1.4260, 1.4618, 1.4388, 1.4148, 1.4712, 1.4496, 1.4185, 1.4035, 1.4625, 1.4069,
             1.4170, 1.4702, 1.4206, 1.4349, 1.4503, 1.3794, 1.3696, 1.3617, 1.4323, 1.4155],
    va_acc=[0.3116, 0.2967, 0.3205, 0.3561, 0.3294, 0.3353, 0.3323, 0.3412, 0.2997, 0.3412,
            0.3234, 0.3145, 0.3383, 0.3591, 0.3383, 0.3442, 0.3650, 0.3739, 0.3323, 0.3680,
            0.3234, 0.3145, 0.3472, 0.3056, 0.3383, 0.3561, 0.3798, 0.3947, 0.3561, 0.3383])
# kan_weighted.ipynb cell 16 -- printed every 5 epochs (50 epochs)
KAN_W = dict(epoch=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
             tr_loss=[2.0102, 1.7279, 1.5936, 1.4861, 1.3912, 1.3243, 1.3008, 1.2588, 1.2336, 1.2204, 1.2068],
             tr_acc=[0.337, 0.328, 0.300, 0.329, 0.360, 0.373, 0.376, 0.399, 0.393, 0.404, 0.410],
             va_loss=[1.8574, 1.7035, 1.6758, 1.6182, 1.6364, 1.5783, 1.5399, 1.5641, 1.5543, 1.5442, 1.5528],
             va_acc=[0.094, 0.501, 0.368, 0.360, 0.274, 0.300, 0.315, 0.308, 0.297, 0.292, 0.282])
# kan.ipynb -- unweighted / no undersampling, early stop at epoch 32
KAN_U = dict(epoch=[1, 5, 10, 15, 20, 25, 30],
             tr_loss=[0.6579, 0.4575, 0.4290, 0.4099, 0.3969, 0.3835, 0.3674],
             tr_acc=[0.883, 0.888, 0.890, 0.894, 0.897, 0.900, 0.904],
             va_loss=[0.4934, 0.4576, 0.4327, 0.4199, 0.4669, 0.4166, 0.4121],
             va_acc=[0.887, 0.887, 0.890, 0.891, 0.891, 0.896, 0.897])

# --------------------------------------------------------------------------
# Multi-class reports
# --------------------------------------------------------------------------
# gcn(final).ipynb cell 34 -- 338-record test split (rows true, cols predicted)
GCN_CLASSES = ['Hominidae', 'Elephantidae', 'Bovidae', 'Giraffidae', 'Equidae']
GCN_CM = [[21, 15, 20, 5, 7], [12, 33, 12, 5, 5], [6, 5, 52, 1, 4],
          [12, 26, 15, 10, 5], [14, 6, 7, 0, 40]]
GCN_F1 = [0.3158, 0.4342, 0.5977, 0.2247, 0.6250]
# kan_weighted.ipynb cell 20 -- 11,601-record test split; IDs 0,1,2,4,5,6,7,9
KAN_CLASSES = ['Hominidae', 'Elephantidae', 'Suidae', 'Numididae', 'Bovidae',
               'Giraffidae', 'Hyaenidae', 'Equidae']
KAN_SUPPORT = [347, 10282, 30, 39, 108, 698, 8, 89]
KAN_F1 = [0.175, 0.462, 0.029, 0.119, 0.117, 0.185, 0.000, 0.050]
KAN_RECALL = [0.450, 0.307, 0.433, 0.667, 0.565, 0.433, 0.000, 0.596]
# kan.ipynb -- same ids plus 3, 8, 11; unweighted
KAN_U_SUMMARY = dict(acc=0.893, macro_f1=0.162, weighted_f1=0.863, n=11603, n_classes=11)

# kan_weighted.ipynb cell 20 -- test confusion matrix (rows true, cols predicted),
# class order as KAN_CLASSES; read from the saved figure and checked against the
# printed per-class precision, recall and support.
KAN_CM = [[156, 44, 8, 25, 35, 52, 0, 27],
          [1168, 3158, 811, 327, 758, 2195, 0, 1865],
          [7, 7, 13, 0, 0, 1, 0, 2],
          [3, 0, 0, 26, 10, 0, 0, 0],
          [10, 12, 7, 5, 61, 4, 0, 9],
          [91, 144, 28, 14, 64, 302, 0, 55],
          [1, 0, 1, 0, 3, 2, 0, 1],
          [0, 20, 0, 1, 8, 7, 0, 53]]

# ml_classifier.ipynb cell 22 -- the eight largest XGBoost importances (printed)
XGB_TOP8 = [('peak_std_h', 0.108835), ('Z_zcr', 0.059897), ('mag_mean', 0.053403),
            ('X_energy', 0.048816), ('mag_energy', 0.045855), ('mag_std', 0.043132),
            ('n_peaks', 0.042706), ('mag_max', 0.042117)]
FEATURE_LABELS = {'peak_std_h': 'Peak height, std.', 'Z_zcr': 'Z zero-crossing rate',
                  'mag_mean': 'Magnitude, mean', 'X_energy': 'X energy',
                  'mag_energy': 'Magnitude energy', 'mag_std': 'Magnitude, std.',
                  'n_peaks': 'Peak count', 'mag_max': 'Magnitude, max.'}

# Acquisition (Szenicer et al., 2022): 200 Hz sampling, so 2,048 samples = 10.24 s
FS_HZ = 200


def balanced_accuracy_ci(cm, z=1.959964):
    """Normal-approximation 95% CI for balanced accuracy, treating the two class
    recalls as independent binomial proportions (test-sampling variance only)."""
    (tn, fp), (fn, tp) = cm
    r1, n1 = tp / (tp + fn), tp + fn
    r0, n0 = tn / (tn + fp), tn + fp
    ba = (r1 + r0) / 2
    se = 0.5 * math.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    return ba - z * se, ba + z * se
