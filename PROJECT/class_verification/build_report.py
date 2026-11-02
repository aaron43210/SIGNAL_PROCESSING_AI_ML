"""Build a per-class evidence report (PDF) for dset_allspec_150_chunks_clean.nc.

Reads only small random subsets of the 29.5 GB file. Run with the venv in PROJECT/PROJECT/venv:
    ../venv/bin/python build_report.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
NC_PATH = "/jupyter/aswin.vs/PROJECT/dset_allspec_150_chunks_clean.nc"
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(0)

official = json.load(open(os.path.join(HERE, "class_dict_official.json")))
official = {v: k for k, v in official.items() if not k.startswith("_")}

# Map used in the earlier notebooks (class1_and_class6_*.ipynb) -- shown only for comparison.
NOTEBOOK_MAP = {0: "Human", 1: "Elephant", 2: "Giraffe", 3: "Lion", 4: "Buffalo", 5: "Zebra",
                6: "Wildebeest", 7: "Hippo", 8: "Cheetah", 9: "Hyena"}

VISUAL_NOTES = {
    0: "People are visible in some frames (two men standing near the sensor, one in a cap), others show an empty scene. Matches HOMINIDAE.",
    1: "Elephants clearly visible in most frames, singly and in herds at a waterhole. Matches ELEPHANTIDAE.",
    2: "Small dark pig-like animals are visible in two frames (consistent with warthog / bushpig); the other frames are empty scenes. Plausible for SUIDAE, not conclusive.",
    3: "Only two night infrared frames, essentially dark; a faint shape in the second one. Cannot be confirmed from these photos.",
    4: "Small dark ground birds are visible in the scrub (consistent with guineafowl). Plausible for NUMIDIDAE.",
    5: "Slender antelope with light bellies, impala-like, in several frames. Matches BOVIDAE (antelope), <b>not</b> 'Zebra' as the old notebooks said.",
    6: "Giraffes clearly visible in every frame. Matches GIRAFFIDAE, <b>not</b> 'Wildebeest' as the old notebooks said.",
    7: "Night infrared frames; an animal of dog / hyena shape is faintly visible in three of them. Consistent with HYAENIDAE, not conclusive.",
    8: "Night infrared frames of the same scene; no animal is clearly visible (a hare would be tiny). Cannot be confirmed from these photos.",
    9: "Zebras (striped) are visible in two frames; others are empty scenes. Matches EQUIDAE, <b>not</b> 'Hyena' as the old notebooks said.",
    11: "Night infrared frames showing a very large rounded body filling the view at 14 m, plus a large animal in the first frame. Consistent with HIPPOPOTAMIDAE.",
}
STATUS = {0: "confirmed", 1: "confirmed", 2: "plausible", 3: "not verifiable", 4: "plausible", 5: "confirmed (antelope)",
          6: "confirmed", 7: "plausible", 8: "not verifiable", 9: "confirmed", 11: "plausible"}

# Common members of each family (my inference -- the dataset only stores the family name).
MEMBERS = {
    "HOMINIDAE": "Human (people walking / running)",
    "ELEPHANTIDAE": "Elephant",
    "SUIDAE": "Pigs: warthog, bushpig",
    "FELIDAE": "Cats: lion, leopard, cheetah",
    "NUMIDIDAE": "Guineafowl (a bird)",
    "BOVIDAE": "Buffalo, wildebeest, antelope, cattle, goats",
    "GIRAFFIDAE": "Giraffe",
    "HYAENIDAE": "Hyenas",
    "LEPORIDAE": "Hares / rabbits",
    "EQUIDAE": "Zebra, donkey, horse",
    "MUSTELIDAE": "Badgers / honey badger (not present in this file)",
    "HIPPOPOTAMIDAE": "Hippopotamus",
}

ds = nc.Dataset(NC_PATH)
cls = ds["class"][:].data.astype(int)
dist = ds["distance"][:].data
classes = [int(c) for c in np.unique(cls)]
counts = {c: int((cls == c).sum()) for c in classes}


def read_rows(var, idx):
    return np.stack([var[int(i)] for i in sorted(idx)]), sorted(idx)


# ---------- per-class photo figures ----------
# The 'image' variable is a 128x128 RGB camera-trap photo (not a spectrogram). Animals are easiest to see when
# they are close, so examples are drawn at random (fixed seed) from the nearer half of each class.
example_idx = {}
for c in classes:
    ids = np.where(cls == c)[0]
    order = ids[np.argsort(dist[ids])]
    pool = order[: max(6, len(order) // 2)]
    example_idx[c] = sorted(rng.choice(pool, size=min(6, len(pool)), replace=False).tolist())

for c in classes:
    pick = example_idx[c]
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        img = np.asarray(ds["image"][int(i)]).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f"event {i}  |  {dist[i]:.0f} m", fontsize=9)
    fig.suptitle(f"Class {c} = {official.get(c, '?')}  (n = {counts[c]})", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"class_{c:02d}_photos.png"), dpi=110)
    plt.close(fig)
    print("photos done for class", c, flush=True)

# ---------- overview: one photo per class ----------
n = len(classes)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6.2))
for ax in axes.ravel():
    ax.axis("off")
for ax, c in zip(axes.ravel(), classes):
    i = example_idx[c][0]
    ax.imshow(np.asarray(ds["image"][int(i)]).astype(np.uint8))
    ax.set_title(f"class {c}\n{official.get(c, '?')}\n(n={counts[c]})", fontsize=9)
fig.suptitle("One photo per class (camera-trap frame stored with each event)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "overview_photos.png"), dpi=110)
plt.close(fig)
print("overview done", flush=True)

# ---------- average amplitude spectra (distance-independent: each event RMS-normalised) ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
for c in classes:
    if counts[c] < 10:
        continue
    ids = np.where(cls == c)[0]
    sel = rng.choice(ids, size=min(200, len(ids)), replace=False)
    x, _ = read_rows(ds["chunk"], sel)
    z = x[:, 2, :]
    z = z / (np.sqrt((z ** 2).mean(1, keepdims=True)) + 1e-30)
    sp = np.abs(np.fft.rfft(z * np.hanning(z.shape[1]), axis=1)).mean(0)
    ax.semilogy(sp[1:400], lw=1.1, label=f"{c} {official.get(c, '?')}")
ax.set_xlabel("FFT bin (sampling rate is not stored in the file)")
ax.set_ylabel("mean |FFT| of RMS-normalised Z")
ax.set_title("Average Z-component spectrum per class (classes with n >= 10)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "spectra_by_class.png"), dpi=110)
plt.close(fig)
print("spectra done", flush=True)

# ---------- PDF ----------
ss = getSampleStyleSheet()
H1, H2, B = ss["Title"], ss["Heading2"], ss["BodyText"]
small = ParagraphStyle("small", parent=B, fontSize=8, leading=10)
doc = SimpleDocTemplate(os.path.join(OUT, "class_map_report.pdf"), pagesize=A4,
                        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                        title="Class map verification", author="")
W = A4[0] - 3 * cm
S = []
S.append(Paragraph("Class number to animal: evidence report", H1))
S.append(Paragraph("File: <b>dset_allspec_150_chunks_clean.nc</b> (70,637 events). All examples below are read "
                   "directly from that file; class numbers are the raw values stored in its <i>class</i> variable.", B))
S.append(Spacer(1, 6))
S.append(Paragraph("Key finding", H2))
S.append(Paragraph(
    "The dataset authors' own <b>class_dict.pkl</b> (Dropbox folder <i>dset_allspec_150</i>) labels classes by "
    "<b>animal family</b>, not by species, and it does <b>not</b> match the species map used in the earlier notebooks. "
    "Examples: class 6 is <b>GIRAFFIDAE</b> (giraffe), not 'Wildebeest'; class 5 is <b>BOVIDAE</b> (buffalo, wildebeest, "
    "antelope, cattle) rather than 'Zebra'; class 9 is <b>EQUIDAE</b> (zebra, donkey, horse) rather than 'Hyena'; "
    "class 7 is <b>HYAENIDAE</b> (hyena), not 'Hippo'; class 11 is <b>HIPPOPOTAMIDAE</b>. "
    "Class 4 is <b>NUMIDIDAE</b>, which is the guineafowl, a bird.", B))
S.append(Spacer(1, 4))
S.append(Paragraph(
    "<b>Photo check (the camera frame stored with each event):</b> classes 0 (humans), 1 (elephants), 5 (antelope), "
    "6 (giraffes) and 9 (zebras) are visibly confirmed; classes 2, 4, 7 and 11 are plausible; classes 3 and 8 "
    "(2 and 10 events, night frames) cannot be verified. The old notebook map is right only for classes 0 and 1.", B))
S.append(Spacer(1, 4))
S.append(Paragraph(
    "Caveats. (1) The pickle comes from the authors' <i>dset_allspec_150</i> folder; I apply it to the "
    "<i>chunks_clean</i> file by assumption (class 10 is absent from this file, which fits a cleaned subset). "
    "(2) The <i>image</i> variable is a 128x128 <b>RGB camera-trap photo</b> taken with each seismic event (it is not a "
    "spectrogram, contrary to the earlier notebooks). A photo shows what the camera saw, so a frame can be empty or show "
    "a different animal than the one that triggered the seismic event; read the photos as evidence, not proof. "
    "(3) Family names cover several species, so a family class is a coarser target than 'species'.", small))
S.append(Spacer(1, 8))

rows = [["Class", "Official family", "Typical members (my inference)", "Old notebook name", "Events", "Photo check"]]
for c in classes:
    rows.append([str(c), official.get(c, "?"), Paragraph(MEMBERS.get(official.get(c, ""), ""), small),
                 NOTEBOOK_MAP.get(c, "-"), f"{counts[c]:,}", Paragraph(STATUS.get(c, ""), small)])
t = Table(rows, colWidths=[1.1 * cm, 3.4 * cm, 5.0 * cm, 2.8 * cm, 1.8 * cm, 3.0 * cm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("FONTSIZE", (0, 0), (-1, -1), 8),
                       ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]))
S.append(t)
S.append(Spacer(1, 4))
S.append(Paragraph("Official map has 12 entries (0-11); class 10 = MUSTELIDAE has no events in this file.", small))
S.append(PageBreak())


def add_img(path, width=W):
    from PIL import Image as PI  # installed with matplotlib
    w, h = PI.open(path).size
    S.append(Image(path, width=width, height=width * h / w))


S.append(Paragraph("Overview: one photo per class", H2))
add_img(os.path.join(OUT, "overview_photos.png"))
S.append(Spacer(1, 6))
S.append(Paragraph("Average spectrum per class", H2))
add_img(os.path.join(OUT, "spectra_by_class.png"))
S.append(Paragraph("Each event was RMS-normalised before the FFT so the curves do not simply reflect distance. The class-average spectra overlap closely, so species separation has to come from finer structure (footstep timing, shape), not from the average spectrum.", small))
S.append(PageBreak())

for c in classes:
    S.append(Paragraph(f"Class {c}: {official.get(c, '?')}", H2))
    S.append(Paragraph(
        f"Events: <b>{counts[c]:,}</b> &nbsp;|&nbsp; typical members: {MEMBERS.get(official.get(c, ''), '')} "
        f"&nbsp;|&nbsp; old notebook name: {NOTEBOOK_MAP.get(c, '-')} &nbsp;|&nbsp; distance range: "
        f"{dist[cls == c].min():.0f} to {dist[cls == c].max():.0f} m", small))
    S.append(Paragraph("Six random events from the nearer half of the class (fixed seed), so the animal is more likely "
                       "to be visible.", small))
    note = VISUAL_NOTES.get(c)
    if note:
        S.append(Paragraph("<b>What the photos show:</b> " + note, small))
    add_img(os.path.join(OUT, f"class_{c:02d}_photos.png"), width=W * 0.92)
    S.append(PageBreak())

doc.build(S)
print("PDF written:", os.path.join(OUT, "class_map_report.pdf"))
