# sklearn MUST be imported before torch/matplotlib on this machine: importing it later
# makes its OpenMP runtime (vcomp140.dll) fail with WinError 1114 and kill the process.
import sklearn  # noqa: F401
import matplotlib
matplotlib.use("Agg")   # headless: save figures, never open a window

# ===== notebook code cell 2 =====
import platform, sys
import torch, torchvision

print(f"python        {platform.python_version()}  ({platform.system()} {platform.release()})")
print(f"torch         {torch.__version__}")
print(f"torchvision   {torchvision.__version__}")
_cuda = torch.cuda.is_available()
print(f"CUDA available{'':>1} {_cuda}")
if _cuda:
    _p = torch.cuda.get_device_properties(0)
    print(f"GPU           {_p.name}")
    print(f"total VRAM    {_p.total_memory / 1024**3:.1f} GB")
    print(f"CUDA runtime  {torch.version.cuda}")
else:
    print("!" * 74)
    print("WARNING: no CUDA device found. Training three models for 20 epochs on CPU")
    print("         will take many hours. Enable a GPU runtime before continuing.")
    print("!" * 74)
print(f"selected device {'cuda' if _cuda else 'cpu'}")
print(f"AMP           {'enabled' if _cuda else 'disabled (needs CUDA)'}")
print("batch size    32   (if you hit CUDA out-of-memory, set BATCH_SIZE = 16 below)")

# ===== notebook code cell 4 =====
# ============================ EDIT THIS ONE LINE ============================ #
DATA_PATH = r"C:\Users\Welcome\Desktop\npn\plantvillage_full\raw\color"   # <- the ONLY line to edit
# ============================================================================ #

OUTPUT_DIR    = "outputs_ensemble38"
SEED          = 42
IMG_SIZE      = 224
BATCH_SIZE    = 32          # RTX 3060 12 GB. On CUDA out-of-memory, set this to 16
EPOCHS        = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.05
USE_AMP       = True
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# USE_PRETRAINED = False  -> random init, trained only on the supplied images.
#   This is the default because "use only the supplied images" is normally read as
#   excluding ImageNet weights too.
# USE_PRETRAINED = True   -> ImageNet weights, far higher accuracy. Only set this if
#   your project rules permit external pretrained weights; say so in your report.
USE_PRETRAINED = False

import hashlib, json, os, random, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import importlib, site

# ---- repair the import system before doing anything else ------------------- #
# Python caches a directory listing per sys.path entry. If a scan of site-packages
# fails once (antivirus or a filesystem filter blocking it, seen here as WinError
# 6714), the EMPTY result is cached and every later import from that directory
# raises ModuleNotFoundError even though the package is present. Drop the caches and
# make sure the real site-packages directories are on sys.path, then check they can
# actually be listed - that distinguishes 'not installed' from 'cannot be read'.
importlib.invalidate_caches()
_site_dirs = []
try:
    _site_dirs = list(site.getsitepackages())
except Exception:
    pass
try:
    _site_dirs.append(site.getusersitepackages())
except Exception:
    pass
for _d in _site_dirs:
    if _d and _d not in sys.path and os.path.isdir(_d):
        sys.path.append(_d)
_unreadable = []
for _d in _site_dirs:
    try:
        if _d and os.path.isdir(_d) and not os.listdir(_d):
            _unreadable.append(_d)
    except OSError as _e:
        _unreadable.append(f"{_d} ({_e})")
if _unreadable:
    raise RuntimeError(NL.join([
        "This kernel cannot read its own site-packages directory:",
        *[f"  {_u}" for _u in _unreadable],
        "",
        "Nothing is missing - the import system has cached a failed directory scan,",
        "so installed packages appear absent. A running kernel cannot recover from",
        "this on its own.",
        "",
        "FIX: restart the kernel (VS Code toolbar Restart, or Command Palette ->",
        "     'Jupyter: Restart Kernel'), then Run All. Do NOT pip install anything.",
        "",
        "To stop it recurring, exclude your Python install from real-time antivirus:",
        f"  {sys.prefix}",
    ]))
importlib.invalidate_caches()

NL = chr(10)

# On Windows an import can fail with OSError (not ImportError) when antivirus or a
# filesystem filter briefly blocks the directory scan the import machinery performs -
# seen here as WinError 6714, ERROR_INVALID_TRANSACTION. That is transient and says
# nothing about whether the package is installed, so it is retried with the import
# caches invalidated. Only a genuine ImportError triggers a pip install.
_TRANSIENT_WINERR = {5, 32, 33, 6714}


def _need(mod, pip=None):
    """Import a module, retrying a flaky filesystem, then report clearly.

    No subprocess pip install is attempted. Inside some Jupyter kernels sys.executable
    cannot be launched at all (Windows raises WinError 2), so that path fails and hides
    the real problem. The Jupyter `%pip` magic installs into the RUNNING kernel and is
    the correct tool, so this reports that instead of guessing.
    """
    last = None
    for attempt in range(5):
        try:
            __import__(mod)
            return
        except (ImportError, OSError) as e:
            last = e
            importlib.invalidate_caches()
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(NL.join([
        f"'{mod}' could not be imported in this kernel after 5 attempts.",
        f"Last error: {type(last).__name__}: {last}",
        f"kernel interpreter: {sys.executable!r}",
        f"kernel prefix    : {sys.prefix!r}",
        "",
        "Fix, in order of likelihood:",
        "  1. Install into THIS kernel. Put this in a cell and run it (the % magic",
        "     installs into the running kernel, unlike a plain pip call):",
        "       %pip install numpy pandas pillow scikit-learn torchmetrics ",
        "       %pip install matplotlib seaborn tqdm torch torchvision",
        "     then restart the kernel.",
        "  2. Or switch kernel: click the kernel name (top right) -> Select Another",
        "     Kernel -> Python Environments -> pick the interpreter that already has",
        "     torch and pandas.",
        "  3. If the package IS installed in this interpreter, antivirus is blocking",
        "     reads of the Python install. Exclude it and restart the kernel.",
    ]))


for _m, _p in [("numpy", None), ("pandas", None), ("PIL", "pillow"), ("sklearn", "scikit-learn"),
               ("torch", None), ("torchvision", None), ("torchmetrics", None),
               ("matplotlib", None), ("seaborn", None), ("tqdm", None)]:
    _need(_m, _p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFile
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False        # a truncated file must fail loudly

# Load every Pillow format plugin now. Pillow imports plugins lazily on first use, and on
# Windows an import that lands mid-scan can fail with OSError WinError 6714 (an antivirus
# or filesystem filter interfering with the directory listing). Doing it once here means
# no import happens inside the 54k-image loop, where such a failure would abort the scan.
Image.init()

DATA_DIR = Path(DATA_PATH).expanduser()
if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"DATA_PATH does not exist: {DATA_DIR.resolve()}\n"
        f"Set DATA_PATH to the folder that directly contains the 38 class folders, e.g.\n"
        f"  local  : DATA_PATH = 'plantvillage_full/raw/color'\n"
        f"  Colab  : DATA_PATH = '/content/plantvillage/raw/color'")
_subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
if len(_subdirs) < 2:
    raise NotADirectoryError(
        f"{DATA_DIR.resolve()} holds {len(_subdirs)} sub-folder(s). It should contain one "
        f"folder per class (38 of them). Point DATA_PATH one level deeper or higher.")

OUT_DIR = Path(OUTPUT_DIR)
(OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

def seed_everything(seed):
    """Every reachable source of randomness, so two runs agree."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)

seed_everything(SEED)
GEN = torch.Generator(); GEN.manual_seed(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ON  = USE_AMP and DEVICE.type == "cuda"
PIN     = DEVICE.type == "cuda"
# Windows spawns DataLoader workers as fresh processes that cannot import classes
# defined in a notebook, so workers stay at 0 there.
NUM_WORKERS = 0 if os.name == "nt" else 2
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

print(f"data      {DATA_DIR.resolve()}")
print(f"outputs   {OUT_DIR.resolve()}")
print(f"device    {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"amp       {AMP_ON} | workers {NUM_WORKERS} | seed {SEED}")
print(f"pretrained {USE_PRETRAINED}" + ("  <- ImageNet weights: allowed by your rules?" if USE_PRETRAINED else "  (from scratch)"))
print(f"epochs {EPOCHS} | batch {BATCH_SIZE} | lr {LEARNING_RATE} | img {IMG_SIZE}")

# ===== notebook code cell 6 =====
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def dhash(img, size=8):
    """64-bit perceptual hash: compare each pixel with its right-hand neighbour."""
    px = list(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS).getdata())
    bits = 0
    for r in range(size):
        base = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | int(px[base + c] > px[base + c + 1])
    return f"{bits:016x}"

# Windows sometimes returns a transient OS error for a file that is perfectly fine -
# antivirus or a filesystem filter briefly locking it. Seen here as WinError 6714
# (ERROR_INVALID_TRANSACTION) and its relatives. Treating one of those as "corrupt"
# would silently drop a good image from the dataset, so they are retried first and only
# reported if they persist.
TRANSIENT_WINERR = {5, 32, 33, 6714}      # denied, sharing violation, lock, transaction
RETRIES, RETRY_WAIT = 3, 0.25


def _is_transient(e):
    return isinstance(e, OSError) and getattr(e, "winerror", None) in TRANSIENT_WINERR


def probe(path):
    """Decode once and record identity, perceptual hash and basic properties.

    A genuinely undecodable file is recorded as an error and quarantined. A transient
    OS-level failure is retried, and only recorded as an error if every attempt fails.
    """
    rec = {"path": str(path), "error": None, "sha256": None, "dhash": None,
           "width": None, "height": None, "mode": None, "img_format": None,
           "bytes": None, "retries": 0}
    for attempt in range(RETRIES):
        try:
            raw = Path(path).read_bytes()
            rec["bytes"] = len(raw)
            rec["sha256"] = hashlib.sha256(raw).hexdigest()
            with Image.open(path) as im:
                im.load()                                  # force a real decode
                rec["img_format"] = im.format
                rec["width"], rec["height"] = im.size
                rec["mode"] = im.mode
                rec["dhash"] = dhash(im.convert("RGB"))     # everything is treated as RGB
            rec["error"] = None
            rec["retries"] = attempt
            return rec
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["retries"] = attempt + 1
            if _is_transient(e) and attempt < RETRIES - 1:
                time.sleep(RETRY_WAIT * (attempt + 1))     # back off, then try again
                continue
            break
    return rec

files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
skipped_non_image = sum(1 for p in DATA_DIR.rglob("*")
                        if p.is_file() and p.suffix.lower() not in IMG_EXT)
if not files:
    raise FileNotFoundError(f"no images with extensions {sorted(IMG_EXT)} under {DATA_DIR.resolve()}")

t0 = time.time()
records = [probe(p) for p in tqdm(files, desc="reading", unit="img")]
df = pd.DataFrame(records)
df["class_raw"] = [Path(p).parent.name for p in df["path"]]
df["filename"]  = [Path(p).name for p in df["path"]]
df["relpath"]   = df["class_raw"] + "/" + df["filename"]
_retried = int((df["retries"] > 0).sum()) if "retries" in df else 0
print(f"read {len(df)} files in {time.time() - t0:.0f}s "
      f"({skipped_non_image} non-image files ignored)")
if _retried:
    print(f"NOTE: {_retried} file(s) needed a retry after a transient OS error "
          f"(antivirus or a filesystem filter). They were re-read, not discarded.")

# ===== notebook code cell 7 =====
# ---- status: only "ok" rows are eligible for the split --------------------- #
df["status"], df["status_reason"] = "ok", ""

def exclude(mask, status, reason):
    hit = mask.fillna(False) & (df["status"] == "ok")
    df.loc[hit, ["status", "status_reason"]] = [status, reason]
    print(f"{status:<12} {int(hit.sum()):>6}  {reason}")
    return int(hit.sum())

n_unreadable = exclude(df["error"].notna(), "quarantined", "unreadable or corrupt")

# Before de-duplicating, look for byte-identical files filed under DIFFERENT classes.
# These are label conflicts: the same bytes cannot be two diseases, so one label is wrong.
# Removing them silently would hide the problem, so they are exported first.
_ok = df[df["status"] == "ok"]
_by_sha = _ok.groupby("sha256")["class_raw"].nunique()
_conflict_sha = set(_by_sha[_by_sha > 1].index)
if _conflict_sha:
    _cx = (_ok[_ok["sha256"].isin(_conflict_sha)]
           .sort_values(["sha256", "class_raw", "relpath"])
           [["sha256", "class_raw", "relpath", "path", "dhash", "bytes"]])
    _cx.to_csv(OUT_DIR / "cross_class_exact_duplicates.csv", index=False)
    print(f"POTENTIAL LABEL CONFLICTS: {len(_conflict_sha)} byte-identical image(s) appear "
          f"under more than one class")
    print(f"  ({len(_cx)} files involved). Identical bytes cannot belong to two classes, "
          f"so at least one")
    print(f"  label is wrong. Written to cross_class_exact_duplicates.csv for review.")
    # Keeping one copy would assign whichever label sorts first - wrong about half the
    # time. Training on a knowingly-wrong label is worse than dropping the image, so
    # EVERY copy is quarantined and the conflict stays visible in the audit trail.
    _n_q = exclude(df["sha256"].isin(_conflict_sha), "quarantined",
                   "exact duplicate with conflicting class labels")
    print(f"  QUARANTINED all {_n_q} conflicting copies - none are used for training,")
    print(f"  validation or test. Fix the labels at source and rerun to recover them.")
else:
    print("no byte-identical images with conflicting class labels")

# exact duplicates: keep one deterministically (lowest relpath) so reruns agree
rank = (df[df["status"] == "ok"].sort_values("relpath", kind="mergesort")
        .groupby("sha256").cumcount().reindex(df.index))
n_dupes = exclude(rank > 0, "dropped", "exact duplicate of another file")
assert df.loc[df["status"] == "ok", "sha256"].is_unique, "duplicates survived"

live = df[df["status"] == "ok"]
CLASS_NAMES = sorted(live["class_raw"].unique())
N_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
assert N_CLASSES == 38, f"expected 38 classes, found {N_CLASSES}: {CLASS_NAMES}"

counts = live["class_raw"].value_counts().sort_index()
print(f"\ntotal files scanned : {len(df):,}")
print(f"unreadable          : {n_unreadable:,}")
print(f"exact duplicates    : {n_dupes:,}")
print(f"usable images       : {len(live):,}")
print(f"classes             : {N_CLASSES}")
print(f"imbalance           : {counts.max() / counts.min():.0f}x "
      f"({counts.idxmax()} {counts.max()} vs {counts.idxmin()} {counts.min()})\n")
print("images per class:")
for c, n in counts.items():
    print(f"  {c:<52} {n:>6}")

# ===== notebook code cell 9 =====
import re
UUID_RE  = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___", re.I)
LEAF_DAY = re.compile(r"^(?P<sp>.*?\bLeaf\s*[\d.]+)\s*Day\s*[\d.]+\s*$", re.I)

def source_name(fn):
    """Original capture name, ingest UUID and extension stripped."""
    return UUID_RE.sub("", Path(fn).stem).strip()

def specimen_key(fn):
    """'GHLB_PS Leaf 39.1 Day 16' -> 'ghlb_ps leaf 39.1' : one physical leaf over time."""
    m = LEAF_DAY.match(source_name(fn))
    return m.group("sp").strip().lower() if m else None

df["source_name"] = df["filename"].map(source_name)
df["specimen"]    = df["filename"].map(specimen_key)

class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        lo, hi = sorted((ra, rb))
        self.p[hi] = lo
        return True

# Five bands, not four. Pigeonhole: with H disjoint bands, two hashes differing in at
# most H-1 bits must agree exactly on at least one band. Four bands therefore guarantee
# recall only up to distance 3 - a pair at exactly distance 4 can differ by one bit in
# every band and be missed. Five bands guarantee distance <= 4, which is the threshold
# used here. Bands are 16/12/12/12/12 bits (4/3/3/3/3 hex characters).
BANDS = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)]


def near_dup_pairs(frame, hamming=4):
    """All pairs within dHash Hamming distance `hamming`.

    Banding only proposes candidates; the true Hamming distance is then computed for
    each candidate, so the result is exact for distance <= 4 and never approximate.
    """
    have = frame[frame["dhash"].notna()]
    buckets = defaultdict(list)
    for idx, h in zip(have.index, have["dhash"]):
        for bi, (lo, hi) in enumerate(BANDS):
            buckets[(bi, h[lo:hi])].append(idx)
    hashes, seen, pairs = have["dhash"].to_dict(), set(), []

    # No bucket is ever skipped: skipping one would silently drop the pairs inside it and
    # make the "exact for distance <= 4" claim false. Large buckets are made affordable
    # instead of being discarded. Inside a bucket, images sharing an identical dHash are
    # collapsed first (distance 0 by definition, so every such pair qualifies and is
    # emitted in linear time via a chain), and the quadratic comparison then runs over
    # DISTINCT hashes only. That is exact and typically shrinks the work by orders of
    # magnitude, because identical dHashes are what make a bucket large in the first place.
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        by_hash = defaultdict(list)
        for idx in bucket:
            by_hash[hashes[idx]].append(idx)
        for same in by_hash.values():             # identical hash -> distance 0
            for k in range(1, len(same)):
                a, b = sorted((same[0], same[k]))
                if (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b))
        reps = [(h, idxs[0]) for h, idxs in by_hash.items()]
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                if bin(int(reps[i][0], 16) ^ int(reps[j][0], 16)).count("1") > hamming:
                    continue
                # representatives are close, so every cross pair between the two hash
                # groups is within `hamming` as well
                for x in by_hash[reps[i][0]]:
                    for y in by_hash[reps[j][0]]:
                        a, b = sorted((x, y))
                        if (a, b) not in seen:
                            seen.add((a, b))
                            pairs.append((a, b))
    return pairs


def hamming_of(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

live = df[df["status"] == "ok"]
uf = UnionFind(live.index.tolist())
edges = Counter()

# Unions are restricted to images of the SAME class. Two images in different classes are
# not the same leaf; a filename that collides across classes is a coincidence, and
# merging on it would create groups that cannot be assigned a class-stratified split.
_cls = live["class_raw"].to_dict()
for a, b in near_dup_pairs(live):
    if _cls[a] == _cls[b]:
        edges["near_identical"] += uf.union(a, b)
    else:
        edges["skipped_cross_class"] += 1
for col, name in (("specimen", "same_leaf"), ("source_name", "same_capture")):
    sub = live[live[col].notna()]
    for _k, idx in sub.groupby(["class_raw", col]).groups.items():
        idx = list(idx)
        for other in idx[1:]:
            edges[name] += uf.union(idx[0], other)

roots = {i: uf.find(i) for i in live.index}
order = {r: n for n, r in enumerate(sorted(set(roots.values())))}
df.loc[live.index, "group_key"] = [f"g{order[roots[i]]:06d}" for i in live.index]

n_groups = df.loc[live.index, "group_key"].nunique()
_bmax = 0
_bk = defaultdict(int)
for _i, _h in zip(live.index, live["dhash"]):
    if _h is not None and isinstance(_h, str):
        for _bi, (_lo, _hi) in enumerate(BANDS):
            _bk[(_bi, _h[_lo:_hi])] += 1
_bmax = max(_bk.values()) if _bk else 0
print(f"near-duplicate search: {len(_bk):,} LSH buckets, largest holds {_bmax:,} images "
      f"(no bucket is skipped, so detection is exact for Hamming <= 4)")
print(f"grouping edges: {dict(edges)}")
print(f"{len(live):,} images -> {n_groups:,} groups "
      f"({len(live) - n_groups:,} images share a group with another)")

# a group must not span two classes, or it could not be assigned a class-stratified split
_span = df[df["status"] == "ok"].groupby("group_key")["class_raw"].nunique()
assert (_span == 1).all(), f"{int((_span > 1).sum())} groups span more than one class"

# ===== notebook code cell 11 =====
RATIOS = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
assert abs(sum(RATIOS.values()) - 1.0) < 1e-9, "ratios must sum to 1"

clean = df[df["status"] == "ok"].copy()
groups = (clean.groupby(["class_raw", "group_key"]).size()
          .rename("n_images").reset_index())

def grouped_split(groups):
    """Give each group to whichever split is furthest below its target for that class."""
    order_ = {s: i for i, s in enumerate(RATIOS)}
    out = {}
    for cls, sub in groups.groupby("class_raw", sort=True):
        sub = (sub.sample(frac=1.0, random_state=SEED)
               .sort_values("n_images", ascending=False, kind="mergesort"))
        total = int(sub["n_images"].sum())
        target = {s: total * r for s, r in RATIOS.items()}
        got = dict.fromkeys(RATIOS, 0)
        if len(sub) < len(RATIOS):
            print(f"  warning: {cls} has only {len(sub)} group(s) - a split may be empty")
        for gk, n in zip(sub["group_key"], sub["n_images"]):
            best = min(RATIOS, key=lambda s: (-(target[s] - got[s]), order_[s]))
            out[gk] = best
            got[best] += n
    return out

assign = grouped_split(groups)
clean["split"] = clean["group_key"].map(assign)
df.loc[clean.index, "split"] = clean["split"]

# ---------------------------- leakage gate --------------------------------- #
def leakage_report(frame):
    ev = frame[frame["split"].isin(["val", "test"])]
    tr = frame[frame["split"] == "train"]
    # Capture and specimen keys are compared per class: the same filename under two
    # different classes is a name collision, not the same physical leaf.
    def keyed(frame, col):
        sub = frame[frame[col].notna()]
        return set(zip(sub["class_raw"], sub[col]))
    same_sha  = len(set(ev["sha256"]) & set(tr["sha256"]))
    same_cap  = len(keyed(ev, "source_name") & keyed(tr, "source_name"))
    same_leaf = len(keyed(ev, "specimen")    & keyed(tr, "specimen"))
    grp = frame.groupby("group_key")["split"].nunique()
    return {"identical_files_across_splits": same_sha,
            "same_capture_across_splits": same_cap,
            "same_leaf_across_splits": same_leaf,
            "groups_spanning_splits": int((grp > 1).sum())}

report = leakage_report(clean)
print("leakage after grouped split:")
print(json.dumps(report, indent=2))
for k, v in report.items():
    assert v == 0, f"LEAKAGE: {k} = {v}"

# near-duplicate pairs must not straddle a split either
# Same-class near-duplicates must not straddle a split - that is what inflates a score.
# A cross-class near-duplicate carries a different label, so it cannot inflate anything;
# it is counted separately as a data-quality note.
_sp = clean["split"].to_dict()
_cl = clean["class_raw"].to_dict()
_pairs = near_dup_pairs(clean)
_cross = sum(1 for a, b in _pairs if _cl[a] == _cl[b] and _sp.get(a) != _sp.get(b))
_cross_class = sum(1 for a, b in _pairs if _cl[a] != _cl[b])
assert _cross == 0, f"LEAKAGE: {_cross} same-class near-duplicate pairs cross splits"
# Cross-class near-duplicates are NOT harmless: two near-identical images carrying
# different labels means at least one label is wrong, which caps achievable accuracy.
# They are written out in full so they can be inspected.
_cc = [(a, b) for a, b in _pairs if _cl[a] != _cl[b]]
if _cc:
    _rows = [{"path_a": clean.loc[a, "path"], "class_a": clean.loc[a, "class_raw"],
              "sha256_a": clean.loc[a, "sha256"], "dhash_a": clean.loc[a, "dhash"],
              "split_a": clean.loc[a, "split"],
              "path_b": clean.loc[b, "path"], "class_b": clean.loc[b, "class_raw"],
              "sha256_b": clean.loc[b, "sha256"], "dhash_b": clean.loc[b, "dhash"],
              "split_b": clean.loc[b, "split"],
              "hamming": hamming_of(clean.loc[a, "dhash"], clean.loc[b, "dhash"])}
             for a, b in _cc]
    pd.DataFrame(_rows).sort_values("hamming").to_csv(
        OUT_DIR / "cross_class_near_duplicates.csv", index=False)
    print(f"POTENTIAL LABEL CONFLICTS: {len(_cc)} near-duplicate pairs carry DIFFERENT "
          f"class labels.")
    print(f"  Two near-identical images with different labels means at least one label "
          f"is wrong.")
    print(f"  They cannot inflate a score (the labels disagree) but they do cap the "
          f"accuracy any")
    print(f"  model can reach. Written to cross_class_near_duplicates.csv for review.")
else:
    print("no cross-class near-duplicate pairs found")

# every class present in every split, and 38 classes exactly
tab = pd.crosstab(clean["class_raw"], clean["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
assert clean["class_raw"].nunique() == 38, "class count changed after splitting"
missing = tab[(tab == 0).any(axis=1)]
assert missing.empty, f"class missing from a split:\n{missing}"

# no test image can appear in training
assert not (set(clean.loc[clean.split == "test", "path"]) &
            set(clean.loc[clean.split == "train", "path"])), "test image found in train"

print("\nALL LEAKAGE CHECKS PASSED\n")
counts_split = clean["split"].value_counts().reindex(["train", "val", "test"])
for s, n in counts_split.items():
    print(f"  {s:<6} {n:>7,}  ({n / len(clean) * 100:.1f}%)")
print("\nper-class counts:")
print(tab.to_string())

# ===== notebook code cell 12 =====
# ---- persist everything, then never re-split ------------------------------- #
COLS = ["relpath", "path", "filename", "split", "class_raw", "group_key", "sha256",
        "dhash", "source_name", "specimen", "width", "height", "mode", "img_format",
        "bytes", "status", "status_reason", "error"]
manifest = df.reindex(columns=COLS).sort_values(["split", "class_raw", "relpath"],
                                                na_position="last").reset_index(drop=True)
manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
manifest[manifest["status"] == "ok"].to_csv(OUT_DIR / "manifest_clean.csv", index=False)
manifest[manifest["status"] != "ok"].to_csv(OUT_DIR / "excluded_images.csv", index=False)

taxonomy = pd.DataFrame({"class_id": range(N_CLASSES), "class_raw": CLASS_NAMES})
taxonomy["crop"]    = [c.split("___")[0] for c in CLASS_NAMES]
taxonomy["disease"] = [c.split("___")[-1] for c in CLASS_NAMES]
taxonomy["n_images"] = [int(counts[c]) for c in CLASS_NAMES]
taxonomy.to_csv(OUT_DIR / "taxonomy.csv", index=False)

fix_report = {
    "data_path": str(DATA_DIR.resolve()),
    "images_scanned": int(len(df)),
    "unreadable": int(n_unreadable),
    "exact_duplicates_removed": int(n_dupes),
    "images_usable": int(len(clean)),
    "classes": int(N_CLASSES),
    "groups": int(n_groups),
    "grouping_edges": {k: int(v) for k, v in edges.items()},
    "ratios": RATIOS,
    "seed": SEED,
    "split_counts": {s: int(n) for s, n in counts_split.items()},
    "leakage_after_split": report,
    "near_duplicate_pairs_crossing_splits": int(_cross),
    "cross_class_exact_duplicate_hashes": int(len(_conflict_sha)),
    "cross_class_exact_duplicate_files_quarantined": int(
        (df["status_reason"] == "exact duplicate with conflicting class labels").sum()),
    "cross_class_near_duplicate_pairs": int(_cross_class),
    "verdict": "PASSED",
}
(OUT_DIR / "fix_report.json").write_text(json.dumps(fix_report, indent=2))
print("saved manifest.csv, manifest_clean.csv, excluded_images.csv, taxonomy.csv, fix_report.json")

# the frozen split - everything downstream reads these three frames
parts = {s: clean[clean["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

# ===== notebook code cell 14 =====
train_counts = parts["train"]["class_raw"].value_counts()
n_train = len(parts["train"])
class_weights = {c: n_train / (N_CLASSES * int(train_counts[c])) for c in CLASS_NAMES}
(OUT_DIR / "class_weights.json").write_text(json.dumps(class_weights, indent=2))

CLASS_WEIGHTS = torch.tensor([class_weights[c] for c in CLASS_NAMES],
                             dtype=torch.float32, device=DEVICE)

# A weighted sampler is deliberately NOT used. Combining it with a weighted loss
# corrects the same imbalance twice: rare classes would be both oversampled and given a
# larger per-sample loss, which encourages memorising the few images those classes have.
# The weighted loss alone is the single correction; minority classes additionally get
# stronger augmentation (section E), which adds variety rather than repetition.
print(f"class weights from {n_train:,} TRAIN images only (val/test never consulted)")
print("imbalance handled by the weighted loss ONLY (no WeightedRandomSampler)")
_ext = sorted(class_weights.items(), key=lambda kv: kv[1])
print(f"  lowest  weight: {_ext[0][0]} = {_ext[0][1]:.3f}  ({train_counts[_ext[0][0]]} imgs)")
print(f"  highest weight: {_ext[-1][0]} = {_ext[-1][1]:.3f}  ({train_counts[_ext[-1][0]]} imgs)")
print("saved class_weights.json")

# ===== notebook code cell 16 =====
MINORITY_THRESHOLD = 500
MINORITY = {c for c in CLASS_NAMES if int(train_counts[c]) < MINORITY_THRESHOLD}

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

standard_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

strong_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.03),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.2))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.15)),
])


class LeafDataset(Dataset):
    """Reads images listed in the manifest. `train=True` picks the per-class policy."""

    def __init__(self, frame, train):
        self.samples = [(r.path, CLASS_TO_IDX[r.class_raw], r.class_raw in MINORITY)
                        for r in frame.itertuples(index=False)]
        self.train = train
        self.classes = list(CLASS_NAMES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, is_minority = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")             # every image becomes RGB
        if not self.train:
            return eval_tf(img), label
        return (strong_tf if is_minority else standard_tf)(img), label


def loader(frame, train, bs=None, generator=None):
    """Training loaders shuffle; validation and test never do, so their order is fixed
    and predictions line up with the manifest rows."""
    return DataLoader(LeafDataset(frame, train), batch_size=bs or BATCH_SIZE,
                      shuffle=train, num_workers=NUM_WORKERS, pin_memory=PIN,
                      worker_init_fn=seed_worker, generator=generator)

# Validation and test loaders are built once and shared - they are deterministic.
# The training loader is rebuilt per model in section F with a fresh generator.
val_loader  = loader(parts["val"],  False, bs=64)
test_loader = loader(parts["test"], False, bs=64)

print(f"minority classes (<{MINORITY_THRESHOLD} train images): {len(MINORITY)}")
for c in sorted(MINORITY, key=lambda c: train_counts[c]):
    print(f"  {c:<52} {int(train_counts[c]):>5}  -> strong augmentation")
print(f"\ntrain {len(parts['train']):,} (augmented, class-weighted loss) | "
      f"val {len(parts['val']):,} | test {len(parts['test']):,} (both eval transform only)")

# ===== notebook code cell 18 =====
def build_model(name, n_classes, pretrained):
    """38-class head on the chosen architecture. GoogLeNet's aux classifiers are removed
    so the forward pass returns a plain tensor in train mode."""
    weights = "DEFAULT" if pretrained else None
    if name == "googlenet":
        m = torchvision.models.googlenet(weights=weights, aux_logits=bool(pretrained),
                                         init_weights=not pretrained)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        if getattr(m, "aux_logits", False):
            m.aux_logits, m.aux1, m.aux2 = False, None, None
    elif name == "resnet18":
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "densenet121":
        m = torchvision.models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    else:
        raise KeyError(name)
    for p in m.parameters():
        p.requires_grad = True                       # all layers trainable
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE))
    out = out if torch.is_tensor(out) else out[0]
    assert tuple(out.shape) == (2, n_classes), f"{name} head wrong: {tuple(out.shape)}"
    return m


def run_epoch(model, dl, criterion, optimizer=None, desc=""):
    """One pass. Optimizer given -> train; omitted -> evaluate under no_grad."""
    training = optimizer is not None
    model.train(training)
    tot_loss = seen = 0
    preds, targets = [], []
    scaler = run_epoch.scaler
    for x, y in tqdm(dl, desc=desc, leave=False, unit="b"):
        x = x.to(DEVICE, non_blocking=PIN); y = y.to(DEVICE, non_blocking=PIN)
        with torch.set_grad_enabled(training):
            with torch.autocast(DEVICE.type, enabled=AMP_ON):
                out = model(x)
                out = out if torch.is_tensor(out) else out[0]
                loss = criterion(out, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        tot_loss += loss.item() * len(y); seen += len(y)
        preds.append(out.argmax(1).detach().cpu()); targets.append(y.cpu())
    preds = torch.cat(preds).numpy(); targets = torch.cat(targets).numpy()
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="macro",
                                                  zero_division=0)
    return {"loss": tot_loss / seen, "accuracy": accuracy_score(targets, preds),
            "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "n": seen, "preds": preds, "targets": targets}


def train_model(name):
    # Every model must start from identical randomness. Reusing one generator would mean
    # the second and third models inherit a stream already advanced by the first, so
    # their batch order - and their weight init - would differ for reasons unrelated to
    # architecture. Seeds are reset and a fresh generator and train loader are built here.
    seed_everything(SEED)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_loader = loader(parts["train"], True, generator=gen)
    model = build_model(name, N_CLASSES, USE_PRETRAINED).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)   # watches VAL macro-F1
    try:                                     # torch >= 2.4 signature
        run_epoch.scaler = torch.amp.GradScaler("cuda", enabled=AMP_ON)
    except TypeError:                            # older torch
        run_epoch.scaler = torch.cuda.amp.GradScaler(enabled=AMP_ON)

    ckpt_path = OUT_DIR / "checkpoints" / f"best_{name}.pth"
    best = {"epoch": 0, "val_macro_f1": -1.0}
    rows, t_start = [], time.time()
    print(f"\n=== {name}  ({n_par:,} params, pretrained={USE_PRETRAINED}) ===")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, f"{name} train {epoch}")
        va = run_epoch(model, val_loader,   criterion, None,      f"{name} val {epoch}")
        scheduler.step(va["macro_f1"])           # validation only
        rows.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                     "seconds": round(time.time() - t0, 1),
                     "train_loss": tr["loss"], "train_acc": tr["accuracy"],
                     "train_macro_precision": tr["macro_precision"],
                     "train_macro_recall": tr["macro_recall"],
                     "train_macro_f1": tr["macro_f1"],
                     "val_loss": va["loss"], "val_acc": va["accuracy"],
                     "val_macro_precision": va["macro_precision"],
                     "val_macro_recall": va["macro_recall"],
                     "val_macro_f1": va["macro_f1"]})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"log_{name}.csv", index=False)
        star = ""
        if va["macro_f1"] > best["val_macro_f1"]:      # save on improvement only
            best = {"epoch": epoch, "val_macro_f1": va["macro_f1"]}
            torch.save({"state_dict": model.state_dict(), "arch": name,
                        "class_names": CLASS_NAMES, "img_size": IMG_SIZE,
                        "normalize_mean": MEAN, "normalize_std": STD,
                        "pretrained": USE_PRETRAINED, "seed": SEED, "epoch": epoch,
                        "val_macro_f1": va["macro_f1"], "params": n_par,
                        "split_counts": {s: len(parts[s]) for s in parts},
                        "ratios": RATIOS}, ckpt_path)
            star = "  <- best"
        print(f"epoch {epoch:>2}/{EPOCHS}  train f1 {tr['macro_f1']:.4f} acc {tr['accuracy']:.4f}"
              f" | val f1 {va['macro_f1']:.4f} acc {va['accuracy']:.4f}{star}")

    log = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(log.epoch, log.train_loss, label="train"); ax[0].plot(log.epoch, log.val_loss, label="val")
    ax[0].set_title(f"{name} loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(log.epoch, log.train_acc, label="train"); ax[1].plot(log.epoch, log.val_acc, label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(log.epoch, log.train_macro_f1, label="train"); ax[2].plot(log.epoch, log.val_macro_f1, label="val")
    ax[2].axvline(best["epoch"], ls=":", c="k", label=f"best ep {best['epoch']}")
    ax[2].set_title("macro-F1"); ax[2].set_xlabel("epoch"); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "figures" / f"curves_{name}.png", dpi=120)
    plt.show()
    print(f"{name}: best epoch {best['epoch']}, val macro-F1 {best['val_macro_f1']:.4f}, "
          f"{(time.time() - t_start)/60:.1f} min")
    return best


MODELS = ["densenet121", "resnet18", "googlenet"]
print("skipping training - using saved checkpoints")


# ===== notebook code cell 20 =====
def load_best(name):
    ck = torch.load(OUT_DIR / "checkpoints" / f"best_{name}.pth",
                    map_location=DEVICE, weights_only=True)
    model = build_model(name, N_CLASSES, pretrained=False)   # architecture only
    model.load_state_dict(ck["state_dict"])
    return model.to(DEVICE).eval(), ck


def probabilities(model, dl):
    """Softmax probabilities for every image in a loader, in loader order."""
    P, Y = [], []
    with torch.no_grad():
        for x, y in tqdm(dl, desc="predict", leave=False, unit="b"):
            with torch.autocast(DEVICE.type, enabled=AMP_ON):
                out = model(x.to(DEVICE, non_blocking=PIN))
            out = out if torch.is_tensor(out) else out[0]
            P.append(out.float().softmax(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


BEST = {}
val_probs, val_y = {}, None
for name in MODELS:
    model, ck = load_best(name)
    BEST[name] = ck
    val_probs[name], y = probabilities(model, val_loader)
    val_y = y if val_y is None else val_y
    f1 = precision_recall_fscore_support(val_y, val_probs[name].argmax(1),
                                         average="macro", zero_division=0)[2]
    print(f"{name:<13} val macro-F1 {f1:.4f}  (best epoch {ck['epoch']})")
    del model; torch.cuda.empty_cache()

# grid over weights that sum to 1; googlenet takes whatever is left
grid = []
for wd in (0.4, 0.5, 0.6):
    for wr in (0.2, 0.3, 0.4):
        wg = round(1.0 - wd - wr, 4)
        if wg >= 0.05:
            grid.append((wd, wr, wg))

results = []
for wd, wr, wg in grid:
    mix = wd * val_probs["densenet121"] + wr * val_probs["resnet18"] + wg * val_probs["googlenet"]
    f1 = precision_recall_fscore_support(val_y, mix.argmax(1), average="macro",
                                         zero_division=0)[2]
    results.append({"densenet121": wd, "resnet18": wr, "googlenet": wg, "val_macro_f1": f1})

grid_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
print(f"\n{len(grid_df)} weight combinations, ranked by VALIDATION macro-F1:")
print(grid_df.head(10).to_string(index=False))

BEST_W = {k: float(grid_df.iloc[0][k]) for k in MODELS}
ENSEMBLE_VAL_F1 = float(grid_df.iloc[0]["val_macro_f1"])

# confidence threshold, also validation-only: 5th percentile of confidence on the
# validation images the ensemble gets right
_mix_val = sum(BEST_W[m] * val_probs[m] for m in MODELS)
_ok = _mix_val.argmax(1) == val_y
CONF_THRESHOLD = float(np.percentile(_mix_val.max(1)[_ok], 5))

json.dump({"weights": BEST_W, "val_macro_f1": ENSEMBLE_VAL_F1,
           "confidence_threshold": CONF_THRESHOLD,
           "selected_on": "validation split only",
           "class_names": CLASS_NAMES, "img_size": IMG_SIZE,
           "normalize_mean": MEAN, "normalize_std": STD,
           "models": MODELS, "pretrained": USE_PRETRAINED},
          open(OUT_DIR / "ensemble_config.json", "w"), indent=2)
grid_df.to_csv(OUT_DIR / "ensemble_weight_search.csv", index=False)

print(f"\nselected weights: {BEST_W}")
print(f"ensemble VAL macro-F1: {ENSEMBLE_VAL_F1:.4f}")
print(f"confidence threshold (5th pct of correct val predictions): {CONF_THRESHOLD:.4f}")
print("saved ensemble_config.json — test data has not been touched yet")

# ===== notebook code cell 22 =====
def wilson_ci(k, n, z=1.96):
    """Wilson score interval. The normal approximation collapses to +/-0 when recall is
    exactly 1.0, claiming a certainty that 26/26 does not support."""
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


test_probs, test_y = {}, None
for name in MODELS:
    model, _ = load_best(name)
    test_probs[name], y = probabilities(model, test_loader)
    test_y = y if test_y is None else test_y
    del model; torch.cuda.empty_cache()

ens_test = sum(BEST_W[m] * test_probs[m] for m in MODELS)
ALL_PROBS = {**test_probs, "ensemble": ens_test}

crit_eval = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTH)
rows, per_class_rows = [], []
for name, P in ALL_PROBS.items():
    pred = P.argmax(1)
    loss = float(crit_eval(torch.log(torch.tensor(P).clamp_min(1e-12)).to(DEVICE),
                           torch.tensor(test_y).to(DEVICE)).item())
    p, r, f1, _ = precision_recall_fscore_support(test_y, pred, average="macro", zero_division=0)
    rows.append({"model": name, "loss": round(loss, 6),
                 "accuracy": accuracy_score(test_y, pred),
                 "macro_precision": p, "macro_recall": r, "macro_f1": f1})
    pc = precision_recall_fscore_support(test_y, pred, labels=range(N_CLASSES), zero_division=0)
    for i, c in enumerate(CLASS_NAMES):
        n_i = int(pc[3][i]); rec = float(pc[1][i])
        lo, hi = wilson_ci(round(rec * n_i), n_i)
        per_class_rows.append({"model": name, "class": c, "support": n_i,
                               "precision": float(pc[0][i]), "recall": rec,
                               "f1": float(pc[2][i]),
                               "recall_ci95_lo": round(lo, 4), "recall_ci95_hi": round(hi, 4)})

comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
per_class = pd.DataFrame(per_class_rows)
comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
per_class.to_csv(OUT_DIR / "per_class_test_metrics.csv", index=False)

print("TEST results (unseen held-out images, same 38 classes):\n")
print(comparison.round(4).to_string(index=False))

print("\n\nclassification report - ensemble:")
print(classification_report(test_y, ens_test.argmax(1), labels=range(N_CLASSES),
                            target_names=CLASS_NAMES, digits=4, zero_division=0))

# ===== notebook code cell 23 =====
# ---- comparison plot ------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(comparison))
ax.bar(x - 0.2, comparison["accuracy"], 0.4, label="accuracy")
ax.bar(x + 0.2, comparison["macro_f1"], 0.4, label="macro-F1")
ax.set_xticks(x); ax.set_xticklabels(comparison["model"], rotation=15)
ax.set_ylim(min(0.9 * comparison[["accuracy", "macro_f1"]].min().min(), 0.95), 1.001)
ax.set_title("Test performance"); ax.legend(); ax.grid(axis="y", alpha=0.3)
for i, (a, f) in enumerate(zip(comparison["accuracy"], comparison["macro_f1"])):
    ax.text(i - 0.2, a, f"{a*100:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.2, f, f"{f*100:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(OUT_DIR / "figures" / "model_comparison.png", dpi=120)
plt.show()

# ---- confusion matrices, counts and row-normalised ------------------------- #
SHORT = [c.replace("___", " ").replace("_", " ")[:26] for c in CLASS_NAMES]
cm = confusion_matrix(test_y, ens_test.argmax(1), labels=range(N_CLASSES))
for norm, tag in ((False, "counts"), (True, "row-normalised %")):
    data = cm.astype(float)
    if norm:
        data = 100 * data / np.maximum(data.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(data, ax=ax, cmap="Blues", cbar=False, square=True, annot=True,
                fmt=".0f", annot_kws={"size": 5},
                xticklabels=SHORT, yticklabels=SHORT, vmin=0, vmax=100 if norm else None)
    ax.set_title(f"Ensemble - test confusion matrix ({tag})")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / f"confusion_test_{'norm' if norm else 'counts'}.png", dpi=110)
    plt.show()

# ---- weakest classes and why their intervals are wide ---------------------- #
ens_pc = per_class[per_class.model == "ensemble"].sort_values("recall")
print("lowest-recall classes on test (ensemble):\n")
print(f"{'class':<50}{'support':>8}{'recall':>9}   95% CI")
for _i, r in ens_pc.head(8).iterrows():
    print(f"{r['class'][:48]:<50}{r['support']:>8}{r['recall']*100:>8.1f}%   "
          f"[{r['recall_ci95_lo']*100:.1f}%, {r['recall_ci95_hi']*100:.1f}%]")
_thin = ens_pc[ens_pc.support < 60]
if len(_thin):
    print(f"\n{len(_thin)} class(es) have fewer than 60 test images. Recall measured on a "
          f"small support\nis imprecise: with n=26 a single extra mistake moves recall by "
          f"~4 points, which is why\nthe Wilson intervals above are wide. Quote the "
          f"interval, and do not read a change inside\nit as an improvement.")

# ===== notebook code cell 24 =====
# ---- example correct and incorrect test predictions ------------------------ #
test_paths = parts["test"]["path"].tolist()
pred_ens = ens_test.argmax(1)
conf_ens = ens_test.max(1)
assert len(test_paths) == len(test_y), "loader order and manifest order disagree"

def grid(idx, title, fname):
    idx = list(idx)[:12]
    if not idx:
        print(f"{title}: none"); return
    cols = 4; rows_ = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 3.7 * rows_))
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        with Image.open(test_paths[i]) as im:
            ax.imshow(im.convert("RGB"))
        good = test_y[i] == pred_ens[i]
        ax.set_title(f"true {SHORT[test_y[i]]}\npred {SHORT[pred_ens[i]]} ({conf_ens[i]*100:.0f}%)",
                     fontsize=7, color="green" if good else "red")
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / fname, dpi=100); plt.show()

rng = np.random.default_rng(SEED)
correct_idx = np.flatnonzero(pred_ens == test_y)
wrong_idx   = np.flatnonzero(pred_ens != test_y)
grid(rng.permutation(correct_idx), f"Correct predictions ({len(correct_idx)} of {len(test_y)})",
     "examples_correct.png")
grid(wrong_idx[np.argsort(-conf_ens[wrong_idx])] if len(wrong_idx) else [],
     f"Incorrect predictions ({len(wrong_idx)} of {len(test_y)}) - most confident first",
     "examples_incorrect.png")

# ===== notebook code cell 26 =====
PHOTO_DIR = Path("my_photos")
PHOTO_DIR.mkdir(exist_ok=True)

cfg = json.loads((OUT_DIR / "ensemble_config.json").read_text())
W_INF, THRESH = cfg["weights"], cfg["confidence_threshold"]
INF_MODELS = {name: load_best(name)[0] for name in MODELS}
print(f"loaded {len(INF_MODELS)} models | weights {W_INF} | threshold {THRESH:.3f}")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
photos = sorted(p for p in PHOTO_DIR.iterdir()
                if p.suffix.lower() in exts and not p.name.startswith("prediction"))
if not photos:
    print(f"\nNo photos found. Copy images into {PHOTO_DIR.resolve()} and rerun this cell.")

for f in photos:
    try:
        with Image.open(f) as im:
            img = im.convert("RGB")
    except Exception as e:
        print(f"{f.name}: not a readable image ({e})"); continue

    x = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = np.zeros(N_CLASSES, dtype=np.float64)
        for name, model in INF_MODELS.items():
            out = model(x)
            out = out if torch.is_tensor(out) else out[0]
            prob += W_INF[name] * out.float().softmax(1)[0].cpu().numpy()
    prob /= prob.sum()
    top5 = np.argsort(prob)[::-1][:5]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax[0].imshow(img); ax[0].axis("off"); ax[0].set_title(f.name[:40])
    ax[1].barh([SHORT[i] for i in top5][::-1], prob[top5][::-1],
               color=["#c7d9f1"] * 4 + ["#1f6fb4"])
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("probability")
    ax[1].set_title(f"ensemble: {CLASS_NAMES[top5[0]]} ({prob[top5[0]]*100:.1f}%)")
    for i, v in enumerate(prob[top5][::-1]):
        ax[1].text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=9)
    fig.tight_layout(); plt.show()

    print(f"{f.name}")
    for rank, i in enumerate(top5, 1):
        print(f"   {rank}. {CLASS_NAMES[i]:<52} {prob[i]*100:6.2f}%")
    if prob[top5[0]] >= THRESH:
        print(f"   PREDICTION: {CLASS_NAMES[top5[0]]}  ({prob[top5[0]]*100:.2f}%)\n")
    else:
        print(f"   UNCERTAIN: this image may not resemble the supported 38-class dataset.")
        print(f"   (best guess {CLASS_NAMES[top5[0]]} at {prob[top5[0]]*100:.2f}%, "
              f"below the {THRESH*100:.1f}% threshold)\n")

# ===== notebook code cell 28 =====
best_single = comparison[comparison.model != "ensemble"].iloc[0]
ens_row = comparison[comparison.model == "ensemble"].iloc[0]
lowest = ens_pc.head(5)

L = []
A = L.append
A("=" * 78)
A("  FINAL REPORT - 38-class closed-set crop disease classifier")
A("=" * 78)
A(f"Dataset source        : provided PlantVillage 38-class data only ({DATA_DIR.resolve()})")
A(f"                        no external datasets were downloaded or used")
A(f"Images scanned        : {fix_report['images_scanned']:,}")
A(f"Unreadable removed    : {fix_report['unreadable']:,}")
A(f"Exact duplicates      : {fix_report['exact_duplicates_removed']:,}")
A(f"Final usable images   : {fix_report['images_usable']:,}  in {fix_report['groups']:,} leakage groups")
A(f"Classes               : {N_CLASSES}")
A("")
A(f"Train / Val / Test    : {len(parts['train']):,} / {len(parts['val']):,} / {len(parts['test']):,}"
  f"   ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}, grouped + stratified, seed {SEED})")
A("")
A("Leakage checks (all must be zero):")
for k, v in fix_report["leakage_after_split"].items():
    A(f"   {k:<38} {v}")
A(f"   {'near_duplicate_pairs_crossing_splits':<38} {fix_report['near_duplicate_pairs_crossing_splits']}")
A("")
A("Training configuration:")
A(f"   epochs {EPOCHS} | batch {BATCH_SIZE} | img {IMG_SIZE} | AdamW lr={LEARNING_RATE} wd={WEIGHT_DECAY}")
A(f"   weighted CrossEntropyLoss (label smoothing {LABEL_SMOOTH}); no weighted sampler")
A(f"   shuffled training loader, fresh generator seeded {SEED} per model")
A(f"   ReduceLROnPlateau on VAL macro-F1 (factor 0.5, patience 3) | AMP {AMP_ON}")
A(f"   stronger augmentation for {len(MINORITY)} classes with <{MINORITY_THRESHOLD} train images")
A(f"Pretrained weights    : {USE_PRETRAINED}"
  + ("  (ImageNet - confirm your project rules allow this)" if USE_PRETRAINED
     else "  (trained from scratch on the supplied images only)"))
A("")
A("Per-model best validation macro-F1 (used for checkpoint selection):")
for name in MODELS:
    A(f"   {name:<14} epoch {BEST[name]['epoch']:>2}   val macro-F1 {BEST[name]['val_macro_f1']:.4f}")
A(f"Best individual model : {best_single['model']}  (test macro-F1 {best_single['macro_f1']:.4f})")
A("")
A(f"Ensemble weights      : " + ", ".join(f"{k}={v:.2f}" for k, v in BEST_W.items()))
A(f"   selected on the VALIDATION split only; test data was not loaded until section H")
A(f"Ensemble val macro-F1 : {ENSEMBLE_VAL_F1:.4f}")
A(f"Ensemble TEST macro-F1: {ens_row['macro_f1']:.4f}   (accuracy {ens_row['accuracy']:.4f})")
A(f"Confidence threshold  : {CONF_THRESHOLD:.4f}  (5th percentile of correct validation predictions)")
A("")
A("Lowest per-class test recall (ensemble):")
for _i, r in lowest.iterrows():
    A(f"   {r['class'][:46]:<48} n={r['support']:<5} recall {r['recall']*100:5.1f}%"
      f"  95% CI [{r['recall_ci95_lo']*100:.1f}%, {r['recall_ci95_hi']*100:.1f}%]")
A("")
A("Possible label conflicts found in the supplied data:")
A(f"   byte-identical images under >1 class : {fix_report['cross_class_exact_duplicate_hashes']}"
  f"   (cross_class_exact_duplicates.csv)")
A(f"      -> {fix_report['cross_class_exact_duplicate_files_quarantined']} file(s) quarantined; no arbitrary label kept")
A(f"   near-duplicate pairs, differing label: {fix_report['cross_class_near_duplicate_pairs']}"
  f"   (cross_class_near_duplicates.csv)")
A("   These cannot inflate a score, but they cap the accuracy any model can reach.")
A("")
A("Limitations:")
A("   The model is evaluated on unseen held-out images from the same 38 supported")
A("   classes and dataset distribution. It does not claim to classify diseases outside")
A("   the 38 classes.")
A("   * Closed set: every input is assigned one of 38 labels. The confidence threshold")
A("     flags unlikely inputs but is not an open-set detector.")
A("   * Small classes carry wide recall intervals - quote the interval, not the point.")
A("   * PlantVillage is uniform studio photography of single detached leaves. Accuracy")
A("     on field photographs taken in other conditions will be substantially lower.")
A("=" * 78)

summary = "\n".join(L)
print(summary)
(OUT_DIR / "final_report.txt").write_text(summary, encoding="utf-8")
print(f"\nsaved {OUT_DIR / 'final_report.txt'}")
print(f"all artefacts in {OUT_DIR.resolve()}")
