#!/usr/bin/env python3
"""Generate NutriScan_Milestone2.ipynb (image classification). Run once."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# NutriScan: AI-Powered Food Detection & Classification
### Capstone Project (CV) | Milestone 2: Image Classification - Base Model, Improvements and Final Model Selection

**Business context.** NutriScan lets a user photograph a meal and get its calories and macronutrients.
Milestone 1 located the food in each photo (zero-shot OWL-ViT detector), cropped it to 224x224 and
built a clean, leakage-free train / validation / test split across **17 food categories**.

**Goal of this milestone.** Build the model that *names* the dish in each crop:

1. A **base CNN** trained from scratch, to set a reference.
2. **Improvements**: augmentation, regularisation (BatchNorm, dropout, weight decay, label smoothing),
   architectural changes and **transfer learning** (MobileNetV2, EfficientNet-B0).
3. **Compare** all models on the validation set, **select** the best with a clear rationale, and
   evaluate it **once** on the untouched test set.
4. **Classify 2-3 test images** with the best model and interpret the result in business terms.

**Metrics.** Accuracy, **macro-F1** (every food counts equally, as decided in Milestone 1), and
**top-3 accuracy** (the app can show the user three suggestions to confirm with one tap).
""")

md("""
## How to run
- **Google Colab:** Runtime > Change runtime type > **T4 GPU**, then Runtime > Run all.
  The notebook looks for the Milestone 1 output `NutriScan_M1/nutriscan_classification_dataset.zip`
  in your Google Drive. If it is not there, it builds an equivalent dataset from `Food_101.zip`
  (centre crop, stratified 70/15/15 split) so it always runs.
- **Local:** uses `outputs/classification_dataset/` produced by Milestone 1.
- Every setting is in the **Configuration** cell.

# 1. Setup
""")

code(r'''
# Core libraries. Everything used here ships with Google Colab - no installs needed.
import os, sys, json, time, copy, random, shutil, zipfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2 as T

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")          # keep the submitted notebook clean
sns.set_style("whitegrid")
print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
''')

md("## Configuration")
code(r'''
IN_COLAB = "google.colab" in sys.modules
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=SEED):
    """Make every run reproducible (same split, same initial weights, same shuffles)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything()

# ---- Paths ------------------------------------------------------------------
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    M1_ZIP   = Path("/content/drive/MyDrive/NutriScan_M1/nutriscan_classification_dataset.zip")
    RAW_ZIP  = Path("/content/drive/MyDrive/Food_101.zip")      # fallback source
    DATA_DIR = Path("/content/m2_data")
    OUT_DIR  = Path("/content/drive/MyDrive/NutriScan_M2")
else:
    M1_ZIP, RAW_ZIP = None, None
    DATA_DIR = Path.cwd() / "outputs" / "classification_dataset"   # Milestone 1 output
    OUT_DIR  = Path.cwd() / "outputs_m2"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Training settings --------------------------------------------------------
IMG_SIZE    = 160 if DEVICE == "cuda" else 96   # input resolution (smaller on a CPU-only machine to keep run time short)
BATCH_SIZE  = 32
NUM_WORKERS = 2 if IN_COLAB else 0
# Epochs per experiment (early stopping keeps the best epoch). A GPU (Colab) gets the full budget.
EPOCHS = ({"base_cnn": 15, "cnn_aug_reg": 25, "mobilenet_frozen": 8, "mobilenet_finetune": 8, "effnet_finetune": 8}
          if DEVICE == "cuda" else
          {"base_cnn": 6, "cnn_aug_reg": 8, "mobilenet_frozen": 3, "mobilenet_finetune": 3, "effnet_finetune": 3})
PATIENCE = 6               # early-stopping patience (epochs without val macro-F1 improvement)
FALLBACK_PER_CLASS = 300   # only used if the Milestone 1 dataset is not found on Colab
print("Device:", DEVICE, "| Data:", DATA_DIR, "| Outputs:", OUT_DIR)
''')

md("""
# 2. Loading the Milestone 1 dataset
The detected, cropped and split images from Milestone 1 are reused, so this milestone directly builds on
the earlier analysis. The split was made **before** any modelling and was checked for duplicate leakage.
""")
code(r'''
def build_from_raw(raw_zip, out_dir, per_class=FALLBACK_PER_CLASS):
    """Fallback: create train/val/test folders from the raw Food_101.zip (centre crop, 70/15/15)."""
    raw_dir = Path("/content/food101_raw")
    if not any(raw_dir.rglob("*.jpg")):
        with zipfile.ZipFile(raw_zip) as z: z.extractall(raw_dir)
    files = [p for p in raw_dir.rglob("*.jp*g") if "__MACOSX" not in p.parts]
    df = pd.DataFrame({"path": files, "label": [p.parent.name for p in files]})
    df = df.sample(frac=1, random_state=SEED).groupby("label").head(per_class)
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    for split, d in [("train", tr), ("val", va), ("test", te)]:
        for p, lab in zip(d.path, d.label):
            dst = out_dir / split / lab; dst.mkdir(parents=True, exist_ok=True)
            im = ImageOps.fit(ImageOps.exif_transpose(Image.open(p)).convert("RGB"), (224, 224))
            im.save(dst / p.name, quality=92)

if IN_COLAB and not (DATA_DIR / "train").exists():
    if M1_ZIP.exists():
        with zipfile.ZipFile(M1_ZIP) as z: z.extractall(DATA_DIR)
        print("Extracted Milestone 1 dataset from Drive.")
    else:
        print("Milestone 1 zip not found - building the dataset from Food_101.zip ...")
        build_from_raw(RAW_ZIP, DATA_DIR)
assert (DATA_DIR / "train").exists(), f"Dataset not found in {DATA_DIR}"

CLASS_NAMES = sorted(p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir())
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)

def list_split(split):
    """Return a dataframe of (path, label, y) for one split."""
    rows = [(p, p.parent.name) for p in sorted((DATA_DIR / split).rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    d = pd.DataFrame(rows, columns=["path", "label"]); d["y"] = d.label.map(CLS2IDX)
    return d

splits = {s: list_split(s) for s in ["train", "val", "test"]}
counts = pd.DataFrame({s: d.label.value_counts() for s, d in splits.items()}).loc[CLASS_NAMES]
print(f"{N_CLASSES} classes | " + " | ".join(f"{s}: {len(d)}" for s, d in splits.items()))
display(counts.T)
''')

md("""
**Observations**
- 17 food classes with an identical class order in all three splits.
- The per-class counts are balanced in the working set, so plain cross-entropy is appropriate; **macro-F1**
  is still reported so that no single weak class can hide behind a good average.
- The test set is **held out**: it is used once, only for the final selected model.
""")

md("""
## Pre-loading images into memory
The crops are small, so they are decoded once into a uint8 tensor. This removes JPEG decoding from every
epoch and makes training several times faster, especially on Colab where disk I/O is the bottleneck.
""")
code(r'''
def load_tensor(paths, size=IMG_SIZE):
    """Decode, resize and stack images as a uint8 tensor of shape (N, 3, size, size)."""
    arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

t0 = time.time()
X = {s: load_tensor(d.path) for s, d in splits.items()}
Y = {s: torch.tensor(d.y.values) for s, d in splits.items()}
print(f"Loaded in {time.time() - t0:.0f}s:", {s: tuple(x.shape) for s, x in X.items()})
''')

md("""
## Transforms: normalisation and augmentation
- **Evaluation transform:** convert to float and normalise with ImageNet statistics (Milestone 1 showed the
  dataset mean is close to ImageNet's, so the pretrained statistics are appropriate).
- **Training augmentation** (decided in Milestone 1 EDA): random resized crop, horizontal flip, small
  rotation, and brightness / contrast / saturation jitter to mimic real phone photos in poor light.
""")
code(r'''
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

eval_tf = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
aug_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
    T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

class TensorDS(Dataset):
    """Wraps pre-loaded uint8 images and applies a transform per sample."""
    def __init__(self, x, y, tf): self.x, self.y, self.tf = x, y, tf
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.tf(self.x[i]), self.y[i]

def make_loader(split, augment=False):
    ds = TensorDS(X[split], Y[split], aug_tf if augment else eval_tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

# Visual check of what augmentation does to the same images
idx = np.random.default_rng(SEED).choice(len(Y["train"]), 4, replace=False)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
inv = lambda t: (t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
for j, i in enumerate(idx):
    ax[0, j].imshow(inv(eval_tf(X["train"][i]))); ax[0, j].set_title(CLASS_NAMES[Y["train"][i]], fontsize=9)
    ax[1, j].imshow(inv(aug_tf(X["train"][i]))); ax[1, j].set_title("augmented", fontsize=9)
for a in ax.flat: a.axis("off")
plt.suptitle("Original (top) vs augmented (bottom)"); plt.tight_layout()
plt.savefig(FIG_DIR / "m2_01_augmentation.png", dpi=150); plt.show()
''')

md("""
# 3. Training and evaluation utilities
One shared training loop is used for every model so that the comparison is fair:
AdamW optimiser, cosine learning-rate schedule, model selection on **validation macro-F1**, early stopping.
""")
code(r'''
@torch.no_grad()
def predict(model, split):
    """Return softmax probabilities and true labels for a split."""
    model.eval(); probs = []
    for xb, _ in make_loader(split):
        probs.append(F.softmax(model(xb.to(DEVICE)), dim=1).cpu())
    return torch.cat(probs).numpy(), Y[split].numpy()

def scores(probs, y):
    """Accuracy, macro-F1 and top-3 accuracy."""
    pred = probs.argmax(1)
    top3 = (np.argsort(-probs, 1)[:, :3] == y[:, None]).any(1).mean()
    return dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"), top3=top3)

def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_model(name, model, epochs, lr, augment, weight_decay=0.0, label_smoothing=0.0, params=None):
    """Train with AdamW + cosine LR, keep the epoch with the best validation macro-F1."""
    seed_everything()
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(params if params is not None else [p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    train_loader = make_loader("train", augment)
    hist, best, best_state, wait, t0 = [], -1, None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward(); opt.step()
            tot += loss.item() * len(yb); correct += (out.argmax(1) == yb).sum().item(); n += len(yb)
        sched.step()
        pv, yv = predict(model, "val"); s = scores(pv, yv)
        val_loss = F.nll_loss(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(yv)).item()
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc=correct / n, val_loss=val_loss, **{f"val_{k}": v for k, v in s.items()}))
        print(f"[{name}] ep {ep:2d} | train loss {tot/n:.3f} acc {correct/n:.3f} | val loss {val_loss:.3f} acc {s['acc']:.3f} F1 {s['macro_f1']:.3f}")
        if s["macro_f1"] > best:
            best, best_state, wait = s["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE: print(f"[{name}] early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    h = pd.DataFrame(hist); h.attrs["train_time_s"] = time.time() - t0
    return model, h

RESULTS, HISTORIES, MODELS = {}, {}, {}

def record(name, model, hist, description):
    """Store validation metrics, size, speed and the generalisation gap for the comparison table."""
    pv, yv = predict(model, "val"); s = scores(pv, yv)
    best = hist.loc[hist.val_macro_f1.idxmax()]
    total, trainable = count_params(model)
    model.eval(); x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(3): model(x)
        t0 = time.time()
        for _ in range(20): model(x)
    RESULTS[name] = dict(description=description, val_acc=s["acc"], val_macro_f1=s["macro_f1"], val_top3=s["top3"],
                         train_acc=best.train_acc, gap=best.train_acc - s["acc"], best_epoch=int(best.epoch),
                         params_M=total / 1e6, trainable_M=trainable / 1e6, size_MB=total * 4 / 1e6,
                         latency_ms=(time.time() - t0) / 20 * 1000, train_time_s=hist.attrs["train_time_s"])
    HISTORIES[name], MODELS[name] = hist, model
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in RESULTS[name].items()})

def plot_history(name):
    h = HISTORIES[name]; fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val"); ax[0].set_title(f"{name}: loss")
    ax[1].plot(h.epoch, h.train_acc, label="train acc"); ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1", ls="--"); ax[1].set_title(f"{name}: accuracy")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / f"m2_curve_{name}.png", dpi=150); plt.show()
''')

md("""
# 4. Model 1 - Base CNN (from scratch)
A simple, conventional CNN: three Conv-ReLU-MaxPool blocks followed by a fully connected classifier.
**No augmentation and no regularisation**, so it shows what a plain CNN can learn from ~1,200 training images.
""")
code(r'''
class BaseCNN(nn.Module):
    """3 conv blocks (32-64-128 filters) + dense head. Deliberately plain."""
    def __init__(self, n_classes):
        super().__init__()
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, n_classes))
    def forward(self, x): return self.head(self.features(x))

m, h = train_model("base_cnn", BaseCNN(N_CLASSES), EPOCHS["base_cnn"], lr=1e-3, augment=False)
record("base_cnn", m, h, "Plain 3-block CNN, no augmentation / regularisation")
plot_history("base_cnn")
''')
code(r'''
# Observations for the base model, computed from its results
r = RESULTS["base_cnn"]
print(f"Base CNN: val accuracy {r['val_acc']:.1%}, macro-F1 {r['val_macro_f1']:.3f}, top-3 {r['val_top3']:.1%}")
print(f"Train accuracy at best epoch {r['train_acc']:.1%} -> generalisation gap {r['gap']:.1%}")
print(f"Random guessing would give {1/N_CLASSES:.1%}; the model is "
      f"{r['val_acc'] * N_CLASSES:.1f}x better than chance.")
''')
md("""
**Observations - base model**
- The plain CNN learns real signal (several times better than the 5.9% chance level), but its accuracy is far
  too low for a product: most meals would be logged wrongly.
- The training curve shows the classic **overfitting** pattern: training accuracy keeps rising while validation
  loss stops improving and then increases. With ~70 images per class the network memorises the training set
  instead of learning general food features.
- Conclusion: we need (a) more effective data via **augmentation**, (b) **regularisation**, and most of all
  (c) features learned from far more images - i.e. **transfer learning**.
""")

md("""
# 5. Improving performance
## Model 2 - CNN + augmentation + regularisation + architectural changes
Changes versus the base model:
- **Architecture:** 4 conv blocks with two convolutions each, **Batch Normalisation**, and global average pooling
  instead of a large dense layer (far fewer parameters to overfit).
- **Regularisation:** dropout 0.4, weight decay 1e-4, label smoothing 0.1.
- **Augmentation:** the training transform from Section 2.
""")
code(r'''
class ImprovedCNN(nn.Module):
    """4 double-conv blocks with BatchNorm, global average pooling and dropout."""
    def __init__(self, n_classes, p_drop=0.4, w=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(8 * w, n_classes))
    def forward(self, x): return self.head(self.features(x))

# Width 32 on a GPU; halved on CPU-only machines to keep training time reasonable.
m, h = train_model("cnn_aug_reg", ImprovedCNN(N_CLASSES, w=32 if DEVICE == "cuda" else 16), EPOCHS["cnn_aug_reg"], lr=2e-3, augment=True,
                   weight_decay=1e-4, label_smoothing=0.1)
record("cnn_aug_reg", m, h, "4-block CNN + BatchNorm + dropout + weight decay + augmentation")
plot_history("cnn_aug_reg")
''')
code(r'''
a, b = RESULTS["base_cnn"], RESULTS["cnn_aug_reg"]
print(f"Macro-F1: {a['val_macro_f1']:.3f} -> {b['val_macro_f1']:.3f} ({b['val_macro_f1'] - a['val_macro_f1']:+.3f})")
print(f"Generalisation gap: {a['gap']:.1%} -> {b['gap']:.1%}")
print(f"Parameters: {a['params_M']:.2f}M -> {b['params_M']:.2f}M")
''')
md("""
**Observations - augmentation and regularisation**
- Augmentation and regularisation **close most of the gap between training and validation accuracy**:
  the model no longer memorises the training images.
- Validation accuracy did **not** improve within this short training budget: the regularised network learns more slowly,
  and a network trained from scratch on ~1,200 images has not seen enough food to learn rich visual features.
  This is the ceiling of the from-scratch approach, and the reason to move to transfer learning.
""")

md("""
## Model 3 - Transfer learning: MobileNetV2, frozen backbone
MobileNetV2 was pretrained on ImageNet (1.3M images, including many foods). We **freeze** all its layers and
train only a new classification head. MobileNetV2 is chosen because it is designed for **mobile phones**,
where NutriScan runs.
""")
code(r'''
def mobilenet(n_classes, trainable_blocks=0, p_drop=0.3):
    """ImageNet MobileNetV2 with a new head. trainable_blocks = how many final feature blocks to unfreeze."""
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters(): p.requires_grad = False
    if trainable_blocks:
        for p in m.features[-trainable_blocks:].parameters(): p.requires_grad = True
    m.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(m.last_channel, n_classes))
    return m

m, h = train_model("mobilenet_frozen", mobilenet(N_CLASSES), EPOCHS["mobilenet_frozen"], lr=1e-3, augment=True,
                   weight_decay=1e-4, label_smoothing=0.1)
record("mobilenet_frozen", m, h, "MobileNetV2 (ImageNet), frozen backbone, new head + augmentation")
plot_history("mobilenet_frozen")
''')
md("""
**Observations - frozen transfer learning**
- Training only the head (about 22k parameters) already gives a **very large jump** over both from-scratch CNNs.
  The pretrained features (textures, shapes, colours of food) transfer directly to this task.
- Training is also much faster, because only the last layer learns.
""")

md("""
## Model 4 - Transfer learning: MobileNetV2, fine-tuned
We now **unfreeze the last 5 blocks** of MobileNetV2 and train them with a **10x smaller learning rate** than
the new head, so the pretrained features adapt to food without being destroyed.
""")
code(r'''
m = mobilenet(N_CLASSES, trainable_blocks=5)
param_groups = [{"params": [p for p in m.features.parameters() if p.requires_grad], "lr": 1e-4},
                {"params": m.classifier.parameters(), "lr": 1e-3}]
m, h = train_model("mobilenet_finetune", m, EPOCHS["mobilenet_finetune"], lr=1e-3, augment=True,
                   weight_decay=1e-4, label_smoothing=0.1, params=param_groups)
record("mobilenet_finetune", m, h, "MobileNetV2, last 5 blocks fine-tuned (discriminative LR) + augmentation")
plot_history("mobilenet_finetune")
''')

md("""
## Model 5 - Transfer learning: EfficientNet-B0, fine-tuned
A second, more recent architecture with a similar size, to check whether the result depends on the backbone.
Same recipe: last blocks unfrozen with a lower learning rate.
""")
code(r'''
def effnet(n_classes, trainable_blocks=3, p_drop=0.3):
    """ImageNet EfficientNet-B0 with a new head and the last blocks unfrozen."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.features.parameters(): p.requires_grad = False
    for p in m.features[-trainable_blocks:].parameters(): p.requires_grad = True
    m.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(m.classifier[1].in_features, n_classes))
    return m

m = effnet(N_CLASSES)
param_groups = [{"params": [p for p in m.features.parameters() if p.requires_grad], "lr": 1e-4},
                {"params": m.classifier.parameters(), "lr": 1e-3}]
m, h = train_model("effnet_finetune", m, EPOCHS["effnet_finetune"], lr=1e-3, augment=True,
                   weight_decay=1e-4, label_smoothing=0.1, params=param_groups)
record("effnet_finetune", m, h, "EfficientNet-B0, last 3 blocks fine-tuned + augmentation")
plot_history("effnet_finetune")
''')

md("""
# 6. Model comparison and final model selection
""")
code(r'''
cmp = pd.DataFrame(RESULTS).T
cmp_view = cmp[["description", "val_acc", "val_macro_f1", "val_top3", "train_acc", "gap", "params_M", "size_MB", "latency_ms", "train_time_s"]].copy()
for c in ["val_acc", "val_top3", "train_acc", "gap"]: cmp_view[c] = (cmp_view[c].astype(float) * 100).round(1)
for c in ["val_macro_f1"]: cmp_view[c] = cmp_view[c].astype(float).round(3)
for c in ["params_M", "size_MB", "latency_ms", "train_time_s"]: cmp_view[c] = cmp_view[c].astype(float).round(1)
cmp_view.columns = ["Description", "Val acc %", "Val macro-F1", "Val top-3 %", "Train acc %", "Gap (pp)", "Params (M)", "Size (MB)", "CPU/GPU latency (ms/img)", "Train time (s)"]
display(cmp_view)

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(cmp)); w = 0.27
ax.bar(x - w, cmp.val_acc.astype(float), w, label="Val accuracy")
ax.bar(x, cmp.val_macro_f1.astype(float), w, label="Val macro-F1")
ax.bar(x + w, cmp.val_top3.astype(float), w, label="Val top-3 accuracy")
ax.set_xticks(x); ax.set_xticklabels(cmp.index, rotation=15); ax.set_ylim(0, 1); ax.legend(loc="upper left")
ax.set_title("Validation performance by model")
plt.tight_layout(); plt.savefig(FIG_DIR / "m2_02_model_comparison.png", dpi=150); plt.show()
''')
code(r'''
# Selection rule: highest validation macro-F1. If two models are within 1 point of F1,
# prefer the smaller one, because the model has to run on a phone.
ranked = cmp.astype({"val_macro_f1": float, "size_MB": float}).sort_values("val_macro_f1", ascending=False)
top = ranked.iloc[0]
near = ranked[ranked.val_macro_f1 >= top.val_macro_f1 - 0.01]
BEST = near.sort_values("size_MB").index[0]
best_model = MODELS[BEST]
print("Ranking by validation macro-F1:"); print(ranked[["val_macro_f1", "size_MB"]].round(3))
print(f"\nSELECTED MODEL: {BEST}")
print(f"  val macro-F1 {RESULTS[BEST]['val_macro_f1']:.3f} | val accuracy {RESULTS[BEST]['val_acc']:.1%} | "
      f"top-3 {RESULTS[BEST]['val_top3']:.1%} | size {RESULTS[BEST]['size_MB']:.1f} MB")
''')
md("""
**Comparison and rationale**
- **From scratch vs transfer learning** is the decisive factor. Augmentation and regularisation reduce
  overfitting, but the jump in accuracy comes from reusing ImageNet features.
- **Fine-tuning beats a frozen backbone**: letting the last blocks adapt to food textures adds accuracy on top of
  the frozen model, with no sign of severe overfitting thanks to augmentation, label smoothing and weight decay.
- **Selection rule:** highest validation **macro-F1** (every food counts equally); if two models are within one
  F1 point, the **smaller** one wins because NutriScan runs on a phone. The selected model is printed above.
- The test set has not been touched so far, so the test score below is an unbiased estimate of real performance.
""")

md("# 7. Final evaluation of the selected model on the test set")
code(r'''
pt, yt = predict(best_model, "test")
test_s = scores(pt, yt); pred_t = pt.argmax(1)
print(f"TEST ({BEST}): accuracy {test_s['acc']:.1%} | macro-F1 {test_s['macro_f1']:.3f} | top-3 {test_s['top3']:.1%}  (n={len(yt)})")
rep = pd.DataFrame(classification_report(yt, pred_t, target_names=CLASS_NAMES, output_dict=True)).T
per_class = rep.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]].sort_values("f1-score")
display(per_class.round(3))

cm = confusion_matrix(yt, pred_t)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Test confusion matrix - {BEST}")
plt.tight_layout(); plt.savefig(FIG_DIR / "m2_03_confusion_matrix.png", dpi=150); plt.show()
''')
code(r'''
# Most confused pairs (off-diagonal cells of the confusion matrix)
off = [(CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j]) for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j and cm[i, j] > 0]
confused = pd.DataFrame(off, columns=["true", "predicted", "count"]).sort_values("count", ascending=False).head(8)
display(confused.reset_index(drop=True))
print("Weakest classes (F1):", per_class["f1-score"].head(3).round(2).to_dict())
print("Strongest classes (F1):", per_class["f1-score"].tail(3).round(2).to_dict())
''')
md("""
**Observations - test performance**
- The test score is close to the validation score, which confirms the model **generalises** and was not tuned
  to the validation set by chance.
- Errors concentrate in the **look-alike groups predicted in Milestone 1** (slice desserts, pancakes vs waffles,
  fried items such as samosa / spring rolls / falafel). These are fine-grained differences that even people
  confuse in a small photo.
- Top-3 accuracy is much higher than top-1: when the model is wrong, the correct dish is often its second or
  third guess. This directly supports a "confirm one of three suggestions" user interface.
""")

md("""
# 8. Classifying sample test images with the best model
Three test images are classified: two the model gets right and one it gets wrong, so both behaviours can be seen.
""")
code(r'''
rng = np.random.default_rng(SEED)
right = np.where(pred_t == yt)[0]; wrong = np.where(pred_t != yt)[0]
sample_idx = list(rng.choice(right, 2, replace=False)) + (list(rng.choice(wrong, 1)) if len(wrong) else list(rng.choice(right, 1)))

fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
sample_rows = []
for a, i in zip(ax, sample_idx):
    top3 = np.argsort(-pt[i])[:3]
    a.imshow(X["test"][i].permute(1, 2, 0).numpy()); a.axis("off")
    ok = "CORRECT" if top3[0] == yt[i] else "WRONG"
    txt = "\n".join(f"{CLASS_NAMES[k]}: {pt[i, k]:.0%}" for k in top3)
    a.set_title(f"True: {CLASS_NAMES[yt[i]]}  [{ok}]\n{txt}", fontsize=10, color="green" if ok == "CORRECT" else "red")
    sample_rows.append(dict(true=CLASS_NAMES[yt[i]], pred=CLASS_NAMES[top3[0]], confidence=round(float(pt[i, top3[0]]), 3),
                            second=CLASS_NAMES[top3[1]], third=CLASS_NAMES[top3[2]]))
plt.suptitle(f"Best model ({BEST}) on test images - top-3 predictions"); plt.tight_layout()
plt.savefig(FIG_DIR / "m2_04_sample_predictions.png", dpi=150); plt.show()
display(pd.DataFrame(sample_rows))
''')
code(r'''
# Confidence analysis: how reliable is the model when it is confident?
conf = pt.max(1); correct = pred_t == yt
for th in [0.5, 0.7, 0.9]:
    m_ = conf >= th
    print(f"confidence >= {th:.0%}: covers {m_.mean():5.1%} of photos, accuracy on them {correct[m_].mean():5.1%}")
print(f"Mean confidence when correct {conf[correct].mean():.2f} vs when wrong {conf[~correct].mean():.2f}")
''')
md("""
**Observations - sample predictions**
- On the correctly classified images the right dish is ranked first, but confidence is moderate at this short training budget.
- On the misclassified image the true dish appears among the alternatives and the confidence is noticeably
  lower. Low confidence is therefore a useful signal: the app should **ask the user to confirm** instead of
  logging silently.
- The confidence table shows the business trade-off: auto-log only above a confidence threshold, and ask for a
  one-tap confirmation below it.
""")

md("# 9. Saving the final model and results")
code(r'''
torch.save(best_model.state_dict(), OUT_DIR / f"best_model_{BEST}.pt")
summary = dict(best_model=BEST, img_size=IMG_SIZE, classes=CLASS_NAMES,
               test_acc=test_s["acc"], test_macro_f1=test_s["macro_f1"], test_top3=test_s["top3"],
               n_train=len(Y["train"]), n_val=len(Y["val"]), n_test=len(Y["test"]),
               comparison={k: {kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in v.items()} for k, v in RESULTS.items()},
               per_class_f1=per_class["f1-score"].round(3).to_dict(),
               confused_pairs=confused.astype({"count": int}).to_dict("records"), samples=sample_rows,
               conf_coverage={str(th): dict(coverage=float((conf >= th).mean()), acc=float(correct[conf >= th].mean())) for th in [0.5, 0.7, 0.9]})
json.dump(summary, open(OUT_DIR / "milestone2_results.json", "w"), indent=2)
print("Saved:", sorted(p.name for p in OUT_DIR.iterdir()))
''')

md("""
# 10. Conclusions and business interpretation

**What we built.** A food classifier for 17 dishes, developed in a controlled sequence: a plain CNN baseline,
a regularised and augmented CNN, and three transfer-learning variants, all compared on the same validation set
and judged on macro-F1.

**Key insights**
1. **Transfer learning is the single biggest lever.** With roughly 70 training photos per dish, models trained
   from scratch cannot learn enough. Reusing ImageNet features multiplies accuracy at a fraction of the training cost.
2. **Augmentation and regularisation fix overfitting**, which was the base model's main failure.
3. **Remaining errors are look-alike dishes** (desserts, pancakes vs waffles, fried snacks) - the same groups flagged
   in Milestone 1 EDA - not random noise.
4. **The model knows when it is unsure.** Confidence separates right from wrong predictions, and the correct dish is
   in the top-3 about two times out of three.

**Business impact**
- Logging a meal changes from typing and searching (around 30-60 seconds) to **one photo and, at most, one tap**.
  Faster logging is the main driver of retention in diet apps.
- Recommended product flow: **auto-log** high-confidence predictions, show **top-3 suggestions** for
  low-confidence ones. This keeps calorie errors low while still removing most manual work.
- The selected model is small and fast enough to run **on the phone**, which means no server cost per photo,
  instant results and better privacy.

**Limitations and next steps**
- Trained on CPU with a reduced budget (96 px images, 3-8 epochs); the GPU settings in this notebook (160 px, more epochs) should be run for the production model.
- Only 17 dishes and a capped working set; scale to the full dataset and more categories (Food-101 has 101).
- Train on more data per class and use the user's one-tap corrections as new labelled data (a feedback loop).
- Add portion-size estimation and link labels to a nutrition database (USDA) to output calories and macros.
- Multi-item plates: classify every detected box from Milestone 1, not only the main one.
""")

md("# Export to HTML")
code(r'''
# Export to HTML for submission.
# Colab: File > Download > Download .ipynb, then run locally:  jupyter nbconvert --to html NutriScan_Milestone2.ipynb
# (or use a free online ipynb-to-HTML converter). Local: run the same command after "Run All".
print("Notebook finished. Best model:", BEST, "| test macro-F1:", round(test_s["macro_f1"], 3))
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "accelerator": "GPU", "colab": {"gpuType": "T4"}}
nbf.write(nb, Path(__file__).with_name("NutriScan_Milestone2.ipynb"))
print("written")
