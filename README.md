# NutriScan — Capstone Milestone 1

Everything needed to run, interpret and submit Milestone 1 of the Food Detection
capstone. The notebook runs **unchanged** on this machine and on Google Colab —
it detects which one it is on and configures itself.

---

## What this project is (the 30-second version)

A food-logging app. You photograph your meal, it tells you what you ate and how
many calories. Two AI steps make that work:

1. **Detection — *where* is the food?** Draw a box around the food, ignore the
   table, cutlery and hands. (Pointing at it.)
2. **Classification — *what* is the food?** Name the dish in that box.
   (Saying its name.)

**Milestone 1 is step 1 only.** No classifier is trained yet. This milestone
explores the photos, locates the food in each one, crops and standardises it,
and packages the result as a clean train/validation/test dataset. Milestone 2
trains the model that names the dish.

**The clever bit:** the dataset says *what* is in each photo but never *where*.
Normally you would need thousands of hand-drawn boxes to train a detector. We
avoid that entirely by using a **zero-shot detector** (OWL-ViT) — a model that
already links pictures to English words, so you simply ask it to find
"a plate of food" and it draws the box. No labelling, no training.

**The data:** 16,256 photos across 17 food categories, one folder per category,
folder name = label. 16 categories have 1,000 photos; `apple_pie` has only 256.
That lopsidedness is **class imbalance**, and it is one of your reportable
findings.

---

## Files in this folder

| File | What it is |
|---|---|
| `NutriScan_Milestone1.ipynb` | **The notebook.** Runs locally and on Colab |
| `NutriScan_Milestone1_ORIGINAL_BACKUP.ipynb` | Your original, untouched |
| `setup_local.sh` | One-time local setup (creates `~/nutriscan-env`) |
| `run_local.sh` | Launches JupyterLab with the right environment |
| `requirements.txt` | Dependency list |
| `collect_findings.py` | Fills your findings table from the notebook's results |
| `milestone1_findings_TEMPLATE.csv` | **The 54 numbers/observations you collect** |
| `NutriScan_Milestone1_Report_TEMPLATE.md` | **The PDF report skeleton you fill in** |

---

## Option A — run locally

This machine has **no CUDA GPU** (AMD integrated graphics), so everything runs
on the CPU. That is fine — it is slower, not blocked.

```bash
cd ~/Downloads/Food_101/NutriScan_Milestone1
./setup_local.sh      # once, ~5 minutes, ~1.5 GB
./run_local.sh        # opens JupyterLab
```

Then **Run → Run All Cells**.

The notebook finds your dataset at `~/Downloads/Food_101/Food_101` automatically.
If you move it, edit `LOCAL_DATA_DIR` in the **Configuration** cell.

### How long it takes on this machine

**Measured, not estimated.** I ran the whole notebook here at
`MAX_PER_CLASS = 10`: it completed with no errors in about 4 minutes, and the
detector ran at **1.13 seconds per image** on 8 CPU cores (4.5 s per batch of 4).

| `MAX_PER_CLASS` | Images | Measured/projected time |
|---|---|---|
| `10` (smoke test) | 170 | **~4 min** (measured) |
| `150` | 2,550 | ~50 min |
| `400` (the default) | ~6,800 | **~2 hours** |
| `None` (everything) | 16,256 | ~5 hours |

**Recommended plan:** run the smoke test locally at `MAX_PER_CLASS = 10` to see
the whole pipeline work and understand each cell, then do the real run at
`400` **on Colab with a T4 GPU**, where the same job takes 10–20 minutes.

Detections are checkpointed, so raising the cap and re-running never recomputes
images already done.

Outputs land in `outputs/` — figures, crops, the final dataset, and
`persist/milestone1_results.json`.

---

## Option B — run on Colab

1. Zip the dataset folder and upload it to your Drive as
   `MyDrive/Food_101.zip`.
2. Upload `NutriScan_Milestone1.ipynb` to Colab.
3. **Runtime → Change runtime type → T4 GPU.** This matters — on a GPU the whole
   notebook finishes in 10–20 minutes instead of over an hour.
4. Run all cells. The Drive mount prompt appears at the "Loading the Data" cell.

Outputs are saved to `MyDrive/NutriScan_M1/` so a disconnect does not lose work.

Running it in both places is a good idea — Colab for speed, locally for
understanding what each cell actually does.

---

## After the run: turning output into marks

This is where most of the marks are, and the notebook cannot do it for you.

### Step 1 — collect the numbers

```bash
~/nutriscan-env/bin/python collect_findings.py
```

This reads `milestone1_results.json` and writes
`milestone1_findings_FILLED.csv` with ~30 of the 54 values already filled.
(On Colab, download `milestone1_results.json` from `MyDrive/NutriScan_M1/` first
and pass it with `--results`.)

### Step 2 — fill in the rest

Open the CSV in LibreOffice or Google Sheets. Rows left blank are the ones that
need your eyes: which categories look alike, what the failure cases have in
common, what you decide to do about imbalance.

### Step 3 — write the interpretation column

`what_it_means_write_one_sentence` is the column that earns marks.

> "detection rate 94%" — that is data, worth nothing.
>
> "94% of photos yielded a usable food crop, beating our 90% target; the
> remaining 6% were close-ups where the whole photo is already food, so keeping
> the full image loses nothing." — that is an insight, and it scores.

### Step 4 — do the manual box review (15 minutes, 10 marks)

Section 4.5 shows a grid of 50 detections with an ID on each. Look at them, note
which boxes are wrong or only partly right, and put those IDs into `BAD_IDS` and
`PARTIAL_IDS`, then re-run that cell.

Because the dataset has no "correct" boxes, **this manual review is the only
accuracy evidence you have** for the highest-weighted criterion. Do not skip it.

### Step 5 — fill in the observation cells

Every cell marked `**Observations (write after running):**` needs replacing with
2–4 real sentences. There are ten. Leaving them as prompts tells the grader the
work is unfinished.

### Step 6 — produce the two submission files

- **Notebook → HTML.** Run the last cell, or File → Save and Export As → HTML.
- **Report → PDF.** Fill in `NutriScan_Milestone1_Report_TEMPLATE.md`, drop the
  figures from `outputs/figures/` in where marked, export to PDF.
  Submit PDF, **not** .docx. No code in the report.

---

## Marking checklist

| Criterion | Marks | Done when |
|---|---|---|
| Business problem | 2 | Section 1 of the report is written in your own words |
| Exploratory data analysis | 4 | All four EDA observation cells written; every figure captioned with a "what this tells us" line |
| Object detection | 10 | Notebook ran clean; manual review filled in; detection rate, fallback reasons and failure cases all discussed |
| Actionable insights | 4 | Imbalance finding + decision; where detection fails + fix; confusable categories named |

**Before submitting, check that none of these strings survive anywhere:**
`write after running` · `Fill in with your actual numbers` · `X images` ·
`ratio Z` · `[WRITE:` · `{{`

---

## What was changed from your original notebook

The original is kept as `NutriScan_Milestone1_ORIGINAL_BACKUP.ipynb`.

1. **Fixed a syntax error** — a stray `Ca` had been typed into the
   Configuration cell (`EXTRACT_DIR = Path(...)   Ca`). The notebook could not
   run at all with that present.
2. **Dual-mode paths** — auto-detects Colab vs local; no Drive mount or unzip
   on a local run.
3. **Unpinned `transformers`** — the pinned `4.46.3` predates Python 3.14 and
   its dependencies have no matching wheels, so the install cell would fail.
4. **`MAX_PER_CLASS` 150 → 400** — at 150 every class is capped equally, which
   flattens `apple_pie` (256 images) and erases the imbalance finding. At 400
   the real imbalance survives.
5. **`SWEEP_PER_CLASS` 3 → 15** — choosing the detection threshold off 51
   images was too thin a basis for a 10-mark decision.
6. **CPU-aware** — uses all 8 cores, smaller detector batches when there is no
   GPU.
7. **Imbalance cell rewritten** — reports the imbalance on the *full* dataset
   and states an explicit action, instead of printing "classes are close to
   balanced" because the cap made them so.
8. **"101 classes" → 17** in three places, so the prose matches the output.
9. **Manual QA cell rewritten** — the originals defaulted `BAD_IDS` and
    `PARTIAL_IDS` to empty lists, which would have silently reported **100%
    detection precision** without you reviewing anything. They now default to
    `None`, so "not reviewed yet" stays distinct from "reviewed, found none".
    It also validates your IDs and records how many boxes you reviewed.
10. **HTML export cell** — works locally as well as on Colab.
11. **Fixed a crash with current `transformers`** — v5 renamed
    `post_process_object_detection` to `post_process_grounded_object_detection`.
    The notebook now tries both, so it runs on Colab's version and on a current
    local install.
12. **`N_WORKERS = 0` on local runs** — Python 3.12+ uses the `forkserver`
    start method on Linux, which crashes DataLoader worker processes inside a
    notebook. On CPU the model forward pass dominates, so this costs nothing.
13. **Summary cell** — the vague `X images, 101 classes ... ratio Z` text is
    now explicit `{{placeholders}}` matching the report template.

### Verified
The notebook was run here end to end at `MAX_PER_CLASS = 10` using the
environment `setup_local.sh` builds. Exit code 0, all 32 code cells, all 13
figures produced, dataset built and verified (118 train / 26 val / 26 test,
batch tensor `(64, 3, 224, 224)`, 17 classes, no duplicate images across
splits, normalisation confirmed reversible).
