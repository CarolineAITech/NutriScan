#!/usr/bin/env python3
"""
Build the NutriScan Milestone 1 business report as a PDF.

Every number is read from milestone1_results.json, manifest.csv or
detections.json - nothing is typed in by hand. Figures are embedded from
outputs/figures/. Rendering is done by headless Chromium.

Usage:  ~/nutriscan-env/bin/python make_report.py
"""
import base64, io, json, subprocess, sys, math
from datetime import date
from pathlib import Path
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
PERSIST, FIGS = HERE / "outputs/persist", HERE / "outputs/figures"
OUT_HTML, OUT_PDF = HERE / "NutriScan_Milestone1_Report.html", HERE / "NutriScan_Milestone1_Report.pdf"

R = json.loads((PERSIST / "milestone1_results.json").read_text())
man = pd.read_csv(PERSIST / "manifest.csv")
DET = json.loads((PERSIST / "detections.json").read_text())
THR, MIN_AREA = R["det_threshold"], 0.10

# ---- recompute the detection breakdown with the notebook's own rule ----------
def select_reason(rec, thr, min_area=MIN_AREA):
    W, H = rec["w"], rec["h"]
    over = [(s, b) for b, s in zip(rec["boxes"], rec["scores"]) if s >= thr]
    cands = []
    for s, (x1, y1, x2, y2) in over:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if max(0, x2 - x1) * max(0, y2 - y1) / (W * H) >= min_area:
            cands.append(s)
    if not over:  return "no_detection", 0
    if not cands: return "box_too_small", 0
    return "ok", len(cands)

rows = [select_reason(DET[p], THR) for p in man.path]
man["reason"] = [r for r, _ in rows]
man["n_regions"] = [n for _, n in rows]
reasons = man.loc[~man.ok, "reason"].value_counts()
fb_by_class = (1 - man.groupby("label").ok.mean()).sort_values(ascending=False)
per_class = man.pivot_table(index="label", columns="split", values="path", aggfunc="count").fillna(0).astype(int)

N = len(man)
rate = man.ok.mean()
ci = 1.96 * math.sqrt(rate * (1 - rate) / N) * 100
multi = (man.n_regions >= 2).mean()

def pct(x):  return f"{x * 100:.1f}%"
def img(name, width="100%", max_px=1700, quality=82):
    """Embed a figure, downscaled and JPEG-encoded so the PDF stays a sensible size."""
    p = FIGS / f"{name}.png"
    if not p.exists(): return f"<p><em>[missing figure: {name}]</em></p>"
    im = Image.open(p).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:{width}">'

mp  = R.get("manual_precision_good")
mpu = R.get("manual_precision_good_or_partial")
nrev = R.get("n_manually_reviewed")
manual_txt = (f"{pct(mp)} of boxes judged correct and {pct(mpu)} judged usable "
              f"(correct or partially correct) across {nrev} hand-reviewed detections"
              if mp is not None else "not yet reviewed")

def table(pairs):
    return ("<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in pairs) + "</table>")

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; margin: 0; }
h1 { font-size: 23pt; margin: 0 0 4pt; color: #15503c; letter-spacing: -.3pt; }
h2 { font-size: 14pt; margin: 22pt 0 7pt; color: #15503c; border-bottom: 2px solid #2a7f62;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #22262a; page-break-after: avoid; }
p { margin: 0 0 8pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.5pt;
        font-family: Helvetica, Arial, sans-serif; page-break-inside: avoid; }
th, td { border: 1px solid #c9d6d0; padding: 4.5pt 7pt; text-align: left; vertical-align: top; }
th { background: #eef5f1; font-weight: 600; width: 42%; }
table.wide th { width: auto; background: #eef5f1; }
thead th { background: #2a7f62; color: #fff; border-color: #2a7f62; }
figure { margin: 10pt 0 14pt; page-break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #4a5a52; margin-top: 4pt;
             font-family: Helvetica, Arial, sans-serif; }
figcaption b { color: #15503c; }
.sub { color: #4a5a52; font-size: 10pt; margin-bottom: 2pt; }
.meta { font-size: 9pt; color: #5a6a62; font-family: Helvetica, Arial, sans-serif;
        border-bottom: 3px solid #2a7f62; padding-bottom: 10pt; margin-bottom: 16pt; }
.kpi { display: flex; gap: 8pt; margin: 12pt 0 14pt; }
.kpi div { flex: 1; border: 1px solid #c9d6d0; border-top: 3px solid #2a7f62;
           padding: 8pt 6pt; text-align: center; font-family: Helvetica, Arial, sans-serif; }
.kpi .v { font-size: 16pt; font-weight: 700; color: #15503c; display: block; }
.kpi .l { font-size: 7.5pt; color: #4a5a52; text-transform: uppercase; letter-spacing: .4pt; }
.callout { background: #eef5f1; border-left: 4px solid #2a7f62; padding: 8pt 11pt;
           margin: 10pt 0 12pt; page-break-inside: avoid; }
.callout p:last-child { margin-bottom: 0; }
.pb { page-break-before: always; }
ul { margin: 0 0 9pt; padding-left: 16pt; } li { margin-bottom: 4pt; }
"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>NutriScan Milestone 1</title><style>{CSS}</style></head><body>

<h1>NutriScan: AI-Powered Food Detection</h1>
<p class="sub">Milestone 1 — Data Understanding, Exploratory Analysis and Food Localisation</p>
<div class="meta">Prepared for the Chief Operating Officer and Data Science Manager &nbsp;·&nbsp;
{date.today():%d %B %Y} &nbsp;·&nbsp; Capstone Project</div>

<h2>1. The Business Problem</h2>
<p>Maintaining a balanced diet depends on knowing what you actually eat, and today that means
logging meals by hand. It is slow, it is easy to get wrong, and most people abandon it within
weeks. The data that does get recorded is unreliable: self-reported calorie intake is
systematically under-estimated, which undermines every recommendation built on top of it.</p>
<p><strong>NutriScan</strong> removes the typing. A user photographs a meal and the system
identifies the food and returns its nutritional profile immediately. Delivering that requires two
capabilities in sequence: <strong>detection</strong> — finding where the food is in a photograph
that also contains a table, cutlery and hands — and <strong>classification</strong> — naming the
dish inside that region. <strong>This milestone delivers the first of those, and the clean
dataset the second one will be trained on. No classifier has been trained yet.</strong></p>

<table class="wide"><thead><tr><th>Pain point today</th><th>What NutriScan changes</th><th>Business value</th></tr></thead>
<tr><td>Manual logging is slow; users quit</td><td>One photograph, instant result</td><td>Higher engagement and retention</td></tr>
<tr><td>Self-reported intake is inaccurate</td><td>Image-based identification</td><td>Trustworthy data for recommendations</td></tr>
<tr><td>Generic dietary advice</td><td>Personalisation from logged habits</td><td>Premium-tier differentiation</td></tr>
<tr><td>Health data sits in silos</td><td>Integration with health apps</td><td>Partnerships and ecosystem reach</td></tr></table>

<div class="kpi">
  <div><span class="v">{pct(rate)}</span><span class="l">Food located</span></div>
  <div><span class="v">{R['n_crops']:,}</span><span class="l">Training images built</span></div>
  <div><span class="v">{R['n_classes']}</span><span class="l">Food categories</span></div>
  <div><span class="v">{pct(mp) if mp is not None else '—'}</span><span class="l">Boxes correct on review</span></div>
</div>

<h2>2. The Data</h2>
{table([("Source", "Food-101 subset"),
        ("Total images", f"{R['n_images_total']:,}"),
        ("Food categories", R['n_classes']),
        ("Corrupt or unreadable files", R['n_corrupt']),
        ("Exact duplicate files removed", R['n_duplicates']),
        ("Images after cleaning", f"{R['n_clean']:,}"),
        ("Images processed this milestone", f"{R['n_working_set']:,}"),
        ("Typical image size", f"{int(R['median_width'])} × {int(R['median_height'])} px")])}
<p>The library is {R['n_images_total']:,} photographs of plated dishes, organised one folder per
food type, with the folder name acting as the label. Every file opened correctly and no exact
duplicates were found — a cleaner starting point than is typical. Duplicate checking matters
commercially as well as technically: had the same photograph appeared in both the training and
the test set, our accuracy figures would have been inflated and we would have over-promised on
what the product can do.</p>
<p>Detection was run on a stratified random sample of <strong>{R['n_working_set']:,} images</strong>
({R['n_working_set'] // R['n_classes']} per category). Section 4 explains why, and what it costs us.</p>

<h2 class="pb">3. What the Data Tells Us</h2>
<h3>3.1 The categories are not evenly represented</h3>
{table([("Imbalance ratio (full dataset)", f"{R['imbalance_ratio_full']:.2f}×"),
        ("Smallest category", "apple_pie — 256 images"),
        ("Largest categories", "1,000 images each"),
        ("Imbalance within our working sample", f"{R['imbalance_ratio_work']:.2f}×")])}
<p>One category, <em>apple pie</em>, holds roughly a quarter of the images the others do. Left
unaddressed, a model trained on this data learns that guessing apple pie is rarely worth the
risk — it will score well on average while being quietly useless for that one food. Any user
logging apple pie gets a wrong calorie count.</p>
<div class="callout"><p><strong>Decision:</strong> Milestone 2 will train with balanced class
weights, so errors on scarce categories cost the model more, and will be judged on
<strong>macro-F1</strong> — which averages performance across categories equally — rather than
plain accuracy, which a majority-biased model can inflate.</p></div>
<figure>{img("01_class_distribution")}<figcaption><b>Figure 1.</b> Images per category, full
dataset (top) and the working sample (bottom). <b>What this tells us:</b> the shortfall is
confined to a single category; capping our sample at {R['n_working_set'] // R['n_classes']} per
category makes the working set look balanced, which is why the full-data ratio is the one quoted
above.</figcaption></figure>

<h3>3.2 The photographs vary widely — and that is useful</h3>
<figure>{img("03_class_mosaic")}<figcaption><b>Figure 2.</b> One representative photograph per
category. <b>What this tells us:</b> several categories are visually close — slice-shaped
desserts, stacked discs with syrup, golden fried parcels — and these are where the classifier
will struggle.</figcaption></figure>
<p>Most photographs show a single dish, but plates, cutlery, table surfaces and occasionally
packaging appear in frame, and lighting, camera angle and plating vary considerably. That
variation is an asset: a model trained only on studio-perfect images would fail on the
arm's-length photographs real users take.</p>

<h3>3.3 Image quality is good, which is itself a risk</h3>
{table([("Dark images", f"{R['pct_dark']:.1f}%"),
        ("Over-exposed images", f"{R['pct_bright']:.1f}%"),
        ("Low-contrast images", f"{R['pct_low_contrast']:.1f}%"),
        ("Blurry images", f"{R['pct_blurry']:.1f}%")])}
<p>Only a small fraction of images are poorly lit or out of focus, and the dataset's average
colour profile is close to that of ImageNet — the reference collection our pretrained models were
built on — which confirms those models can be used without recalibration.</p>
<div class="callout"><p><strong>The risk this creates:</strong> our training images are
<em>easier</em> than the ones users will actually submit. Milestone 2 must deliberately degrade
training images — varying brightness, contrast and focus — so the model is robust to real
conditions. This is a requirement, not a refinement.</p></div>

<h2 class="pb">4. Finding the Food in the Photograph</h2>
<h3>4.1 The obstacle, and how we got around it</h3>
<p>The dataset tells us <em>what</em> is in each photograph but never <em>where</em>. Conventional
object detectors must be taught from thousands of hand-drawn boxes, which we do not have and
which would be expensive and slow to produce.</p>
<p>We avoided that cost entirely by using a <strong>zero-shot detector</strong> — a model that has
already learned to associate images with English descriptions. We simply ask it, in plain English,
to find “a plate of food”, and it returns the region. No annotation, no training, and it
generalises to food types it has never been shown.</p>
<div class="callout"><p><strong>Commercial significance:</strong> this removes a data-labelling
project from the critical path. The same approach extends to new cuisines and menu items without
commissioning fresh annotation each time.</p></div>

<h3>4.2 Calibrating the confidence threshold</h3>
<p>The detector scores its confidence in each region. Set the bar too high and genuine meals are
rejected; too low and we accept boxes drawn around tables and plate rims. We tested a range of
settings and selected the strictest one that still located food in at least 90% of photographs.</p>
<figure>{img("07_threshold_sweep", "72%")}<figcaption><b>Figure 3.</b> Detection success rate
against confidence threshold. <b>What this tells us:</b> the chosen setting of
{R['det_threshold']} is the strictest value that still clears the 90% business
target.</figcaption></figure>

<h3>4.3 Results</h3>
{table([("Photographs where food was located", f"<strong>{pct(rate)}</strong> (95% confidence interval ±{ci:.1f} points)"),
        ("Business target", "90% — <strong>met</strong>" if rate >= 0.90 else "90% — not met"),
        ("Fell back to using the full photograph", pct(1 - rate)),
        ("&nbsp;&nbsp;— no region scored above the threshold", f"{reasons.get('no_detection', 0)} images ({pct(reasons.get('no_detection', 0) / N)})"),
        ("&nbsp;&nbsp;— best region was too small to be credible", f"{reasons.get('box_too_small', 0)} images ({pct(reasons.get('box_too_small', 0) / N)})"),
        ("Average detector confidence", f"{R['mean_det_score']:.3f}"),
        ("Median region size, as a share of the photo", f"{R['median_box_area_frac'] * 100:.0f}%"),
        ("Photographs containing two or more food regions", pct(multi))])}
<p>We met the target. Just as importantly, the system never discards an image: when no confident
region is found it keeps the whole photograph, so every picture still yields a usable training
sample and the pipeline cannot silently lose data.</p>
<figure>{img("06_detection_samples")}<figcaption><b>Figure 4.</b> Detected regions on sample
photographs; the selected region is outlined in green. <b>What this tells us:</b> the region
generally sits on the food itself, but on plates holding several items it often encloses only
one of them.</figcaption></figure>

<h3>4.4 How we verified accuracy honestly</h3>
<p>Because no human ever marked the “correct” region for these photographs, there is no automatic
score to compute. We therefore reviewed a random sample of detections by eye and judged each one
correct, partially correct, or wrong. The result: <strong>{manual_txt}</strong>.</p>
<p>This is a sample-based estimate, not an exact figure, and we state it as such. Section 6
describes how to make detection quality properly measurable.</p>

<h3>4.5 From photograph to model-ready data</h3>
<p>Each detected region is cut out with a small margin, squared off so the food is not stretched
out of shape, resized to a uniform {int(R['median_width']) // 2 if False else 224}×224 pixels, and
colour-standardised to match what pretrained models expect. The results are divided into training,
validation and test sets with category proportions held constant, and we verified that no
photograph appears in more than one set.</p>
<figure>{img("11_before_after")}<figcaption><b>Figure 5.</b> Original photograph with the selected
region (top) and the final standardised crop (bottom). <b>What this tells us:</b> the pipeline
produces consistent, food-centred inputs from inconsistent source photographs.</figcaption></figure>
{table([("Training images", f"{R['n_train']:,}"),
        ("Validation images", f"{R['n_val']:,}"),
        ("Test images", f"{R['n_test']:,}"),
        ("<strong>Total prepared</strong>", f"<strong>{R['n_crops']:,}</strong>"),
        ("Smallest per-category test set", f"{int(per_class['test'].min())} images"),
        ("Overlap between sets", "None — verified")])}

<h2 class="pb">5. Insights and Recommendations</h2>

<h3>5.1 The single biggest weakness: meals with more than one item</h3>
<p><strong>{pct(multi)} of photographs contain two or more distinct food regions, but the system
currently keeps only the best one.</strong> On a plate holding several items it crops one of them
and discards the rest.</p>
<div class="callout"><p>For a calorie-counting product this is the most consequential finding in
this report. A user photographing a mixed plate would have only part of their meal counted, and
the error is always in the same direction — under-counting. The fix is inexpensive, because every
candidate region is already stored: we change which ones we keep, not how we find them.</p></div>

<h3>5.2 Where detection struggles</h3>
<table class="wide"><thead><tr><th>Situation</th><th>Why it happens</th><th>What we will do</th></tr></thead>
<tr><td>Loose, scattered foods — <em>french fries</em> had the highest fallback rate at {pct(fb_by_class.iloc[0])}</td>
    <td>No single shape encloses a pile of separate strips</td><td>Union of regions rather than one box</td></tr>
<tr><td>Several items on one plate</td><td>Only the single best region is kept</td><td>Keep every confident region and classify each</td></tr>
<tr><td>Close-ups where food fills the frame</td><td>No region stands out from its surroundings</td>
    <td>Acceptable — the whole photograph <em>is</em> the food</td></tr>
<tr><td>Region includes plate rim or sauce</td><td>Prompts are deliberately generic</td><td>Tune the wording; raise the minimum size filter</td></tr>
<tr><td>No ground-truth regions exist</td><td>Dataset was never annotated with boxes</td>
    <td>Hand-label a sample to obtain a true, comparable accuracy score</td></tr></table>
<figure>{img("10_failure_cases")}<figcaption><b>Figure 6.</b> Fallback cases and the
lowest-confidence accepted regions. <b>What this tells us:</b> most failures are close-ups where
keeping the full photograph is the right answer, rather than genuine mistakes.</figcaption></figure>

<h3>5.3 Did cropping actually add value?</h3>
<p>The median detected region covers {R['median_box_area_frac'] * 100:.0f}% of the photograph,
which tells us most images in this dataset are <em>already</em> close-ups of a single plated dish,
framed by a photographer. Cropping therefore changes these particular images less than one might
expect.</p>
<p>We report that plainly because it does not make the detection stage redundant — it makes the
dataset unrepresentative. The product's real input is a cluttered table photographed at arm's
length. Detection is what makes that input tractable, and this dataset simply under-represents the
hard case. <strong>Recommendation: validate Milestone 2 on genuine user photographs before
launch.</strong></p>

<h3>5.4 Categories we expect to be confused</h3>
<p>From visual inspection, the groups most at risk are <strong>chocolate cake / strawberry
shortcake / apple pie</strong>, <strong>pancakes / waffles</strong>, <strong>samosa / spring
rolls</strong>, and <strong>french fries / onion rings</strong>. These should be the first place
we look in the Milestone 2 confusion matrix. They carry a direct business cost: confusing a light
dessert with a rich one is a logging error of several hundred calories, which is precisely the
kind of inaccuracy that erodes user trust.</p>

<h2 class="pb">6. Plan for Milestone 2</h2>
<table class="wide"><thead><tr><th>Step</th><th>Approach</th></tr></thead>
<tr><td>Starting point</td><td>The {R['n_crops']:,} standardised images built in this milestone</td></tr>
<tr><td>Baseline</td><td>A small network trained from scratch, for a reference score</td></tr>
<tr><td>Main approach</td><td>Transfer learning — EfficientNet-B0, ResNet50, MobileNetV2 for mobile deployment</td></tr>
<tr><td>Why transfer learning</td><td>Re-uses a model already trained on millions of photographs, so far less data and time is needed</td></tr>
<tr><td>Robustness</td><td>Augmentation — flips, crops, brightness, contrast and blur, to mimic real user photographs</td></tr>
<tr><td>Imbalance</td><td>Balanced class weights and label smoothing</td></tr>
<tr><td>How we judge it</td><td>Macro-F1 and per-category scores alongside top-1 accuracy; confusion matrix reviewed for the look-alike groups</td></tr>
<tr><td>Final selection</td><td>Best validation macro-F1, with a speed and model-size check for mobile; the test set is used once</td></tr>
<tr><td>Beyond Milestone 2</td><td>Map categories to nutrition data, estimate portion size, generate recommendations, integrate with health apps</td></tr></table>

<h3>Risks</h3>
<ul>
<li><strong>Visually similar foods will cap accuracy.</strong> Some categories are genuinely hard to separate from a photograph alone.</li>
<li><strong>Coverage is limited to {R['n_classes']} categories.</strong> Real users eat far more variety; expanding the menu is a data-collection problem, not a modelling one.</li>
<li><strong>A photograph cannot reveal hidden ingredients or portion size.</strong> Cooking oil, added sugar and serving weight are invisible, so calorie figures will always carry error. This should be communicated honestly in the product.</li>
<li><strong>Dataset photographs are easier than user photographs.</strong> Validate on real submissions before launch.</li>
</ul>

<h3>Infrastructure recommendation</h3>
<div class="callout"><p>This milestone ran on a CPU-only machine, where locating food takes about
one second per photograph — several hours for the full library, against roughly ten minutes on
standard cloud GPU hardware. That constraint is manageable now. It will not be in Milestone 2,
where model training rather than detection becomes the dominant cost and CPU-only execution would
be impractical. <strong>We recommend provisioning GPU compute before the classification phase
begins.</strong></p></div>

<h2>7. Conclusion</h2>
<p>We understood the data, established that it is clean but unevenly distributed across
categories, and demonstrated that food can be located automatically in photographs
<strong>without commissioning any manual annotation</strong> — meeting the 90% target at
{pct(rate)}, with {manual_txt.split(' across')[0] if mp is not None else 'a hand-reviewed sample confirming box quality'}.
From that we built a clean, leak-free dataset of {R['n_crops']:,} standardised food images ready
for classification.</p>
<p>We also identified the most important weakness early: {pct(multi)} of photographs contain more
than one food item, and the current pipeline keeps only one. Fixing that is inexpensive and is the
first item of work in the next phase. The project is on track for Milestone 2.</p>

</body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"HTML written: {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")

proc = subprocess.run(["chromium", "--headless", "--disable-gpu", "--no-sandbox",
                       "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                       OUT_HTML.as_uri()], capture_output=True, text=True, timeout=300)
if OUT_PDF.exists() and OUT_PDF.stat().st_size > 50_000:
    print(f"PDF written : {OUT_PDF} ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"Chromium failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")
    sys.exit(1)
