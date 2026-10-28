# NutriScan — Milestone 1 Business Report

**Project:** AI-Powered Food Detection & Classification
**Prepared by:** <your name>
**Date:** <date>
**Audience:** COO / CMO / Data Science Manager

> **HOW TO USE THIS TEMPLATE**
> - Replace every `{{...}}` with a real number from your notebook run.
> - Replace every `[WRITE: ...]` with 2-4 sentences in your own words.
> - Delete all these grey instruction boxes before exporting.
> - **Do not paste code.** This document is for a business leader.
> - Target length: 8-12 pages including figures. Every figure needs a caption
>   AND a one-line "what this tells us".
> - Export to PDF when done (see the note at the bottom).

---

## 1. The Business Problem and Why It Matters
*(Rubric: Introduction of the business problem — 2 marks)*

### 1.1 The problem today

[WRITE: 3-4 sentences. People want to eat well but tracking food is manual,
slow and inaccurate. Most food-diary users quit within weeks. Self-reported
calorie intake is known to be badly under-estimated. Say why this is a problem
worth money.]

### 1.2 What NutriScan does about it

[WRITE: 2-3 sentences. The user photographs a meal; the system identifies the
food and returns calories and macronutrients instantly. One photo replaces
manual logging.]

### 1.3 The value to the business

| Pain point today | What NutriScan changes | Value to NutriScan |
|---|---|---|
| Manual logging is slow, users quit | One photo, instant result | Higher engagement and retention |
| Self-reported intake is inaccurate | Image-based identification | More trustworthy data |
| Generic diet advice | Personalised suggestions from logged habits | Premium-tier differentiation |
| Siloed health data | Integration with health apps | Partnerships, ecosystem reach |

### 1.4 What this milestone delivers

[WRITE: 2-3 sentences. Milestone 1 = find the food in the photo, cut it out,
standardise it, and build a clean dataset. Milestone 2 = train the model that
names the dish. State clearly that no classifier is trained yet.]

**Success measures agreed for this phase**

| Measure | Target | Achieved |
|---|---|---|
| Images yielding a usable food crop | ≥ 90% | {{detection_rate}} |
| Clean, leak-free training dataset | Yes / No | {{yes_no}} |
| Classification accuracy (Milestone 2) | ≥ 85% | Not yet — next milestone |

---

## 2. The Data We Are Working With
*(Rubric: Exploratory data analysis — part of 4 marks)*

### 2.1 Overview

| Item | Value |
|---|---|
| Source | Food-101 (subset) |
| Total images | {{n_images_total}} |
| Food categories (classes) | {{n_classes}} |
| Corrupt / unreadable files | {{n_corrupt}} |
| Exact duplicate files removed | {{n_duplicates}} |
| Images after cleaning | {{n_clean}} |
| Images used in this milestone | {{n_working_set}} |
| Typical image size | {{median_width}} x {{median_height}} px |
| Largest side (all images) | 512 px |

[WRITE: 2-3 sentences explaining in plain English what this dataset is —
photographs of plated dishes, one dish per photo, organised in one folder per
food type, with the folder name acting as the label. Say why the number of
images per class matters.]

### 2.2 Data quality checks we ran

[WRITE: Explain, without code, that you checked for (a) files that will not
open, (b) exact duplicate photos, (c) photos that are too dark, too bright,
washed out or out of focus. Say WHY duplicates matter — if the same photo
appears in both the training and the testing set, the test score is inflated
and the business is misled about how good the model is.]

> **FIGURE 1** — insert `01_class_distribution.png`
> *Caption:* Number of images available per food category.
> *What this tells us:* [WRITE one line]

---

## 3. What We Learned From the Images
*(Rubric: Exploratory data analysis — 4 marks)*

### 3.1 Are the categories balanced?

| Measure | Value |
|---|---|
| Smallest category | {{smallest_class}} ({{smallest_count}} images) |
| Largest category | {{largest_class}} ({{largest_count}} images) |
| Imbalance ratio (largest ÷ smallest) | {{imbalance_ratio_full}} |

[WRITE: 3-4 sentences. Is it balanced? Which class is short of data? What is
the business risk of an under-represented class — the model sees it rarely,
predicts it rarely, and users logging that food get wrong calorie counts.]

### 3.2 What the photos actually look like

> **FIGURE 2** — insert `02_sample_grid.png`
> *Caption:* Five random photos from eight categories.
> *What this tells us:* [WRITE one line]

> **FIGURE 3** — insert `03_class_mosaic.png`
> *Caption:* One representative photo per category.
> *What this tells us:* [WRITE one line]

[WRITE: 4-6 sentences covering:
 - Is there usually one dish per photo, or several?
 - What else is in frame — plates, cutlery, tables, hands, packaging, text?
 - How much do angle, lighting and plating vary within the same food type?
 - Which categories look similar to each other and will be hard to tell apart?
   (e.g. waffles vs pancakes, samosa vs spring rolls, the two cake types)]

### 3.3 Image size, shape and quality

| Measure | Value |
|---|---|
| Photos that are dark | {{pct_dark}}% |
| Photos that are over-exposed | {{pct_bright}}% |
| Photos with low contrast | {{pct_low_contrast}}% |
| Photos that are blurry | {{pct_blurry}}% |
| Darkest categories | {{darkest_classes}} |

> **FIGURE 4** — insert `04_image_sizes.png`
> **FIGURE 5** — insert `05_brightness_contrast.png`
> *What these tell us:* [WRITE one line each]

[WRITE: 3-4 sentences. Photos come in different shapes, so simply squashing
them to a square would distort the food — explain that this is why we crop to
a square instead. Mention that a small share of dark/blurry photos is actually
GOOD news, because real users take dark, blurry photos too, and the model must
cope with them.]

---

## 4. Finding the Food in the Photo
*(Rubric: Object Detection — 10 marks. THIS IS THE BIGGEST SECTION.)*

### 4.1 The challenge

[WRITE: 3-4 sentences. The dataset tells us WHAT is in each photo but not
WHERE. There are no boxes drawn around the food. A normal object detector has
to be taught with thousands of hand-drawn boxes, which we do not have and
cannot afford to create. Explain this blocker in plain words.]

### 4.2 How we solved it

[WRITE: 3-4 sentences. We used a "zero-shot" detector — a model that has
already learned to connect pictures and words, so you can simply ask it in
English to find "a plate of food" and it draws the box. No training, no
hand-labelling, works for every food type including ones we have not seen.]

| Option considered | Why / why not |
|---|---|
| Standard detector trained on everyday objects | Knows only a handful of foods — rejected |
| **Zero-shot text-prompted detector (OWL-ViT)** | No labelling needed, covers all dishes — **chosen** |
| Train our own detector | Most accurate, but needs thousands of hand-drawn boxes — future option |

### 4.3 Setting the confidence level

[WRITE: 3-4 sentences. The detector gives each box a confidence score. Set the
bar too high and we reject good food photos; too low and we accept boxes
around tables and plates. Explain that we tested a range of settings and chose
the strictest one that still found food in at least 90% of photos.]

> **FIGURE 6** — insert `07_threshold_sweep.png`
> *Caption:* Detection success rate at different confidence settings.
> *What this tells us:* [WRITE one line]

**Chosen confidence threshold: {{det_threshold}}**

### 4.4 Results

| Measure | Value | Target |
|---|---|---|
| Photos where food was located | {{detection_rate}} | ≥ 90% |
| Photos falling back to the full image | {{fallback_rate}} | — |
| Average confidence of accepted boxes | {{mean_det_score}} | — |
| Share of the photo covered by the box (median) | {{median_box_area_frac}} | — |
| Photos containing 2+ separate food regions | {{pct_multi_region}}% | — |
| Manual review: boxes judged correct | {{manual_precision_good}} | — |

> **FIGURE 7** — insert `06_detection_samples.png`
> **FIGURE 8** — insert `08_detection_summary.png`
> *What these tell us:* [WRITE one line each]

[WRITE: 3-4 sentences. Did we hit the 90% target? Explain the safety net: if
no confident box is found, we keep the whole photo rather than throwing the
image away, so no training data is ever lost.]

### 4.5 How we checked the accuracy

[WRITE: 3-4 sentences. Because nobody drew "correct" boxes for these photos,
there is no automatic score to compute. So we reviewed {{n_reviewed}} boxes by
eye and judged each one correct, partly correct, or wrong. Report the result
and be honest that this is a sample-based estimate, not an exact figure.]

### 4.6 Turning boxes into a model-ready dataset

[WRITE: 4-5 sentences describing the pipeline in plain English:
 1. Cut out the food using the box, with a small margin so nothing is clipped.
 2. Square off the crop so the food is not stretched out of shape.
 3. Resize every crop to the same 224x224 size the model expects.
 4. Standardise the colours to the range pre-trained models are built for.
 5. Split into training / validation / test sets, keeping category
    proportions equal and making sure no photo appears in two sets.]

> **FIGURE 9** — insert `11_before_after.png`
> *Caption:* Original photo with the detected box (top) and the final
> standardised crop fed to the model (bottom).
> *What this tells us:* [WRITE one line]

| Final dataset | Images |
|---|---|
| Training | {{n_train}} |
| Validation | {{n_val}} |
| Test | {{n_test}} |
| **Total** | **{{n_crops}}** |

[WRITE: 1-2 sentences confirming no photo appears in more than one split, and
why a business leader should care — it means the reported accuracy will be
honest.]

---

## 5. Insights and Recommendations
*(Rubric: Actionable Insights — 4 marks)*

### 5.1 Category imbalance

[WRITE: 3-4 sentences. State the imbalance ratio {{imbalance_ratio_full}} and
name the short category. Then state the DECISION: use class weights so the
model is penalised more for mistakes on rare foods, and judge the model on
macro-F1 (which treats every food equally) rather than plain accuracy (which
can be high while completely ignoring a rare food).]

### 5.2 Where detection works, and where it does not

| Where it struggles | Why | What we will do |
|---|---|---|
| Several dishes on one plate | We keep only the single best box | Keep every confident box and classify each separately |
| Soups, stews, sauces | No clear edges, box drifts to the bowl or table | Try segmentation, or bowl-specific prompts |
| Dark or blurry photos | Detector confidence drops | Brightness augmentation, lower the bar, test-time augmentation |
| Close-ups where food fills the frame | Box is nearly the whole photo | Acceptable — the full photo IS the food |
| No "correct" boxes exist | Cannot compute an exact accuracy score | Hand-label a small sample to measure properly |

> **FIGURE 10** — insert `10_failure_cases.png`
> *Caption:* Examples where detection failed or was low-confidence.
> *What this tells us:* [WRITE one line]

[WRITE: 3-4 sentences. Which food categories failed most often, and what do
they have in common? Were dark or blurry photos measurably harder? Quote the
numbers from the notebook.]

### 5.3 Did detection actually add value?

[WRITE: 3-4 sentences — THIS IS A HIGH-VALUE INSIGHT. The median box covers
{{median_box_area_frac}} of the photo. If that is very high, most photos were
already tight close-ups of food and cropping changed little. Say so honestly,
then explain why detection is still essential for the real product: a real
user photographs a cluttered table, not a studio-styled dish.]

### 5.4 Categories we expect the model to confuse

[WRITE: name the look-alike groups you spotted, and give a business
consequence — e.g. confusing ice cream with strawberry shortcake is roughly a
200-calorie logging error, which erodes user trust.]

### 5.5 Other observations

[WRITE: 2-3 bullets — anything else you noticed. Image quality, mislabelled
photos, unusual backgrounds, text/watermarks, anything surprising.]

---

## 6. Plan for the Next Phase
*(Rubric: supports Actionable Insights + shows methodology)*

| Step | Approach |
|---|---|
| Starting point | The {{n_crops}} standardised crops built in this milestone |
| Baseline | A small network trained from scratch, for a reference score |
| Main approach | Transfer learning — EfficientNet-B0, ResNet50, MobileNetV2 |
| Why transfer learning | Re-uses a model already trained on millions of photos; far less data and time needed |
| Making it robust | Augmentation — flips, crops, brightness and blur changes to mimic real user photos |
| Handling imbalance | Class weights, label smoothing |
| How we will judge it | Top-1 accuracy, macro-F1, per-category scores, confusion matrix |
| Choosing the final model | Best validation macro-F1, plus a speed/size check for mobile |
| Beyond Milestone 2 | Map each food to nutrition data, estimate portion size, generate recommendations, integrate with health apps |

**Key risks**

[WRITE: 3-4 bullets.
 - Look-alike foods will cap accuracy.
 - Only {{n_classes}} categories are covered — real users eat far more.
 - A photo cannot reveal hidden oil, sugar or portion size, so calorie
   estimates will carry error.
 - Model trained on well-lit dataset photos may do worse on real user photos.]

---

## 7. Conclusion

[WRITE: 4-6 sentences. We understood the data, proved we can automatically
locate food without any hand-drawn boxes, built a clean and leak-free dataset
of {{n_crops}} standardised food images, and set out a concrete plan for the
classifier. The project is on track for Milestone 2.]

---

### Export to PDF
Open this file in VS Code, Typora, or paste into Google Docs / Word, insert the
figures from `figures/` where marked, then **File > Export / Print > PDF**.
Submit the PDF — **not** a .docx.
