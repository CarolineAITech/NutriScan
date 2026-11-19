#!/usr/bin/env python3
"""Build the NutriScan final capstone presentation on the Great Learning template.

Numbers come from outputs/persist/milestone1_results.json and outputs_m2/milestone2_results.json.
Usage:  ~/nutriscan-env/bin/python make_presentation.py
"""
import copy, json
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

HERE = Path(__file__).parent
TEMPLATE = Path.home() / "Downloads" / "Template Capstone PPT-1-2 (1).pptx"
OUT = HERE / "m2" / "NutriScan_Final_Presentation.pptx"
R1 = json.loads((HERE / "outputs/persist/milestone1_results.json").read_text())
S = json.loads((HERE / "outputs_m2/milestone2_results.json").read_text())
C = S["comparison"]
FIG = HERE / "outputs_m2/figures"

BLUE, DARK, GREY, LIGHT, ORANGE, WHITE = (RGBColor(0x00, 0x70, 0xC0), RGBColor(0x1F, 0x2A, 0x37), RGBColor(0x55, 0x60, 0x6E),
                                          RGBColor(0xE8, 0xF2, 0xFB), RGBColor(0xE8, 0x7A, 0x1E), RGBColor(0xFF, 0xFF, 0xFF))
FONT = "Arial"
X0, W = 0.6, 11.0            # content area: stays clear of the template's blue band and "gl" logo circle
pct = lambda v: f"{100 * v:.0f}%"

prs = Presentation(str(TEMPLATE))
L_CONTENT = prs.slides[1].slide_layout      # "3_Blank": blue band + gl logo
L_TITLE = prs.slides[0].slide_layout        # "4_Blank": wave background

def clear(slide, keep_pictures=False):
    for sh in list(slide.shapes):
        if keep_pictures and sh.shape_type == 13: continue
        sh._element.getparent().remove(sh._element)

def text(slide, x, y, w, h, runs, size=16, color=DARK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    """Add a text box. runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(6)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = t; f = r.font
            f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", color)
    return tb

def title(slide, t):
    text(slide, X0, 0.85, W, 0.9, t, size=36, color=BLUE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Title")

def box(slide, x, y, w, h, fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.fill.background(); s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = 0.08
    if name: s.name = name
    return s

def badge(slide, x, y, label, d=0.6, fill=BLUE, size=20):
    """Numbered circle - the deck's visual motif (echoes the template's ovals)."""
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label; r.font.name = FONT; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE

def card(slide, x, y, w, h, num, head, body, body_size=15):
    box(slide, x, y, w, h)
    badge(slide, x + 0.25, y + 0.25, num)
    text(slide, x + 1.0, y + 0.22, w - 1.2, 0.66, head, size=19, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.25, y + 1.0, w - 0.5, h - 1.15, body, size=body_size, color=DARK)

def new_slide(layout=None):
    return prs.slides.add_slide(layout or L_CONTENT)

def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t

best, base, reg = C[S["best_model"]], C["base_cnn"], C["cnn_aug_reg"]
cc50 = S["conf_coverage"]["0.5"]
s_title, s_problem, s_approach, s_insights, s_recs = list(prs.slides)

# ---------------- 1. Title ----------------
clear(s_title, keep_pictures=True)
text(s_title, 0.45, 2.35, 10.5, 1.2, "NutriScan: AI Food Recognition", size=48, color=BLUE, bold=True)
text(s_title, 0.5, 3.55, 10.0, 0.6, "Snap a meal, know what you ate - in seconds", size=24, color=DARK)
text(s_title, 0.5, 4.35, 10.0, 0.9, [[("Capstone Project - Computer Vision  |  Final Presentation", {"size": 18, "color": GREY})],
                                     [("Presented by: [Your Name]", {"size": 18, "color": GREY, "bold": True})]])
notes(s_title, "~20 sec. Good morning. I'm presenting NutriScan: an AI system that recognises food from a photo so users can log meals in seconds.")

# ---------------- 2. Key takeaways (new) ----------------
s_key = new_slide(); title(s_key, "Key Takeaways")
cw, gap, y = 2.6, 0.2, 2.1
items = [("1", "The problem", "People give up on food diaries because typing in every meal is slow and error-prone."),
         ("2", "Our solution", "Photo in, dish name out. AI first finds the food on the plate, then names the dish."),
         ("3", "What we found", f"Reusing a model pre-trained on 1.3M photos lifted accuracy {best['val_acc'] / base['val_acc']:.1f}x versus building from scratch."),
         ("4", "Our advice", f"Launch as \"snap & confirm\": show the top 3 guesses (right {pct(S['test_top3'])} of the time) and learn from every tap.")]
for i, (n, h, b) in enumerate(items):
    x = X0 + i * (cw + gap)
    box(s_key, x, y, cw, 4.0)
    badge(s_key, x + 0.25, y + 0.3, n)
    text(s_key, x + 0.2, y + 1.1, cw - 0.4, 0.5, h, size=20, color=BLUE, bold=True)
    text(s_key, x + 0.2, y + 1.7, cw - 0.4, 2.2, b, size=16)
notes(s_key, "~50 sec. Four messages to remember. (1) Manual logging is the reason people quit. (2) We built a two-step AI: find the food, then name it. "
             "(3) The biggest win came from reusing a model already trained on millions of photos. (4) We recommend launching as snap-and-confirm, not fully automatic.")

# ---------------- 3. Business problem ----------------
clear(s_problem); title(s_problem, "Business Problem Understanding")
cw2, ch2 = 5.4, 2.3
grid = [("1", "Problem", "Manual meal logging is slow, tedious and inaccurate, so users log inconsistently and abandon diet apps."),
        ("2", "Objectives", "Detect where the food is in a photo, identify the dish, and lay the foundation for real-time calories and macros."),
        ("3", "Scope", f"{R1['n_classes']} dishes (pizza, tacos, samosa, waffles...), {R1['n_images_total']:,} photos. Nutrition lookup and portion size are later phases."),
        ("4", "Constraints", "No boxes marking the food in the data; apple pie has only 256 photos vs 1,000; look-alike dishes; must run fast on a phone.")]
for i, (n, h, b) in enumerate(grid):
    card(s_problem, X0 + (i % 2) * (cw2 + 0.2), 2.0 + (i // 2) * (ch2 + 0.2), cw2, ch2, n, h, b)
notes(s_problem, "~1.5 min. The problem: logging meals by hand is slow, so people stop. Our objective is to turn a photo into a dish name - the first building block for calories. "
                 "Scope: 17 dishes, about 16 thousand photos. Constraints: the photos were never marked with where the food is, one dish has 4x fewer photos, "
                 "some dishes look alike, and the model must be small enough for a phone.")

# ---------------- 4. Modelling approach: the pipeline ----------------
clear(s_approach); title(s_approach, "Modelling Approach Used & Why")
steps = [("1", "Find the food", f"An AI detector that understands plain English is asked to find \"a plate of food\". No manual labelling. Food located in {pct(R1['detection_rate'])} of photos; {pct(R1['manual_precision_good'])} of boxes judged good in review."),
         ("2", "Clean & prepare", "Crop to the food, resize, and split the photos into learning, tuning and final-exam sets - with no duplicates leaking between them."),
         ("3", "Name the dish", "An image-recognition network (CNN). We tried 5 designs and picked the best on photos it had never seen.")]
sw = 3.37
for i, (n, h, b) in enumerate(steps):
    x = X0 + i * (sw + 0.45)
    card(s_approach, x, 2.0, sw, 3.05, n, h, b, body_size=14)
    if i < 2:
        a = s_approach.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + sw + 0.07), Inches(3.3), Inches(0.31), Inches(0.45))
        a.fill.solid(); a.fill.fore_color.rgb = BLUE; a.line.fill.background()
text(s_approach, X0, 5.3, W, 0.45, "Why this approach", size=20, color=BLUE, bold=True)
whys = [("No labelling cost", "the detector needs no hand-drawn boxes"), ("Learns from millions", "reuses knowledge from 1.3M photos"),
        ("Built for phones", "small, fast model; photos stay on the device")]
for i, (h, b) in enumerate(whys):
    text(s_approach, X0 + i * 3.82, 5.8, 3.5, 1.0, [[(h, {"bold": True, "color": DARK})], [(b, {"color": GREY, "size": 14})]], size=16)
notes(s_approach, "~1.5 min. Two AI steps. Step 1 finds the food: we used a detector that understands English, so we simply asked it for 'a plate of food' - "
                  "no expensive manual labelling. It found the food in 90% of photos. Step 2 cleans and splits the data so the final test is fair. "
                  "Step 3 names the dish; we compared five designs.")

# ---------------- 5. Model comparison (new) ----------------
s_cmp = new_slide(); title(s_cmp, "How We Chose the Model")
labels = {"base_cnn": "Basic CNN (from scratch)", "cnn_aug_reg": "CNN + augmentation & regularisation",
          "mobilenet_frozen": "MobileNetV2 pre-trained (frozen)", "mobilenet_finetune": "MobileNetV2 fine-tuned (chosen)",
          "effnet_finetune": "EfficientNet-B0 fine-tuned"}
order = list(C)
cd = CategoryChartData(); cd.categories = [labels[k] for k in order]
cd.add_series("Accuracy on unseen photos (%)", [round(100 * C[k]["val_acc"], 1) for k in order])
gf = s_cmp.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(X0), Inches(2.0), Inches(6.6), Inches(4.9), cd)
ch = gf.chart; ch.has_legend = False; ch.has_title = True
ch.chart_title.text_frame.text = "% of unseen photos named correctly"
tr = ch.chart_title.text_frame.paragraphs[0].runs[0].font; tr.size = Pt(15); tr.bold = True; tr.name = FONT; tr.color.rgb = DARK
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = '0.0"%"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(14); dl.font.name = FONT; dl.font.color.rgb = DARK; dl.font.bold = True
ser = pl.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor(0x9D, 0xC3, 0xE6)
pt = ser.points[order.index(S["best_model"])]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = BLUE
ca = ch.category_axis; ca.reverse_order = True; ca.tick_labels.font.size = Pt(12); ca.tick_labels.font.name = FONT; ca.tick_labels.font.color.rgb = DARK
ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.maximum_scale = 55; va.minimum_scale = 0
lessons = [[("From scratch: ", {"bold": True}), (f"the model memorised its training photos ({pct(base['train_acc'])} there, {pct(base['val_acc'])} on new ones). About 70 photos per dish is too few.", {})],
           [("Augmentation & regularisation: ", {"bold": True}), ("stopped the memorising, but could not add knowledge.", {})],
           [("Transfer learning: ", {"bold": True}), (f"start from a model that has seen 1.3M photos - a {best['val_acc'] / base['val_acc']:.1f}x jump.", {})],
           [("Chosen: MobileNetV2 fine-tuned. ", {"bold": True, "color": BLUE}), (f"Most accurate, only {best['size_MB']:.0f} MB, built to run on phones.", {})]]
text(s_cmp, 7.4, 2.1, 4.2, 4.8, lessons, size=15)
notes(s_cmp, "~1.5 min. We built up step by step. A basic network built from scratch only got about 16% right - it memorised its training photos. "
             "Adding augmentation (flipped, rotated, re-lit copies of photos) and regularisation stopped the memorising but could not add knowledge. "
             "The breakthrough was transfer learning: starting from a model that has already seen 1.3 million photos. "
             "We chose fine-tuned MobileNetV2: the most accurate, and at 9 MB it runs on a phone. Selection was made on validation photos; the test photos were used only once at the end.")

# ---------------- 6. Insights: key numbers ----------------
clear(s_insights); title(s_insights, "Insights from Analysis")
stats = [(pct(R1["detection_rate"]), "of photos: food located automatically"),
         (pct(S["test_acc"]), "of new photos: dish right on the first guess"),
         (pct(S["test_top3"]), "of new photos: right dish in the top 3 suggestions"),
         (pct(cc50["acc"]), f"accuracy when the model is confident ({pct(cc50['coverage'])} of photos)")]
for i, (v, l) in enumerate(stats):
    x = X0 + i * 2.8
    box(s_insights, x, 2.0, 2.6, 2.0)
    text(s_insights, x + 0.1, 2.1, 2.4, 0.95, v, size=44, color=BLUE if i != 3 else ORANGE, bold=True, align=PP_ALIGN.CENTER)
    text(s_insights, x + 0.15, 3.05, 2.3, 0.9, l, size=14, color=GREY, align=PP_ALIGN.CENTER)
f1 = sorted(S["per_class_f1"].items(), key=lambda kv: kv[1])
nice = lambda k: k.replace("_", " ")
cp = S["confused_pairs"]
ins = [[("Distinctive dishes are easiest: ", {"bold": True}), (", ".join(nice(k) for k, _ in f1[-3:][::-1]) + ".", {})],
       [("Look-alikes cause most errors: ", {"bold": True}), ("; ".join(f"{nice(p['true'])} vs {nice(p['predicted'])}" for p in cp[:3]) + ".", {})],
       [("The model knows when it is unsure: ", {"bold": True}), ("confidence tells us when to ask the user instead of guessing.", {})],
       [("Accuracy is a first-prototype figure: ", {"bold": True}), ("trained on 1,700 photos with laptop-level compute.", {})]]
text(s_insights, X0, 4.35, W, 2.6, ins, size=16)
notes(s_insights, "~1.5 min. Four numbers. The AI finds the food in 90% of photos. It names the dish right first time in 44% of new photos - "
                  "but the right dish is in its top three 66% of the time. And when it is confident, it is right 84% of the time. "
                  "Errors come mostly from look-alike dishes. Important: this is a first prototype trained on only 1,700 photos on a laptop; "
                  "the full 16,000-photo dataset on a GPU is the obvious next lever.")

# ---------------- 7. What the app would show (new) ----------------
s_demo = new_slide(); title(s_demo, "What the User Would See")
# Crop the matplotlib titles off the figure; the captions below carry that information in larger text.
src = Image.open(FIG / "m2_04_sample_predictions.png"); sw_, sh_ = src.size
im = HERE / "outputs_m2" / "demo_photos.png"; src.crop((0, int(sh_ * 0.17), sw_, int(sh_ * 0.97))).save(im)
iw, ih = Image.open(im).size
pw = W; ph = pw * ih / iw
if ph > 2.7: ph = 2.7; pw = ph * iw / ih
s_demo.shapes.add_picture(str(im), Inches(X0 + (W - pw) / 2), Inches(1.95), Inches(pw), Inches(ph))
caps = []
for s in S["samples"]:
    if s["true"] == s["pred"]:
        act = "auto-log" if s["confidence"] >= 0.5 else "ask to confirm"
        caps.append(f"{nice(s['true']).title()}: correct at {s['confidence']:.0%} confidence -> {act}")
    else:
        caps.append(f"{nice(s['true']).title()}: first guess {nice(s['pred'])}, right answer offered as option 2 -> one tap fixes it")
yb = 1.95 + ph + 0.25
for i, c in enumerate(caps):
    x = X0 + i * 3.73
    box(s_demo, x, yb, 3.53, 5.75 - yb)   # all captions end above the template's logo circle
    text(s_demo, x + 0.15, yb + 0.05, 3.25, 5.75 - yb - 0.1, c, size=14, anchor=MSO_ANCHOR.MIDDLE)
notes(s_demo, "~1 min. Three real test photos. Samosa: correct and confident, so it can be logged automatically. "
              "Chocolate cake: correct but less confident, so we ask the user to confirm. Nachos: the first guess was tacos, "
              "but nachos was the second suggestion - one tap fixes it, and that tap becomes new training data.")

# ---------------- 8. Recommendations ----------------
clear(s_recs); title(s_recs, "Recommendations")
recs = [("Launch as \"snap & confirm\"", f"Show the top 3 dishes and let the user confirm with one tap (right {pct(S['test_top3'])} of the time). Don't auto-log everything yet."),
        ("Auto-log only confident photos", f"{pct(cc50['acc'])} accurate above 50% confidence. Widen this as the model improves."),
        ("Retrain on the full dataset with a GPU", "Today's model used 1,700 photos and short laptop training. All 16,256 photos is the cheapest accuracy gain."),
        ("Turn every tap into training data", "User confirmations are free labels, especially for look-alike dishes like nachos vs tacos.")]
for i, (h, b) in enumerate(recs):
    y = 2.0 + i * 1.25
    badge(s_recs, X0, y + 0.15, str(i + 1))
    text(s_recs, X0 + 0.85, y, W - 0.85, 0.45, h, size=19, color=BLUE, bold=True)
    text(s_recs, X0 + 0.85, y + 0.45, W - 0.85, 0.7, b, size=16)
notes(s_recs, "~1.5 min. Four recommendations. One: launch as snap-and-confirm with three suggestions. Two: auto-log only when the model is confident. "
              "Three: retrain on all 16 thousand photos with a GPU - the cheapest accuracy gain. Four: every user tap is a free label, so the product improves itself.")

# ---------------- 9. Benefits & next steps (new) ----------------
s_ben = new_slide(); title(s_ben, "Potential Benefits & Next Steps")
text(s_ben, X0, 1.95, 5.4, 0.5, "Benefits of implementing", size=20, color=BLUE, bold=True)
bens = [("Faster logging", "one photo + one tap instead of searching a food list"),
        ("Higher engagement", "easier logging keeps users tracking - the core retention driver"),
        ("Low running cost", f"{best['size_MB']:.0f} MB model runs on the phone - no server cost per photo"),
        ("Privacy", "photos never need to leave the user's device")]
for i, (h, b) in enumerate(bens):
    y = 2.55 + i * 1.1
    badge(s_ben, X0, y + 0.05, "+", d=0.5, fill=ORANGE, size=20)
    text(s_ben, X0 + 0.7, y, 4.8, 1.0, [[(h, {"bold": True})], [(b, {"color": GREY, "size": 14})]], size=16)
box(s_ben, 6.4, 1.95, 5.2, 3.85)
text(s_ben, 6.65, 2.1, 5.2, 0.5, "Roadmap", size=20, color=BLUE, bold=True)
road = [("Next 1-2 months", "Retrain on full data (GPU); pilot with a user group"),
        ("3-6 months", "101 dishes + local cuisine; calories via a nutrition database"),
        ("6-12 months", "Portion size, multi-dish plates, healthier suggestions")]
for i, (h, b) in enumerate(road):
    y = 2.65 + i * 1.03
    badge(s_ben, 6.65, y + 0.05, str(i + 1), d=0.5, size=18)
    text(s_ben, 7.3, y, 4.2, 1.0, [[(h, {"bold": True, "color": DARK})], [(b, {"size": 14, "color": DARK})]], size=16)
notes(s_ben, "~1 min. Benefits: faster logging, which drives engagement; low running cost because the model runs on the phone; and privacy. "
             "Roadmap: retrain and pilot in the next two months, then add more dishes and calorie lookup, then portion sizes and personalised suggestions.")

# ---------------- 10. Thank you / Q&A (new) ----------------
s_end = new_slide(L_TITLE)
for sh in s_title.shapes:
    if sh.shape_type == 13: s_end.shapes._spTree.append(copy.deepcopy(sh._element))
text(s_end, 0.45, 2.55, 10.0, 1.2, "Thank You", size=48, color=BLUE, bold=True)
text(s_end, 0.5, 3.7, 10.0, 0.6, "Questions & Discussion", size=24, color=DARK)
notes(s_end, "Q&A PREP\n"
      "Q: Why is accuracy only 44%? A: First prototype - 1,700 photos, short CPU training. The right dish is in the top 3 66% of the time, "
      "and the confident predictions are 84% right. Retraining on all 16k photos with a GPU is the next step.\n"
      "Q: Why not build a model from scratch? A: We tried - 16%. With ~70 photos per dish it memorises; transfer learning reuses 1.3M photos of knowledge.\n"
      "Q: Why MobileNetV2 over EfficientNet? A: It scored higher on validation (44% vs 36%) and is smaller (9 MB vs 16 MB) - fits on a phone.\n"
      "Q: How did you avoid cheating on the test? A: 70/15/15 split, duplicates removed across sets, model chosen on validation, test used once.\n"
      "Q: Class imbalance? A: Apple pie had 256 vs 1,000 photos; we capped each class at the same count and judged by macro-F1 so every dish counts equally.\n"
      "Q: How do you get calories? A: Next phase - map each dish to a nutrition database (e.g. USDA) and add portion-size estimation.\n"
      "Q: Detection without labels? A: Zero-shot OWL-ViT detector prompted with 'a plate of food'; 90% detection rate, 92% of boxes judged good on manual review of 50.")

# Final order: title, key takeaways, problem, approach, comparison, insights, demo, recommendations, benefits, end
want = [s_title, s_key, s_problem, s_approach, s_cmp, s_insights, s_demo, s_recs, s_ben, s_end]
lst = prs.slides._sldIdLst
ids = {id(s): e for s, e in zip(prs.slides, list(lst))}
for e in list(lst): lst.remove(e)
for s in want: lst.append(ids[id(s)])
OUT.parent.mkdir(exist_ok=True); prs.save(str(OUT)); print("saved", OUT)
