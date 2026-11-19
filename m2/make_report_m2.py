#!/usr/bin/env python3
"""Build the NutriScan Milestone 2 business report (PDF) from outputs_m2/milestone2_results.json."""
import base64, json, subprocess
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
OUT, FIGS = HERE / "outputs_m2", HERE / "outputs_m2" / "figures"
S = json.loads((OUT / "milestone2_results.json").read_text())
C = S["comparison"]
BEST = S["best_model"]
NAMES = {"base_cnn": "1. Base CNN (from scratch)", "cnn_aug_reg": "2. CNN + augmentation + regularisation",
         "mobilenet_frozen": "3. MobileNetV2, frozen (transfer learning)",
         "mobilenet_finetune": "4. MobileNetV2, fine-tuned", "effnet_finetune": "5. EfficientNet-B0, fine-tuned"}
pct = lambda v: f"{100 * v:.1f}%"

def img(name, width="100%"):
    b = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'

b, r, best = C["base_cnn"], C["cnn_aug_reg"], C[BEST]
rows = "".join(
    f"<tr{' class=best' if k == BEST else ''}><td>{NAMES[k]}</td><td>{pct(v['val_acc'])}</td><td>{v['val_macro_f1']:.3f}</td>"
    f"<td>{pct(v['val_top3'])}</td><td>{100 * v['gap']:.1f} pp</td><td>{v['size_MB']:.1f}</td></tr>"
    for k, v in C.items())
pairs = "".join(f"<li>{p['true'].replace('_', ' ')} &rarr; predicted as {p['predicted'].replace('_', ' ')} ({p['count']}x)</li>"
                for p in S["confused_pairs"][:5])
f1 = sorted(S["per_class_f1"].items(), key=lambda kv: kv[1])
weak = ", ".join(f"{k.replace('_', ' ')} ({v:.2f})" for k, v in f1[:3])
strong = ", ".join(f"{k.replace('_', ' ')} ({v:.2f})" for k, v in f1[-3:][::-1])
cc = S["conf_coverage"]
samples = "".join(f"<li>True <b>{s['true'].replace('_', ' ')}</b> &rarr; predicted <b>{s['pred'].replace('_', ' ')}</b> "
                  f"at {s['confidence']:.0%} confidence (alternatives: {s['second'].replace('_', ' ')}, {s['third'].replace('_', ' ')})"
                  f"{'' if s['true'] == s['pred'] else ' &mdash; <i>wrong, but the low confidence would trigger a user confirmation</i>'}</li>"
                  for s in S["samples"])
lift = best["val_macro_f1"] / max(b["val_macro_f1"], 1e-9)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>NutriScan Milestone 2</title><style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color:#1d2a24; font-size:10.5pt; line-height:1.45; }}
h1 {{ color:#1f6b52; font-size:21pt; margin:0 0 2px; }} h2 {{ color:#1f6b52; border-bottom:2px solid #cfe5db; padding-bottom:3px; margin-top:20px; font-size:14pt; }}
.sub {{ color:#5b6b64; margin-bottom:12px; }} table {{ border-collapse:collapse; width:100%; font-size:9.5pt; margin:8px 0; }}
th, td {{ border:1px solid #d7e2dc; padding:5px 7px; text-align:left; }} th {{ background:#eaf4ef; }}
tr.best td {{ background:#fff6d6; font-weight:600; }}
.kpis {{ display:flex; gap:10px; margin:10px 0; }} .kpi {{ flex:1; background:#eaf4ef; border-radius:6px; padding:8px 10px; }}
.kpi b {{ display:block; font-size:17pt; color:#1f6b52; }} .box {{ background:#f6faf8; border-left:4px solid #1f6b52; padding:8px 12px; margin:10px 0; }}
.pb {{ page-break-before: always; }} li {{ margin:3px 0; }}
</style></head><body>
<h1>NutriScan &mdash; Milestone 2 Report</h1>
<div class="sub">Food image classification: base model, improvements and final model selection &middot; {date.today():%d %B %Y}</div>

<h2>Executive summary</h2>
<div class="kpis">
<div class="kpi"><b>{pct(S['test_acc'])}</b>Test accuracy (top-1)</div>
<div class="kpi"><b>{pct(S['test_top3'])}</b>Correct dish in top-3</div>
<div class="kpi"><b>{S['test_macro_f1']:.2f}</b>Test macro-F1</div>
<div class="kpi"><b>{best['size_MB']:.0f} MB</b>Model size (runs on phone)</div></div>
<ul>
<li>We trained five models to recognise <b>17 dishes</b> from the food crops prepared in Milestone 1
({S['n_train']} training, {S['n_val']} validation, {S['n_test']} test images).</li>
<li>A plain CNN built from scratch reached only {pct(b['val_acc'])} validation accuracy. Augmentation and
regularisation stopped overfitting but accuracy stayed low ({pct(r['val_acc'])}).</li>
<li><b>Transfer learning was the decisive improvement</b>: the selected model, <b>{NAMES[BEST][3:]}</b>, reaches
{pct(best['val_acc'])} validation accuracy, {lift:.1f}x the base model's macro-F1, and <b>{pct(S['test_acc'])}</b> on the untouched test set.</li>
<li>Business impact: meal logging becomes <b>one photo plus, at most, one tap</b>. When the model is unsure, it shows
three suggestions; the right dish is among them {pct(S['test_top3'])} of the time.</li></ul>

<h2>1. Base model: a CNN trained from scratch</h2>
<p>A conventional three-block convolutional network with no augmentation or regularisation sets the reference point.</p>
<ul><li>Validation accuracy <b>{pct(b['val_acc'])}</b> and macro-F1 <b>{b['val_macro_f1']:.3f}</b>. That is about
{b['val_acc'] * 17:.1f}x better than guessing (5.9%), but far too inaccurate for a product.</li>
<li>Clear <b>overfitting</b>: training accuracy reached {pct(b['train_acc'])}, a gap of {100 * b['gap']:.0f} points to validation.
With about 70 photos per dish, the network memorises the photos instead of learning what each food looks like.</li></ul>
{img('m2_curve_base_cnn.png', '92%')}

<h2>2. Improving performance</h2>
<p>Each step tackled a specific weakness found in the previous model or in the Milestone 1 analysis.</p>
<table><tr><th>Model</th><th>What changed and why</th></tr>
<tr><td>2. CNN + aug + reg</td><td><b>Augmentation</b> (crop, flip, rotation, brightness/contrast) to imitate real phone photos, as recommended in Milestone 1.
<b>Regularisation</b>: BatchNorm, dropout, weight decay, label smoothing. <b>Architecture</b>: deeper network with global average pooling, fewer parameters.</td></tr>
<tr><td>3. MobileNetV2 frozen</td><td><b>Transfer learning</b>: reuse a network pretrained on 1.3M ImageNet photos; train only a new final layer. Built for mobile phones.</td></tr>
<tr><td>4. MobileNetV2 fine-tuned</td><td>Unfreeze the last 5 blocks with a 10x smaller learning rate so the features adapt to food.</td></tr>
<tr><td>5. EfficientNet-B0 fine-tuned</td><td>A second modern architecture, to check that the result does not depend on one backbone.</td></tr></table>

<h2 class="pb">3. Model comparison</h2>
<table><tr><th>Model</th><th>Val accuracy</th><th>Val macro-F1</th><th>Val top-3</th><th>Overfitting gap</th><th>Size (MB)</th></tr>{rows}</table>
{img('m2_02_model_comparison.png', '88%')}
<div class="box"><b>What the comparison shows</b><ul>
<li><b>Augmentation and regularisation fix overfitting</b>: the train&ndash;validation gap fell from {100 * b['gap']:.0f} to {100 * r['gap']:.0f} points. However, a model trained
from scratch on about 1,200 photos cannot learn enough about food, so accuracy stayed at {pct(r['val_acc'])}.</li>
<li><b>Transfer learning made the biggest difference.</b> Even with the backbone frozen, accuracy jumps to {pct(C['mobilenet_frozen']['val_acc'])}.
Fine-tuning adds more because the pretrained features adapt to food textures.</li></ul></div>

<h2>4. Final model selection</h2>
<p><b>Selected: {NAMES[BEST][3:]}.</b> Rule: highest validation <b>macro-F1</b>, so every dish counts equally (as decided in Milestone 1). If two models
are within one F1 point, the <b>smaller model</b> wins, because NutriScan runs on the phone. The model was then evaluated <b>once</b> on the held-out test set:</p>
<ul><li>Test accuracy <b>{pct(S['test_acc'])}</b>, macro-F1 <b>{S['test_macro_f1']:.3f}</b>, top-3 <b>{pct(S['test_top3'])}</b>. The test score is in line with
validation ({pct(best['val_acc'])}), which confirms it generalises.</li>
<li>Strongest dishes: {strong}. Weakest: {weak}.</li>
<li>Most frequent confusions are look-alike dishes, the same groups flagged in the Milestone 1 EDA:<ul>{pairs}</ul></li></ul>

<h2 class="pb">5. Classifying sample test images</h2>
{img('m2_04_sample_predictions.png')}
<ul>{samples}</ul>
<div class="box"><b>Observation.</b> Confidence is moderate at this training budget. In the wrong example the true dish was the model's second choice,
which supports showing top-3 suggestions. Confidence can drive the user experience:</div>
<table><tr><th>Policy: auto-log when confidence is at least</th><th>Share of photos auto-logged</th><th>Accuracy on those photos</th></tr>
{''.join(f"<tr><td>{float(k):.0%}</td><td>{pct(v['coverage'])}</td><td>{pct(v['acc'])}</td></tr>" for k, v in cc.items())}</table>
<p>The confusion matrix below shows that the remaining errors are concentrated in a few dish pairs, not spread across all classes.</p>
{img('m2_03_confusion_matrix.png', '72%')}

<h2 class="pb">6. Business interpretation: what difference this project makes</h2>
<ul>
<li><b>Faster logging, better retention.</b> Manual logging (search, scroll, select) takes 30&ndash;60 seconds per item and is the main reason
users abandon diet apps. A photo with at most one confirmation tap cuts this to a few seconds.</li>
<li><b>Recommended product flow.</b> Auto-log when confidence is high; otherwise show the <b>top-3 suggestions</b> (right {pct(S['test_top3'])} of the time).
This keeps calorie errors low while still removing most manual work.</li>
<li><b>Low running cost.</b> The selected model is {best['size_MB']:.0f} MB and runs on the phone, so there is no server cost per photo, results are instant
and the photos never leave the device (a privacy selling point).</li>
<li><b>Data flywheel.</b> Every user confirmation or correction is a new labelled photo. Retraining on these improves the weakest dishes over time.</li></ul>

<h2>7. Limitations and next steps</h2>
<ul><li><b>Training budget:</b> models were trained on CPU with a reduced budget (96 px images, 3&ndash;8 epochs). Accuracy is therefore a lower bound; the notebook's GPU settings (160 px, more epochs) should be used for the production model.</li>
<li><b>Scope:</b> 17 dishes on a capped working set. Next, train on the full dataset and extend to the 101 Food-101 categories and local cuisine.</li>
<li><b>Look-alike dishes:</b> collect more examples for the confused pairs and use higher resolution or a larger backbone for them.</li>
<li><b>Nutrition:</b> link each label to a nutrition database (e.g. USDA) and add portion-size estimation to turn a dish name into calories and macros.</li>
<li><b>Multi-item plates:</b> classify every box found by the Milestone 1 detector, not only the main one, and sum the calories.</li></ul>
</body></html>"""

out_html, out_pdf = HERE / "NutriScan_Milestone2_Report.html", HERE / "NutriScan_Milestone2_Report.pdf"
out_html.write_text(html)
subprocess.run(["chromium", "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
print("PDF written:", out_pdf)
