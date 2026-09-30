// SELMA3D 2026 talk (7-8 min): metric results only, no rankings.
const pptxgen = require("pptxgenjs");
const path = require("path");
const NOTES = require("./script.json"); // talk script, one entry per slide

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Validating for the Unseen - SELMA3D 2026";

const C = {
  dark: "0E1A2E", ink: "14213D", blue: "2B6CB0", lblue: "8FB3DC", orange: "E0782A",
  tint: "EEF3FA", otint: "FDF1E7", muted: "5A6477", white: "FFFFFF", line: "D9DEE7",
};
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.333, M = 0.6;
const img = (f) => path.join(__dirname, "img", f);
const RATIO = {
  "calibration_drift.png": 2.442, "calibration_prelim.png": 2.687, "cfos_detection.png": 2.326,
  "dataset_annotated.png": 2.653, "dataset_pool.png": 1.667, "fg_ratio_candidates.png": 2.381,
  "flowchart.png": 2.32, "qual_holdout.png": 0.977, "qual_unseen.png": 1.875, "router_space.png": 1.532,
  "ssl_v2_loss.png": 2.424, "ttt_split.png": 2.326,
};

function title(slide, text, opts = {}) {
  const x = opts.x ?? M;
  slide.addText(text, {
    x, y: 0.35, w: W - x - M, h: 0.85, fontFace: HEAD, fontSize: opts.size ?? 30, bold: true,
    color: opts.color ?? C.ink, valign: "middle", margin: 0, isTextBox: true,
  });
}
function subtitle(slide, text, y = 1.18, x = M) {
  slide.addText(text, {
    x, y, w: W - x - M, h: 0.45, fontFace: BODY, fontSize: 16, color: C.muted, margin: 0, isTextBox: true,
  });
}
function findingBadge(slide, n) {
  slide.addShape(pres.shapes.OVAL, { x: M, y: 0.47, w: 0.62, h: 0.62, fill: { color: C.orange }, line: { color: C.orange } });
  slide.addText(String(n), {
    x: M, y: 0.47, w: 0.62, h: 0.62, fontFace: HEAD, fontSize: 22, bold: true, color: C.white,
    align: "center", valign: "middle", margin: 0, isTextBox: true,
  });
}
function picture(slide, file, x, y, w) {
  const h = w / RATIO[file];
  slide.addImage({ path: img(file), x, y, w, h });
  return h;
}
function stat(slide, x, y, w, h, value, label, color = C.blue, fill = C.tint) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  slide.addText(value, {
    x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h * 0.5, fontFace: HEAD, fontSize: 26, bold: true, color,
    valign: "bottom", margin: 0, isTextBox: true,
  });
  slide.addText(label, {
    x: x + 0.2, y: y + 0.1 + h * 0.5, w: w - 0.4, h: h * 0.5 - 0.2, fontFace: BODY, fontSize: 13, color: C.ink,
    valign: "top", margin: 0, isTextBox: true,
  });
}
function caption(slide, text, x, y, w, align = "left") {
  slide.addText(text, { x, y, w, h: 0.35, fontFace: BODY, fontSize: 11, italic: true, color: C.muted, align, margin: 0, isTextBox: true });
}

// 1 ---------------------------------------------------------------- title
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addText("Validating for the Unseen", {
    x: M, y: 1.75, w: 5.9, h: 1.3, fontFace: HEAD, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true,
  });
  s.addText("Finding cells, nerves and vessels in 3D microscope images, including types the model was never taught", {
    x: M, y: 3.15, w: 5.7, h: 1.3, fontFace: BODY, fontSize: 18, color: "CADCFC", margin: 0, isTextBox: true,
  });
  s.addText("Team Four and Furious  ·  SELMA3D 2026 challenge", {
    x: M, y: 4.85, w: 5.7, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.orange, margin: 0, isTextBox: true,
  });
  s.addText("Ayush Gupta  ·  Harshita Gupta  ·  Anushka Dixit", {
    x: M, y: 5.3, w: 5.7, h: 0.4, fontFace: BODY, fontSize: 14, color: "CADCFC", margin: 0, isTextBox: true,
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.75, y: 1.85, w: 5.95, h: 3.25, rectRadius: 0.1, fill: { color: C.white }, line: { color: C.white } });
  picture(s, "qual_unseen.png", 6.875, 1.955, 5.7);
  s.addText("Our results on structure types that nobody labelled", {
    x: 6.75, y: 5.22, w: 5.95, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: "AFC3E3", align: "center", margin: 0, isTextBox: true,
  });
  s.addNotes(NOTES[0]);
}

// 2 ---------------------------------------------------------------- challenge
{
  const s = pres.addSlide();
  title(s, "The challenge: find structures nobody labelled");
  subtitle(s, "Learn from unlabelled images, train on a few labelled ones, then test on mostly new types");
  const y0 = 1.95, cw = 3.9, ch = 1.3, gap = 0.3;
  stat(s, M, y0, cw, ch, "3,400", "unlabelled images of 17 structure types, for the model to learn from on its own");
  stat(s, M, y0 + ch + gap, cw, ch, "82", "labelled images of just 6 types (the only labels we could use)");
  stat(s, M, y0 + 2 * (ch + gap), cw, ch, "17 of 26", "test images show types that had no labels at all", C.orange, C.otint);
  const h = picture(s, "dataset_pool.png", 4.85, 1.9, 7.85);
  caption(s, "One example of each of the 17 structure types", 4.85, 1.9 + h + 0.05, 7.85, "center");
  s.addNotes(NOTES[1]);
}

// 3 ---------------------------------------------------------------- annotated data
{
  const s = pres.addSlide();
  title(s, "Two groups, two ways of scoring");
  const h = picture(s, "dataset_annotated.png", (W - 11.4) / 2, 1.3, 11.4);
  const y = 1.3 + h + 0.3, cw = (W - 2 * M - 0.4) / 2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: cw, h: 1.05, rectRadius: 0.08, fill: { color: C.tint }, line: { color: C.tint } });
  s.addText([
    { text: "Separate objects  ", options: { bold: true, color: C.blue } },
    { text: "plaques, nuclei, neurons: the score checks whether each single object is found correctly", options: { color: C.ink } },
  ], { x: M + 0.2, y: y + 0.1, w: cw - 0.4, h: 0.85, fontFace: BODY, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + cw + 0.4, y, w: cw, h: 1.05, rectRadius: 0.08, fill: { color: C.otint }, line: { color: C.otint } });
  s.addText([
    { text: "Connected structures  ", options: { bold: true, color: C.orange } },
    { text: "vessels, nerves, axons: the score checks whether the tubes stay connected", options: { color: C.ink } },
  ], { x: M + cw + 0.6, y: y + 0.1, w: cw - 0.4, h: 0.85, fontFace: BODY, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes(NOTES[2]);
}

// 4 ---------------------------------------------------------------- pipeline
{
  const s = pres.addSlide();
  title(s, "Our method");
  const h = picture(s, "flowchart.png", 0.9, 1.3, W - 1.8);
  caption(s, "Blue: our additions. Orange: the rule that decides \u201ctube-like or not\u201d. Almost everything was trained on one laptop graphics card.", M, 1.3 + h + 0.12, W - 2 * M, "center");
  s.addNotes(NOTES[3]);
}

// 5 ---------------------------------------------------------------- SSL
{
  const s = pres.addSlide();
  title(s, "Teaching the model with unlabelled images");
  subtitle(s, "We hide parts of an image and the model fills them in; but these images are about 97% empty background");
  const rows = [
    ["×10", "Real structures count ten times more, so the model cannot just learn background"],
    ["50%", "of the training pieces focus on real structures, not empty space"],
    ["=", "Images are prepared the same way as in later training"],
  ];
  rows.forEach(([k, t], i) => {
    const y = 2.0 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.95, h: 0.95, fill: { color: C.blue }, line: { color: C.blue } });
    s.addText(k, { x: M, y, w: 0.95, h: 0.95, fontFace: HEAD, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: M + 1.2, y, w: 4.6, h: 0.95, fontFace: BODY, fontSize: 15, color: C.ink, valign: "middle", margin: 0, isTextBox: true });
  });
  const h = picture(s, "ssl_v2_loss.png", 6.75, 2.05, 6.0);
  caption(s, "The model keeps improving over about 19,000 training steps", 6.75, 2.05 + h + 0.05, 6.0, "center");
  s.addNotes(NOTES[4]);
}

// 6 ---------------------------------------------------------------- finding 1
{
  const s = pres.addSlide();
  findingBadge(s, 1);
  title(s, "The usual check fooled us; a simple count did not", { x: M + 0.85, size: 28 });
  const h = picture(s, "calibration_prelim.png", M, 1.45, 8.3);
  caption(s, "Test score vs. how much each model marks on new types (compared with v15)", M, 1.45 + h + 0.05, 8.3);
  const x = 9.25, cw = 3.5, ch = 1.25;
  stat(s, x, 1.5, cw, ch, "×1.19 → 0.537", "v11 marks too much on new types", C.muted);
  stat(s, x, 1.5 + ch + 0.25, cw, ch, "×1.00 → 0.616", "v15: our main model", C.blue);
  stat(s, x, 1.5 + 2 * (ch + 0.25), cw, ch, "×0.75 → 0.573", "v17 marks too little, even though the usual check liked it", C.orange, C.otint);
  s.addText("On familiar types, all three models mark almost the same amount.", {
    x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true,
  });
  s.addNotes(NOTES[5]);
}

// 7 ---------------------------------------------------------------- finding 2
{
  const s = pres.addSlide();
  findingBadge(s, 2);
  title(s, "Not forgetting, just marking the wrong amount", { x: M + 0.85, size: 28 });
  subtitle(s, "We hid two types (cells and nerves) during training and tested only on them", 1.2, M + 0.85);
  const h = picture(s, "calibration_drift.png", M, 1.95, 7.6);
  caption(s, "Left: scores with different cut-offs. Right: how much is marked, compared with the truth", M, 1.95 + h + 0.05, 7.6);
  const x = 8.6, cw = 4.15, ch = 1.2;
  stat(s, x, 1.95, cw, ch, "0.432 vs 0.413", "with the right amount marked, the fully trained model is better", C.blue);
  stat(s, x, 1.95 + ch + 0.25, cw, ch, "×3.0 vs ×0.6", "nerves marked 3 times too much, cells about half as much", C.orange, C.otint);
  stat(s, x, 1.95 + 2 * (ch + 0.25), cw, ch, "0.334 → 0.360", "fixed by matching the amount to a steadier model, no labels needed", C.blue);
  s.addNotes(NOTES[6]);
}

// 8 ---------------------------------------------------------------- routing + anchoring
{
  const s = pres.addSlide();
  title(s, "Sending tube-like images to bigger models");
  subtitle(s, "Bigger models were better on nerves but worse on cells, so we decide by the shape of what was found");
  const h1 = picture(s, "router_space.png", M, 1.85, 5.2);
  caption(s, "The rule: is it one big network, or long thin objects?", M, 1.85 + h1 + 0.05, 5.2);
  const h2 = picture(s, "fg_ratio_candidates.png", 6.2, 1.85, 6.55);
  caption(s, "How much each version marks, compared with v15", 6.2, 1.85 + h2 + 0.05, 6.55);
  const y = 1.85 + h2 + 0.55;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.2, y, w: 6.55, h: 1.45, rectRadius: 0.08, fill: { color: C.tint }, line: { color: C.tint } });
  s.addText([
    { text: "The rule: ", options: { bold: true, color: C.blue } },
    { text: "got all 26 test images right.", options: { color: C.ink, breakLine: true } },
    { text: "The fix: ", options: { bold: true, color: C.orange } },
    { text: "bigger models decide where, v15 decides how much, so the amount stays the same.", options: { color: C.ink } },
  ], { x: 6.4, y: y + 0.1, w: 6.15, h: 1.25, fontFace: BODY, fontSize: 14, valign: "middle", paraSpaceAfter: 6, margin: 0, isTextBox: true });
  s.addNotes(NOTES[7]);
}

// 9 ---------------------------------------------------------------- finding 3
{
  const s = pres.addSlide();
  findingBadge(s, 3);
  title(s, "Learning from each test image: good for tubes, bad for cells", { x: M + 0.85, size: 28 });
  subtitle(s, "The model trains briefly on its own most confident guesses for each image", 1.2, M + 0.85);
  const h = picture(s, "ttt_split.png", M, 1.95, 7.7);
  caption(s, "Each line is one image (coloured: better, grey: worse)", M, 1.95 + h + 0.05, 7.7);
  const x = 8.7, cw = 4.05, ch = 1.2;
  stat(s, x, 1.95, cw, ch, "+0.042", "better on nerves, in 6 out of 6 images", C.blue);
  stat(s, x, 1.95 + ch + 0.25, cw, ch, "−0.038", "worse on cells, in 8 out of 9 images", C.orange, C.otint);
  stat(s, x, 1.95 + 2 * (ch + 0.25), cw, ch, "0.334 → 0.353", "when used only on tube-like images", C.blue);
  s.addNotes(NOTES[8]);
}

// 10 --------------------------------------------------------------- results (native chart)
{
  const s = pres.addSlide();
  title(s, "Our results");
  subtitle(s, "Both of our entries scored almost exactly the same, so the credit goes to the v15 model");
  s.addChart(pres.charts.BAR, [
    { name: "First test (26 images)", labels: ["Objects: overlap", "Objects: found", "Tubes: overlap", "Tubes: connected"], values: [0.629, 0.453, 0.633, 0.747] },
    { name: "Final test", labels: ["Objects: overlap", "Objects: found", "Tubes: overlap", "Tubes: connected"], values: [0.547, 0.360, 0.653, 0.737] },
  ], {
    x: M, y: 1.85, w: 7.6, h: 4.9, barDir: "col", barGapWidthPct: 60,
    chartColors: [C.lblue, C.blue], valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: C.ink,
    catAxisLabelColor: C.ink, catAxisLabelFontSize: 12, valAxisLabelColor: C.muted, valAxisLabelFontSize: 11,
    valGridLine: { color: "E5E9F0", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: C.ink,
  });
  const x = 8.6, cw = 4.15, ch = 1.1, g = 0.2;
  stat(s, x, 1.9, cw, ch, "0.653", "overlap score for tubes (vessels, nerves)", C.blue);
  stat(s, x, 1.9 + (ch + g), cw, ch, "0.738", "connection score for tubes", C.blue);
  stat(s, x, 1.9 + 2 * (ch + g), cw, ch, "±0.15", "steady results from image to image", C.blue);
  stat(s, x, 1.9 + 3 * (ch + g), cw, ch, "0.547 / 0.360", "separate objects like cells were harder", C.orange, C.otint);
  s.addNotes(NOTES[9]);
}

// 11 --------------------------------------------------------------- qualitative
{
  const s = pres.addSlide();
  title(s, "What our results look like");
  const hh = 4.95, wh = hh * RATIO["qual_holdout.png"];
  s.addImage({ path: img("qual_holdout.png"), x: M, y: 1.35, w: wh, h: hh });
  caption(s, "Compared with the correct answers, on images not used in training", M, 1.35 + hh + 0.08, wh);
  const x2 = M + wh + 0.4, w2 = W - M - x2;
  const h2 = picture(s, "qual_unseen.png", x2, 1.75, w2);
  caption(s, "Completely new types, with no labels available", x2, 1.75 + h2 + 0.08, w2);
  s.addNotes(NOTES[10]);
}

// 12 --------------------------------------------------------------- failure mode
{
  const s = pres.addSlide();
  title(s, "Where it still struggles: faint and small cells");
  subtitle(s, "We checked more than 12,000 cells of a type the model never saw labels for");
  const h = picture(s, "cfos_detection.png", M, 1.9, 7.9);
  caption(s, "How many cells were found, by size (left) and brightness (right)", M, 1.9 + h + 0.05, 7.9);
  const x = 8.9, cw = 3.85, ch = 1.2;
  stat(s, x, 1.95, cw, ch, "39%", "of the cells were found at all", C.orange, C.otint);
  stat(s, x, 1.95 + ch + 0.25, cw, ch, "0.50×", "found cells were drawn about half their real size", C.orange, C.otint);
  stat(s, x, 1.95 + 2 * (ch + 0.25), cw, ch, "3% vs 20%", "faintest vs. brightest cells found correctly", C.orange, C.otint);
  s.addNotes(NOTES[11]);
}

// 13 --------------------------------------------------------------- take-aways
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addText("Three simple lessons", { x: M, y: 0.5, w: 8, h: 0.9, fontFace: HEAD, fontSize: 36, bold: true, color: C.white, margin: 0, isTextBox: true });
  const items = [
    ["Test on types your model has never seen", "The usual check on familiar types fooled us; a simple count on new types did not."],
    ["Watch how much your model marks", "On new types, marking too much or too little both cost points."],
    ["Getting worse is often fixable", "Late in training the model marks the wrong amount, and this can be fixed without labels."],
  ];
  items.forEach(([h, t], i) => {
    const y = 1.7 + i * 1.45;
    s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.75, h: 0.75, fill: { color: C.orange }, line: { color: C.orange } });
    s.addText(String(i + 1), { x: M, y, w: 0.75, h: 0.75, fontFace: HEAD, fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: M + 1.05, y: y, w: 11.0, h: 0.45, fontFace: HEAD, fontSize: 21, bold: true, color: C.white, valign: "top", margin: 0, isTextBox: true });
    s.addText(t, { x: M + 1.05, y: y + 0.45, w: 11.0, h: 0.6, fontFace: BODY, fontSize: 15, color: "CADCFC", valign: "top", margin: 0, isTextBox: true });
  });
  s.addText("Next: find faint and small cells better.", {
    x: M, y: 6.25, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 14, italic: true, color: C.orange, margin: 0, isTextBox: true,
  });
  s.addNotes(NOTES[12]);
}

pres.writeFile({ fileName: path.join(__dirname, "SELMA3D_2026_presentation.pptx") }).then((f) => console.log("wrote", f));
