const pptxgen = require("pptxgenjs");

// ---------- Palette: "Teal Trust" -- fits the trust/misinformation theme ----------
const TEAL = "028090";
const SEAFOAM = "00A896";
const MINT = "02C39A";
const DARK = "013A40";
const WHITE = "FFFFFF";
const INK = "1B1B1B";
const MUTED = "5A6B6E";
const CARD_BG = "F2F7F6";

const FIG = "reports/figures";
const ICON = "assets";

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

const W = 13.33, H = 7.5;

// ---------- Helpers ----------

function darkSlide() {
    const s = pres.addSlide();
    s.background = { color: DARK };
    return s;
}
function lightSlide() {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    return s;
}

function title(s, text, opts = {}) {
    s.addText(text, {
        x: 0.6, y: 0.45, w: W - 1.2, h: 0.9,
        fontFace: "Cambria", fontSize: 30, bold: true, color: opts.color || INK,
        align: "left", margin: 0,
    });
}
function kicker(s, text, opts = {}) {
    s.addText(text.toUpperCase(), {
        x: 0.6, y: 0.18, w: W - 1.2, h: 0.35,
        fontFace: "Calibri", fontSize: 13, bold: true, color: opts.color || TEAL,
        charSpacing: 2, align: "left", margin: 0,
    });
}
function icon(s, name, color, x, y, size) {
    s.addImage({ path: `${ICON}/${name}_${color}.png`, x, y, w: size, h: size });
}
function iconCircle(s, name, x, y, diameter, circleColor, iconColor) {
    s.addShape("ellipse", { x, y, w: diameter, h: diameter, fill: { color: circleColor }, line: { type: "none" } });
    const pad = diameter * 0.26;
    icon(s, name, iconColor, x + pad, y + pad, diameter - pad * 2);
}
function pageNum(s, n) {
    s.addText(String(n), {
        x: W - 0.7, y: H - 0.45, w: 0.4, h: 0.3,
        fontFace: "Calibri", fontSize: 10, color: MUTED, align: "right", margin: 0,
    });
}

// ================= Slide 1: Title =================
{
    const s = darkSlide();
    iconCircle(s, "newspaper", 0.9, 0.85, 0.9, SEAFOAM, DARK);
    s.addText("AI-Powered Fake News Detection", {
        x: 0.9, y: 2.5, w: 11.5, h: 1.3,
        fontFace: "Cambria", fontSize: 44, bold: true, color: WHITE, margin: 0,
    });
    s.addText("A From-Scratch Machine Learning Pipeline for Binary News Classification", {
        x: 0.9, y: 3.65, w: 11, h: 0.6,
        fontFace: "Calibri", fontSize: 20, italic: true, color: MINT, margin: 0,
    });
    s.addShape("line", { x: 0.9, y: 4.55, w: 3.2, h: 0, line: { color: SEAFOAM, width: 2 } });
    s.addText("Akash  |  AI & ML Summer Internship Program 2026", {
        x: 0.9, y: 4.8, w: 10, h: 0.4,
        fontFace: "Calibri", fontSize: 16, color: WHITE, margin: 0,
    });
    s.addText("Indian Institute of Computing and Technology (IICT)", {
        x: 0.9, y: 5.2, w: 10, h: 0.4,
        fontFace: "Calibri", fontSize: 14, color: "9FC9C7", margin: 0,
    });
}

// ================= Slide 2: Problem & Motivation =================
{
    const s = lightSlide();
    kicker(s, "The Problem");
    title(s, "Misinformation Outpaces Manual Fact-Checking");
    const items = [
        { i: "chartline", h: "Scale", t: "Millions of articles published daily -- far beyond what human fact-checkers can review." },
        { i: "speed", h: "Speed", t: "Misinformation spreads fastest in the hours right after publication, before fact-checks land." },
        { i: "balance", h: "Triage", t: "Even an imperfect classifier helps direct scarce human review to the content that needs it most." },
    ];
    const colW = 3.7, gap = 0.4, startX = 0.7, y = 2.1;
    items.forEach((it, idx) => {
        const x = startX + idx * (colW + gap);
        s.addShape("roundRect", { x, y, w: colW, h: 3.6, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
        iconCircle(s, it.i, x + 0.35, y + 0.4, 0.9, TEAL, WHITE);
        s.addText(it.h, { x: x + 0.35, y: y + 1.5, w: colW - 0.7, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: DARK, margin: 0 });
        s.addText(it.t, { x: x + 0.35, y: y + 2.05, w: colW - 0.7, h: 1.4, fontFace: "Calibri", fontSize: 14, color: MUTED, margin: 0, valign: "top" });
    });
    pageNum(s, 2);
}

// ================= Slide 3: Pipeline Overview =================
{
    const s = lightSlide();
    kicker(s, "Project Overview");
    title(s, "A Complete Pipeline, Built From Scratch");
    const steps = [
        { i: "database", t: "Collect &\nClean Data" },
        { i: "broom", t: "Preprocess &\nTokenize" },
        { i: "cogs", t: "Feature\nEngineering" },
        { i: "brain", t: "Train 4\nModels" },
        { i: "chartline", t: "Evaluate &\nValidate" },
    ];
    const n = steps.length, boxW = 2.15, gap = 0.35;
    const totalW = n * boxW + (n - 1) * gap;
    const startX = (W - totalW) / 2, y = 2.6;
    steps.forEach((st, idx) => {
        const x = startX + idx * (boxW + gap);
        s.addShape("roundRect", { x, y, w: boxW, h: 2.3, rectRadius: 0.12, fill: { color: idx % 2 === 0 ? TEAL : SEAFOAM }, line: { type: "none" } });
        icon(s, st.i, WHITE, x + boxW / 2 - 0.45, y + 0.35, 0.9);
        s.addText(st.t, { x: x + 0.1, y: y + 1.35, w: boxW - 0.2, h: 0.85, fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, align: "center", margin: 0 });
        if (idx < n - 1) {
            s.addShape("rightArrow", { x: x + boxW + 0.03, y: y + 1.0, w: gap - 0.06, h: 0.3, fill: { color: MUTED }, line: { type: "none" } });
        }
    });
    s.addText("No pre-built vectorizers or end-to-end libraries -- tokenization, Bag-of-Words, and TF-IDF are all implemented from their mathematical definitions.", {
        x: 1.2, y: 5.3, w: 10.9, h: 0.7, fontFace: "Calibri", fontSize: 14, italic: true, color: MUTED, align: "center", margin: 0,
    });
    pageNum(s, 3);
}

// ================= Slide 4: Dataset Overview =================
{
    const s = lightSlide();
    kicker(s, "Dataset");
    title(s, "38,604 Articles, After Careful Cleaning");
    const stats = [
        { n: "44,898", l: "raw articles\n(Fake + Real)" },
        { n: "6,434", l: "removed: corrupted,\nempty, or duplicate" },
        { n: "38,604", l: "final clean\ndataset" },
        { n: "54.9 / 45.1", l: "Real / Fake\nclass split" },
    ];
    const colW = 2.75, gap = 0.3, startX = 0.7, y = 2.15;
    stats.forEach((st, idx) => {
        const x = startX + idx * (colW + gap);
        s.addShape("roundRect", { x, y, w: colW, h: 2.0, rectRadius: 0.1, fill: { color: CARD_BG }, line: { type: "none" } });
        s.addText(st.n, { x: x + 0.1, y: y + 0.25, w: colW - 0.2, h: 0.75, fontFace: "Cambria", fontSize: 26, bold: true, color: TEAL, align: "center", margin: 0 });
        s.addText(st.l, { x: x + 0.15, y: y + 1.05, w: colW - 0.3, h: 0.8, fontFace: "Calibri", fontSize: 12, color: MUTED, align: "center", margin: 0 });
    });
    s.addImage({ path: `${FIG}/01_class_balance.png`, x: 4.1, y: 4.4, w: 3.3, h: 2.75 });
    s.addText("Kaggle: Fake and Real News Dataset (clmentbisaillon)", {
        x: 0.7, y: 7.05, w: 8, h: 0.3, fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
    });
    pageNum(s, 4);
}

// ================= Slide 5: Feature Leakage Discovery =================
{
    const s = darkSlide();
    kicker(s, "A Key Finding", { color: MINT });
    title(s, "Two Leakage Traps -- Caught Before They Mattered", { color: WHITE });
    const cards = [
        { h: "\u201CSubject\u201D Category", t: "Fake and Real articles used completely separate subject tags with zero overlap -- a model could hit ~100% accuracy without reading a single word." },
        { h: "Reuters Dateline", t: "99.8% of Real articles opened with a wire-service dateline (\u201CWASHINGTON (Reuters) - \u201D). Left in, the model would key off source formatting, not content." },
    ];
    const colW = 5.6, gap = 0.5, startX = 0.9, y = 2.3;
    cards.forEach((c, idx) => {
        const x = startX + idx * (colW + gap);
        s.addShape("roundRect", { x, y, w: colW, h: 3.5, rectRadius: 0.12, fill: { color: "024E58" }, line: { type: "none" } });
        iconCircle(s, "warning", x + 0.4, y + 0.4, 0.85, MINT, DARK);
        s.addText(c.h, { x: x + 0.4, y: y + 1.5, w: colW - 0.8, h: 0.55, fontFace: "Cambria", fontSize: 20, bold: true, color: WHITE, margin: 0 });
        s.addText(c.t, { x: x + 0.4, y: y + 2.05, w: colW - 0.8, h: 1.3, fontFace: "Calibri", fontSize: 14, color: "CFEBE9", margin: 0, valign: "top" });
    });
    s.addText("Both were identified during EDA and corrected before modeling began.", {
        x: 0.9, y: 6.1, w: 11, h: 0.5, fontFace: "Calibri", fontSize: 14, italic: true, color: MINT, align: "center", margin: 0,
    });
    pageNum(s, 5);
}

// ================= Slide 6: Feature Engineering =================
{
    const s = lightSlide();
    kicker(s, "Methodology");
    title(s, "Three Feature Representations, Built From Scratch");
    const cols = [
        { h: "Bag-of-Words", d: "5,000-word vocabulary.\nRaw word counts per document.", tag: "Sparse \u00B7 5,000 dims" },
        { h: "TF-IDF", d: "Length-normalized term frequency \u00D7 smoothed inverse document frequency, L2-normalized.", tag: "Sparse \u00B7 5,000 dims" },
        { h: "Word2Vec", d: "Skip-gram embeddings trained on the corpus. Documents = averaged word vectors.", tag: "Dense \u00B7 100 dims" },
    ];
    const colW = 3.7, gap = 0.4, startX = 0.7, y = 2.15;
    cols.forEach((c, idx) => {
        const x = startX + idx * (colW + gap);
        s.addShape("roundRect", { x, y, w: colW, h: 3.9, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
        s.addShape("roundRect", { x, y, w: colW, h: 0.55, rectRadius: 0.12, fill: { color: SEAFOAM }, line: { type: "none" } });
        s.addShape("rect", { x, y: y + 0.28, w: colW, h: 0.27, fill: { color: SEAFOAM }, line: { type: "none" } });
        s.addText(c.h, { x: x + 0.2, y, w: colW - 0.4, h: 0.55, fontFace: "Cambria", fontSize: 17, bold: true, color: WHITE, valign: "middle", margin: 0 });
        s.addText(c.d, { x: x + 0.25, y: y + 0.8, w: colW - 0.5, h: 1.9, fontFace: "Calibri", fontSize: 13.5, color: INK, margin: 0, valign: "top" });
        s.addText(c.tag, { x: x + 0.25, y: y + 3.15, w: colW - 0.5, h: 0.4, fontFace: "Calibri", fontSize: 11.5, bold: true, italic: true, color: TEAL, margin: 0 });
    });
    pageNum(s, 6);
}

// ================= Slide 7: Models Evaluated =================
{
    const s = lightSlide();
    kicker(s, "Methodology");
    title(s, "Four Models, Four Modeling Philosophies");
    const rows = [
        { i: "search", h: "K-Nearest Neighbors", tag: "Non-Parametric", d: "Classifies by majority vote among the k closest training points." },
        { i: "balance", h: "Logistic Regression", tag: "Parametric", d: "Learns a fixed set of weights defining a linear decision boundary." },
        { i: "diagram", h: "Random Forest", tag: "Ensemble", d: "Combines many decision trees trained on bootstrapped samples." },
        { i: "brain", h: "Neural Network", tag: "Deep Learning", d: "Feedforward network with dropout regularization and early stopping." },
    ];
    let y = 1.95;
    const rowH = 1.18;
    rows.forEach((r) => {
        iconCircle(s, r.i, 0.8, y + 0.09, 0.85, TEAL, WHITE);
        s.addText(r.h, { x: 1.9, y: y - 0.02, w: 4.0, h: 0.5, fontFace: "Cambria", fontSize: 18, bold: true, color: DARK, margin: 0 });
        s.addShape("roundRect", { x: 5.95, y: y + 0.02, w: 2.0, h: 0.4, rectRadius: 0.2, fill: { color: MINT }, line: { type: "none" } });
        s.addText(r.tag, { x: 5.95, y: y + 0.02, w: 2.0, h: 0.4, fontFace: "Calibri", fontSize: 11.5, bold: true, color: DARK, align: "center", valign: "middle", margin: 0 });
        s.addText(r.d, { x: 8.2, y: y - 0.02, w: 4.4, h: 0.6, fontFace: "Calibri", fontSize: 13, color: MUTED, margin: 0, valign: "top" });
        y += rowH;
    });
    pageNum(s, 7);
}

// ================= Slide 8: Final Leaderboard (native chart) =================
{
    const s = lightSlide();
    kicker(s, "Results");
    title(s, "Logistic Regression Leads -- By a Fine Margin");
    const chartData = [
        {
            name: "Test Accuracy (%)",
            labels: ["KNN", "Random Forest", "Neural Network", "Logistic Regression"],
            values: [94.56, 98.01, 98.17, 98.36],
        },
    ];
    s.addChart(pres.ChartType.bar, chartData, {
        x: 0.8, y: 2.0, w: 8.2, h: 4.6,
        barDir: "bar",
        chartColors: [TEAL],
        showTitle: false,
        showValue: true,
        dataLabelPosition: "outEnd",
        dataLabelColor: DARK,
        dataLabelFontSize: 12,
        catAxisLabelColor: INK,
        catAxisLabelFontSize: 13,
        valAxisLabelColor: MUTED,
        valAxisMinVal: 90,
        valAxisMaxVal: 100,
        valGridLine: { color: "E5E5E5", size: 1 },
        catGridLine: { style: "none" },
        showLegend: false,
        barGapWidthPct: 40,
    });
    const notes = [
        { l: "Best Feature Set", v: "TF-IDF (3 of 4 models)" },
        { l: "Top ROC AUC", v: "0.9985 (Logistic Regression)" },
        { l: "Robustness", v: "All models: std dev < 0.3%" },
    ];
    let ny = 2.3;
    notes.forEach(n => {
        s.addText(n.l, { x: 9.4, y: ny, w: 3.2, h: 0.35, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0 });
        s.addText(n.v, { x: 9.4, y: ny + 0.32, w: 3.2, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: TEAL, margin: 0 });
        ny += 1.05;
    });
    pageNum(s, 8);
}

// ================= Slide 9: Curse of Dimensionality =================
{
    const s = lightSlide();
    kicker(s, "Key Finding");
    title(s, "Feature Choice Swung KNN's Accuracy by 12 Points");
    s.addImage({ path: `${FIG}/09_model_feature_heatmap.png`, x: 0.8, y: 2.0, w: 6.4, h: 4.0 });
    s.addShape("roundRect", { x: 7.6, y: 2.1, w: 5.0, h: 3.9, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
    iconCircle(s, "warning", 7.95, 2.4, 0.7, TEAL, WHITE);
    s.addText("The Curse of Dimensionality", { x: 8.85, y: 2.5, w: 3.6, h: 0.5, fontFace: "Cambria", fontSize: 16, bold: true, color: DARK, margin: 0 });
    s.addText("KNN scored just 82.2% on 5,000-dim TF-IDF, but 94.6% on 100-dim Word2Vec embeddings -- the same algorithm, only the feature space changed.", {
        x: 7.95, y: 3.3, w: 4.5, h: 1.5, fontFace: "Calibri", fontSize: 13.5, color: INK, margin: 0, valign: "top",
    });
    s.addText("In high-dimensional sparse spaces, distances between points become nearly uniform -- \u201Cnearest neighbor\u201D stops being meaningful.", {
        x: 7.95, y: 4.75, w: 4.5, h: 1.1, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0, valign: "top",
    });
    pageNum(s, 9);
}

// ================= Slide 10: Simplicity Wins =================
{
    const s = darkSlide();
    kicker(s, "Key Finding", { color: MINT });
    title(s, "More Complexity Didn't Mean More Accuracy", { color: WHITE });
    s.addImage({ path: `${FIG}/11_training_time_comparison.png`, x: 0.7, y: 2.0, w: 6.5, h: 4.4 });
    const stats = [
        { n: "0.17s", l: "Logistic Regression\ntraining time" },
        { n: "~120s", l: "Random Forest\ntraining time" },
        { n: "\u2264 0.35%", l: "Accuracy gap between\ntop 3 models" },
    ];
    let sy = 2.3;
    stats.forEach(st => {
        s.addShape("roundRect", { x: 7.6, y: sy, w: 5.0, h: 1.15, rectRadius: 0.1, fill: { color: "024E58" }, line: { type: "none" } });
        s.addText(st.n, { x: 7.85, y: sy + 0.12, w: 1.9, h: 0.9, fontFace: "Cambria", fontSize: 26, bold: true, color: MINT, valign: "middle", margin: 0 });
        s.addText(st.l, { x: 9.7, y: sy + 0.12, w: 2.75, h: 0.9, fontFace: "Calibri", fontSize: 12.5, color: WHITE, valign: "middle", margin: 0 });
        sy += 1.35;
    });
    pageNum(s, 10);
}

// ================= Slide 11: ROC / PR Curves =================
{
    const s = lightSlide();
    kicker(s, "Results");
    title(s, "All Models Discriminate Well -- KNN Trails Modestly");
    s.addImage({ path: `${FIG}/07_roc_curves.png`, x: 0.6, y: 1.95, w: 5.9, h: 5.15 });
    s.addImage({ path: `${FIG}/08_precision_recall_curves.png`, x: 6.8, y: 1.95, w: 5.9, h: 5.15 });
    pageNum(s, 11);
}

// ================= Slide 12: What Is the Model Learning? =================
{
    const s = darkSlide();
    kicker(s, "Honest Limitation", { color: MINT });
    title(s, "Style vs. Substance: A Necessary Caveat", { color: WHITE });
    s.addShape("roundRect", { x: 0.9, y: 2.15, w: 11.5, h: 1.5, rectRadius: 0.12, fill: { color: "024E58" }, line: { type: "none" } });
    s.addText("Top predictive words: via, pic, featured, getty, image (Fake)  \u00B7  reporters, citing, weekday names (Real)", {
        x: 1.3, y: 2.4, w: 10.7, h: 1.0, fontFace: "Cambria", fontSize: 17, bold: true, color: MINT, align: "center", valign: "middle", margin: 0,
    });
    s.addText("These are formatting and sourcing conventions -- not claims about factual accuracy.", {
        x: 1.3, y: 4.0, w: 10.7, h: 0.6, fontFace: "Calibri", fontSize: 16, color: WHITE, align: "center", margin: 0,
    });
    s.addText("The ~98% accuracy reflects strong performance at detecting this dataset's stylistic signature of Real vs. Fake --\na narrower, more honest claim than \u201Cdetecting fake news\u201D in general.", {
        x: 1.5, y: 4.9, w: 10.3, h: 1.3, fontFace: "Calibri", fontSize: 15, italic: true, color: "CFEBE9", align: "center", margin: 0,
    });
    pageNum(s, 12);
}

// ================= Slide 13: Key Insights =================
{
    const s = lightSlide();
    kicker(s, "Conclusion");
    title(s, "Key Insights");
    const insights = [
        { i: "search", t: "Feature representation matters more than model choice for distance-based methods." },
        { i: "speed", t: "Model complexity did not translate to better accuracy on this task." },
        { i: "warning", t: "Dataset metadata can silently solve a task without genuine content understanding." },
        { i: "lightbulb", t: "What the model learns is narrower than general-purpose fake news detection." },
    ];
    let y = 2.05;
    insights.forEach(it => {
        iconCircle(s, it.i, 0.8, y, 0.7, TEAL, WHITE);
        s.addText(it.t, { x: 1.8, y: y - 0.05, w: 10.6, h: 0.85, fontFace: "Calibri", fontSize: 16, color: INK, valign: "middle", margin: 0 });
        y += 1.15;
    });
    pageNum(s, 13);
}

// ================= Slide 14: Future Work =================
{
    const s = lightSlide();
    kicker(s, "Future Scope");
    title(s, "Where This Project Could Go Next");
    const items = [
        "Test generalization on an independent, more recent fake-news dataset",
        "Train Word2Vec on a larger corpus to test if the gap with TF-IDF closes",
        "Add non-leaking metadata features as a secondary experiment",
        "Extend to a sequence-based architecture (LSTM / attention) for richer context",
    ];
    let y = 2.2;
    items.forEach((t, idx) => {
        s.addShape("roundRect", { x: 0.9, y, w: 0.55, h: 0.55, rectRadius: 0.1, fill: { color: SEAFOAM }, line: { type: "none" } });
        s.addText(String(idx + 1), { x: 0.9, y, w: 0.55, h: 0.55, fontFace: "Cambria", fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
        s.addText(t, { x: 1.75, y: y - 0.02, w: 10.7, h: 0.65, fontFace: "Calibri", fontSize: 15.5, color: INK, valign: "middle", margin: 0 });
        y += 1.05;
    });
    pageNum(s, 14);
}

// ================= Slide 15: Thank You =================
{
    const s = darkSlide();
    iconCircle(s, "rocket", W / 2 - 0.55, 2.1, 1.1, SEAFOAM, DARK);
    s.addText("Thank You", {
        x: 0, y: 3.5, w: W, h: 1.0, fontFace: "Cambria", fontSize: 40, bold: true, color: WHITE, align: "center", margin: 0,
    });
    s.addText("Questions?", {
        x: 0, y: 4.4, w: W, h: 0.6, fontFace: "Calibri", fontSize: 20, italic: true, color: MINT, align: "center", margin: 0,
    });
    s.addText("Akash  \u00B7  AI & ML Summer Internship Program 2026  \u00B7  IICT", {
        x: 0, y: 6.6, w: W, h: 0.4, fontFace: "Calibri", fontSize: 13, color: "9FC9C7", align: "center", margin: 0,
    });
}

pres.writeFile({ fileName: "Fake_News_Detection_Presentation.pptx" }).then(() => {
    console.log("Presentation created.");
});