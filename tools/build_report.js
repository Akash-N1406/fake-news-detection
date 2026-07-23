const fs = require("fs");
const {
    Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
    Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
    BorderStyle, PageBreak, TableOfContents, LevelFormat, convertInchesToTwip
} = require("docx");

const FIG = "reports/figures";

// ---------- Helpers ----------

function h1(text) {
    return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
    return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
    return new Paragraph({
        children: [new TextRun({ text, size: 22, ...opts })],
        spacing: { after: 160 },
        alignment: AlignmentType.JUSTIFIED,
    });
}
function pMixed(runs) {
    return new Paragraph({
        children: runs.map(r => new TextRun({ size: 22, ...r })),
        spacing: { after: 160 },
        alignment: AlignmentType.JUSTIFIED,
    });
}
function bullet(text) {
    return new Paragraph({
        children: [new TextRun({ text, size: 22 })],
        bullet: { level: 0 },
        spacing: { after: 100 },
    });
}
function caption(text) {
    return new Paragraph({
        children: [new TextRun({ text, italics: true, size: 20 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
    });
}
function image(filename, widthPx, heightPx, maxWidthIn = 5.5) {
    const aspect = heightPx / widthPx;
    const widthIn = maxWidthIn;
    const heightIn = widthIn * aspect;
    return new Paragraph({
        children: [
            new ImageRun({
                type: "png",
                data: fs.readFileSync(`${FIG}/${filename}`),
                transformation: { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) },
            }),
        ],
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 80 },
    });
}
function cell(text, opts = {}) {
    return new TableCell({
        width: { size: opts.width || 2000, type: WidthType.DXA },
        shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2E8" } : undefined,
        children: [new Paragraph({
            children: [new TextRun({ text, bold: !!opts.header, size: 20 })],
            alignment: AlignmentType.CENTER,
        })],
    });
}
function table(headers, rows, colWidths) {
    const widths = colWidths || headers.map(() => Math.floor(9000 / headers.length));
    return new Table({
        width: { size: 9000, type: WidthType.DXA },
        columnWidths: widths,
        rows: [
            new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: widths[i] })) }),
            ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: widths[i] })) })),
        ],
    });
}
function spacer(size = 120) {
    return new Paragraph({ text: "", spacing: { after: size } });
}
function codeBlock(lines) {
    return new Paragraph({
        children: lines.split("\n").flatMap((line, i) => [
            ...(i > 0 ? [new TextRun({ break: 1 })] : []),
            new TextRun({ text: line, font: "Courier New", size: 18 }),
        ]),
        shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
        spacing: { before: 120, after: 200 },
        border: {
            top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
            bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
            left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
            right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
        },
    });
}
function eqn(text) {
    return new Paragraph({
        children: [new TextRun({ text, italics: true, size: 22 })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 100, after: 200 },
    });
}

// ---------- Document Content ----------

const titleBlock = [
    new Paragraph({
        children: [new TextRun({ text: "AI-Powered Fake News Detection Using Text Classification", bold: true, size: 32 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
    }),
    new Paragraph({
        children: [new TextRun({ text: "A From-Scratch Machine Learning Pipeline for Binary News Classification", italics: true, size: 24 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
    }),
    new Paragraph({
        children: [new TextRun({ text: "Akash", size: 22 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
    }),
    new Paragraph({
        children: [new TextRun({ text: "AI & ML Summer Internship Program 2026", size: 20 })],
        alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
        children: [new TextRun({ text: "Indian Institute of Computing and Technology (IICT)", size: 20 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
    }),
];

const abstract = [
    h2("Abstract"),
    p("This project presents a complete, from-scratch machine learning pipeline for classifying news articles as real or fake, built without relying on pre-built end-to-end preprocessing or feature-extraction solutions. Using the Fake and Real News Dataset (38,604 articles after cleaning), we implement manual text cleaning and tokenization, three feature representations (Bag-of-Words, TF-IDF, and Word2Vec embeddings), and four classification algorithms spanning non-parametric (K-Nearest Neighbors), parametric (Logistic Regression), ensemble (Random Forest), and deep learning (Neural Network) paradigms. Logistic Regression achieved the best overall performance (98.36% test accuracy, 0.9985 ROC AUC), closely followed by the Neural Network and Random Forest, while KNN's accuracy varied by 12 percentage points depending on feature representation, illustrating the curse of dimensionality in distance-based classification. Two feature-leakage issues were identified and corrected during exploratory analysis, and all results were validated via cross-validation and repeated-seed robustness checks. We conclude that model complexity did not improve accuracy on this task, and that the strongest predictive signals reflect stylistic and sourcing conventions rather than semantic claims about factual accuracy -- an important scope limitation for interpreting the results."),
    new Paragraph({
        children: [
            new TextRun({ text: "Keywords: ", bold: true, size: 22 }),
            new TextRun({ text: "fake news detection, text classification, natural language processing, TF-IDF, Word2Vec, logistic regression, random forest, neural networks, k-nearest neighbors", italics: true, size: 22 }),
        ],
        spacing: { after: 300 },
    }),
];

// ---- Section 1: Introduction ----
const introSection = [
    h1("1. Introduction"),
    h2("1.1 Problem Statement"),
    p("The proliferation of online news has been accompanied by a corresponding rise in deliberately fabricated or misleading articles -- commonly termed \u201Cfake news\u201D -- that can influence public opinion, election outcomes, and individual decision-making at scale. Manual fact-checking cannot keep pace with the volume of content published daily, motivating automated approaches to flag likely-fake articles for review."),
    p("This project builds a machine learning pipeline, implemented from first principles rather than relying on pre-built end-to-end solutions, to classify news articles as real or fake based on their textual content. The pipeline covers data acquisition and cleaning, manual text preprocessing and tokenization, three distinct feature engineering approaches (Bag-of-Words, TF-IDF, and Word2Vec embeddings), and four classification algorithms spanning non-parametric, parametric, ensemble, and deep learning paradigms."),
    h2("1.2 Importance of Fake News Detection"),
    p("Automated fake news detection matters for three practical reasons: (1) scale -- the volume of published content vastly exceeds human fact-checking capacity; (2) speed -- misinformation often spreads fastest in the hours immediately after publication, before manual fact-checks can be completed; and (3) triage -- even an imperfect classifier is useful as a first-pass filter that directs limited human fact-checking resources toward the content most likely to need it, rather than replacing human judgment entirely."),
    h2("1.3 Related Work"),
    p("Automated fake news detection has been approached from several angles in the literature. Content-based approaches, the focus of this project, classify articles using only their textual features -- word frequencies, syntactic patterns, or learned embeddings -- and are attractive because they require no external verification infrastructure. Propagation-based approaches instead model how a story spreads through a social network, on the premise that fabricated stories often exhibit distinctive sharing patterns independent of their content. Hybrid approaches combine both signals, and knowledge-graph or fact-verification approaches attempt to cross-reference specific claims within an article against a trusted knowledge base, at substantially higher implementation cost."),
    p("Within content-based approaches, classical machine learning pipelines built on Bag-of-Words or TF-IDF features paired with linear or tree-based classifiers remain a strong and widely used baseline, frequently matching or exceeding more complex deep learning approaches on moderately sized, single-domain datasets -- a pattern this project's own results (Sections 4-5) independently reproduce. More recent work has explored transformer-based language models for this task, which can capture longer-range contextual dependencies than the averaged Word2Vec embeddings used here, at substantially higher computational cost and reduced interpretability. This project positions itself deliberately at the classical end of this spectrum, in keeping with the assignment's requirement to implement preprocessing and feature extraction from first principles rather than relying on pre-trained, black-box language models."),
    h2("1.4 Project Scope and Approach"),
    p("Per the assignment brief, this project deliberately avoids black-box, pre-built pipelines. Text cleaning and tokenization were implemented as custom functions rather than library calls such as nltk.word_tokenize; Bag-of-Words and TF-IDF were implemented directly from their underlying mathematical definitions rather than via CountVectorizer / TfidfVectorizer; and all four classification algorithms were evaluated, tuned, and cross-validated using a consistent, leakage-aware methodology described in Section 3."),
    h2("1.5 Report Structure"),
    p("The remainder of this report is organized as follows. Section 2 describes the dataset, its cleaning, and two feature-leakage issues discovered during exploratory analysis. Section 3 details the preprocessing, feature engineering, and modeling methodology, including mathematical formulations for each technique. Section 4 presents quantitative results across all four models and three feature representations, including robustness validation. Section 5 discusses the parametric-versus-non-parametric comparison at the heart of the assignment brief, alongside an honest treatment of what the models are actually learning. Section 6 concludes with key insights, limitations, and future work, and Section 7 provides an appendix of pipeline scripts, hyperparameter search spaces, and representative code excerpts."),
];

// ---- Section 2: Dataset Description ----
const datasetSection = [
    h1("2. Dataset Description"),
    h2("2.1 Source"),
    p("This project uses the Fake and Real News Dataset (Kaggle, published by user clmentbisaillon), comprising two CSV files: Fake.csv (23,481 articles labeled as fake news) and True.csv (21,417 articles labeled as real news). Each record contains four fields: title, text, subject, and date."),
    h2("2.2 Data Cleaning"),
    p("Raw data required several cleaning steps before use: 10 corrupted rows were removed, where the title field contained a malformed URL rather than an article title; 631 rows with empty article text were removed; and 5,793 exact duplicate articles (identical title and text) were removed to prevent train/test leakage. After cleaning, 38,653 articles remained (54.8% real, 45.2% fake), subsequently reduced to 38,604 after a further cleaning step (Section 2.3) removed 49 articles that became empty once a non-content prefix was stripped."),
    h2("2.3 Feature Leakage Investigation"),
    p("Two significant feature leakage issues were identified and addressed during exploratory data analysis, both illustrating that dataset metadata can trivially \u201Csolve\u201D a classification task without reflecting genuine content understanding."),
    pMixed([
        { text: "Subject category. ", bold: true },
        { text: "The subject field perfectly separates the two classes in this dataset: Fake articles are exclusively labeled News, politics, left-news, Government News, US_News, or Middle-east, while Real articles are exclusively labeled politicsNews or worldnews, with zero overlap. A model using subject as a feature would classify articles with near-perfect accuracy without learning anything about fake-versus-real language. Subject was therefore excluded from all feature representations and used only as descriptive metadata." },
    ]),
    pMixed([
        { text: "Reuters wire-service dateline. ", bold: true },
        { text: "During EDA, word-frequency analysis revealed that 99.8% of Real articles (21,159 of 21,196) contained the word \u201CReuters,\u201D typically as a dateline prefix (e.g., \u201CWASHINGTON (Reuters) - \u201D), compared to only 1.3% of Fake articles. Left unaddressed, a model could reach near-99% accuracy simply by detecting this token, rather than learning distinguishing content patterns. A regex-based dateline stripper was implemented to remove this prefix from article text, with \u201Creuters\u201D additionally added to the stopword list to neutralize residual mid-article mentions." },
    ]),
    h2("2.4 Class Balance"),
    p("The final cleaned dataset (38,604 articles) is reasonably balanced: 54.9% real, 45.1% fake (Figure 1). This mild imbalance was accounted for via stratified train/test splitting and stratified k-fold cross-validation throughout, ensuring the class ratio was preserved in every train/validation/test partition."),
    image("01_class_balance.png", 900, 750, 3.2),
    caption("Figure 1. Class balance of the cleaned dataset."),
    h2("2.5 Article and Title Length"),
    p("Fake articles run slightly longer on average (232.5 tokens vs. 220.1 for Real, post-cleaning) and have substantially longer titles (10.1 tokens vs. 7.7 for Real) -- see Figures 2-3. Longer, more elaborate headlines are consistent with the sensationalized framing often associated with fake news content."),
    image("02_article_length_distribution.png", 1200, 750, 4.5),
    caption("Figure 2. Article body length distribution, Real vs. Fake."),
    image("03_title_length_distribution.png", 1200, 750, 4.5),
    caption("Figure 3. Title length distribution, Real vs. Fake."),
    image("04_top_words_by_label.png", 2100, 1050, 5.8),
    caption("Figure 4. Top 20 most frequent words by label, after cleaning."),
];

// ---- Section 3: Methodology ----
const methodologySection = [
    h1("3. Methodology"),
    h2("3.1 Preprocessing and Tokenization"),
    p("All text cleaning and tokenization was implemented manually: (1) lowercasing and removal of URLs, HTML artifacts, and non-alphabetic characters via regular expressions; (2) whitespace tokenization, valid here since punctuation was already stripped in step 1; (3) stopword removal using a hand-curated list of approximately 150 common English function words, plus the corpus-specific addition of \u201Creuters\u201D; and (4) minimum token length filtering (tokens under 3 characters dropped)."),
    h2("3.2 Train/Test Split"),
    p("An 80/20 stratified split (30,883 train / 7,721 test articles, random_state=42) was performed before any vocabulary or feature construction, ensuring no test-set information could leak into feature engineering -- a common methodological error when vocabulary is built from the full dataset prior to splitting."),
    h2("3.3 Feature Engineering"),
    pMixed([
        { text: "Bag-of-Words (BoW). ", bold: true },
        { text: "A 5,000-word vocabulary was built from the training set's most frequent words. Each document was represented as a sparse vector of raw word counts over this vocabulary." },
    ]),
    pMixed([
        { text: "TF-IDF. ", bold: true },
        { text: "Using the same 5,000-word vocabulary, term frequency was computed as count(word, doc) / total_tokens(doc) (length-normalized), and inverse document frequency as the smoothed formula ln((1+N)/(1+df(word))) + 1, with document frequency computed from the training set only. The final TF-IDF vectors were L2-normalized row-wise." },
    ]),
    pMixed([
        { text: "Word2Vec Embeddings. ", bold: true },
        { text: "A skip-gram Word2Vec model (100 dimensions, window size 5, minimum word count 5) was trained on the training corpus using gensim. Document-level vectors were computed as the mean of their constituent words' embeddings." },
    ]),
    h2("3.4 Models"),
    p("Four classification algorithms were implemented via scikit-learn (and TensorFlow/Keras for the neural network), consistent with the assignment's provided code skeleton and spanning four distinct modeling paradigms."),
    h2("3.4.1 K-Nearest Neighbors (Non-Parametric)"),
    p("KNN makes no assumption about the functional form of the decision boundary and stores the entire training set at inference time. For a query document x, KNN computes its distance to every training document, identifies the k nearest neighbors, and assigns the majority class among them:"),
    eqn("\u0177 = mode( { y\u1D62 : x\u1D62 \u2208 N\u2096(x) } )"),
    p("where N\u2096(x) denotes the set of k training points closest to x under Euclidean distance. Because TF-IDF vectors were L2-normalized (Section 3.3), Euclidean distance between them is a monotonic function of cosine similarity, so KNN effectively performs cosine-similarity-based neighbor search on that feature set without additional computation. k was tuned across {3, 5, 7, 9, 11} via a validation split carved from the training data."),
    h2("3.4.2 Logistic Regression (Parametric)"),
    p("Logistic Regression models the log-odds of the positive class (Fake) as a linear function of the input features, learning a fixed-size weight vector w and bias b regardless of training set size -- the defining property of a parametric model:"),
    eqn("P(y=1 | x) = \u03C3(w\u1D40x + b) = 1 / (1 + e\u207B\u207D\u02B7\u1D40\u02E3\u207A\u1D47\u207E)"),
    p("Parameters are fit by minimizing the L2-regularized binary cross-entropy (log) loss over the training set:"),
    eqn("L(w) = \u2212(1/n)\u03A3[y\u1D62 log(\u0177\u1D62) + (1\u2212y\u1D62)log(1\u2212\u0177\u1D62)] + (1/C)\u2016w\u2016\u00B2"),
    p("where C is the inverse regularization strength, tuned across {0.01, 0.1, 1.0, 10.0}. Smaller C applies stronger regularization (shrinking weights toward zero, reducing overfitting risk); larger C allows the model to fit the training data more closely."),
    h2("3.4.3 Random Forest (Ensemble)"),
    p("Random Forest is an ensemble of decision trees trained via bagging (bootstrap aggregating): each of T trees is trained on an independent bootstrap resample of the training set, and at each split, only a random subset of features is considered as split candidates (reducing correlation between trees). Final predictions are made by majority vote across all trees:"),
    eqn("\u0177 = mode( { h\u209C(x) : t = 1, ..., T } )"),
    p("where h\u209C(x) is the prediction of the t-th tree. This project tuned the number of trees T (n_estimators \u2208 {100, 200}) and the maximum tree depth (max_depth \u2208 {None, 30}), where None allows trees to expand until leaves are pure or contain fewer than 2 samples."),
    h2("3.4.4 Neural Network (Deep Learning)"),
    p("The neural network is a feedforward architecture with two hidden layers: Dense(128, ReLU) \u2192 Dropout(0.3) \u2192 Dense(64, ReLU) \u2192 Dropout(0.3) \u2192 Dense(1, sigmoid). Each hidden layer computes:"),
    eqn("h = ReLU(Wx + b),   ReLU(z) = max(0, z)"),
    p("Dropout randomly zeroes 30% of activations during training as a regularizer, reducing co-adaptation between neurons. The network is trained by minimizing binary cross-entropy loss via the Adam optimizer (an adaptive-learning-rate variant of stochastic gradient descent), with early stopping on validation loss (patience = 3 epochs) to halt training once the model stops improving -- both a computational saving and an implicit regularizer, since it prevents the network from continuing to fit noise in the training set after it has converged on the generalizable signal."),
    spacer(),
    p("All hyperparameter tuning used a validation split carved from the training set only; the held-out test set was touched exactly once per model for final evaluation, preserving an unbiased performance estimate."),
    h2("3.5 Evaluation Metrics"),
    p("Each model was evaluated using five standard classification metrics, computed from the confusion matrix of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), where the positive class is defined as \u201CFake\u201D:"),
    eqn("Accuracy = (TP + TN) / (TP + TN + FP + FN)"),
    eqn("Precision = TP / (TP + FP)"),
    eqn("Recall = TP / (TP + FN)"),
    eqn("F1 = 2 \u00D7 (Precision \u00D7 Recall) / (Precision + Recall)"),
    p("Precision answers \u201Cof the articles flagged as Fake, what fraction actually were?\u201D while Recall answers \u201Cof the articles that were actually Fake, what fraction did the model catch?\u201D F1 is their harmonic mean, penalizing models that trade one off heavily for the other. In a deployment context where false accusations of \u201Cfake\u201D carry reputational cost, precision may be weighted more heavily; where missing genuine misinformation is the greater risk, recall matters more (see Section 5.6)."),
    p("In addition, the Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate across all possible classification thresholds, and the Area Under the Curve (AUC) summarizes this into a single threshold-independent number: an AUC of 1.0 indicates perfect separation between classes, while 0.5 indicates performance no better than random guessing. The Precision-Recall curve and its summary statistic, Average Precision, provide a complementary view that is more sensitive to performance on the minority/positive class specifically -- useful here given the dataset's mild class imbalance (Section 2.4)."),
    h2("3.6 Robustness Validation"),
    p("Beyond a single train/test evaluation, robustness was assessed via 5-fold stratified cross-validation for KNN, Logistic Regression, and Random Forest, reporting mean and standard deviation across folds; and repeated training across 5 random seeds for the Neural Network, since its dominant source of run-to-run variance is weight initialization and stochastic gradient descent rather than data partitioning."),
];

// ---- Section 4: Results ----
const resultsSection = [
    h1("4. Results"),
    h2("4.1 Overall Performance"),
    table(
        ["Model", "Type", "Best Features", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
        [
            ["Logistic Regression", "Parametric", "TF-IDF", "98.36%", "98.89%", "97.44%", "98.16%", "0.9985"],
            ["Neural Network", "Deep Learning", "TF-IDF", "~98.2%", "~98.1%", "~97.7%", "~98.0%", "0.997-0.998"],
            ["Random Forest", "Ensemble", "TF-IDF", "98.01%", "99.29%", "96.27%", "97.75%", "0.9980"],
            ["KNN", "Non-Parametric", "Embeddings", "94.56%", "96.28%", "91.47%", "93.84%", "0.9867"],
        ],
        [1900, 1500, 1200, 1100, 1100, 1000, 1000, 1200]
    ),
    spacer(200),
    h2("4.2 Robustness (Cross-Validation / Repeated Seeds)"),
    table(
        ["Model", "Validation Method", "Mean Accuracy", "Std Dev"],
        [
            ["KNN (embeddings)", "5-fold CV", "94.26%", "\u00B10.29%"],
            ["Logistic Regression (TF-IDF)", "5-fold CV", "98.10%", "\u00B10.11%"],
            ["Random Forest (TF-IDF)", "5-fold CV", "97.55%", "\u00B10.18%"],
            ["Neural Network", "5 random seeds", "~98.2%", "\u00B10.04-0.08%"],
        ],
        [3000, 2500, 2000, 1500]
    ),
    spacer(200),
    p("All standard deviations are small (under 0.3 percentage points), indicating each model's reported performance is a stable property of its configuration rather than a result of a favorable single split or initialization."),
    h2("4.3 Feature Set Comparison"),
    p("Accuracy varied substantially by feature representation, most dramatically for KNN (82.24% on TF-IDF vs. 94.56% on embeddings -- Figure 9). For the other three models, TF-IDF was consistently the strongest or near-strongest feature set, with BoW close behind and embeddings trailing by 1-4 percentage points."),
    image("09_model_feature_heatmap.png", 1200, 750, 5.0),
    caption("Figure 9. Accuracy by model and feature set."),
    h2("4.4 Confusion Matrices and Error Patterns"),
    p("Confusion matrices for each model's best configuration are shown in Figure 6. Random Forest showed the highest precision (99.29%) but comparatively lower recall (96.27%), indicating a more conservative model that under-flags some fake articles in exchange for very few false alarms on real news. Logistic Regression offered the most balanced precision/recall trade-off among the top performers."),
    image("06_confusion_matrices_grid.png", 1500, 1500, 5.0),
    caption("Figure 6. Confusion matrices for the best configuration of each model."),
    h2("4.5 Per-Class Classification Report"),
    p("Table 3 breaks down precision, recall, and F1 separately for each class (Real vs. Fake), derived from each model's confusion matrix. This finer-grained view shows that Random Forest and the Neural Network are slightly better at correctly identifying Real articles (higher Real-class recall) than Fake articles, while Logistic Regression is the most balanced across both classes -- consistent with it offering the best overall F1 in Table 1."),
    table(
        ["Model", "Class", "Precision", "Recall", "F1"],
        [
            ["KNN", "Real", "93.27%", "97.10%", "95.15%"],
            ["KNN", "Fake", "96.28%", "91.47%", "93.81%"],
            ["Logistic Regression", "Real", "97.93%", "99.10%", "98.51%"],
            ["Logistic Regression", "Fake", "98.89%", "97.44%", "98.16%"],
            ["Random Forest", "Real", "97.01%", "99.43%", "98.21%"],
            ["Random Forest", "Fake", "99.29%", "96.27%", "97.75%"],
            ["Neural Network", "Real", "98.26%", "98.56%", "98.41%"],
            ["Neural Network", "Fake", "98.24%", "97.87%", "98.06%"],
        ],
        [2400, 1200, 1500, 1500, 1400]
    ),
    caption("Table 3. Per-class precision, recall, and F1 for each model's best configuration."),
    h2("4.6 Model Comparison Overview"),
    p("Figure 5 summarizes single-split test accuracy alongside the robustness-validated (cross-validation or repeated-seed) mean accuracy with error bars, for all four models side by side. The close agreement between the two bars per model, and the small error bars, together confirm that a single test-set evaluation was not misleading for any of the four algorithms in this study."),
    image("05_final_model_comparison.png", 1500, 900, 5.2),
    caption("Figure 5. Test accuracy vs. robustness-validated accuracy, all models."),
    h2("4.7 ROC and Precision-Recall Analysis"),
    p("All four models achieved ROC AUC above 0.98 (Figure 7), with Logistic Regression, Random Forest, and the Neural Network clustering tightly near-perfect discrimination and KNN trailing modestly. Precision-Recall curves (Figure 8) show the same ordering."),
    image("07_roc_curves.png", 1200, 1050, 4.3),
    caption("Figure 7. ROC curves for all four models."),
    image("08_precision_recall_curves.png", 1200, 1050, 4.3),
    caption("Figure 8. Precision-Recall curves for all four models."),
    h2("4.8 Precision / Recall / F1 Comparison"),
    p("Figure 10 presents the same precision, recall, and F1 values from Table 3's best configurations as a grouped bar chart, making the relative trade-offs across models visually direct: Random Forest's precision-recall gap is the widest among the top three, while Logistic Regression and the Neural Network are both comparatively balanced."),
    image("10_metric_comparison.png", 1500, 900, 5.2),
    caption("Figure 10. Precision, Recall, and F1 by model (best configuration)."),
    h2("4.9 Training Time"),
    p("Training time varied by roughly three orders of magnitude across models (Figure 11): Logistic Regression trained in a fraction of a second, while Random Forest required substantially longer for its largest configuration. This cost/benefit trade-off is discussed further in Section 5."),
    image("11_training_time_comparison.png", 1200, 750, 4.5),
    caption("Figure 11. Training time comparison across models (log scale)."),
];

// ---- Section 5: Discussion ----
const discussionSection = [
    h1("5. Discussion: Parametric vs. Non-Parametric Models"),
    h2("5.1 Why the Non-Parametric Model (KNN) Struggled"),
    p("KNN's performance depended dramatically on which feature representation it was given -- far more than any other model in this study. On TF-IDF (5,000 dimensions, sparse), KNN reached only 82.24% accuracy. On Word2Vec embeddings (100 dimensions, dense), the same algorithm reached 94.56% accuracy -- a 12-point swing from a feature representation change alone."),
    p("This is a direct illustration of the curse of dimensionality. KNN's entire decision rule rests on the assumption that \u201Cnearby\u201D points in feature space are similar. In high-dimensional sparse spaces, distances between points become increasingly uniform, which strips the neighbor ranking of much of its meaning. Word2Vec's 100-dimensional dense space, by contrast, compresses semantic information into far fewer dimensions, preserving meaningful distance relationships and letting KNN's neighbor-voting logic actually work."),
    h2("5.2 Why the Parametric Model (Logistic Regression) Won"),
    p("Logistic Regression -- the simplest and most parametric model in this study -- was the best-performing model overall, at 98.36% accuracy and a 0.9985 ROC AUC. Text classification via TF-IDF or Bag-of-Words features is very often close to linearly separable: individual words push the decision toward \u201CFake\u201D or \u201CReal\u201D largely independently of one another, which is precisely the assumption a linear model encodes. Where the data genuinely rewards a linear decision boundary, adding model complexity buys little and costs a great deal: Logistic Regression trained in well under one second, while Random Forest and the Neural Network took substantially longer for a comparable or slightly lower accuracy."),
    p("Logistic Regression's coefficients also offered direct interpretability -- the top coefficients pushing toward \u201CFake\u201D (via, read, pic, watch, com, featured) and toward \u201CReal\u201D (weekday names, reporters, citing) were corroborated independently by Random Forest's feature importances, giving two structurally different models converging on the same signal."),
    h2("5.3 An Honest Limitation: What Is the Model Actually Learning?"),
    p("The words driving classification lean heavily toward formatting and sourcing artifacts rather than semantic content about truthfulness itself. This suggests the ~98% accuracy achieved here reflects strong performance at distinguishing this dataset's Real and Fake articles by their stylistic and sourcing conventions, which is a genuinely useful and demonstrable signal, but is a narrower claim than \u201Cdetecting fake news\u201D in general. A model trained here may generalize less well to fake news that mimics wire-service formatting, or real news published without it."),
    h2("5.4 Ensemble and Deep Learning in Context"),
    p("Random Forest and the Neural Network both performed competitively without matching Logistic Regression's top result. The Neural Network, despite having the most representational capacity of the four models, converged quickly and did not translate that capacity into a measurable accuracy advantage on this task -- a useful finding that additional model capacity is not automatically beneficial when the underlying decision boundary is close to linear."),
    h2("5.5 Practical Deployment Considerations"),
    p("Beyond raw accuracy, the choice of model in a real deployment would also depend on operational constraints this project surfaced concretely. Training time differed by roughly three orders of magnitude (Section 4.9): Logistic Regression retrains in a fraction of a second, making it practical to retrain frequently as new labeled data arrives, whereas Random Forest's multi-minute training cost would make frequent retraining considerably more expensive at scale. Inference-time cost is a separate consideration from training cost: KNN, despite being cheap to \u201Ctrain\u201D (it simply stores the training set), must compare each new article against the full training corpus at prediction time, making it the most expensive of the four models to deploy for high-throughput, low-latency use cases such as real-time social media triage."),
    p("The precision/recall trade-off documented in Table 3 also has direct operational relevance. A deployment that automatically removes or hides flagged content should weight precision heavily, since a high false-positive rate risks suppressing legitimate journalism -- Random Forest's 99.29% precision would be attractive here. Conversely, a deployment that merely routes flagged articles to human fact-checkers for review can tolerate more false positives in exchange for higher recall, since a human reviewer provides a second check -- making Logistic Regression's more balanced profile, or even a lower decision threshold on any of the top three models, a reasonable choice. This distinction between an automated-action system and a human-in-the-loop triage system is a design decision that a raw accuracy or F1 number alone does not resolve, and would need to be made explicitly before deploying any of these models in practice."),
    h2("5.6 Conclusion of Discussion"),
    p("Across single-split evaluation, cross-validation, and repeated-seed robustness checks, the ranking was consistent: Logistic Regression \u2273 Neural Network \u2273 Random Forest \u226B KNN. The central lesson is not that simpler models are always better, but that model choice should follow from the structure of the problem and its features, and that practical deployment constraints -- training cost, inference cost, and the relative cost of false positives versus false negatives -- matter as much as the headline accuracy figure."),
];

// ---- Section 6: Conclusion ----
const conclusionSection = [
    h1("6. Conclusion"),
    h2("6.1 Summary of Findings"),
    p("This project built a complete, from-scratch text classification pipeline for fake news detection, evaluating four algorithms across three feature representations. Logistic Regression achieved the best overall performance, closely followed by the Neural Network and Random Forest. KNN performed substantially worse on high-dimensional sparse features but recovered to a respectable accuracy once given a lower-dimensional dense embedding space. All results were validated via cross-validation and repeated-seed training, with all four models showing low variance."),
    h2("6.2 Key Insights"),
    bullet("Feature representation matters more than model choice for distance-based methods."),
    bullet("Model complexity did not translate to better accuracy on this task."),
    bullet("Dataset metadata can silently solve a classification task without genuine content understanding."),
    bullet("What the model learns is narrower than \u201Cfake news detection\u201D in general -- it reflects this dataset's stylistic signature."),
    spacer(),
    h2("6.3 Limitations"),
    bullet("Dataset scope limited primarily to U.S. political news from 2016-2017."),
    bullet("Word2Vec trained on a modest corpus (~30,883 documents) relative to typical usage."),
    bullet("Strongest predictive signals reflect style/sourcing rather than fact-checking."),
    bullet("No adversarial or out-of-distribution testing was performed."),
    spacer(),
    h2("6.4 Future Scope"),
    bullet("Evaluate generalization on an independent, more recent fake-news dataset."),
    bullet("Train Word2Vec on a larger corpus to test whether the gap with TF-IDF closes."),
    bullet("Incorporate non-leaking metadata features as a secondary experiment."),
    bullet("Extend the Neural Network to a sequence-based architecture (e.g., LSTM or attention-based)."),
];

// ---- Section 7: Appendix ----
const appendixSection = [
    h1("7. Appendix"),
    h2("7.1 Pipeline Script Index"),
    table(
        ["Script", "Purpose"],
        [
            ["day1_load_data.py", "Load, merge, and clean raw Fake.csv/True.csv"],
            ["day2_clean_tokenize.py", "Manual text cleaning, dateline stripping, tokenization"],
            ["day3_eda.py", "Exploratory data analysis and visualization"],
            ["day4_bag_of_words.py", "Train/test split; Bag-of-Words feature construction"],
            ["day5_tfidf.py", "TF-IDF feature construction"],
            ["day6_embeddings.py", "Word2Vec embedding training and document vectorization"],
            ["day7_consolidate_features.py", "Feature set validation and unified loader"],
            ["day8_knn.py", "K-Nearest Neighbors training and evaluation"],
            ["day9_logistic_regression.py", "Logistic Regression training and evaluation"],
            ["day10_random_forest.py", "Random Forest training and evaluation"],
            ["day11_neural_network.py", "Neural Network (Keras) training and evaluation"],
            ["day12_cv_tuning.py", "Cross-validation-based hyperparameter tuning"],
            ["day13_nn_robustness.py", "Neural Network robustness via repeated seeds"],
            ["day14_consolidate_results.py", "Final model comparison table and charts"],
            ["day15_roc_curves.py", "ROC and Precision-Recall curve generation"],
            ["day16_additional_viz.py", "Heatmap, metric comparison, training time charts"],
        ],
        [3200, 5800]
    ),
    spacer(200),
    h2("7.2 Hyperparameter Search Spaces"),
    table(
        ["Model", "Hyperparameter", "Values Tested"],
        [
            ["KNN", "k (n_neighbors)", "3, 5, 7, 9, 11"],
            ["Logistic Regression", "C (inverse regularization)", "0.01, 0.1, 1.0, 10.0"],
            ["Random Forest", "n_estimators", "100, 200"],
            ["Random Forest", "max_depth", "None, 30"],
            ["Neural Network", "Architecture", "Dense(128)-Dropout(0.3)-Dense(64)-Dropout(0.3)-Dense(1)"],
            ["Neural Network", "Early stopping patience", "3 epochs (monitor: validation loss)"],
        ],
        [2500, 2800, 3700]
    ),
    spacer(200),
    h2("7.3 Software Environment"),
    bullet("Python 3.12"),
    bullet("scikit-learn (KNN, Logistic Regression, Random Forest, evaluation metrics)"),
    bullet("TensorFlow/Keras (Neural Network)"),
    bullet("gensim (Word2Vec)"),
    bullet("pandas, numpy, scipy.sparse (data handling)"),
    bullet("matplotlib, seaborn (visualization)"),
    spacer(200),
    h2("7.4 Representative Code Excerpts"),
    p("The following excerpts illustrate the \u201Cfrom scratch\u201D implementation approach used throughout this project, in place of pre-built library calls."),
    pMixed([{ text: "Reuters dateline removal (Section 2.3), from day2_clean_tokenize.py:", bold: true }]),
    codeBlock(
        `DATELINE_RE = re.compile(
    r"^\\s*[A-Z][A-Za-z\\.,/'\\-\\s]*?\\(Reuters\\)\\s*-\\s*"
    r"|^\\s*\\(Reuters\\)\\s*-\\s*"
)

def strip_dateline(text: str) -> str:
    return DATELINE_RE.sub("", text, count=1)`
    ),
    pMixed([{ text: "Smoothed IDF computation (Section 3.3), from day5_tfidf.py:", bold: true }]),
    codeBlock(
        `def compute_idf(token_lists, vocabulary):
    """Document frequency + smooth IDF, from TRAINING set only."""
    n_docs = len(token_lists)
    doc_freq = np.zeros(len(vocabulary), dtype=np.int64)

    for tokens in token_lists:
        seen = set(tokens)  # count each word once per document
        for word in seen:
            col_idx = vocabulary.get(word)
            if col_idx is not None:
                doc_freq[col_idx] += 1

    idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
    return idf, doc_freq`
    ),
    pMixed([{ text: "Bag-of-Words vectorization (Section 3.3), from day4_bag_of_words.py:", bold: true }]),
    codeBlock(
        `def vectorize(token_lists, vocabulary):
    """Sparse document-term matrix of raw word counts."""
    n_docs, n_vocab = len(token_lists), len(vocabulary)
    matrix = sparse.lil_matrix((n_docs, n_vocab), dtype=np.int32)

    for row_idx, tokens in enumerate(token_lists):
        counts = Counter(tokens)
        for word, count in counts.items():
            col_idx = vocabulary.get(word)
            if col_idx is not None:
                matrix[row_idx, col_idx] = count

    return matrix.tocsr()`
    ),
    spacer(200),
    h2("7.5 Sample Test Predictions"),
    p("Table 4 shows a representative sample of test-set articles alongside the true label and the Logistic Regression model's prediction (the best-performing configuration), illustrating both a correct and an incorrect classification case."),
    table(
        ["Article Title (truncated)", "True Label", "Predicted Label", "Correct?"],
        [
            ["Iran's Foreign Minister tweeted on Friday...", "Real", "Real", "Yes"],
            ["ART CONTEST WINNER Disqualified for Being...", "Fake", "Fake", "Yes"],
            ["New York lawmakers on Monday passed an...", "Real", "Real", "Yes"],
            ["WATCH: Trump Supporter Says Something...", "Fake", "Real", "No"],
        ],
        [4500, 1500, 1700, 1300]
    ),
    caption("Table 4. Sample test predictions (Logistic Regression, TF-IDF)."),
    p("The misclassified example is illustrative of the limitation discussed in Section 5.3: articles that do not follow the stylistic and sourcing conventions the model has learned to associate with their true class (for instance, a Fake article written in a more restrained register, without embedded social-media captions) are more likely to be misclassified, since the model's strongest signals are stylistic rather than fact-based."),
];

// ---- References ----
const referencesSection = [
    h1("References"),
    p("[1] A. Ahmed, T. Traore, and S. Saad, \u201CFake and Real News Dataset,\u201D Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"),
    p("[2] F. Pedregosa et al., \u201CScikit-learn: Machine Learning in Python,\u201D Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011."),
    p("[3] M. Abadi et al., \u201CTensorFlow: Large-Scale Machine Learning on Heterogeneous Systems,\u201D 2015. [Online]. Available: https://www.tensorflow.org/"),
    p("[4] R. \u0158eh\u016F\u0159ek and P. Sojka, \u201CSoftware Framework for Topic Modelling with Large Corpora,\u201D in Proc. LREC Workshop on New Challenges for NLP Frameworks, 2010."),
    p("[5] T. Mikolov, K. Chen, G. Corrado, and J. Dean, \u201CEfficient Estimation of Word Representations in Vector Space,\u201D arXiv:1301.3781, 2013."),
];

// ---------- Assemble Document ----------

const doc = new Document({
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 }, // US Letter
                margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
            },
        },
        children: [
            ...titleBlock,
            ...abstract,
            new Paragraph({ children: [new PageBreak()] }),
            h2("Table of Contents"),
            new TableOfContents("Table of Contents", {
                hyperlink: true,
                headingStyleRange: "1-2",
            }),
            new Paragraph({ children: [new PageBreak()] }),
            ...introSection,
            ...datasetSection,
            new Paragraph({ children: [new PageBreak()] }),
            ...methodologySection,
            ...resultsSection,
            new Paragraph({ children: [new PageBreak()] }),
            ...discussionSection,
            ...conclusionSection,
            new Paragraph({ children: [new PageBreak()] }),
            ...appendixSection,
            ...referencesSection,
        ],
    }],
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("Fake_News_Detection_IEEE_Report.docx", buffer);
    console.log("Document created: Fake_News_Detection_IEEE_Report.docx");
});