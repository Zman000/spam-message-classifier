import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def create_presentation(output_path="SMS_Spam_Classifier_Presentation.pptx"):
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    PRIMARY = RGBColor(30, 58, 138)    # Deep Navy Blue
    SECONDARY = RGBColor(59, 130, 246) # Blue Accent
    DARK_TEXT = RGBColor(31, 41, 55)   # Charcoal Gray
    LIGHT_BG = RGBColor(248, 250, 252) # Soft Off-White
    WHITE = RGBColor(255, 255, 255)
    CARD_BG = RGBColor(241, 245, 249)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="MACHINE LEARNING COURSE PROJECT"):
        # Top banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = category.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = SECONDARY

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(pptx.enum.shapes.MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY
    bg1.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11), Inches(3.5))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "SMS Spam & Scam Detection Using Supervised Machine Learning"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p_sub = tf1.add_paragraph()
    p_sub.text = "An Explainable, Reproducible Coursework Project for TA Evaluation"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(191, 219, 254)
    p_sub.space_before = Pt(15)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Technologies: Python | Scikit-Learn | Bag-of-Words | Streamlit | Pandas"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = RGBColor(226, 232, 240)
    p_meta.space_before = Pt(35)

    # ==========================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Project Objectives")

    content_box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf2 = content_box2.text_frame
    tf2.word_wrap = True

    points2 = [
        ("The Problem:", "SMS remains a vital communication channel for OTPs, transactions, and personal chats. However, lack of sender authentication makes it a frequent target for smishing (SMS phishing) and financial scams."),
        ("Binary Classification Goal:", "Classify incoming raw SMS texts into Legitimate (Ham = 0) vs. Fraudulent/Promotional (Spam = 1)."),
        ("Key Coursework Requirements:", "Demonstrate fundamental supervised learning concepts taught in the course—strictly avoid unexplainable black-box deep learning or transformers."),
        ("Core Deliverables:", "Data cleaning & duplicate leakage removal, Bag-of-Words feature extraction, 4-model benchmarking, ablation studies, and an interactive Streamlit application.")
    ]

    for title, desc in points2:
        p_t = tf2.add_paragraph() if tf2.paragraphs[0].text else tf2.paragraphs[0]
        p_t.text = f"•  {title}"
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY
        p_t.space_before = Pt(14)

        p_d = tf2.add_paragraph()
        p_d.text = f"    {desc}"
        p_d.font.size = Pt(14)
        p_d.font.color.rgb = DARK_TEXT

    # ==========================================
    # SLIDE 3: WORKFLOW PIPELINE
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "End-to-End Machine Learning Pipeline")

    steps = [
        ("1. Dataset Ingestion", "Load UCI SMS dataset, drop 414 exact duplicates to prevent train-test data leakage."),
        ("2. Exploratory Analysis", "Analyze class balance, message lengths, and top keywords per category."),
        ("3. Stratified Split", "80% Train (4,128) and 20% Test (1,032) preserving exact class ratios (stratify=y)."),
        ("4. Text Preprocessing", "Lowercase, normalize URLs (urladdr) and currency (moneyprize), strip punctuation."),
        ("5. Bag of Words (BoW)", "CountVectorizer (max_features=3000) fitted ONLY on training split."),
        ("6. Supervised Training", "Train Multinomial NB, Logistic Regression, Decision Tree, and Random Forest."),
        ("7. Evaluation & Experiments", "Evaluate on unseen test set, test vocab sizes, analyze false positives/negatives."),
        ("8. Deployment", "Save best pipeline (CountVectorizer + MultinomialNB) and serve via Streamlit UI.")
    ]

    for i, (title, desc) in enumerate(steps):
        col = i % 2
        row = i // 2
        x = Inches(0.8 + col * 5.9)
        y = Inches(1.6 + row * 1.3)

        box = s3.shapes.add_textbox(x, y, Inches(5.6), Inches(1.1))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_st = tf.paragraphs[0]
        p_st.text = title
        p_st.font.size = Pt(15)
        p_st.font.bold = True
        p_st.font.color.rgb = SECONDARY

        p_sd = tf.add_paragraph()
        p_sd.text = desc
        p_sd.font.size = Pt(12)
        p_sd.font.color.rgb = DARK_TEXT

    # ==========================================
    # SLIDE 4: DATASET & DATA HYGIENE
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Dataset Exploration & Duplicate Removal")

    box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf4 = box4.text_frame
    tf4.word_wrap = True

    p4_1 = tf4.paragraphs[0]
    p4_1.text = "UCI SMS Spam Collection Benchmark"
    p4_1.font.size = Pt(16)
    p4_1.font.bold = True
    p4_1.font.color.rgb = PRIMARY

    points4 = [
        "Raw messages: 5,574 labeled SMS messages.",
        "Crucial Data Hygiene: 414 exact duplicates removed. Leaving duplicates causes the same message to appear in both train and test sets (data leakage).",
        "Clean unique total: 5,160 messages.",
        "Ham (Legitimate): 4,518 messages (87.6%).",
        "Spam (Scam): 642 messages (12.4%).",
        "Length asymmetry: Ham messages average 70.9 characters, while Spam averages 137.4 characters (nearly 2x longer)."
    ]
    for pt in points4:
        p = tf4.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(8)

    # Insert class count figure if present
    fig_path = "reports/figures/01_class_counts.png"
    if os.path.exists(fig_path):
        s4.shapes.add_picture(fig_path, Inches(7.2), Inches(1.7), width=Inches(5.3))

    # ==========================================
    # SLIDE 5: FEATURE EXTRACTION: BAG OF WORDS
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Feature Extraction: Bag-of-Words (BoW)")

    box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf5 = box5.text_frame
    tf5.word_wrap = True

    bow_points = [
        ("Bag-of-Words Representation (CountVectorizer):", "Converts raw text into a matrix of token counts. Word order and grammar are disregarded, while frequency of informative tokens is preserved."),
        ("Vocabulary Size (max_features = 3,000):", "Limits feature space to the top 3,000 most frequent tokens across the corpus, eliminating rare noise words and keeping feature dimensions bounded."),
        ("Strict Train/Test Separation (Anti-Leakage):", "The CountVectorizer is fitted ONLY on the 80% training data (fit_transform). The 20% test data is strictly projected using transform()."),
        ("Memory Efficiency (Sparse CSR Matrix):", "Because each SMS contains only a few dozen unique words out of 3,000 vocabulary slots, the resulting matrix is 99.55% zeros. SciPy CSR sparse representation keeps memory consumption minimal.")
    ]

    for title, desc in bow_points:
        p_t = tf5.add_paragraph() if tf5.paragraphs[0].text else tf5.paragraphs[0]
        p_t.text = f"•  {title}"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY
        p_t.space_before = Pt(12)

        p_d = tf5.add_paragraph()
        p_d.text = f"    {desc}"
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = DARK_TEXT

    # ==========================================
    # SLIDE 6: SUPERVISED ALGORITHMS COMPARED
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Four Supervised Learning Algorithms")

    models_info = [
        ("1. Multinomial Naive Bayes", "MultinomialNB()", "Applies Bayes' theorem with feature independence assumption. Computes conditional token probabilities P(word|spam). Extremely fast, uses Laplace smoothing (alpha=1.0) to handle unseen words."),
        ("2. Logistic Regression", "LogisticRegression(max_iter=1000)", "Linear model mapping weighted word counts through the sigmoid function to output posterior probability P(spam|x). Learns positive weights for scam cues and negative weights for ham words."),
        ("3. Decision Tree Classifier", "DecisionTreeClassifier(max_depth=30)", "Non-linear hierarchical partitioning based on Gini impurity. Regularized with max_depth=30 to curb overfitting and branch memorization."),
        ("4. Random Forest Classifier", "RandomForestClassifier(n_estimators=100)", "Ensemble bagging method aggregating 100 decorrelated decision trees. Drastically reduces variance compared to a single decision tree.")
    ]

    for i, (m_name, m_code, m_desc) in enumerate(models_info):
        col = i % 2
        row = i // 2
        x = Inches(0.8 + col * 5.9)
        y = Inches(1.6 + row * 2.5)

        box = s6.shapes.add_textbox(x, y, Inches(5.6), Inches(2.2))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_m = tf.paragraphs[0]
        p_m.text = m_name
        p_m.font.size = Pt(16)
        p_m.font.bold = True
        p_m.font.color.rgb = PRIMARY

        p_c = tf.add_paragraph()
        p_c.text = f"Config: {m_code}"
        p_c.font.size = Pt(12)
        p_c.font.bold = True
        p_c.font.color.rgb = SECONDARY

        p_d = tf.add_paragraph()
        p_d.text = m_desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_TEXT
        p_d.space_before = Pt(4)

    # ==========================================
    # SLIDE 7: MODEL COMPARISON & RESULTS
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Experimental Results on Unseen Test Data (1,032 msgs)")

    box7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.2))
    tf7 = box7.text_frame
    tf7.word_wrap = True

    p7_1 = tf7.paragraphs[0]
    p7_1.text = "Model Evaluation Summary:"
    p7_1.font.size = Pt(16)
    p7_1.font.bold = True
    p7_1.font.color.rgb = PRIMARY

    table_data = [
        "Multinomial Naive Bayes (BEST):",
        "  • Accuracy: 98.45% | Precision: 93.75% | Recall: 93.75% | F1: 0.9375",
        "  • Caught 120 of 128 test spam messages (Only 8 False Negatives).",
        "Logistic Regression:",
        "  • Accuracy: 97.87% | Precision: 95.69% | Recall: 86.72% | F1: 0.9098",
        "Decision Tree:",
        "  • Accuracy: 97.87% | Precision: 94.92% | Recall: 87.50% | F1: 0.9106",
        "Random Forest:",
        "  • Accuracy: 97.09% | Precision: 99.00% | Recall: 77.34% | F1: 0.8684",
        "  • Extreme precision, but missed 29 spam messages."
    ]

    for line in table_data:
        p = tf7.add_paragraph()
        p.text = line
        if ":" in line and not line.startswith(" "):
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = SECONDARY
            p.space_before = Pt(6)
        else:
            p.font.size = Pt(11)
            p.font.color.rgb = DARK_TEXT

    # Insert model comparison chart
    comp_fig = "reports/figures/06_model_comparison.png"
    if os.path.exists(comp_fig):
        s7.shapes.add_picture(comp_fig, Inches(6.8), Inches(1.7), width=Inches(5.7))

    # ==========================================
    # SLIDE 8: ABLATION EXPERIMENTS & ERROR ANALYSIS
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Ablation Studies & Misclassification Analysis")

    box8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf8 = box8.text_frame
    tf8.word_wrap = True

    abl_points = [
        ("Experiment 2: Vocabulary Size Impact:",
         "Tested 500, 1000, 3000, and 5000 terms. Performance improves steadily up to 3,000 words (F1: 0.907 -> 0.9375). Above 3,000 words, scores plateau because the cleaned corpus has ~3,700 unique words."),
        ("Experiment 3: Preprocessing Variations:",
         "Standard cleaning (normalizing URLs and currency) and stopword removal reduced background noise, improving the F1-score from 0.933 (minimal raw) to 0.941."),
        ("Experiment 4: Misclassification (Error) Analysis:",
         "False Positives: Legitimate messages flagged as spam due to promo keywords (e.g. 'FREE SMS', telecom provider names 'airtel').\n"
         "False Negatives: Sophisticated smishing messages intentionally disguised in conversational, friendly language (e.g. 'Sorry I missed your call let's talk').")
    ]

    for title, desc in abl_points:
        p_t = tf8.add_paragraph() if tf8.paragraphs[0].text else tf8.paragraphs[0]
        p_t.text = f"•  {title}"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY
        p_t.space_before = Pt(12)

        for sub_line in desc.split("\n"):
            p_d = tf8.add_paragraph()
            p_d.text = f"    {sub_line}"
            p_d.font.size = Pt(13)
            p_d.font.color.rgb = DARK_TEXT

    # ==========================================
    # SLIDE 9: STREAMLIT APP & RUNTIME STATS
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Deployment, Inference & Performance Stats")

    box9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf9 = box9.text_frame
    tf9.word_wrap = True

    stats_points = [
        ("Streamlit Interactive UI (app.py):", "Clean web interface allowing real-time SMS input, sample message quick-testing, instant Ham/Spam classification, and confidence scoring."),
        ("Pipeline Serialization (models/best_model.pkl):", "Packaged Scikit-Learn Pipeline combining CountVectorizer and MultinomialNB via Joblib. App loads the model in < 0.1s without retraining."),
        ("Training & Inference Speed:", "All 4 models train in under 5 seconds total on CPU. Real-time inference takes < 5 milliseconds per message."),
        ("Reproducibility & Codebase Quality:", "Fully modular structure (src/data_loader.py, preprocessing.py, train.py, evaluate.py), git-versioned, and zero overengineering.")
    ]

    for title, desc in stats_points:
        p_t = tf9.add_paragraph() if tf9.paragraphs[0].text else tf9.paragraphs[0]
        p_t.text = f"•  {title}"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY
        p_t.space_before = Pt(12)

        p_d = tf9.add_paragraph()
        p_d.text = f"    {desc}"
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = DARK_TEXT

    # ==========================================
    # SLIDE 10: CONCLUSION & TA QUESTIONS DEFENSE
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Summary & Key Defense Questions for TA Evaluation")

    box10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf10 = box10.text_frame
    tf10.word_wrap = True

    qa_points = [
        ("Q: Why did you drop duplicate messages before splitting?",
         "A: In SMS datasets, identical spam/promos repeat. If not dropped, the exact same text appears in both train and test splits, creating data leakage and artificially inflated scores."),
        ("Q: Why does Naive Bayes beat Random Forest on F1-score?",
         "A: Random Forest was too conservative (99% precision, but missed 29 spam messages). Naive Bayes achieved 93.75% recall and caught 120 of 128 spams, yielding a superior balance."),
        ("Q: Why CountVectorizer and not TF-IDF or Word2Vec?",
         "A: Coursework focus is on explainable supervised learning from scratch. Bag-of-Words directly demonstrates discrete word frequency distributions and Laplace smoothing principles."),
        ("Q: What is the primary limitation of this model?",
         "A: It is word-order oblivious (Bag-of-Words) and relies on text patterns. It does not inspect network headers or verify real-time phone number ownership.")
    ]

    for title, desc in qa_points:
        p_t = tf10.add_paragraph() if tf10.paragraphs[0].text else tf10.paragraphs[0]
        p_t.text = f"{title}"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY
        p_t.space_before = Pt(8)

        p_d = tf10.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_TEXT

    # Save presentation
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_presentation()
