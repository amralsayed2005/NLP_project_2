# Individual Contribution Report — Project 2: Toxic Spans Detection (SemEval-2021 Task 5)

- **Student Name:** Omar Elgamal
- **Student ID:** `2309116447`
- **GitHub Handle:** `@omarelgamal-web`
- **Primary Role:** Discriminative Sequence Modeling & Feature Engineering Lead (`03_Advanced_CRF.ipynb`)

---

## 1. Core Technical Responsibilities
- **Contextual & Morphological Feature Engineering:** Designed the discriminative feature extractor (`token_to_features` & `word_shape`) over a $\pm 2$ token context window, capturing 2/3/4-character prefixes and suffixes, capitalization, punctuation obfuscation flags (`has_nonalpha`), condensed word shapes (`Xx+`), and adjacent lexical bigrams.
- **Linear-Chain CRF Optimization:** Configured and trained `sklearn_crfsuite.CRF` using L-BFGS optimization with Elastic-Net regularization ($c_1 = 0.1, c_2 = 0.1$) and `all_possible_transitions=True`.
- **Transition Weight & Feature Attribution Analysis:** Extracted and visualized the learned transition weight matrix (verifying $w(\text{O} \to \text{I-TOXIC}) = -10.3399$ and $w(\text{I-TOXIC} \to \text{I-TOXIC}) = +0.6884$) and top discriminative toxic state features.

---

## 2. Key Code & Deliverable Contributions
- Authored and executed `03_Advanced_CRF.ipynb` and exported `crf_metrics.json`.
- Achieved **0.6207 Exact Span Micro-F1** and **0.6859 Exact Span Micro-Recall** (1,269 True Positives, +21 exact spans over the HMM baseline, and **0 invalid BIO transitions**).
