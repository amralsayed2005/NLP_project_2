# Individual Contribution Report — Project 2: Toxic Spans Detection (SemEval-2021 Task 5)

- **Student Name:** Adham Mostafa
- **Student ID:** `2309115774`
- **GitHub Handle:** `@AdhamK1`
- **Primary Role:** Probabilistic Graphical Models Lead — Generative HMM (`02_Baseline_HMM.ipynb`)

---

## 1. Core Technical Responsibilities
- **First-Order HMM Formulation:** Designed and implemented `HMMSequenceLabeler` from scratch to model the joint distribution $P(\mathbf{x}, \mathbf{y}) = \prod_{t=1}^T P(y_t \mid y_{t-1}) P(x_t \mid y_t)$ across BIO states.
- **Laplace Smoothing & OOV Handling:** Implemented additive smoothing for state transitions ($\alpha_{\text{trans}} = 10^{-6}$) and lexical emissions ($\alpha_{\text{emit}} = 10^{-3}$), mapping rare singleton tokens ($\text{freq} \le 1$) to `<UNK>` for robust unseen token handling on test comments.
- **Log-Space Viterbi Decoder:** Implemented dynamic programming decoding with backpointers in log-probability space to prevent floating-point underflow and recover exact global MAP tag sequences.

---

## 2. Key Code & Deliverable Contributions
- Authored and executed `02_Baseline_HMM.ipynb` and exported `hmm_metrics.json`.
- Achieved **0.6316 Exact Span Micro-F1** (0.5937 Precision, 0.6746 Recall, 0.5948 Macro-F1, 1,248 True Positives) with **0 invalid BIO transitions** on the held-out test set.
- Visualized the learned $3 \times 3$ state transition probability matrix and conducted qualitative span error analysis.
