# Individual Contribution Report — Project 2: Toxic Spans Detection (SemEval-2021 Task 5)

- **Student Name:** Amr Alsayed
- **Student ID:** `2309115638`
- **GitHub Handle:** `@amralsayed2005`
- **Primary Role:** Project Lead & Deep Learning Architecture Engineer (`04_DeepLearning_BiLSTM_CRF.ipynb`)

---

## 1. Core Technical Responsibilities
- **BiLSTM-CRF Architecture Design:** Designed and implemented the end-to-end PyTorch `BiLSTM_CRF` sequence labeling model combining learned word embeddings ($d_w = 128$, $|V_w| = 18,221$) and 3-character morphological suffix embeddings ($d_s = 32$, $|V_s| = 2,041$).
- **Regularization & Channel Dropout:** Implemented `SpatialDropout1D` (`nn.Dropout2d`, $p = 0.30$) across sequence embedding channels and `nn.LayerNorm` atop the 2-layer Bidirectional LSTM ($256	ext{-d}$) encoder.
- **Vectorized Linear-Chain CRF Layer:** Built the batched log-space Forward-Backward negative log-likelihood loss with toxic-sequence upweighting ($1.8	imes$) and hard structural BIO transition masks ($-10000.0$ on `<START> -> I-TOXIC` and `O -> I-TOXIC`).
- **Training Pipeline & Early Stopping:** Configured `AdamW` optimization, gradient clipping (`max_norm = 5.0`), `ReduceLROnPlateau` scheduling, and Early Stopping on the 10% validation split.

---

## 2. Key Code & Deliverable Contributions
- Authored and executed `04_DeepLearning_BiLSTM_CRF.ipynb` and exported `bilstm_crf_metrics.json`.
- Achieved the project's top performance on the 2,000-comment test set: **0.6578 Exact Span Micro-F1**, **0.7351 Micro-Recall** (1,360 exact True Positives), **0.6137 Macro-F1**, and **0 invalid BIO transitions**.
- Co-authored the methodological bridge connecting Project 1 (toxic comment classification) to Project 2 (fine-grained BIO toxic span extraction).
