# Individual Contribution Report — Project 2: Toxic Spans Detection (SemEval-2021 Task 5)

- **Student Name:** Yahia Alhamwi
- **Student ID:** `2309115151`
- **GitHub Handle:** `@9yavi`
- **Primary Role:** Evaluation Metrics, Cross-Model Benchmarking & Presentation Lead

---

## 1. Core Technical Responsibilities
- **Exact Span Match Evaluation Suite:** Implemented the strict boundary-level evaluation framework (`extract_exact_spans`, `compute_exact_span_metrics`, and `count_invalid_bio_transitions`) used consistently across all models (`02_Baseline_HMM`, `03_Advanced_CRF`, and `04_DeepLearning_BiLSTM_CRF`), ensuring 100% boundary match evaluation instead of misleading token-level accuracy.
- **Cross-Model Statistical Benchmarking:** Aggregated Micro/Macro Precision, Recall, F1, True/False Positives, and bootstrap confidence intervals across all three paradigms.
- **Repository Governance & Visual Presentation:** Authored `README.md`, `CONTRIBUTIONS.md`, and `generate_presentation.py` to build the 10-slide defense deck (`presentation.pptx`) with Project 1-inspired training dynamics and comparative dot-and-bar charts.

---

## 2. Key Code & Deliverable Contributions
- Standardized strict IOB2 evaluation across all notebooks and verified **0 invalid BIO transitions** across all 80,481 test tokens.
- Produced the final comparative figures (`presentation_assets/`), `README.md`, and `presentation.pptx`.
