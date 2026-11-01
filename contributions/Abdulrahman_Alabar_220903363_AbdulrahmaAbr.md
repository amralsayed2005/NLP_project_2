# Individual Contribution Report — Project 2: Toxic Spans Detection (SemEval-2021 Task 5)

- **Student Name:** Abdulrahman Alabar
- **Student ID:** `220903363`
- **GitHub Handle:** `@AbdulrahmaAbr`
- **Primary Role:** Data Engineering & BIO Alignment Lead (`01_Data_Engineering_BIO.ipynb`)

---

## 1. Core Technical Responsibilities
- **Dataset Acquisition & Offset Parsing:** Built the automated ingestion pipeline for the official SemEval-2021 Task 5 datasets (`tsd_train.csv` with 7,939 comments and `tsd_test.csv` with 2,000 comments) using `ast.literal_eval` to safely parse character index lists.
- **Contiguous Character-Interval Reconstruction:** Developed `char_offsets_to_intervals` to group contiguous character indices into closed intervals $[a_k, b_k]$, accurately distinguishing multi-word toxic spans from adjacent separate spans.
- **Offset-Preserving Regex Tokenization & BIO Tagging:** Implemented `text_and_offsets_to_bio` using `re.finditer(r"\w+|[^\w\s]", text)` with a $\ge 50\%$ character overlap threshold to map character offsets into strict `O`, `B-TOXIC`, and `I-TOXIC` tags.
- **Corpus EDA & Imbalance Analysis:** Quantified the extreme class imbalance (92.63% `O` in train, 96.85% `O` in test) and token length distributions.

---

## 2. Key Code & Deliverable Contributions
- Authored and executed `01_Data_Engineering_BIO.ipynb`.
- Generated and verified the standardized datasets `tsd_train_bio.csv` (352,139 tokens) and `tsd_test_bio.csv` (80,481 tokens) with **0 illegal `O -> I-TOXIC` ground-truth transitions**.
