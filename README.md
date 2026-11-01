# SemEval-2021 Task 5: Toxic Spans Detection via Sequence Labeling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM--CRF-EE4C2C.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/SemEval--2021-Task%205%20Toxic%20Spans-2b5c8f.svg)](https://competitions.codalab.org/competitions/25623)

## 1. Executive Summary & Project Overview

Unlike coarse document-level toxicity classification (explored in **Project 1**), **Toxic Spans Detection (SemEval-2021 Task 5)** formulates toxicity attribution as a fine-grained **Sequence Labeling** problem. Given an raw user comment $C = (c_1, c_2, \dots, c_L)$ of length $L$ characters and ground-truth toxic character indices $S \subseteq \{0, 1, \dots, L-1\}$, our objective is to identify the exact token boundaries of every toxic expression.

To solve this rigorously across classical probabilistic graphical models and deep neural architectures, we transform character-level offset lists into word-level **BIO (Beginning-Inside-Outside)** sequences ($\mathcal{Y} = \{\text{O}, \text{B-TOXIC}, \text{I-TOXIC}\}$) and systematically benchmark three sequence labeling paradigms:
1. **Generative Sequence Labeling**: First-Order **Hidden Markov Model (HMM)** with Laplace smoothing, `<UNK>` lexical backoff, and Log-Space Viterbi decoding.
2. **Discriminative Feature-Engineered Graphical Model**: **Linear-Chain Conditional Random Field (CRF)** trained via L-BFGS with Elastic-Net ($L_1 + L_2$) regularization over orthographic, morphological, and $\pm 2$ contextual window features.
3. **Deep Neural Sequence Labeling**: **Bidirectional LSTM-CRF (`BiLSTM-CRF`)** combining word + 3-character suffix embeddings, **1D Spatial Dropout (`nn.Dropout2d`)**, a 2-layer BiLSTM encoder with Layer Normalization, and a vectorized **Linear-Chain CRF layer** with hard structural BIO transition constraints.

---

## 2. Repository Structure

```text
project2_nlp_dataset/
├── 01_Data_Engineering_BIO.ipynb      # Phase 1: Character-offset parsing, contiguous interval reconstruction & BIO tagging
├── 02_Baseline_HMM.ipynb              # Phase 2: Generative First-Order HMM & Log-Space Viterbi Decoding
├── 03_Advanced_CRF.ipynb              # Phase 3: Linear-Chain CRF with ±2 contextual & morphological features
├── 04_DeepLearning_BiLSTM_CRF.ipynb   # Phase 4: PyTorch BiLSTM-CRF with SpatialDropout1D, LayerNorm & Early Stopping
├── tsd_train.csv                      # Official SemEval-2021 Task 5 raw training corpus (7,939 comments)
├── tsd_test.csv                       # Official SemEval-2021 Task 5 raw test corpus (2,000 comments)
├── tsd_train_bio.csv                  # Processed BIO-tagged training dataset (352,139 tokens)
├── tsd_test_bio.csv                   # Processed BIO-tagged test dataset (80,481 tokens)
├── hmm_metrics.json                   # Serialized Test Exact Span evaluation metrics for Baseline HMM
├── crf_metrics.json                   # Serialized Test Exact Span evaluation metrics for Linear-Chain CRF
├── bilstm_crf_metrics.json            # Serialized Test Exact Span evaluation metrics for BiLSTM-CRF
├── generate_presentation.py           # Automated script generating charts and the 10-slide presentation.pptx
├── presentation.pptx                  # Academic defense slide deck (10 slides, widescreen 16:9)
├── contributions/                     # Individual team member contribution reports (.md)
│   ├── Amr_Alsayed_2309115638_amralsayed2005.md
│   ├── Abdulrahman_Alabar_220903363_AbdulrahmaAbr.md
│   ├── Adham_Mostafa_2309115774_AdhamK1.md
│   ├── OMAR_ELGAMAL_2309116447_omarelgamal-web.md
│   └── YAHIA_ALHAMWI_2309115151_9yavi.md
├── CONTRIBUTIONS.md                   # Team roles, research governance, and task distribution
├── LICENSE                            # Open-source MIT License
└── README.md                          # Comprehensive project documentation
```

### Research Team & Individual Contributions

| Student Name | Student ID | GitHub Handle | Primary Role | Individual Report |
| :--- | :---: | :---: | :--- | :--- |
| **Amr Alsayed** | `2309115638` | `@amralsayed2005` | Deep Learning Lead (`04_DeepLearning_BiLSTM_CRF.ipynb`) | [Amr_Alsayed_2309115638_amralsayed2005.md](Amr_Alsayed_2309115638_amralsayed2005.md) |
| **Abdulrahman Alabar** | `220903363` | `@AbdulrahmaAbr` | Data Engineering & BIO Lead (`01_Data_Engineering_BIO.ipynb`) | [Abdulrahman_Alabar_220903363_AbdulrahmaAbr.md](Abdulrahman_Alabar_220903363_AbdulrahmaAbr.md) |
| **Adham Mostafa** | `2309115774` | `@AdhamK1` | Generative HMM Lead (`02_Baseline_HMM.ipynb`) | [Adham_Mostafa_2309115774_AdhamK1.md](Adham_Mostafa_2309115774_AdhamK1.md) |
| **Omar Elgamal** | `2309116447` | `@omarelgamal-web` | Discriminative CRF Lead (`03_Advanced_CRF.ipynb`) | [OMAR_ELGAMAL_2309116447_omarelgamal-web.md](OMAR_ELGAMAL_2309116447_omarelgamal-web.md) |
| **Yahia Alhamwi** | `2309115151` | `@9yavi` | Exact Span Evaluation, Benchmarking & Presentation Lead | [YAHIA_ALHAMWI_2309115151_9yavi.md](YAHIA_ALHAMWI_2309115151_9yavi.md) |

---

## 3. Dataset & Phase 1: Data Engineering (`01_Data_Engineering_BIO.ipynb`)

### 3.1 Corpus Statistics & Severe Class Imbalance
The SemEval-2021 Task 5 dataset comprises **7,939 training comments** and **2,000 test comments**. After parsing stringified character offsets via `ast.literal_eval` and applying offset-preserving regex tokenization (`\w+|[^\w\s]`), we obtain the following BIO label distribution:

| BIO Tag | Train Token Count | Train Share (%) | Test Token Count | Test Share (%) | Semantic Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`O`** | 326,184 | **92.63%** | 77,946 | **96.85%** | Non-toxic (Outside) token |
| **`B-TOXIC`** | 10,295 | **2.92%** | 1,850 | **2.30%** | Beginning token of a contiguous toxic span |
| **`I-TOXIC`** | 15,660 | **4.45%** | 685 | **0.85%** | Inside continuation token of a multi-word toxic span |
| **Total** | **352,139** | **100.00%** | **80,481** | **100.00%** | — |

### 3.2 Contiguous Interval Alignment Algorithm
1. **Interval Reconstruction**: Sorted character indices $S$ are grouped into contiguous intervals $[a_k, b_k]$ so that multi-word toxic expressions (whose internal spaces are included in $S$) share a unified interval ID $k$, whereas separate adjacent toxic spans receive distinct interval IDs.
2. **Overlap Thresholding**: Each regex token $w_i$ spanning characters $[s_i, e_i)$ is marked toxic if at least $50\%$ of its character coordinates fall within $S$.
3. **Strict BIO Assignment**: The first token entering interval $k$ is assigned `B-TOXIC`; subsequent contiguous tokens belonging to the same interval $k$ receive `I-TOXIC`. This guarantees **0 invalid `O -> I-TOXIC` transitions** in the ground-truth corpus.

---

## 4. Sequence Labeling Methodology Across Phases 2–4

### 4.1 Phase 2: Generative Baseline — Hidden Markov Model (`02_Baseline_HMM.ipynb`)
- **Formulation**: Models the joint distribution $P(\mathbf{x}, \mathbf{y}) = \prod_{t=1}^T P(y_t \mid y_{t-1}) P(x_t \mid y_t)$ over hidden BIO states and observed tokens.
- **Smoothing & OOV Handling**: Applies additive smoothing ($\alpha_{\text{trans}} = 10^{-6}$, $\alpha_{\text{emit}} = 10^{-3}$) and maps singleton training tokens ($\text{freq} \le 1$) to `<UNK>` so unseen test tokens have calibrated emission probabilities across `O`, `B-TOXIC`, and `I-TOXIC`.
- **Inference**: Exact dynamic programming via the **Log-Space Viterbi Algorithm** prevents numerical underflow over long comments.

### 4.2 Phase 3: Discriminative Graphical Model — Linear-Chain CRF (`03_Advanced_CRF.ipynb`)
- **Formulation**: Directly models the global conditional probability $P(\mathbf{y} \mid \mathbf{x}; \mathbf{w}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^T \sum_k w_k f_k(y_{t-1}, y_t, \mathbf{x}, t)\right)$.
- **Contextual & Morphological Features ($\pm 2$ Window)**: Extracts lowercased/raw word forms, token length, capitalization (`isupper`, `istitle`), digits, non-alphabetic obfuscation flags, condensed word shape (`Xx+`), 2/3/4-character prefixes and suffixes, and lexical bigrams across $[t-2, t+2]$.
- **Learned Structural Transitions**: L-BFGS optimization with Elastic-Net regularization ($c_1 = 0.1, c_2 = 0.1$) learns a strong negative transition weight for the illegal transition `O -> I-TOXIC` ($w = -10.3399$) and positive continuation weight for `I-TOXIC -> I-TOXIC` ($w = +0.6884$).

### 4.3 Phase 4: Deep Learning Architecture — BiLSTM-CRF (`04_DeepLearning_BiLSTM_CRF.ipynb`)
- **Dual Lexical + Subword Representation**: Concatenates learned word embeddings ($d_w = 128$, $|V_w| = 18,221$) with 3-character suffix embeddings ($d_s = 32$, $|V_s| = 2,041$) to capture morphological inflections and censored profanity.
- **1D Spatial Dropout (`SpatialDropout1D`, $p = 0.30$)**: Drops entire embedding channels across all time steps of a sequence via `nn.Dropout2d`, preventing co-adaptation of adjacent token features.
- **Stacked BiLSTM + LayerNorm**: A 2-layer Bidirectional LSTM ($d_h = 128$ per direction $\implies 256$-dim hidden state) with inter-layer dropout ($0.25$) and `nn.LayerNorm` projects contextual representations into emission logits.
- **CRF Layer with Hard BIO Masks & Sequence-Weighted NLL**: Enforces $-10000.0$ transition masks on `<START> -> I-TOXIC` and `O -> I-TOXIC`, optimizes exact forward-algorithm Negative Log-Likelihood with toxic-sequence upweighting ($1.8\times$), and decodes via batched Viterbi.

---

## 5. Evaluation Protocol: Why Exact Span Match F1 Replaces Token Accuracy

Because **96.85% of test tokens belong to the `O` class**, standard token-level accuracy is academically invalid: a trivial majority-class predictor that outputs `O` everywhere achieves **96.85% accuracy** while yielding **0.0000 F1** on toxic spans.

We strictly evaluate all models using **Exact Span Match Precision, Recall, and F1-Score** (strict IOB2 entity boundary evaluation):
- A predicted toxic span $(\text{TOXIC}, s_{\text{pred}}, e_{\text{pred}})$ is counted as a **True Positive (TP)** if and only if both its start token index $s_{\text{pred}}$ and inclusive end token index $e_{\text{pred}}$ match a ground-truth span $(\text{TOXIC}, s_{\text{true}}, e_{\text{true}})$ **exactly**. Partial overlaps are penalized as both a False Positive (FP) and a False Negative (FN).

---

## 6. Experimental Results & Cross-Model Comparison

All models were evaluated on the held-out official test split (`tsd_test_bio.csv`, 2,000 comments, 1,850 ground-truth toxic spans):

| Model Paradigm | Notebook | Exact Span Micro-P | Exact Span Micro-R | **Exact Span Micro-F1** | Exact Span Macro-P | Exact Span Macro-R | **Exact Span Macro-F1** | TP / FP / FN | Invalid BIO (`O->I`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **02. Baseline HMM** (Generative + Viterbi) | `02_Baseline_HMM.ipynb` | 0.5937 | 0.6746 | **0.6316** | 0.5892 | 0.6221 | **0.5948** | 1,248 / 854 / 602 | **0** |
| **03. Linear-Chain CRF** ($\pm 2$ Context + L-BFGS) | `03_Advanced_CRF.ipynb` | 0.5668 | 0.6859 | **0.6207** | 0.5848 | 0.6219 | **0.5911** | 1,269 / 970 / 581 | **0** |
| **04. BiLSTM-CRF (Ours)** (SpatialDropout + LayerNorm) | `04_DeepLearning_BiLSTM_CRF.ipynb` | **0.5952** | **0.7351** | **0.6578** | **0.6087** | **0.6406** | **0.6137** | **1,360 / 925 / 490** | **0** |

### Key Empirical Findings
1. **Substantial Gain from Neural Contextualization (`BiLSTM-CRF`)**: Our `BiLSTM-CRF` achieves the highest **Exact Span Micro-F1 (0.6578)** and **Macro-F1 (0.6137)**, outperforming the generative HMM baseline by **+2.62% Micro-F1** (+6.05% Micro-Recall) and the feature-engineered Linear-Chain CRF by **+3.71% Micro-F1**.
2. **Recall Breakthrough on Multi-Word & Obfuscated Spans**: Combining dense word + 3-character suffix embeddings with bidirectional recurrent context increases exact span True Positives from 1,248 (HMM) and 1,269 (CRF) to **1,360 exact spans** (+73.51% recall).
3. **Zero Structural BIO Violations**: Across all 80,481 test tokens, all three models achieved **0 invalid `O -> I-TOXIC` transitions**, confirming the effectiveness of transition smoothing in HMM, discriminative transition weights in CRF ($w_{\text{O}\to\text{I}} = -10.34$), and hard transition masking in `BiLSTM-CRF`.

---

## 7. Reproducibility & Execution Instructions

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn sklearn-crfsuite seqeval torch python-pptx

# 2. Execute notebooks sequentially or regenerate presentation slides
python generate_presentation.py
```
