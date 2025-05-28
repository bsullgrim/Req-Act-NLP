# Requirements-to-Activities Matching Tool

**Theoretically Enhanced Matching System using spaCy Transformers and Information Retrieval Techniques**

This repository provides a tool for automatically matching system requirements to engineering activities using a hybrid of semantic, lexical, syntactic, and domain-aware similarity techniques. The tool is built for traceability, documentation support, and structured analysis in technical domains such as aerospace, defense, and systems engineering.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [How It Works](#how-it-works)
* [Installation](#installation)
* [Input Format](#input-format)
* [Execution](#execution)
* [Output Format](#output-format)
* [Evaluation Framework](#evaluation-framework)
* [Sample Output Row](#sample-output-row)
* [Libraries Used](#libraries-used)
* [Configuration Options](#configuration-options)
* [Future Enhancements](#future-enhancements)

---

## Overview

This tool identifies the most relevant activity or set of activities that correspond to a given requirement. It combines natural language processing (NLP) with information retrieval (IR) theory to ensure high-quality traceability between functional descriptions and operational tasks.

---

## Key Features

* **Semantic Similarity**: Deep contextual understanding using RoBERTa transformer embeddings via spaCy.
* **Lexical Matching**: Term-based scoring using BM25, an advanced alternative to TF-IDF.
* **Syntactic Analysis**: Dependency relations, POS sequences, and verb frame structures capture linguistic parallels.
* **Domain-Specific Term Weighting**: Increases weight of important engineering and system-specific terminology.
* **Query Expansion**: Incorporates related terms using embedding-based similarity to overcome terminology mismatches.
* **Robust Preprocessing**: Lemmatization, stopword removal, and optional synonym expansion ensure clean input processing.
* **Encoding Detection**: Automatically detects character encodings in CSVs to prevent file read failures.

---

## How It Works

### Step-by-Step Process

1. **Data Loading**

   * Requirements and activities are loaded from CSV files.
   * Encoding is automatically detected to handle various file formats.

2. **Text Preprocessing**

   * Inputs are lowercased, lemmatized, stripped of stopwords and punctuation.
   * Phrases are collapsed, noun/verb features are extracted, and query expansion is applied.

3. **Domain Term Extraction**

   * Noun chunks and named entities are analyzed to identify domain-specific terms, which are weighted and normalized.

4. **Document Representation**

   * Each requirement and activity is processed using `spaCy`’s `en_core_web_trf` transformer model.
   * Embeddings are extracted from the final transformer layer.

5. **Similarity Computation**
   For each requirement–activity pair, the following scores are computed:

   * **Dense Semantic**: Cosine similarity between mean transformer embeddings.
   * **BM25**: Lexical match score accounting for term saturation and document length.
   * **Syntactic**: Structural similarity using Jaccard similarity over syntactic features.
   * **Domain Weighted**: Emphasizes overlap on domain-specific terms.
   * **Query Expansion**: Measures how many semantically similar terms overlap.

6. **Score Aggregation**

   * Weighted sum of all similarity components (user configurable).
   * Only matches above a minimum score threshold are retained.

7. **Result Filtering**

   * Top-N matches per requirement are selected and output for analysis.

---

## Installation

### Environment Setup

```bash
conda create -n reqmatch python=3.12
conda activate reqmatch
pip install pandas numpy openpyxl spacy scikit-learn matplotlib seaborn chardet
python -m spacy download en_core_web_trf
```

---

## Input Format

### `requirements.csv`

| Column Name        | Description                                 |
| ------------------ | ------------------------------------------- |
| `ID`               | Unique requirement identifier               |
| `Requirement Name` | Short requirement title                     |
| `Requirement Text` | Full textual description of the requirement |

### `activities.csv`

| Column Name     | Description                                     |
| --------------- | ----------------------------------------------- |
| `Activity Name` | Operational task or system function description |

---

## Execution

### Matching

```bash
python matcher.py
```

### Evaluation

```bash
python evaluator.py
```

---

## Output Format

Matching results are saved as CSV files in a `results/` directory, typically named after their configuration:

* `results/semantic_focused.csv`
* `results/lexical_focused.csv`
* `results/balanced.csv`

Each row contains:

* The matched requirement and activity
* Scores for each similarity component
* The final combined score

---

## Evaluation Framework

The `evaluator.py` script evaluates the accuracy of automated matches against a manually curated reference file (`manual_matches.csv`). It computes a set of standard Information Retrieval metrics:

### Metrics Explained

| Metric                                              | Description                                                                                                                             |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision\@k**                                    | The fraction of the top-k predicted activities that are relevant. Higher precision means fewer false positives.                         |
| **Recall\@k**                                       | The fraction of all relevant activities that appear in the top-k predictions. Higher recall means fewer false negatives.                |
| **F1\@k**                                           | The harmonic mean of precision and recall at rank k. Provides a single value to balance precision and recall.                           |
| **MRR (Mean Reciprocal Rank)**                      | Measures how high the first correct match appears in the ranked list. A value of 1.0 means the correct activity is always ranked first. |
| **NDCG\@k (Normalized Discounted Cumulative Gain)** | Captures both relevance and rank, rewarding correct predictions that appear earlier in the list.                                        |

### Sample Evaluation Output (Console)

```
MATCHING EVALUATION SUMMARY
======================================================================
Dataset Coverage:
  Total requirements: 120
  Requirements with predictions: 120
  Coverage: 100.0%

Key Performance Metrics:
  MRR (Mean Reciprocal Rank): 0.389 ± 0.102

  Precision@k:
    P@1: 0.301 ± 0.12
    P@3: 0.217 ± 0.09
    P@5: 0.192 ± 0.08

  Recall@k:
    R@1: 0.187 ± 0.10
    R@3: 0.254 ± 0.12
    R@5: 0.296 ± 0.14

  F1@k:
    F1@1: 0.226 ± 0.11
    F1@3: 0.232 ± 0.10
    F1@5: 0.213 ± 0.09

  NDCG@k:
    NDCG@1: 0.301 ± 0.08
    NDCG@3: 0.272 ± 0.07
    NDCG@5: 0.290 ± 0.09
```

The script also offers:

* Visual plots for each metric
* Side-by-side comparison across configurations
* Detailed per-requirement results in CSV

---

## Sample Output Row

Here is an example row from the output file:

| Requirement ID | Requirement Text                       | Activity Name | Combined Score | Dense Semantic | BM25 Score | Syntactic Score | Domain Weighted | Query Expansion |
| -------------- | -------------------------------------- | ------------- | -------------- | -------------- | ---------- | --------------- | --------------- | --------------- |
| REQ-001        | The system shall store encrypted logs. | store logs    | 0.842          | 0.89           | 0.61       | 0.75            | 0.58            | 0.25            |

This row indicates that the requirement “store encrypted logs” was strongly matched to the activity “store logs” with a high combined confidence score. Each component score is also included to support traceability and tuning.

---

## Libraries Used

| Library        | Purpose                                                      |
| -------------- | ------------------------------------------------------------ |
| `spaCy`        | NLP processing, transformer embeddings, POS tagging, parsing |
| `pandas`       | Data loading and CSV handling                                |
| `numpy`        | Vector math and statistical functions                        |
| `scikit-learn` | TF-IDF vectorization, cosine similarity                      |
| `matplotlib`   | Metric visualization                                         |
| `seaborn`      | Enhanced plotting                                            |
| `chardet`      | Automatic file encoding detection                            |
| `openpyxl`     | Excel output support for `.xlsx` formats                     |

---

## Configuration Options

Modify configuration parameters within `matcher.py` or pass explicitly:

```python
matcher.run_enhanced_matcher(
    weights={
        'dense_semantic': 0.4,
        'bm25': 0.2,
        'syntactic': 0.2,
        'domain_weighted': 0.1,
        'query_expansion': 0.1
    },
    min_sim=0.35,
    top_n=5,
    out_file="results/semantic_focused"
)
```

---

## Future Enhancements

* Integration of synonym dictionaries and controlled vocabularies
* Fine-tuning embeddings on engineering documents (e.g., specifications, CONOPS)
* Cross-encoder reranker for top-N results
* Web-based UI for interactive matching and validation
* Active learning with user feedback for continuous improvement

---

## License

This project is open for educational and non-commercial use. For other uses, please contact the author.
