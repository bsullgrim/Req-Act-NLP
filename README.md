# Requirements-to-Activities Hybrid Matching Tool (spaCy Transformer Edition)

This tool matches system requirements to activities using a **hybrid approach**:  
- **Linguistic matching** (verb/noun overlap)  
- **Semantic similarity** (transformer-based, using spaCy’s `en_core_web_trf`)

It is designed for traceability and explainability in engineering, aerospace, and systems projects.

---

## General Overview

**Purpose:**  
Automatically link requirements to the most relevant activities, combining both explicit word matching and deep semantic language understanding.

**Inputs:**  
- `requirements.csv` (columns: `Requirement Name`, `Requirement Text`)
- `activities.csv` (column: `Activity Name`)

**Outputs:**  
- `hybrid_matches_trf.csv`
- `hybrid_matches_trf.xlsx`

**What you get:**  
For each requirement, a ranked list of activities with scores showing how well each activity matches, based on both explicit verb/noun overlap and semantic similarity. The requirement ID is preserved in the output for traceability.

## evaluator.py

The `evaluator.py` script provides a comprehensive evaluation framework to assess the performance of automated activity matching systems by comparing predicted matches against a manually curated reference. The main function, `evaluator_with_prf`, calculates various metrics to evaluate matching accuracy, including precision, recall, F1 score, top-N accuracy, and Mean Reciprocal Rank (MRR).

### Key Functionalities:

- **File Inputs**:
  - `manual_file`: Path to a CSV file containing the manually matched activities (`manual_matches.csv` by default).
  - `auto_file`: Path to a CSV file containing the automatically generated matches (`hybrid_matches_trf.csv` by default).

- **Preprocessing**:
  - Loads and normalizes activity names to lowercase and strips any context markers to ensure consistent matching.
  - Groups automatic matches by requirement ID and extracts the top-N highest scoring predictions.

- **Evaluation Metrics**:
  - **Top-N Accuracy**: Measures whether any of the manually matched activities appear in the top-N predictions.
  - **Top-1 Accuracy**: Measures whether the top prediction exactly matches the first manual match.
  - **Precision, Recall, F1 Score**: Computed per row and averaged across all examples.
  - **MRR (Mean Reciprocal Rank)**: Evaluates how high the first correct match appears in the predicted ranking.

- **Outputs**:
  - A summary dictionary with all computed metrics.
  - A DataFrame with detailed evaluation results per requirement.
  - Prints performance statistics to the console.
  - Displays up to 10 examples where the automated matcher failed to include the correct result in the top-N list.

- **Robustness**:
  - Gracefully handles missing or empty input files.
  - Applies defensive programming techniques to avoid crashing on malformed data.

This script is intended to be executed as a standalone module, providing immediate feedback on matching model performance.

## evaluator.py

The `evaluator.py` script provides a comprehensive evaluation framework to assess the performance of automated activity matching systems by comparing predicted matches against a manually curated reference. The main function, `evaluator_with_prf`, calculates various metrics to evaluate matching accuracy, including precision, recall, F1 score, top-N accuracy, and Mean Reciprocal Rank (MRR).

### Key Functionalities:

- **File Inputs**:
  - `manual_file`: Path to a CSV file containing the manually matched activities (`manual_matches.csv` by default).
  - `auto_file`: Path to a CSV file containing the automatically generated matches (`hybrid_matches_trf.csv` by default).

- **Preprocessing**:
  - Loads and normalizes activity names to lowercase and strips any context markers to ensure consistent matching.
  - Groups automatic matches by requirement ID and extracts the top-N highest scoring predictions.

- **Evaluation Metrics**:
  - **Top-N Accuracy**: Measures whether any of the manually matched activities appear in the top-N predictions.
  - **Top-1 Accuracy**: Measures whether the top prediction exactly matches the first manual match.
  - **Precision, Recall, F1 Score**: Computed per row and averaged across all examples.
  - **MRR (Mean Reciprocal Rank)**: Evaluates how high the first correct match appears in the predicted ranking.

- **Outputs**:
  - A summary dictionary with all computed metrics.
  - A DataFrame with detailed evaluation results per requirement.
  - Prints performance statistics to the console.
  - Displays up to 10 examples where the automated matcher failed to include the correct result in the top-N list.

- **Robustness**:
  - Gracefully handles missing or empty input files.
  - Applies defensive programming techniques to avoid crashing on malformed data.

This script is intended to be executed as a standalone module, providing immediate feedback on matching model performance.

---

## grid_search.py

The `grid_search.py` script automates the process of hyperparameter tuning for the matching system by running multiple combinations of vector-based and semantic similarity weights, similarity thresholds, and top-N values. It leverages the `run_matcher` and `evaluator_with_prf` functions from the `matcher` and `evaluator` modules, respectively.

### Key Functionalities:

- **Hyperparameters**:
  - `vn_weights`: List of weights for vector similarity (e.g., [0.0, 0.2, ..., 1.0]).
  - `sem_weights`: Complementary weights for semantic similarity.
  - `min_sims`: List of minimum similarity thresholds (e.g., [0.2, 0.3, ..., 0.6]).
  - `top_ns`: Number of top matches to evaluate (currently fixed to [5]).

- **Execution Loop**:
  - Iterates over the Cartesian product of the parameter values using `itertools.product`.
  - Runs the matching algorithm for each combination.
  - Evaluates the result against the manual match file.

- **Output**:
  - Stores evaluation metrics for each parameter set in a list.
  - Saves a summary DataFrame of all evaluations to `grid_search_results.csv`.
  - Prints progress and alerts if a run returns no usable results.

This script is ideal for identifying the best-performing configuration of the matching system based on quantitative evaluation metrics.

---

## Quick Start

### 1. Install Anaconda  
Download and install the [Anaconda Individual Edition](https://www.anaconda.com/products/distribution) from the official site.

### 2. Create and Activate a New Environment  
(Optional but recommended for isolation.)
```bash
conda create -n reqmatch python=3.12
conda activate reqmatch
```
### 3. **Install dependencies:**  
   Open a terminal and run:
   ```
   conda install pandas numpy openpyxl spacy
   python -m spacy download en_core_web_trf
   ```
### 4. Prepare Input Files
Place your requirements.csv and activities.csv in the same folder as the script.

    requirements.csv must include: ID, Requirement Name, Requirement Text

    activities.csv must include: Activity Name
### 5. **Run the script:**
   ```
   python Req-Act-NLP.py
   ```
### 6. Review the Output

Results are written to:

    hybrid_matches_trf.csv

    hybrid_matches_trf.xlsx
---

## How It Works 

### Libraries Used

| Library      | Purpose                                                                                 |
|--------------|-----------------------------------------------------------------------------------------|
| pandas       | Data manipulation, reading/writing CSV and Excel files                                  |
| spacy        | Natural Language Processing (NLP): transformer embeddings, POS tagging, lemmatization   |
| numpy        | Vector math for cosine similarity                                                       |
| openpyxl     | Excel file writing through pandas                                                       |

### Model Used

- **spaCy `en_core_web_trf`**  
  - Transformer-based, uses RoBERTa under the hood for deep contextual understanding.
  - Provides: POS tagging, lemmatization, dependency parsing, NER, and transformer embeddings.
  - No static word vectors; all similarity is calculated from transformer output.

### Algorithm Steps

1. **Load Data**  
   - Reads requirements and activities from CSV files.
2. **Extract Verbs and Nouns**  
   - For each activity, extracts lemmatized verbs and nouns using spaCy’s POS tagger.
3. **Precompute Activity Documents**  
   - Uses `nlp.pipe` for efficient batch processing of activity names.
4. **Match Each Requirement to Activities**
   - For each requirement:
     - Lemmatizes and collects all lemmas.
     - For each activity:
       - **Verb/Noun Score:** Fraction of activity verbs/nouns present in requirement.
       - **Semantic Similarity:** Cosine similarity of mean transformer embeddings (using `.last_hidden_layer_state.data`).
       - **Combined Score:** Weighted sum (default: 80% verb/noun, 20% semantic).
     - Keeps top N matches per requirement above a minimum threshold.
5. **Output**  
   - Saves results to CSV and Excel, including all component scores for transparency.

---

### Key Function Calls & Their Purpose

| Function/Call                                    | Purpose                                                                                 |
|--------------------------------------------------|-----------------------------------------------------------------------------------------|
| `spacy.load("en_core_web_trf")`                  | Loads transformer-based English pipeline                                                |
| `nlp(text)`                                      | Processes text: tokenization, tagging, transformer embedding                            |
| `extract_action_parts(text)`                     | Extracts lemmatized verbs/nouns from activity name                                      |
| `doc._.trf_data.last_hidden_layer_state.data`    | Gets transformer output as numpy array for similarity calculation                       |
| `trf_similarity(doc1, doc2)`                     | Calculates cosine similarity between two texts using transformer embeddings              |
| `pandas.DataFrame.to_csv/to_excel`               | Writes results to CSV/Excel                                                             |

---

### Configuration Parameters

- `VERB_NOUN_WEIGHT`: Weight for verb/noun matching (default 0.8)
- `SEMANTIC_WEIGHT`: Weight for semantic similarity (default 0.2)
- `MIN_SIMILARITY`: Minimum combined score to consider a match (default 0.4)
- `TOP_N`: Max number of matches per requirement (default 5)

---

### Example Output Row

| Requirement ID | Requirement Text         | Activity Name | Activity Verbs | Activity Nouns | VerbNoun Score | Similarity Score | Combined Score |
|----------------|-------------------------|---------------|---------------|---------------|---------------|-----------------|---------------|
| REQ-001        | The system shall store… | store data    | store         | data          | 1.0           | 0.82            | 0.856         |

---

## Notes and Best Practices

- **The transformer model (`en_core_web_trf`) is slower and more resource-intensive** than spaCy’s medium model, but yields much better semantic similarity, especially for complex or nuanced requirements.
- **No static word vectors:** All similarity is computed from transformer outputs.
- **You can adjust weights and thresholds** at the top of the script to tune for your domain.

---


## References

- [spaCy en_core_web_trf Model Card](https://spacy.io/models/en#en_core_web_trf)
- [spaCy English Models Overview](https://spacy.io/models/en)
- [spaCy Documentation: Vectors & Similarity](https://spacy.io/usage/vectors-similarity)

