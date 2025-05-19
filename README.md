# Requirements-to-Activities Hybrid Matching Tool (spaCy Transformer Edition)

This tool matches system requirements to activities using a **hybrid approach**:  
- **Linguistic matching** (verb/noun overlap)  
- **Semantic similarity** (transformer-based, using spaCy’s `en_core_web_trf`)

It is designed for traceability and explainability in engineering, aerospace, and systems projects.

---

## General Overview (For Non-Python Users)

**Purpose:**  
Automatically link requirements to the most relevant activities, combining both explicit word matching and deep semantic language understanding.

**Inputs:**  
- `requirements.csv` (columns: `Requirement Name`, `Requirement Text`)
- `activities.csv` (column: `Activity Name`)

**Outputs:**  
- `hybrid_matches_trf.csv`
- `hybrid_matches_trf.xlsx`

**What you get:**  
For each requirement, a ranked list of activities with scores showing how well each activity matches, based on both explicit verb/noun overlap and semantic similarity.

---

## Quick Start

1. **Install Python 3.8+** (recommended).
2. **Install dependencies:**  
   Open a terminal and run:
   ```
   pip install pandas spacy openpyxl numpy
   python -m spacy download en_core_web_trf
   ```
3. **Place your `requirements.csv` and `activities.csv` in the same folder as the script.**
4. **Run the script:**
   ```
   python Req-Act-NLP.py
   ```
5. **Open the output files (`hybrid_matches_trf.csv` or `.xlsx`) in Excel or your preferred tool.**

---

## How It Works (For Python Professionals)

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

