# Requirements-to-Activities Matching and Quality Analysis Tool

**Advanced Text Matching System with Requirement Quality Assessment using spaCy, Sentence Transformers, and Information Retrieval Techniques**

This comprehensive tool provides automated matching between system requirements and engineering activities, while simultaneously analyzing requirement quality to improve overall traceability and documentation. Built for engineering teams in technical domains such as aerospace, defense, and systems engineering.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Architecture and Components](#architecture-and-components)
* [How It Works](#how-it-works)
* [Installation](#installation)
* [Input Format](#input-format)
* [Execution](#execution)
* [Output Format](#output-format)
* [Understanding the Results](#understanding-the-results)
* [Evaluation Framework](#evaluation-framework)
* [Requirements Quality Analysis](#requirements-quality-analysis)
* [Sample Output Examples](#sample-output-examples)
* [Libraries Used and Why](#libraries-used-and-why)
* [Configuration Options](#configuration-options)
* [Troubleshooting](#troubleshooting)
* [Performance Benchmarks](#performance-benchmarks)
* [Reference and Industry Standards](#references-and-industry-standards)

---

## Overview

This tool combines two critical capabilities for engineering documentation:

1. **Intelligent Requirement-Activity Matching**: Automatically identifies which engineering activities satisfy specific system requirements using advanced NLP techniques
2. **Requirement Quality Assessment**: Analyzes requirement text quality to identify clarity, completeness, and testability issues

The system uses a hybrid approach combining semantic understanding, lexical matching, domain-specific knowledge, and query expansion to ensure high-quality traceability between functional descriptions and operational tasks.

---

## Key Features

### Matching Capabilities
* **Semantic Similarity**: Deep contextual understanding using sentence-transformer embeddings (all-MiniLM-L6-v2) with spaCy vector fallback
* **Lexical Matching**: Term-based scoring using BM25 with aerospace term boosting
* **Domain-Specific Similarity**: Multi-evidence scoring using aerospace vocabulary, learned co-occurrences, phrase patterns, and weighted domain terms
* **Query Expansion**: Synonym-based activity expansion using domain resources to bridge vocabulary gaps
* **Comprehensive Explainability**: Detailed explanations for every match decision

### Quality Analysis Capabilities
* **Multi-Dimensional Quality Assessment**: Analyzes clarity, completeness, verifiability, atomicity, and consistency
* **Automatic Issue Detection**: Identifies vague language, missing components, and structural problems
* **Severity Classification**: Categorizes issues from critical to low priority
* **Quality Scoring**: Provides 0-100 scores for each quality dimension
* **Improvement Recommendations**: Specific guidance for enhancing requirement quality

### Engineering Workflow Integration
* **Excel Review Packages**: Formatted workbooks for engineering team review
* **Interactive HTML Reports**: Visual summaries with filtering and explanation details
* **JSON Integration Data**: Machine-readable formats for tool integration
* **Action Item Tracking**: Project management-ready CSV files
* **Quality Impact Analysis**: Shows how requirement quality affects matching accuracy

### Visualization and Demos
* **Matching Journey Visualization**: Layer-by-layer HTML visualizations showing how each scoring component contributes to final match decisions
* **Presentation-Ready Output**: 16:9 format visuals suitable for technical presentations
* **Color-Coded Quality Indicators**: Visual feedback on requirement quality and match confidence

### Domain Resources and Extensibility
* **Aerospace Vocabulary**: 100+ categorized terms across 9 technical categories
* **Synonym Mappings**: 40+ domain-specific term equivalencies
* **Abbreviation Handling**: Common aerospace acronyms (S/C, GN&C, ADCS, TCS, etc.)
* **Learned Knowledge**: Patterns and synonyms automatically extracted from manual trace data
* **Synthetic Data Generator**: Generate test datasets across multiple domains (aerospace, automotive, medical)
* **Offline Transformer Support**: Manage and use sentence-transformer models without internet access

---

## Architecture and Components

### Project Structure

```
Req-Act-NLP/
├── src/                              # Main source code
│   ├── matching/
│   │   ├── matcher.py                # AerospaceMatcher - core matching engine
│   │   └── domain_resources.py       # Domain knowledge management
│   ├── quality/
│   │   └── reqGrading.py             # EnhancedRequirementAnalyzer - quality analysis
│   ├── evaluation/
│   │   └── simple_evaluation.py      # FixedSimpleEvaluator - performance metrics
│   ├── knowledge/
│   │   └── domain_knowledge_builder.py  # Learn patterns from manual traces
│   ├── demos/
│   │   ├── html_viz_v1.py            # HTML visualization of matching journey
│   │   └── vizV1.py                  # Legacy visualization
│   ├── utils/
│   │   ├── repository_setup.py       # Directory structure management
│   │   ├── file_utils.py             # SafeFileHandler - encoding-aware file I/O
│   │   ├── path_resolver.py          # SmartPathResolver - file location resolution
│   │   ├── matching_workbook_generator.py  # Excel output generation
│   │   ├── synthetic_dataset_generator.py  # Generate synthetic test data
│   │   └── TRF_manager.py            # Transformer model management (online/offline)
│   └── deprecated/                   # Legacy code preserved for reference
├── data/
│   └── raw/
│       ├── requirements.csv          # Input: System requirements
│       ├── activities.csv            # Input: Engineering activities
│       ├── manual_matches.csv        # Ground truth for evaluation
│       └── bad_requirements.csv      # Example poor-quality requirements
├── outputs/
│   ├── matching_results/             # Match CSVs and explanation JSONs
│   ├── quality_analysis/             # Quality reports (CSV, JSON, Excel)
│   ├── engineering_review/           # Excel workbooks for engineer review
│   ├── evaluation_results/           # Performance metric JSONs
│   ├── visuals/                      # HTML visualization outputs
│   └── archive/                      # Historical results
├── resources/
│   └── aerospace/
│       ├── vocabulary.json           # Categorized aerospace terms
│       ├── synonyms.json             # Domain-specific synonym mappings
│       ├── abbreviations.json        # Aerospace acronyms (S/C, GN&C, etc.)
│       └── domain_knowledge/         # Learned patterns from manual traces
└── README.md
```

### Core Modules

1. **`src/matching/matcher.py`** - `AerospaceMatcher` class with four-component hybrid scoring and full explainability
2. **`src/quality/reqGrading.py`** - `EnhancedRequirementAnalyzer` with INCOSE pattern validation and semantic analysis
3. **`src/evaluation/simple_evaluation.py`** - `FixedSimpleEvaluator` for IR-standard performance metrics
4. **`src/knowledge/domain_knowledge_builder.py`** - Learns domain patterns from manual traces
5. **`src/demos/html_viz_v1.py`** - `TechnicalJourneyVisualizerHTML` for interactive matching visualizations

### Utility Modules

* **`src/utils/repository_setup.py`** - `RepositoryStructureManager` for standardized output directories
* **`src/utils/file_utils.py`** - `SafeFileHandler` for encoding-aware file operations
* **`src/utils/path_resolver.py`** - `SmartPathResolver` for intelligent file discovery
* **`src/utils/matching_workbook_generator.py`** - Excel workbook generation for engineering review
* **`src/utils/synthetic_dataset_generator.py`** - Synthetic dataset generation for testing across domains
* **`src/utils/TRF_manager.py`** - Transformer model management with offline/online support

### Domain Resources

* **`resources/aerospace/vocabulary.json`** - 100+ categorized aerospace terms across 9 categories
* **`resources/aerospace/synonyms.json`** - 40+ domain-specific synonym mappings
* **`resources/aerospace/abbreviations.json`** - Aerospace acronyms and abbreviations
* **`resources/aerospace/domain_knowledge/`** - Extracted patterns and learned synonyms from manual traces

---

## How It Works

### Step-by-Step Process

1. **Data Loading and Preprocessing**
   * Requirements and activities loaded from CSV files with automatic encoding detection
   * Text preprocessing: lemmatization, stopword removal, phrase normalization
   * Quality analysis: each requirement analyzed for structural and linguistic issues

2. **Domain Knowledge Extraction**
   * Auto-detection of technical terms using frequency analysis and linguistic patterns
   * Domain-specific weighting based on term characteristics (length, frequency, technical indicators)
   * Synonym expansion using pre-built dictionaries and semantic similarity

3. **Multi-Modal Similarity Computation**
   For each requirement-activity pair, four similarity scores are computed:

   * **Semantic (Weight: 0.2)**: Sentence-transformer or spaCy-based meaning similarity
   * **BM25 Lexical (Weight: 0.4)**: Statistical relevance ranking with aerospace term boosting
   * **Domain (Weight: 0.2)**: Multi-evidence aerospace term overlap, learned relationships, and phrase patterns
   * **Query Expansion (Weight: 0.2)**: Synonym-based activity expansion to bridge vocabulary gaps

4. **Quality-Enhanced Scoring**
   * Requirement quality scores influence match confidence
   * Poor quality requirements flagged for improvement
   * Quality-match correlation analysis provided

5. **Explainable Results Generation**
   * Detailed explanations for each similarity component
   * Quality impact analysis for poor matches
   * Evidence presentation (shared terms, similarity levels)

6. **Engineering Workflow Integration**
   * Results categorized by review priority and confidence
   * Excel workbooks with embedded explanations
   * HTML reports with interactive filtering
   * Action items for project management

---

## Installation

### Environment Setup

```bash
# Create and activate environment
conda create -n reqmatch python=3.12
conda activate reqmatch

# Install core dependencies
pip install pandas numpy openpyxl spacy scikit-learn matplotlib seaborn chardet

# Install spaCy model (default model used by the matcher)
python -m spacy download en_core_web_lg

# Recommended: Install sentence-transformers for enhanced semantic similarity
pip install sentence-transformers torch

# Optional: Install additional spaCy models for experimentation
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
# Test spaCy installation
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('Installation successful!')"
```

---

## Input Format

### Required Files

Place input files in the `data/raw/` directory.

#### `data/raw/requirements.csv`
| Column Name        | Description                                 | Required |
| ------------------ | ------------------------------------------- | -------- |
| `ID`               | Unique requirement identifier               | Yes      |
| `Requirement Name` | Short requirement title                     | Optional |
| `Requirement Text` | Full textual description of the requirement | Yes      |

**Example:**
```csv
ID,Requirement Name,Requirement Text
REQ-001,Login Performance,"The system shall authenticate users within 2 seconds of credential submission"
REQ-002,Data Encryption,"The system shall encrypt all stored user data using AES-256 encryption"
```

#### `data/raw/activities.csv`
| Column Name     | Description                                     | Required |
| --------------- | ----------------------------------------------- | -------- |
| `Activity Name` | Operational task or system function description | Yes      |

**Example:**
```csv
Activity Name
Implement user authentication module
Deploy encryption services
Configure database security
Test login performance
```

### Optional Files

#### `data/raw/manual_matches.csv` (for evaluation)
| Column Name    | Description                           |
| -------------- | ------------------------------------- |
| `ID`           | Requirement ID                        |
| `Satisfied By` | Comma-separated list of activity names |

---

## Execution

### Basic Matching
```bash
# Run with default settings
python src/matching/matcher.py
```

### Quality Analysis Only
```bash
# Analyze requirement quality
python src/quality/reqGrading.py

# With custom column name
python src/quality/reqGrading.py -c "Requirements Description"

# With verbose output
python src/quality/reqGrading.py -v
```

### Evaluation
```bash
# Evaluate matching performance
python src/evaluation/simple_evaluation.py
```

### Visualization
```bash
# Generate HTML visualization of matching journey
python src/demos/html_viz_v1.py
```

### Synthetic Data Generation
```bash
# Generate synthetic datasets for testing across domains
python src/utils/synthetic_dataset_generator.py
```

---

## Output Format

### Core Results

#### `outputs/matching_results/aerospace_matches.csv`
Contains detailed match results with explanations:

| Column | Description |
|--------|-------------|
| `Requirement_ID` | Requirement identifier |
| `Requirement_Text` | Full requirement text |
| `Activity_Name` | Matched activity |
| `Combined_Score` | Weighted total similarity score |
| `Semantic_Score` | Semantic similarity (0-1) |
| `BM25_Score` | Lexical relevance score |
| `Domain_Score` | Domain-specific term overlap score |
| `Query_Expansion_Score` | Synonym expansion matching score |

#### `outputs/matching_results/aerospace_matches_explanations.json`
Machine-readable explanations for every match decision, including:
* Detailed score breakdowns
* Evidence (shared terms, similarity levels)
* Quality impact analysis
* Match confidence levels

### Engineering Workflow Outputs

#### `outputs/engineering_review/matching_workbook.xlsx`
Excel workbook with multiple sheets:
* **Auto-Approve**: High-confidence matches
* **Quick Review**: Moderate-confidence matches requiring brief review
* **Detailed Review**: Complex matches needing thorough analysis
* **Manual Analysis**: Low-confidence matches requiring human expertise
* **Summary**: Aggregate statistics and quality metrics

### Quality Analysis Outputs

#### `outputs/quality_analysis/requirements_quality_report.csv` / `.json` / `.xlsx`
Enhanced requirement analysis with:
* **Quality Scores**: Clarity, Completeness, Verifiability, Atomicity, Consistency (0-100)
* **Issue Lists**: Specific problems identified in each requirement
* **Severity Breakdown**: Critical, High, Medium, Low issue counts
* **Overall Quality Score**: Weighted composite quality metric
* **INCOSE Compliance**: Pattern validation against INCOSE standards
* **Semantic Analysis**: Contextual ambiguity detection

### Evaluation Outputs

#### `outputs/evaluation_results/`
* **evaluation_metrics.json**: Full IR metrics (Precision@k, Recall@k, F1@k, MRR, NDCG@k)
* **fixed_simple_metrics.json**: Simplified metric summaries

### Visualization Outputs

#### `outputs/visuals/`
* HTML visualizations showing the layer-by-layer matching journey
* Score breakdowns with color-coded quality indicators
* 16:9 presentation-ready format

---

## Understanding the Results

### Match Confidence Levels

| Level | Score Range | Meaning | Action Required |
|-------|-------------|---------|-----------------|
| **EXCELLENT** | >=0.6 | Strong multi-component alignment | Minimal review |
| **GOOD** | 0.45-0.6 | Solid match with good evidence | Brief review |
| **MODERATE** | 0.3-0.45 | Reasonable match with some uncertainty | Standard review |
| **WEAK** | <0.3 | Limited connection, possible false positive | Detailed analysis |

### Score Component Interpretation

#### **Semantic Score (0-1)**
Sentence-transformer or spaCy-based meaning similarity:
* **>0.7**: Very High - strong conceptual match
* **0.5-0.7**: High - good conceptual alignment
* **0.3-0.5**: Medium - moderate conceptual similarity
* **<0.3**: Low - weak conceptual connection

#### **BM25 Score (0-1, normalized)**
Statistical relevance ranking with aerospace term boosting:
* Higher scores indicate more shared technical terms
* Accounts for term rarity, document length, and coverage
* Aerospace terms receive a 1.3x boost

#### **Domain Score (0-1)**
Multi-evidence aerospace domain similarity:
* Combines aerospace vocabulary overlap, learned co-occurrence patterns, phrase matching, and weighted domain terms
* Multi-evidence bonus when 3+ evidence types align
* **>0.5**: Strong technical vocabulary overlap
* **0.2-0.5**: Moderate technical alignment
* **<0.2**: Minimal technical term sharing

#### **Query Expansion Score (0-1)**
Synonym-based activity expansion matching:
* Expands activity terms using domain synonym mappings
* Measures what fraction of requirement terms are covered by expanded activities
* Addresses vocabulary mismatch between requirements and short activity names

---

## Evaluation Framework

### Standard Information Retrieval Metrics

#### **Precision@k**
Fraction of top-k predictions that are correct
* **Interpretation**: How accurate are our top recommendations?
* **Good Performance**: P@5 > 0.3 for engineering domains

#### **Recall@k**  
Fraction of correct answers found in top-k predictions
* **Interpretation**: How complete is our recommendation coverage?
* **Good Performance**: R@5 > 0.4 for comprehensive coverage

#### **F1@k**
Harmonic mean of Precision@k and Recall@k
* **Interpretation**: Balanced accuracy and completeness measure
* **Target Performance**: F1@5 ≥ 0.21 (project benchmark)

#### **MRR (Mean Reciprocal Rank)**
Average of 1/rank for first correct match
* **Interpretation**: How quickly do we find the right answer?
* **Range**: 0-1, higher is better

#### **NDCG@k**
Normalized Discounted Cumulative Gain
* **Interpretation**: Rewards correct matches appearing earlier in rankings
* **Accounts for**: Position bias and graded relevance

### Sample Evaluation Output
```
MATCHING EVALUATION SUMMARY
======================================================================
Dataset Coverage:
  Total requirements: 120
  Requirements with predictions: 120
  Coverage: 100.0%

Key Performance Metrics:
  MRR (Mean Reciprocal Rank): 0.389 ± 0.102

  F1@k:
    F1@1: 0.226 ± 0.11
    F1@3: 0.232 ± 0.10
    F1@5: 0.213 ± 0.09  ← Target: ≥0.21 ✓

  NDCG@k:
    NDCG@5: 0.290 ± 0.09
```

---

## Requirements Quality Analysis

### Quality Dimensions Explained

#### **Clarity Score (0-100)**
Measures how clear and unambiguous the requirement is.

**❌ Poor Clarity (Score: ~30)**
> "The system should provide appropriate response times for users."

**✅ Good Clarity (Score: ~90)**  
> "The system shall display the user login screen within 2 seconds of startup."

**What Hurts Clarity:**
* Vague words: "appropriate," "adequate," "sufficient," "reasonable"
* Passive voice: "Data will be processed" vs "The system shall process data"
* Complex, hard-to-read sentences

#### **Completeness Score (0-100)**
Checks if the requirement has all necessary components.

**❌ Incomplete (Score: ~40)**
> "Send notifications when needed."
> * Missing: who, what kind, when exactly

**✅ Complete (Score: ~95)**
> "The system shall send an email notification to the user within 5 minutes when their order is confirmed."
> * **Who**: The system
> * **What**: send email notification
> * **To whom**: the user  
> * **When**: within 5 minutes
> * **Trigger**: when order is confirmed
> * **Strength**: "shall" (mandatory)

#### **Verifiability Score (0-100)**
Measures whether you can test if the requirement has been met.

**❌ Not Verifiable (Score: ~20)**
> "The system shall have good performance."

**✅ Verifiable (Score: ~100)**
> "The database shall support at least 1000 concurrent users with response times under 3 seconds."

**What Makes Requirements Verifiable:**
* Specific numbers: "within 2 seconds," "at least 100 users," "99.9% uptime"
* Measurable criteria: percentages, time limits, quantities
* Clear success conditions

#### **Atomicity Score (0-100)**
Checks if the requirement describes only one thing.

**❌ Multiple Requirements (Score: ~40)**
> "The system shall validate user passwords, send confirmation emails, and log all login attempts while maintaining security standards."

**✅ Atomic (Score: ~100)**
> "The system shall validate user passwords during login."

#### **Consistency Score (0-100)**
Measures use of standard requirement language.

**Modal Verb Meanings:**
* **shall/must/will**: Mandatory (required)
* **should**: Recommended (preferred but not required)  
* **may/can/might**: Optional (allowed but not required)

### Quality Score Interpretation

| Score Range | Grade | Meaning | Action Required |
|-------------|-------|---------|-----------------|
| **90-100** | EXCELLENT | Ready for implementation | No action needed |
| **80-89** | GOOD | Minor improvements beneficial | Optional cleanup |
| **70-79** | FAIR | Acceptable with some issues | Recommended fixes |
| **60-69** | POOR | Significant problems | Required improvements |
| **<60** | CRITICAL | Major rewrite needed | Must fix before use |

### Issue Severity Levels

#### **Critical Issues** 🚨
Fundamental problems making requirements unusable
* **Example**: Empty requirements, completely missing structure
* **Action**: Must fix immediately

#### **High Issues** ⚠️  
Significant problems affecting implementation or testing
* **Examples**: "adequate security" (not testable), missing essential components
* **Action**: Fix before development begins

#### **Medium Issues** ⚡
Moderate problems causing potential confusion
* **Examples**: Vague timing, unclear requirement strength, passive voice
* **Action**: Fix when possible, prioritize high-impact

#### **Low Issues** ℹ️
Minor style or clarity improvements
* **Examples**: Very long requirements, minor readability issues
* **Action**: Fix during reviews or updates

---

## Sample Output Examples

### High-Quality Match Example
```csv
Requirement_ID,Requirement_Text,Activity_Name,Combined_Score,Semantic_Score,BM25_Score,Domain_Score,Query_Expansion_Score
REQ-001,"The system shall authenticate users within 2 seconds using multi-factor authentication",implement user authentication module,0.72,0.89,0.61,0.58,0.45
```

**Explanation:**
* **High semantic similarity (0.89)**: Strong conceptual alignment between "authenticate users" and "authentication module"
* **Good BM25 score (0.61)**: Shared terms like "authenticate," "users," "multi-factor"
* **Strong overall confidence**: Combined score of 0.72 indicates EXCELLENT match quality

### Quality Analysis Example

**Original Requirement (Poor Quality - Score: 35)**
> "The system should provide adequate security and good performance while ensuring users have appropriate access to necessary functions as needed."

**Quality Issues Identified:**
* Clarity (high): ambiguous terms 'adequate', 'good', 'appropriate', 'necessary', 'as needed'
* Verifiability (high): no measurable criteria found
* Atomicity (high): multiple conjunctions suggest compound requirements
* Completeness (medium): weak modal verb 'should' vs required 'shall'

**Improved Requirements (Quality Score: ~90 each)**
1. "The system shall encrypt all user data using AES-256 encryption."
2. "The system shall respond to user requests within 2 seconds under normal load."
3. "The system shall authenticate users using multi-factor authentication before granting access to personal data."

---

## Libraries Used and Why

### Core NLP and Machine Learning

| Library | Purpose | Why This Choice |
|---------|---------|-----------------|
| **spaCy** | NLP processing, text preprocessing, linguistic analysis | Industry-standard NLP library. Default model `en_core_web_lg` provides word vectors, dependency parsing, POS tagging, and lemmatization for text preprocessing and fallback similarity. |
| **scikit-learn** | TF-IDF vectorization, cosine similarity, clustering | Well-established ML library with robust text processing utilities. Used for BM25 implementation and baseline similarity measures. |
| **numpy** | Vector mathematics, statistical computations | Essential for efficient numerical operations on embeddings and similarity calculations. Enables fast cosine similarity and norm computations. |

### Data Processing and I/O

| Library | Purpose | Why This Choice |
|---------|---------|-----------------|
| **pandas** | Data manipulation, CSV handling, DataFrame operations | Standard for tabular data in Python. Excellent CSV reading with encoding detection, data filtering, and export capabilities. |
| **chardet** | Automatic character encoding detection | Prevents file reading failures with international characters or different encoding standards. Essential for robust CSV processing. |
| **openpyxl** | Excel file creation and formatting | Enables rich Excel output with multiple sheets, formatting, and styling for engineering review workflows. |
| **json** | Structured data serialization | Machine-readable format for explanations and integration with other tools. |

### Visualization and Analysis

| Library | Purpose | Why This Choice |
|---------|---------|-----------------|
| **matplotlib** | Basic plotting and visualization | Standard Python plotting library for evaluation metrics visualization. |
| **seaborn** | Enhanced statistical visualizations | Built on matplotlib with better defaults and statistical plotting capabilities for evaluation analysis. |

### Advanced Features

| Library | Purpose | Why This Choice |
|---------|---------|-----------------|
| **pathlib** | Modern file path handling | Python 3.4+ standard for robust cross-platform file operations. |
| **logging** | Structured logging and debugging | Essential for production-grade tools. Provides detailed execution tracking and error diagnosis. |
| **dataclasses** | Type-safe data structures | Clean, maintainable code for complex data structures like match explanations and quality metrics. |
| **collections** | Specialized data structures (Counter, defaultdict) | Efficient counting and frequency analysis for domain term extraction and statistics. |

### Optional Advanced Dependencies

| Library | Purpose | Why This Choice |
|---------|---------|-----------------|
| **sentence-transformers** | Enhanced semantic embeddings | Provides access to specialized sentence-level embedding models (e.g., all-MiniLM-L6-v2) for improved semantic matching, with offline model support via `TRF_manager.py`. |
| **torch (PyTorch)** | Deep learning backend | Required by sentence-transformers for model inference. |

### Why Not Other Options?

**NLTK vs spaCy**: spaCy chosen for better performance, industrial-grade transformer support, and more consistent API.

**Sentence-Transformers vs spaCy transformers**: Both are now supported. spaCy provides a unified pipeline with linguistic features, while sentence-transformers offers specialized sentence embeddings. The matcher can use either or both.

**Basic TF-IDF vs BM25**: BM25 provides better performance for shorter texts (requirements) with term saturation handling and length normalization.

---

## Configuration Options

### Matching Configuration
```python
# Default aerospace-optimized weights (BM25-heavy for technical domains)
default_config = {
    'weights': {
        'semantic': 0.2,
        'bm25': 0.4,
        'domain': 0.2,
        'query_expansion': 0.2
    },
    'min_similarity': 0.3,
    'top_n': 5
}

# Semantic-focused configuration (when using strong transformer models)
semantic_config = {
    'weights': {
        'semantic': 0.4,
        'bm25': 0.2,
        'domain': 0.2,
        'query_expansion': 0.2
    },
    'min_similarity': 0.3,
    'top_n': 5
}

# Lexical-focused configuration
lexical_config = {
    'weights': {
        'semantic': 0.1,
        'bm25': 0.5,
        'domain': 0.2,
        'query_expansion': 0.2
    },
    'min_similarity': 0.3,
    'top_n': 3
}
```

### Quality Analysis Configuration
```python
# Strict quality standards
strict_quality = {
    'min_clarity_score': 80,
    'min_completeness_score': 85,
    'min_verifiability_score': 75,
    'require_modal_verbs': True,
    'max_sentence_length': 40
}

# Flexible quality standards
flexible_quality = {
    'min_clarity_score': 60,
    'min_completeness_score': 65,  
    'min_verifiability_score': 55,
    'require_modal_verbs': False,
    'max_sentence_length': 60
}
```

---

## Troubleshooting

### Common Installation Issues

```bash
# spaCy model not found
python -m spacy download en_core_web_lg

# Encoding errors when reading CSV
# Solution: Tool automatically detects and handles multiple encodings

# Memory issues with large datasets
# Solution: Adjust batch_size parameter in nlp.pipe() calls
```

### Common Data Issues

#### File Format Problems
```bash
# Column not found error
python src/quality/reqGrading.py -c "Your_Actual_Column_Name"

# Empty or malformed CSV
# Ensure CSV has proper headers and content
```

#### Quality Analysis Issues
* **All scores are low**: Check if requirements contain actual requirement text vs IDs
* **No domain terms found**: Verify requirements contain technical vocabulary
* **Encoding problems**: `SafeFileHandler` handles multiple encodings automatically

#### Matching Performance Issues
* **Poor semantic scores**: Ensure transformer model is properly installed
* **No BM25 matches**: Check for shared vocabulary between requirements and activities
* **Low domain scores**: May indicate vocabulary mismatch; update `resources/aerospace/synonyms.json`

### Performance Optimization

```python
# For large datasets, increase min_similarity to reduce computation
from src.utils.repository_setup import RepositoryStructureManager
from src.matching.matcher import AerospaceMatcher

repo_manager = RepositoryStructureManager("outputs")
repo_manager.setup_repository_structure()
matcher = AerospaceMatcher(repo_manager=repo_manager)
matcher.run_matching(
    min_similarity=0.4,  # Higher threshold reduces computation
    top_n=3              # Fewer results per requirement
)
```

---

## Performance Benchmarks

### Target Performance Metrics

| Metric | Target | Excellent | Industry Standard |
|--------|--------|-----------|-------------------|
| **F1@5** | ≥0.21 | ≥0.30 | ≥0.15 |
| **Precision@5** | ≥0.25 | ≥0.35 | ≥0.20 |
| **MRR** | ≥0.35 | ≥0.45 | ≥0.25 |

### Quality Benchmarks

| Organization Maturity | Avg Quality Score | % with Issues | Excellent Quality (%) |
|----------------------|-------------------|---------------|----------------------|
| **New** | 50-65 | 60-70% | <10% |
| **Mature** | 75-85 | 30-40% | 30-50% |
| **Best-in-Class** | >85 | <20% | >60% |

### Processing Speed
* **Requirements**: ~10-20 per second (with transformer model)
* **Quality Analysis**: ~50-100 per second
* **Memory Usage**: ~2-4GB for 1000 requirements (transformer model)

---

## Quick Start Checklist

- [ ] Install Python 3.12+ and dependencies
- [ ] Download spaCy model: `python -m spacy download en_core_web_lg`
- [ ] Place `requirements.csv` and `activities.csv` in `data/raw/`
- [ ] Run matching: `python src/matching/matcher.py`
- [ ] Run quality analysis: `python src/quality/reqGrading.py`
- [ ] Review Excel output in `outputs/engineering_review/`
- [ ] Check quality scores and address POOR/CRITICAL requirements
- [ ] Evaluate performance: `python src/evaluation/simple_evaluation.py`
- [ ] Configure weights and thresholds as needed
