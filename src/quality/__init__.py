"""
Requirements Quality Analysis Package

Automated quality assessment for requirements using NLP and machine learning.
Provides comprehensive analysis across multiple quality dimensions with
specific recommendations for improvement.

Key Components:
- RequirementAnalyzer: Main quality analysis engine using transformer models
- QualityMetrics: Structured quality assessment results
- Excel report generation for engineering teams
- Integration with project utils and matching pipeline

Quality Dimensions Analyzed:
- Clarity: Ambiguous terms, passive voice detection
- Completeness: Missing components, modal verb analysis  
- Verifiability: Measurable criteria identification (enhanced with units detection)
- Atomicity: Compound requirement detection
- Consistency: Modal verb pattern analysis
- Design: Implementation detail detection
"""

# Import main classes (with fallback for standalone operation)
try:
    from .reqGrading import EnhancedRequirementAnalyzer as RequirementAnalyzer, QualityMetrics
except ImportError:
    # Fallback for standalone operation
    try:
        from reqGrading import RequirementAnalyzer, QualityMetrics
    except ImportError:
        # Final fallback - might be running reqGrading directly
        RequirementAnalyzer = None
        QualityMetrics = None

__version__ = "2.0.0"
__author__ = "Requirements Quality Team"

# Export main classes
__all__ = [
    "RequirementAnalyzer", 
    "QualityMetrics",
    "analyze_requirements",
    "create_quality_report"
]

def analyze_requirements(requirements_file, output_file=None, 
                        requirement_column="Requirement Text",
                        create_excel=True, repo_manager=None):
    """
    Convenience function for comprehensive requirements quality analysis.
    
    Args:
        requirements_file: CSV file path with requirements
        output_file: Output file path (auto-generated if None)
        requirement_column: Column name containing requirements
        create_excel: Whether to create Excel dashboard
        repo_manager: Repository structure manager
        
    Returns:
        pd.DataFrame: Enhanced requirements DataFrame with quality analysis
        
    Example:
        >>> from src.quality import analyze_requirements
        >>> enhanced_df = analyze_requirements("requirements.csv")
        >>> print(f"Analyzed {len(enhanced_df)} requirements")
    """
    if RequirementAnalyzer is None:
        raise ImportError("RequirementAnalyzer not available. Check imports.")
    
    # Initialize analyzer with transformer model
    analyzer = RequirementAnalyzer(spacy_model="en_core_web_trf", repo_manager=repo_manager)
    
    # Run analysis
    enhanced_df = analyzer.analyze_file(
        input_file=requirements_file,
        output_file=output_file,
        requirement_column=requirement_column,
        excel_report=create_excel
    )
    
    return enhanced_df

def create_quality_report(enhanced_df, output_file=None, repo_manager=None):
    """
    Create Excel quality report from enhanced DataFrame.
    
    Args:
        enhanced_df: DataFrame with quality analysis results
        output_file: Custom output path (optional)
        repo_manager: Repository structure manager
        
    Returns:
        str: Path to created Excel report
    """
    if RequirementAnalyzer is None:
        raise ImportError("RequirementAnalyzer not available. Check imports.")
        
    analyzer = RequirementAnalyzer(repo_manager=repo_manager)
    return analyzer.create_excel_report(enhanced_df, output_file)

def get_quality_thresholds():
    """
    Get updated quality score thresholds for classification.
    
    Returns:
        Dict[str, Dict[str, float]]: Quality thresholds by category
    """
    return {
        'grades': {
            'EXCELLENT': 95.0,    # Near perfect requirements
            'GOOD': 85.0,         # Minor improvements needed  
            'FAIR': 70.0,         # Moderate revision required
            'POOR': 50.0,         # Significant rewrite needed
            'CRITICAL': 0.0       # Complete rewrite required
        },
        'components': {
            'clarity_min': 80.0,         # Minimum for clear requirements
            'completeness_min': 85.0,    # Minimum for complete requirements
            'verifiability_min': 70.0,   # Most critical - minimum for testable requirements
            'atomicity_min': 75.0,       # Minimum for atomic requirements
            'consistency_min': 80.0      # Minimum for consistent requirements
        },
        'weights': {
            'clarity': 0.2,              # 20% weight
            'completeness': 0.2,         # 20% weight
            'verifiability': 0.35,       # 35% weight (most important!)
            'atomicity': 0.15,           # 15% weight
            'consistency': 0.1           # 10% weight
        }
    }

def get_quality_recommendations():
    """
    Get updated quality improvement recommendations.
    
    Returns:
        Dict[str, List[str]]: Recommendations by quality issue type
    """
    return {
        'clarity': [
            "Replace ambiguous terms (appropriate, sufficient, good) with specific criteria",
            "Rewrite passive voice constructions in active voice",
            "Define technical terms and acronyms clearly",
            "Use precise language instead of subjective qualifiers",
            "Improve readability by shortening complex sentences"
        ],
        'completeness': [
            "Add modal verbs (shall/must/will) to indicate requirement priority",
            "Ensure each requirement has clear subject, verb, and object",
            "Specify acceptance criteria and success conditions",
            "Include necessary context and constraints",
            "Avoid requirements that are too short or incomplete"
        ],
        'verifiability': [
            "Add specific performance targets with units (±1%, 50 Nm, 10^-6 g)", 
            "Include measurable criteria and quantifiable metrics",
            "Specify test methods and validation approaches",
            "Define quantifiable success indicators",
            "Avoid comparative statements without baselines ('faster than previous')",
            "Add compliance standards (ISO 9001, DO-178B Level A)"
        ],
        'atomicity': [
            "Split compound requirements into separate statements",
            "Remove multiple conjunctions (and/or) from single requirements",
            "Create one requirement per functional need",
            "Break down overly long requirements (>50 words)",
            "Eliminate embedded sub-requirements"
        ],
        'consistency': [
            "Use consistent modal verb patterns throughout",
            "Maintain uniform terminology and definitions",
            "Apply consistent formatting and structure",
            "Ensure alignment with organizational standards"
        ],
        'design': [
            "Remove implementation details (specific technologies, brands)",
            "Focus on WHAT the system must do, not HOW to build it",
            "Replace technology specifications with performance requirements",
            "Distinguish between requirements and verification methods"
        ]
    }

def get_issue_severity_guide():
    """
    Get guidance on issue severity levels.
    
    Returns:
        Dict[str, Dict]: Severity level descriptions and examples
    """
    return {
        'critical': {
            'description': 'Fundamental flaws that make requirements unusable',
            'examples': ['Empty requirements', 'Missing essential components'],
            'penalty': 25  # Points deducted per issue
        },
        'high': {
            'description': 'Serious issues that significantly impact quality',
            'examples': ['No measurable criteria', 'Implementation details', 'Major ambiguity'],
            'penalty': 15  # Points deducted per issue
        },
        'medium': {
            'description': 'Moderate issues that need attention',
            'examples': ['Passive voice', 'Minor ambiguous terms', 'Missing modal verbs'],
            'penalty': 5   # Points deducted per issue
        },
        'low': {
            'description': 'Minor issues that could be improved',
            'examples': ['Requirements too long', 'Minor readability issues'],
            'penalty': 2   # Points deducted per issue
        }
    }

def validate_quality_config(config):
    """
    Validate quality analysis configuration.
    
    Args:
        config: Dictionary with quality analysis settings
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    required_keys = ['spacy_model']
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    if 'spacy_model' in config:
        valid_models = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf']
        if config['spacy_model'] not in valid_models:
            errors.append(f"Invalid spacy_model. Must be one of: {valid_models}")
    
    return len(errors) == 0, errors

# Updated quality analysis configuration
DEFAULT_CONFIG = {
    'spacy_model': 'en_core_web_trf',  # Use transformer model by default
    'output_format': 'excel',
    'include_explanations': True,
    'severity_weights': {
        'critical': 25,   # Heavy penalty
        'high': 15,       # Significant penalty
        'medium': 5,      # Moderate penalty  
        'low': 2          # Light penalty
    },
    'component_weights': {
        'clarity': 0.2,
        'completeness': 0.2,
        'verifiability': 0.35,  # Most important!
        'atomicity': 0.15,
        'consistency': 0.1
    }
}

def get_default_config():
    """Get default quality analysis configuration."""
    return DEFAULT_CONFIG.copy()

# Enhanced quality analysis best practices
QUALITY_BEST_PRACTICES = """
Requirements Quality Best Practices (v2.0):

1. Verifiability (Most Critical - 35% weight):
   - Include specific measurements with units: "±1%", "50 Nm", "10^-6 g"
   - Add performance thresholds: "less than 5%", "at least 99.9%"
   - Specify standards compliance: "ISO 9001", "DO-178B Level A"
   - Avoid unmeasurable terms: "good", "appropriate", "sufficient"
   - Use quantifiable criteria for all requirements

2. Clarity (20% weight):
   - Use specific, unambiguous language
   - Avoid subjective terms (good, bad, appropriate, sufficient)
   - Write in active voice to clarify responsibility
   - Define all technical terms and acronyms
   - Keep sentences readable and concise

3. Completeness (20% weight):
   - Include modal verbs (shall/must/will) for requirement strength
   - Ensure subject-verb-object structure
   - Specify all necessary conditions and constraints
   - Include error and exception handling
   - Provide complete context for understanding

4. Atomicity (15% weight):
   - One requirement per statement
   - Avoid compound requirements with multiple "and" conjunctions
   - Split complex requirements into focused statements
   - Keep requirements under 50 words when possible
   - Remove embedded sub-requirements

5. Consistency (10% weight):
   - Use standard terminology throughout
   - Follow organizational templates and style guides
   - Maintain uniform modal verb patterns
   - Apply consistent formatting and structure

6. Design Constraints (Quality Check):
   - Focus on WHAT the system must do, not HOW to build it
   - Avoid specifying technologies, brands, or implementation methods
   - Distinguish between requirements and verification methods
   - Keep requirements solution-independent

Key Insight: Verifiability is the most critical quality dimension. 
If you can't test it, it's not a good requirement!
"""

def print_best_practices():
    """Print enhanced requirements quality best practices."""
    print(QUALITY_BEST_PRACTICES)

def run_quality_analysis_cli():
    """
    CLI entry point for quality analysis.
    Allows running: python -m src.quality requirements.csv
    """
    import sys
    import argparse
    
    if RequirementAnalyzer is None:
        print("❌ RequirementAnalyzer not available. Run reqGrading.py directly instead.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Requirements Quality Analysis CLI")
    parser.add_argument("input_file", help="Input CSV file with requirements")
    parser.add_argument("--excel", action="store_true", help="Create Excel report")
    parser.add_argument("--column", default="Requirement Text", help="Requirements column name")
    parser.add_argument("--model", default="en_core_web_trf", help="spaCy model to use")
    
    args = parser.parse_args()
    
    print("🔍 Requirements Quality Analysis (Package Mode)")
    print("=" * 50)
    
    try:
        enhanced_df = analyze_requirements(
            args.input_file,
            requirement_column=args.column,
            create_excel=args.excel
        )
        
        print(f"✅ Analysis complete! Processed {len(enhanced_df)} requirements")
        
        # Print summary stats
        grades = enhanced_df['Quality_Grade'].value_counts()
        print(f"\nQuality Distribution:")
        for grade, count in grades.items():
            print(f"  {grade}: {count}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

# Make CLI available
if __name__ == "__main__":
    run_quality_analysis_cli()