"""
Requirements Traceability Analysis Framework

A comprehensive framework for automated requirements-to-activities traceability
using state-of-the-art NLP and machine learning techniques.

Package Structure:
- matching/: Enhanced matching algorithms with performance optimizations
- quality/: Requirements quality analysis and improvement recommendations  
- evaluation/: Comprehensive evaluation framework with interactive dashboards
- dashboard/: Interactive web-based visualization and exploration tools
- utils/: Shared utilities for file handling, path resolution, and repository management

Key Features:
- Sentence Transformers for semantic similarity
- Enhanced BM25 with technical text preprocessing
- Neural query expansion with domain adaptation
- Comprehensive quality analysis across 5 dimensions
- Adaptive evaluation supporting validation and exploration modes
- Interactive HTML dashboards with drill-down capabilities
- Excel reports for engineering and management teams
"""

# Version information
__version__ = "2.0.0"
__author__ = "Requirements Traceability Team"
__description__ = "Advanced Requirements Traceability Analysis Framework"

# Import main classes for convenience
try:
    from .matching import FinalCleanMatcher, MatchExplanation
    from .quality import RequirementAnalyzer, QualityMetrics  
    from .evaluation import EnhancedMatchingEvaluator
    CORE_IMPORTS_AVAILABLE = True
except ImportError:
    CORE_IMPORTS_AVAILABLE = False

# Export main functionality
__all__ = [
    # Core classes
    "FinalCleanMatcher",
    "MatchExplanation", 
    "RequirementAnalyzer",
    "QualityMetrics",
    "EnhancedMatchingEvaluator",
    
    # Convenience functions
    "run_complete_analysis",
    "create_matcher",
    "analyze_quality",
    "evaluate_results",
    "check_system_status"
]

def run_complete_analysis(requirements_file, activities_file, 
                         ground_truth_file=None, output_dir="outputs",
                         create_dashboard=True):
    """
    Run complete end-to-end requirements traceability analysis.
    
    Args:
        requirements_file: Path to requirements CSV
        activities_file: Path to activities CSV  
        ground_truth_file: Path to manual traces CSV (optional)
        output_dir: Base directory for outputs
        create_dashboard: Whether to create interactive dashboard
        
    Returns:
        Dict: Complete analysis results with file paths
        
    Example:
        >>> from src import run_complete_analysis
        >>> results = run_complete_analysis("reqs.csv", "acts.csv", "manual.csv")
        >>> print(f"Dashboard: {results['dashboard_path']}")
    """
    if not CORE_IMPORTS_AVAILABLE:
        raise ImportError("Core modules not available. Check package installation.")
    
    from .utils.repository_setup import RepositoryStructureManager
    
    # Setup repository structure
    repo_manager = RepositoryStructureManager(output_dir)
    repo_manager.setup_repository_structure()
    
    print("🚀 Starting Complete Requirements Traceability Analysis")
    print("=" * 60)
    
    # Step 1: Enhanced Matching
    print("1. Running Enhanced Matching...")
    matcher = FinalCleanMatcher(repo_manager=repo_manager)
    predictions_df = matcher.run_final_matching(
        requirements_file=requirements_file,
        activities_file=activities_file,
        save_explanations=True
    )
    
    # Step 2: Load requirements for context
    requirements_df = None
    try:
        requirements_df = matcher.file_handler.safe_read_csv(requirements_file)
        print(f"✅ Loaded {len(requirements_df)} requirements for context")
    except Exception as e:
        print(f"⚠️ Could not load requirements context: {e}")
    
    # Step 3: Comprehensive Evaluation  
    print("2. Running Comprehensive Evaluation...")
    evaluator = EnhancedMatchingEvaluator(
        ground_truth_file=ground_truth_file,
        repo_manager=repo_manager
    )
    evaluation_results = evaluator.evaluate_predictions(predictions_df, requirements_df)
    
    # Compile results
    results = {
        'enhanced_predictions': evaluation_results['enhanced_predictions'],
        'evaluation_results': evaluation_results['evaluation_results'],
        'dashboard_path': evaluation_results['dashboard_path'],
        'has_ground_truth': evaluation_results['has_ground_truth'],
        'repository_structure': repo_manager.get_output_paths()
    }
    
    print("3. Analysis Complete!")
    print(f"✅ Enhanced predictions: {len(results['enhanced_predictions'])} matches")
    print(f"✅ Dashboard created: {results['dashboard_path']}")
    
    if results['has_ground_truth']:
        f1_score = evaluation_results['evaluation_results'].get('aggregate_metrics', {}).get('f1_at_5', {}).get('mean', 0)
        print(f"✅ Algorithm F1@5 Score: {f1_score:.3f}")
    
    return results

def create_matcher(repo_manager=None, **kwargs):
    """Create enhanced matcher with optimized configuration."""
    if not CORE_IMPORTS_AVAILABLE:
        raise ImportError("Matching module not available")
    
    if repo_manager is None:
        from .utils.repository_setup import RepositoryStructureManager
        repo_manager = RepositoryStructureManager("outputs")
        repo_manager.setup_repository_structure()
    
    return FinalCleanMatcher(repo_manager=repo_manager, **kwargs)

def analyze_quality(requirements_data, repo_manager=None, **kwargs):
    """Run comprehensive quality analysis on requirements."""
    if not CORE_IMPORTS_AVAILABLE:
        raise ImportError("Quality module not available")
    
    from .quality import analyze_requirements
    return analyze_requirements(requirements_data, repo_manager=repo_manager, **kwargs)

def evaluate_results(predictions_df, ground_truth_file=None, repo_manager=None, **kwargs):
    """Run comprehensive evaluation of matching results.""" 
    if not CORE_IMPORTS_AVAILABLE:
        raise ImportError("Evaluation module not available")
    
    from .evaluation import evaluate_predictions
    return evaluate_predictions(predictions_df, ground_truth_file, repo_manager=repo_manager, **kwargs)

def check_system_status():
    """
    Check system status and available components.
    
    Returns:
        Dict: System status information
    """
    status = {
        'core_modules': CORE_IMPORTS_AVAILABLE,
        'optional_dependencies': {},
        'spacy_models': {},
        'system_info': {}
    }
    
    # Check optional dependencies
    try:
        import sentence_transformers
        status['optional_dependencies']['sentence_transformers'] = sentence_transformers.__version__
    except ImportError:
        status['optional_dependencies']['sentence_transformers'] = "Not installed"
    
    try:
        import sklearn
        status['optional_dependencies']['scikit_learn'] = sklearn.__version__
    except ImportError:
        status['optional_dependencies']['scikit_learn'] = "Not installed"
    
    # Check spaCy models
    try:
        import spacy
        for model in ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf']:
            try:
                nlp = spacy.load(model)
                status['spacy_models'][model] = "Available"
            except OSError:
                status['spacy_models'][model] = "Not installed"
    except ImportError:
        status['spacy_models'] = "spaCy not available"
    
    # System info
    import sys
    status['system_info'] = {
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    return status

def print_system_status():
    """Print comprehensive system status report."""
    status = check_system_status()
    
    print("🔧 Requirements Traceability System Status")
    print("=" * 50)
    
    print(f"Core Modules: {'✅ Available' if status['core_modules'] else '❌ Not Available'}")
    
    print("\nOptional Dependencies:")
    for dep, version in status['optional_dependencies'].items():
        icon = "✅" if version != "Not installed" else "❌"
        print(f"  {icon} {dep}: {version}")
    
    print("\nspaCy Models:")
    for model, status_text in status['spacy_models'].items():
        icon = "✅" if status_text == "Available" else "❌"
        print(f"  {icon} {model}: {status_text}")
    
    print(f"\nPython: {status['system_info']['python_version']}")
    print(f"Platform: {status['system_info']['platform']}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if status['optional_dependencies']['sentence_transformers'] == "Not installed":
        print("  • Install sentence-transformers: pip install sentence-transformers")
    if status['optional_dependencies']['scikit_learn'] == "Not installed":
        print("  • Install scikit-learn: pip install scikit-learn")
    
    missing_models = [k for k, v in status['spacy_models'].items() if v == "Not installed"]
    if missing_models:
        print(f"  • Install spaCy models: python -m spacy download {missing_models[0]}")

# Quick start guide
QUICK_START_GUIDE = """
🚀 Quick Start Guide - Requirements Traceability Analysis

1. Basic Analysis:
   >>> from src import run_complete_analysis
   >>> results = run_complete_analysis("requirements.csv", "activities.csv")
   
2. With Ground Truth Validation:
   >>> results = run_complete_analysis("reqs.csv", "acts.csv", "manual_traces.csv")
   
3. Step-by-Step Analysis:
   >>> from src import create_matcher, analyze_quality, evaluate_results
   >>> matcher = create_matcher()
   >>> predictions = matcher.run_final_matching("reqs.csv", "acts.csv")
   >>> quality_results = analyze_quality("requirements.csv")
   >>> evaluation = evaluate_results(predictions, "manual_traces.csv")

4. Check System Status:
   >>> from src import print_system_status
   >>> print_system_status()

For detailed documentation, see individual module docstrings.
"""

def print_quick_start():
    """Print quick start guide."""
    print(QUICK_START_GUIDE)

# Package metadata for setup.py
PACKAGE_INFO = {
    'name': 'requirements-traceability',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'python_requires': '>=3.8',
    'install_requires': [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'spacy>=3.4.0',
        'scikit-learn>=1.0.0',
        'openpyxl>=3.0.0',
        'pathlib',
        'logging',
        'json',
        'chardet'
    ],
    'extras_require': {
        'performance': [
            'sentence-transformers>=2.0.0',
            'scikit-learn>=1.0.0'
        ],
        'gpu': [
            'torch>=1.9.0',
            'transformers>=4.0.0'
        ]
    }
}

def get_package_info():
    """Get package information for setup.py."""
    return PACKAGE_INFO.copy()