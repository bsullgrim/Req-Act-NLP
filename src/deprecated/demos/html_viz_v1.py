"""
Technical Journey Visualization - HTML Version
Converts matplotlib-based visualizations to HTML for PowerPoint presentations (16:9)
"""

import pandas as pd
import numpy as np
import json
import textwrap
import re
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import csv
from datetime import datetime

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import project modules following conventions
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer

class TechnicalJourneyVisualizerHTML:
    """
    Creates technical layer-by-layer visualization of requirement processing
    using HTML/CSS for PowerPoint-ready 16:9 presentations.
    """
    
    def __init__(self):
        """Initialize with project conventions"""
        self.repo_manager = RepositoryStructureManager("outputs")
        self.file_handler = SafeFileHandler(self.repo_manager)
        
        # Create output directory for visualizations
        self.output_dir = self.repo_manager.structure['visuals_output']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.quality_analyzer = EnhancedRequirementAnalyzer(repo_manager=self.repo_manager)
        self.domain = DomainResources()
        
        # Color scheme for algorithms
        self.colors = {
            'semantic': '#3498db',      # Blue
            'bm25': '#e74c3c',          # Red
            'domain': '#27ae60',        # Green
            'query_expansion': '#f39c12', # Orange
            'good': '#27ae60',          # Green for good scores
            'warning': '#f39c12',       # Orange for medium scores
            'bad': '#e74c3c'            # Red for bad scores
        }
        
        # Track data for exports
        self.layer_data = {}
        
        print(f"✅ Initialized TechnicalJourneyVisualizerHTML")
        print(f"   Output directory: {self.output_dir}")
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score threshold"""
        if score >= 0.8:
            return self.colors['good']
        elif score >= 0.6:
            return self.colors['warning']
        else:
            return self.colors['bad']
    
    def _get_score_percentage(self, score: float) -> int:
        """Convert score to percentage for progress bars"""
        return min(100, max(0, int(score * 100)))
    
    def load_real_data(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Load real matching results, explanations, and quality analysis"""
        
        # Load matches
        matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
        if not matches_path.exists():
            raise FileNotFoundError(f"Run matcher.py first to generate matches at {matches_path}")
        matches_df = pd.read_csv(matches_path)
        
        # Load explanations
        explanations_path = self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json"
        explanations = {}
        if explanations_path.exists():
            with open(explanations_path, 'r', encoding='utf-8') as f:
                explanations_list = json.load(f)
                # Convert list to dict keyed by (req_id, activity)
                for item in explanations_list:
                    key = (item['requirement_id'], item['activity_name'])
                    explanations[key] = item
        
        # Load requirements for quality analysis - try multiple possible locations
        requirements_df = pd.DataFrame()
        possible_req_paths = [
            Path("requirements.csv"),  # Root directory
            Path("data/raw/requirements.csv"),  # Standard data directory
            self.repo_manager.structure.get('raw_data', Path("data/raw")) / "requirements.csv"  # Repository structure
        ]
        
        for req_path in possible_req_paths:
            if req_path.exists():
                requirements_df = pd.read_csv(req_path)
                break
        
        return matches_df, explanations, requirements_df
    
    def create_processing_journey_html(self, requirement_id: str = None, label: str = None) -> str:
        """
        Create HTML technical visualization showing each processing step.
        Returns path to generated HTML file.
        """
        # Load real data
        matches_df, explanations, requirements_df = self.load_real_data()
        
        # Select a real requirement to visualize
        if requirement_id:
            match = matches_df[matches_df['Requirement_ID'] == requirement_id]
            if match.empty:
                print(f"⚠️ {requirement_id} not found, selecting best match")
                match = matches_df.nlargest(1, 'Combined_Score')
        else:
            # Get the highest scoring match for best demonstration
            match = matches_df.nlargest(1, 'Combined_Score')
        
        if match.empty:
            raise ValueError("No matches found in aerospace_matches.csv")
        
        match = match.iloc[0]
        req_id = match['Requirement_ID']
        req_text = match['Requirement_Text']
        activity_name = match['Activity_Name']
        
        print(f"📊 Visualizing journey for: {req_id}")
        print(f"   Requirement: {req_text[:80]}...")
        print(f"   Activity: {activity_name}")
        print(f"   Score: {match['Combined_Score']:.3f}")

        # Clear previous data
        self.layer_data = {}

        # Get explanation data
        explanation = explanations.get((req_id, activity_name), {})
        
        # Create HTML content
        html_content = self._generate_html_journey(
            req_id, req_text, activity_name, match, explanation, label
        )
        
        # Save HTML file
        output_path = self.output_dir / f"{req_id}_technical_journey.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Saved HTML journey: {output_path}")
        return str(output_path)
    
    def _generate_html_journey(self, req_id: str, req_text: str, activity_name: str, 
                              match: pd.Series, explanation: Dict, label: str = None) -> str:
        """Generate complete HTML document for technical journey"""
        
        # Extract algorithm scores
        algorithm_scores = {
            'semantic': float(match.get('Semantic_Score', 0.0)),
            'bm25': float(match.get('BM25_Score', 0.0)),
            'domain': float(match.get('Domain_Score', 0.0)),
            'query_expansion': float(match.get('Query_Expansion_Score', 0.0))
        }
        combined_score = float(match['Combined_Score'])
        
        # Generate layer content
        layer1_html = self._layer1_raw_inputs_html(req_text, activity_name, req_id)
        layer2_html = self._layer2_preprocessing_html(req_text, activity_name)
        layer3_html = self._layer3_algorithms_html(algorithm_scores, explanation)
        layer4_html = self._layer4_combining_html(algorithm_scores, combined_score)
        
        # Create title
        title = f'Technical Processing Journey: {req_id} (Score: {combined_score:.3f})'
        if label:
            title = f"{label} | {title}"
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1920px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            padding: 40px;
            aspect-ratio: 16/9;
            display: flex;
            flex-direction: column;
        }}
        
        .title {{
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        
        .layer {{
            margin-bottom: 25px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            background: #fafbfc;
            flex: 1;
        }}
        
        .layer-title {{
            font-size: 1.4em;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        
        .input-box {{
            background: #e8f4fd;
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .input-label {{
            font-weight: bold;
            color: #2980b9;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .input-text {{
            color: #34495e;
            line-height: 1.4;
            font-size: 1em;
        }}
        
        .algorithm-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 15px 0;
        }}
        
        .algorithm-box {{
            border: 2px solid;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            background: white;
        }}
        
        .algo-title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .algo-score {{
            font-size: 1.8em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        
        .algo-details {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 10px;
            text-align: left;
        }}
        
        .combination-section {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        
        .formula-box {{
            background: #e3f2fd;
            border: 2px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
        }}
        
        .final-score {{
            text-align: center;
            background: #e8f5e8;
            border: 3px solid #4caf50;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        
        .final-score-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .preprocess-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }}
        
        .preprocess-box {{
            background: #fff3e0;
            border: 2px solid #ff9800;
            border-radius: 8px;
            padding: 15px;
        }}
        
        .preprocess-title {{
            font-weight: bold;
            color: #e65100;
            margin-bottom: 10px;
        }}
        
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }}
        
        .keyword {{
            background: #e1f5fe;
            color: #0277bd;
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 0.9em;
            border: 1px solid #0288d1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">{title}</div>
        
        <div class="layer">
            <div class="layer-title">Layer 1: Raw Inputs</div>
            {layer1_html}
        </div>
        
        <div class="layer">
            <div class="layer-title">Layer 2: Text Preprocessing</div>
            {layer2_html}
        </div>
        
        <div class="layer">
            <div class="layer-title">Layer 3: Algorithm Processing</div>
            {layer3_html}
        </div>
        
        <div class="layer">
            <div class="layer-title">Layer 4: Score Combination</div>
            {layer4_html}
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def _layer1_raw_inputs_html(self, req_text: str, activity_name: str, req_id: str) -> str:
        """Generate HTML for raw inputs layer"""
        self.layer_data['raw_inputs'] = {
            'requirement_id': req_id,
            'requirement_text': req_text,
            'activity_name': activity_name
        }
        
        return f"""
        <div class="input-box">
            <div class="input-label">🎯 Requirement ({req_id})</div>
            <div class="input-text">{req_text}</div>
        </div>
        
        <div class="input-box">
            <div class="input-label">⚙️ Activity</div>
            <div class="input-text">{activity_name}</div>
        </div>
        """
    
    def _layer2_preprocessing_html(self, req_text: str, activity_name: str) -> str:
        """Generate HTML for preprocessing layer"""
        # Simulate preprocessing steps
        req_tokens = re.findall(r'\b\w+\b', req_text.lower())
        activity_tokens = re.findall(r'\b\w+\b', activity_name.lower())
        
        # Get some key terms (simplified)
        key_terms = [token for token in req_tokens if len(token) > 3][:8]
        
        self.layer_data['preprocessing'] = {
            'requirement_tokens': req_tokens,
            'activity_tokens': activity_tokens,
            'key_terms': key_terms
        }
        
        key_terms_html = ''.join([f'<span class="keyword">{term}</span>' for term in key_terms])
        
        return f"""
        <div class="preprocess-grid">
            <div class="preprocess-box">
                <div class="preprocess-title">📝 Requirement Processing</div>
                <div>Tokens extracted: {len(req_tokens)}</div>
                <div>Key terms identified: {len(key_terms)}</div>
                <div class="keyword-list">{key_terms_html}</div>
            </div>
            
            <div class="preprocess-box">
                <div class="preprocess-title">⚙️ Activity Processing</div>
                <div>Tokens extracted: {len(activity_tokens)}</div>
                <div>Normalized text: "{activity_name.lower()}"</div>
            </div>
        </div>
        """
    
    def _layer3_algorithms_html(self, algorithm_scores: Dict[str, float], explanation: Dict) -> str:
        """Generate HTML for algorithms layer"""
        algo_names = {
            'semantic': 'Semantic Similarity',
            'bm25': 'BM25 Relevance',
            'domain': 'Domain Knowledge',
            'query_expansion': 'Query Expansion'
        }
        
        self.layer_data['algorithms'] = algorithm_scores.copy()
        self.layer_data['algorithms']['explanation'] = explanation
        
        algo_boxes_html = ""
        for algo, score in algorithm_scores.items():
            color = self.colors[algo]
            score_color = self._get_score_color(score)
            percentage = self._get_score_percentage(score)
            
            # Get explanation details
            algo_explanation = explanation.get(algo, 'No detailed explanation available')
            if isinstance(algo_explanation, dict):
                algo_explanation = str(algo_explanation)[:100] + "..."
            elif len(str(algo_explanation)) > 100:
                algo_explanation = str(algo_explanation)[:100] + "..."
            
            algo_boxes_html += f"""
            <div class="algorithm-box" style="border-color: {color};">
                <div class="algo-title" style="color: {color};">{algo_names[algo]}</div>
                <div class="algo-score" style="color: {score_color};">{score:.3f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%; background-color: {color};"></div>
                </div>
                <div class="algo-details">{algo_explanation}</div>
            </div>
            """
        
        return f"""
        <div class="algorithm-grid">
            {algo_boxes_html}
        </div>
        """
    
    def _layer4_combining_html(self, algorithm_scores: Dict[str, float], combined_score: float) -> str:
        """Generate HTML for score combination layer"""
        # Assume equal weights for now (can be made configurable)
        weights = {'semantic': 1.0, 'bm25': 1.0, 'domain': 1.0, 'query_expansion': 1.0}
        total_weight = sum(weights.values())
        
        self.layer_data['combination'] = {
            'weights': weights,
            'total_weight': total_weight,
            'combined_score': combined_score
        }
        
        # Build formula display
        calc_parts = []
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = algorithm_scores.get(algo, 0.0)
            weight = weights[algo]
            calc_parts.append(f"({score:.3f} × {weight:.1f})")
        
        formula_text = f"Combined = [{' + '.join(calc_parts)}] / {total_weight:.1f}"
        score_color = self._get_score_color(combined_score)
        
        return f"""
        <div class="combination-section">
            <h4>Weighted Average Formula:</h4>
            <div class="formula-box">
                {formula_text}
            </div>
            
            <div class="final-score">
                <div style="font-size: 1.2em; font-weight: bold;">Final Combined Score</div>
                <div class="final-score-value" style="color: {score_color};">{combined_score:.3f}</div>
            </div>
        </div>
        """
    
    def create_evaluation_summary_html(self) -> str:
        """
        Create HTML summary of the algorithm matching evaluation from simple_evaluator.
        Returns path to generated HTML file.
        """
        # Load evaluation results from the correct path
        eval_results_path = Path("outputs/evaluation_results/evaluation_metrics.json")
        eval_report_path = Path("outputs/evaluation_results/matching_evaluation_report.txt")
        
        # Also try the repository manager path as fallback
        if not eval_results_path.exists():
            eval_results_path = self.repo_manager.structure['evaluation_results'] / "evaluation_metrics.json"
            eval_report_path = self.repo_manager.structure['evaluation_results'] / "matching_evaluation_report.txt"
        
        if not eval_results_path.exists():
            raise FileNotFoundError(f"Run simple_evaluation.py first to generate metrics. Expected at: {eval_results_path}")
        
        # Load metrics
        with open(eval_results_path, 'r') as f:
            metrics = json.load(f)
        
        # Load report text (for additional context)
        report_text = ""
        if eval_report_path.exists():
            with open(eval_report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
        
        # Generate HTML
        html_content = self._generate_evaluation_html(metrics, report_text)
        
        # Save HTML file
        output_path = self.output_dir / "evaluation_summary.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Saved evaluation summary: {output_path}")
        return str(output_path)
    
    def _generate_evaluation_html(self, metrics: Dict, report_text: str) -> str:
        """Generate HTML for evaluation summary"""
        
        # Extract key metrics
        f1_scores = {
            'F1@1': metrics.get('f1_at_1', 0),
            'F1@3': metrics.get('f1_at_3', 0),
            'F1@5': metrics.get('f1_at_5', 0)
        }
        
        hit_rates = {
            'Hit@1': metrics.get('hit_at_1', 0),
            'Hit@3': metrics.get('hit_at_3', 0),
            'Hit@5': metrics.get('hit_at_5', 0)
        }
        
        precision_scores = {
            'Precision@1': metrics.get('precision_at_1', 0),
            'Precision@3': metrics.get('precision_at_3', 0),
            'Precision@5': metrics.get('precision_at_5', 0)
        }
        
        recall_scores = {
            'Recall@1': metrics.get('recall_at_1', 0),
            'Recall@3': metrics.get('recall_at_3', 0),
            'Recall@5': metrics.get('recall_at_5', 0)
        }
        
        # Overall metrics
        overall_metrics = {
            'Coverage': metrics.get('coverage', 0),
            'MRR': metrics.get('mrr', 0),
            'Success@1': metrics.get('success_at_1', 0),
            'Perfect Match Rate': metrics.get('perfect_match_rate', 0)
        }
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm Matching Evaluation Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            aspect-ratio: 16/9;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 30px;
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        
        .title {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            font-size: 1.2em;
            color: #7f8c8d;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            height: 100%;
        }}
        
        .metric-section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid;
            display: flex;
            flex-direction: column;
        }}
        
        .section-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            padding: 8px 12px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric-name {{
            font-weight: 600;
            color: #34495e;
        }}
        
        .metric-value {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .value-excellent {{ color: #27ae60; }}
        .value-good {{ color: #f39c12; }}
        .value-poor {{ color: #e74c3c; }}
        
        .f1-section {{ border-left-color: #3498db; }}
        .hit-section {{ border-left-color: #27ae60; }}
        .precision-section {{ border-left-color: #e74c3c; }}
        .recall-section {{ border-left-color: #f39c12; }}
        
        .f1-section .section-title {{ color: #3498db; }}
        .hit-section .section-title {{ color: #27ae60; }}
        .precision-section .section-title {{ color: #e74c3c; }}
        .recall-section .section-title {{ color: #f39c12; }}
        
        .overall-stats {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }}
        
        .overall-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .overall-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        .overall-metric {{
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
        }}
        
        .overall-metric-name {{
            font-size: 0.9em;
            margin-bottom: 5px;
            opacity: 0.9;
        }}
        
        .overall-metric-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🎯 Algorithm Matching Evaluation</div>
            <div class="subtitle">Performance Analysis & Key Metrics</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-section f1-section">
                <div class="section-title">📈 F1 Scores</div>
                {self._generate_metric_items(f1_scores)}
            </div>
            
            <div class="metric-section hit-section">
                <div class="section-title">🎯 Hit Rates</div>
                {self._generate_metric_items(hit_rates, is_percentage=True)}
            </div>
            
            <div class="metric-section precision-section">
                <div class="section-title">🔍 Precision</div>
                {self._generate_metric_items(precision_scores)}
            </div>
            
            <div class="metric-section recall-section">
                <div class="section-title">📊 Recall</div>
                {self._generate_metric_items(recall_scores)}
            </div>
        </div>
        
        <div class="overall-stats">
            <div class="overall-title">🌟 Overall Performance Summary</div>
            <div class="overall-grid">
                {self._generate_overall_metrics(overall_metrics)}
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def _generate_metric_items(self, metrics: Dict[str, float], is_percentage: bool = False) -> str:
        """Generate HTML for metric items"""
        items_html = ""
        for name, value in metrics.items():
            if is_percentage:
                display_value = f"{value:.1%}"
                css_class = self._get_percentage_class(value)
            else:
                display_value = f"{value:.3f}"
                css_class = self._get_score_class(value)
            
            items_html += f"""
            <div class="metric-item">
                <span class="metric-name">{name}</span>
                <span class="metric-value {css_class}">{display_value}</span>
            </div>
            """
        
        return items_html
    
    def _generate_overall_metrics(self, metrics: Dict[str, float]) -> str:
        """Generate HTML for overall metrics"""
        items_html = ""
        for name, value in metrics.items():
            if name in ['Coverage', 'Success@1', 'Perfect Match Rate']:
                display_value = f"{value:.1%}"
            else:  # MRR
                display_value = f"{value:.3f}"
            
            items_html += f"""
            <div class="overall-metric">
                <div class="overall-metric-name">{name}</div>
                <div class="overall-metric-value">{display_value}</div>
            </div>
            """
        
        return items_html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score-based values"""
        if score >= 0.8:
            return "value-excellent"
        elif score >= 0.6:
            return "value-good"
        else:
            return "value-poor"
    
    def _get_percentage_class(self, percentage: float) -> str:
        """Get CSS class for percentage values"""
        if percentage >= 0.8:
            return "value-excellent"
        elif percentage >= 0.6:
            return "value-good"
        else:
            return "value-poor"
    
    def create_algorithm_contribution_html(self) -> str:
        """
        Create HTML visualization of algorithm contribution analysis.
        Returns path to generated HTML file.
        """
        # Load matches data
        matches_df, _, _ = self.load_real_data()
        
        # Calculate algorithm statistics
        algorithm_stats = self._calculate_algorithm_stats(matches_df)
        
        # Generate HTML
        html_content = self._generate_algorithm_contribution_html(algorithm_stats, matches_df)
        
        # Save HTML file
        output_path = self.output_dir / "algorithm_contribution_analysis.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📈 Saved algorithm contribution analysis: {output_path}")
        return str(output_path)
    
    def _calculate_algorithm_stats(self, matches_df: pd.DataFrame) -> Dict:
        """Calculate statistics for algorithm contribution analysis"""
        
        algo_columns = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        stats = {}
        
        for col in algo_columns:
            if col in matches_df.columns:
                scores = matches_df[col].dropna()
                algo_name = col.replace('_Score', '').replace('_', ' ')
                
                stats[algo_name] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'median': float(scores.median()),
                    'count': len(scores),
                    'above_threshold': len(scores[scores > 0.6]),
                    'excellent': len(scores[scores > 0.8])
                }
        
        # Calculate correlations
        correlations = {}
        for i, col1 in enumerate(algo_columns):
            if col1 in matches_df.columns:
                for col2 in algo_columns[i+1:]:
                    if col2 in matches_df.columns:
                        corr = matches_df[col1].corr(matches_df[col2])
                        algo1 = col1.replace('_Score', '')
                        algo2 = col2.replace('_Score', '')
                        correlations[f"{algo1}-{algo2}"] = float(corr)
        
        return {
            'algorithm_stats': stats,
            'correlations': correlations,
            'total_matches': len(matches_df)
        }
    
    def _generate_algorithm_contribution_html(self, stats: Dict, matches_df: pd.DataFrame) -> str:
        """Generate HTML for algorithm contribution analysis"""
        
        algorithm_stats = stats['algorithm_stats']
        correlations = stats['correlations']
        
        # Generate algorithm performance cards
        algo_cards_html = ""
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
        
        for i, (algo_name, algo_stats) in enumerate(algorithm_stats.items()):
            color = colors[i % len(colors)]
            mean_score = algo_stats['mean']
            
            # Calculate performance rating
            if mean_score >= 0.7:
                rating = "Excellent"
                rating_color = "#27ae60"
            elif mean_score >= 0.5:
                rating = "Good"
                rating_color = "#f39c12"
            else:
                rating = "Needs Improvement"
                rating_color = "#e74c3c"
            
            algo_cards_html += f"""
            <div class="algo-card" style="border-left-color: {color};">
                <div class="algo-card-header">
                    <h3 style="color: {color};">{algo_name}</h3>
                    <div class="rating" style="background-color: {rating_color};">{rating}</div>
                </div>
                
                <div class="algo-stats">
                    <div class="stat-item">
                        <span class="stat-label">Average Score</span>
                        <span class="stat-value">{mean_score:.3f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Score Range</span>
                        <span class="stat-value">{algo_stats['min']:.3f} - {algo_stats['max']:.3f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Above Threshold (0.6)</span>
                        <span class="stat-value">{algo_stats['above_threshold']}/{algo_stats['count']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Excellent (>0.8)</span>
                        <span class="stat-value">{algo_stats['excellent']}/{algo_stats['count']}</span>
                    </div>
                </div>
                
                <div class="score-bar">
                    <div class="score-fill" style="width: {mean_score * 100}%; background-color: {color};"></div>
                </div>
            </div>
            """
        
        # Generate correlation matrix
        correlation_items_html = ""
        for pair, corr_value in correlations.items():
            if corr_value >= 0.7:
                corr_class = "corr-high"
                corr_desc = "Strong"
            elif corr_value >= 0.4:
                corr_class = "corr-medium"
                corr_desc = "Moderate"
            else:
                corr_class = "corr-low"
                corr_desc = "Weak"
            
            correlation_items_html += f"""
            <div class="correlation-item {corr_class}">
                <div class="corr-pair">{pair.replace('-', ' ↔ ')}</div>
                <div class="corr-value">{corr_value:.3f}</div>
                <div class="corr-desc">{corr_desc}</div>
            </div>
            """
        
        # Calculate top performing pairs
        if 'Combined_Score' in matches_df.columns:
            top_matches = matches_df.nlargest(5, 'Combined_Score')
            top_matches_html = ""
            
            for idx, match in top_matches.iterrows():
                req_id = match.get('Requirement_ID', 'Unknown')
                activity = match.get('Activity_Name', 'Unknown')
                combined_score = match.get('Combined_Score', 0)
                
                top_matches_html += f"""
                <div class="top-match-item">
                    <div class="match-info">
                        <strong>{req_id}</strong>
                        <span class="activity-name">{activity[:50]}...</span>
                    </div>
                    <div class="match-score">{combined_score:.3f}</div>
                </div>
                """
        else:
            top_matches_html = "<div>Combined scores not available</div>"
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm Contribution Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            aspect-ratio: 16/9;
            display: grid;
            grid-template-rows: auto 1fr auto;
            gap: 30px;
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #74b9ff;
            padding-bottom: 20px;
        }}
        
        .title {{
            font-size: 2.8em;
            font-weight: bold;
            color: #2d3436;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            font-size: 1.3em;
            color: #636e72;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            height: 100%;
        }}
        
        .algorithms-section {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        
        .algo-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .algo-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .algo-card-header h3 {{
            margin: 0;
            font-size: 1.4em;
        }}
        
        .rating {{
            padding: 4px 12px;
            border-radius: 15px;
            color: white;
            font-size: 0.9em;
            font-weight: bold;
        }}
        
        .algo-stats {{
            flex-grow: 1;
            margin-bottom: 15px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #495057;
        }}
        
        .score-bar {{
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .score-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        
        .insights-section {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .insight-panel {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            flex: 1;
        }}
        
        .panel-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 15px;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 8px;
        }}
        
        .correlation-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        
        .corr-high {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .corr-medium {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .corr-low {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        
        .corr-pair {{
            font-weight: 600;
            flex-grow: 1;
        }}
        
        .corr-value {{
            font-weight: bold;
            margin: 0 10px;
        }}
        
        .corr-desc {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .top-match-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .match-info {{
            flex-grow: 1;
        }}
        
        .activity-name {{
            display: block;
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 4px;
        }}
        
        .match-score {{
            font-weight: bold;
            font-size: 1.1em;
            color: #28a745;
        }}
        
        .summary-stats {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 15px;
        }}
        
        .summary-item {{
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
        }}
        
        .summary-label {{
            font-size: 0.9em;
            margin-bottom: 5px;
            opacity: 0.9;
        }}
        
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🔬 Algorithm Contribution Analysis</div>
            <div class="subtitle">Performance Breakdown & Algorithm Interactions</div>
        </div>
        
        <div class="main-content">
            <div class="algorithms-section">
                {algo_cards_html}
            </div>
            
            <div class="insights-section">
                <div class="insight-panel">
                    <div class="panel-title">🔗 Algorithm Correlations</div>
                    {correlation_items_html}
                </div>
                
                <div class="insight-panel">
                    <div class="panel-title">🏆 Top Performing Matches</div>
                    {top_matches_html}
                </div>
            </div>
        </div>
        
        <div class="summary-stats">
            <div style="font-size: 1.4em; font-weight: bold; margin-bottom: 10px;">📊 Analysis Summary</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Total Matches</div>
                    <div class="summary-value">{stats['total_matches']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Algorithms</div>
                    <div class="summary-value">{len(algorithm_stats)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Correlations</div>
                    <div class="summary-value">{len(correlations)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Best Avg Score</div>
                    <div class="summary-value">{max(s['mean'] for s in algorithm_stats.values()):.3f}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def save_layer_data(self, req_id: str):
        """Save layer data to files for analysis"""
        if not self.layer_data:
            print("⚠️ No layer data to save. Run create_processing_journey_html first.")
            return
        
        # Save as JSON
        json_path = self.output_dir / f"{req_id}_layer_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_data, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        csv_path = self.output_dir / f"{req_id}_layer_data.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Key', 'Summary', 'Full_Value'])
            
            for layer_name, content in self.layer_data.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                writer.writerow([layer_name, f"{key}.{sub_key}", str(sub_value)[:100], str(sub_value)])
                        else:
                            writer.writerow([layer_name, key, str(value)[:100], str(value)])
                else:
                    writer.writerow([layer_name, 'data', str(content)[:100], str(content)])
        
        # Save as formatted text for easy copy-paste
        txt_path = self.output_dir / f"{req_id}_layer_text.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"TECHNICAL JOURNEY LAYERS - {req_id}\n")
            f.write("="*80 + "\n\n")
            
            for layer_name, content in self.layer_data.items():
                f.write(f"LAYER: {layer_name}\n")
                f.write("-"*40 + "\n")
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"  {key}:\n")
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                f.write(f"    - {sub_key}: {sub_value}\n")
                        elif isinstance(value, list):
                            for item in value[:5]:  # Limit to first 5 items
                                f.write(f"    - {item}\n")
                        else:
                            # Wrap long text
                            wrapped = textwrap.wrap(str(value), width=70)
                            for line in wrapped[:3]:  # Limit to 3 lines
                                f.write(f"    {line}\n")
                else:
                    f.write(f"  {content}\n")
                f.write("\n")
        
        print(f"📄 Saved layer data:")
        print(f"   - JSON: {json_path}")
        print(f"   - CSV: {csv_path}")
        print(f"   - TXT: {txt_path}")

def main():
    """Demo function to test the HTML visualizer"""
    visualizer = TechnicalJourneyVisualizerHTML()
    
    try:
        # Create processing journey for highest scoring match
        print("🚀 Creating HTML processing journey...")
        journey_path = visualizer.create_processing_journey_html(label="DEMO")
        
        # Create evaluation summary
        print("\n📊 Creating evaluation summary...")
        eval_path = visualizer.create_evaluation_summary_html()
        
        # Create algorithm contribution analysis
        print("\n📈 Creating algorithm contribution analysis...")
        contrib_path = visualizer.create_algorithm_contribution_html()
        
        print(f"\n✅ All HTML visualizations created:")
        print(f"   🎯 Processing Journey: {journey_path}")
        print(f"   📊 Evaluation Summary: {eval_path}")
        print(f"   📈 Algorithm Analysis: {contrib_path}")
        print(f"\n💡 Open these files in your browser for PowerPoint-ready 16:9 visualizations!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to run matcher.py and simple_evaluation.py first!")

if __name__ == "__main__":
    main()