

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
        layer5_html = self._layer5_incose_analysis_html(req_text)
        layer6_html = self._layer6_quality_dimensions_html(req_text)
        layer7_html = self._layer7_final_decision_html(combined_score, req_text, match)
        
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
            padding: 8px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1920px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            padding: 12px;
            aspect-ratio: 16/9;
            display: grid;
            grid-template-columns: 0.8fr 1.2fr 1.3fr;
            grid-template-rows: auto 1.2fr 1.3fr 0.9fr;
            gap: 6px;
            overflow: hidden;
        }}

        .title {{
            grid-column: 1 / -1;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 4px;
        }}

        .layer {{
            border: 1px solid #ecf0f1;
            border-radius: 5px;
            padding: 6px;
            background: #fafbfc;
            min-height: 0;
            overflow: auto;
        }}

        .layer-title {{
            font-size: 0.8em;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 4px;
            border-left: 3px solid #3498db;
            padding-left: 5px;
        }}

        /* Grid layout for layers - adjusted proportions */
        .layer-1 {{ grid-column: 1 / 2; grid-row: 2 / 3; }}
        .layer-2 {{ grid-column: 2 / 3; grid-row: 2 / 3; }}
        .layer-3 {{ grid-column: 3 / 4; grid-row: 2 / 3; }}
        .layer-4 {{ grid-column: 1 / 2; grid-row: 3 / 4; }}
        .layer-5 {{ grid-column: 2 / 4; grid-row: 3 / 4; }}
        .layer-6 {{ grid-column: 1 / 2; grid-row: 4 / 5; }}
        .layer-7 {{ grid-column: 2 / 4; grid-row: 4 / 5; }}
        
        .input-box {{
            background: #e8f4fd;
            border: 1px solid #3498db;
            border-radius: 4px;
            padding: 6px;
            margin: 3px 0;
        }}

        .input-label {{
            font-weight: bold;
            color: #2980b9;
            font-size: 0.85em;
            margin-bottom: 3px;
        }}

        .input-text {{
            color: #34495e;
            line-height: 1.2;
            font-size: 0.8em;
        }}

        .algorithm-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px;
            margin: 3px 0;
        }}

        .algorithm-box {{
            border: 1px solid;
            border-radius: 3px;
            padding: 4px;
            text-align: center;
            background: white;
        }}

        .algo-title {{
            font-weight: bold;
            font-size: 0.7em;
            margin-bottom: 2px;
        }}

        .algo-score {{
            font-size: 1em;
            font-weight: bold;
            margin: 2px 0;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin: 2px 0;
        }}

        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .algo-details {{
            font-size: 0.6em;
            color: #7f8c8d;
            margin-top: 2px;
            text-align: left;
            line-height: 1;
            max-height: 2.4em;
            overflow: hidden;
        }}
        
        .combination-section {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 4px;
            margin: 2px 0;
        }}

        .formula-box {{
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 3px;
            padding: 4px;
            margin: 3px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.65em;
        }}

        .final-score {{
            text-align: center;
            background: #e8f5e8;
            border: 2px solid #4caf50;
            border-radius: 3px;
            padding: 3px;
            margin: 2px 0;
        }}

        .final-score-value {{
            font-size: 1em;
            font-weight: bold;
            margin: 1px 0;
        }}

        .preprocess-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 4px;
            margin: 3px 0;
        }}

        .preprocess-box {{
            background: #fff3e0;
            border: 1px solid #ff9800;
            border-radius: 3px;
            padding: 4px;
        }}

        .preprocess-title {{
            font-weight: bold;
            color: #e65100;
            margin-bottom: 2px;
            font-size: 0.7em;
        }}
        
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            margin: 4px 0;
        }}

        .keyword {{
            background: #e1f5fe;
            color: #0277bd;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.7em;
            border: 1px solid #0288d1;
        }}
        
        /* Layer 5: INCOSE Analysis Styles */
        .incose-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }}

        .pattern-info h4 {{
            margin: 0;
            color: #2c3e50;
            font-size: 0.8em;
        }}

        .compliance-score {{
            font-size: 1em;
            font-weight: bold;
            padding: 3px 8px;
            background: #f8f9fa;
            border-radius: 4px;
            border: 2px solid;
        }}
        
        .pattern-template {{
            background: #fffef0;
            border: 1px dashed #f39c12;
            border-radius: 3px;
            padding: 4px;
            margin: 4px 0;
            font-style: italic;
            color: #555;
            font-size: 0.65em;
        }}

        .components-analysis {{
            margin-top: 5px;
        }}

        .component-summary {{
            display: flex;
            gap: 6px;
            margin-bottom: 5px;
        }}

        .summary-item {{
            padding: 3px 6px;
            border-radius: 3px;
            font-weight: bold;
            text-align: center;
            flex: 1;
            font-size: 0.7em;
        }}

        .required-summary {{
            background: #fadbd8;
            border: 1px solid #e74c3c;
            color: #c0392b;
        }}

        .optional-summary {{
            background: #e8f4fd;
            border: 1px solid #3498db;
            color: #2980b9;
        }}

        .components-list {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 3px;
        }}

        .component-item {{
            display: flex;
            justify-content: space-between;
            padding: 3px 5px;
            border-radius: 3px;
            border: 1px solid;
            font-size: 0.65em;
        }}
        
        .component-item.required {{
            border-color: #e74c3c;
        }}
        
        .component-item.optional {{
            border-color: #3498db;
        }}
        
        .component-item.present {{
            background: #d5f4e6;
        }}
        
        .component-item.missing {{
            background: #f8f9fa;
            opacity: 0.7;
        }}
        
        .component-name {{
            font-weight: bold;
        }}
        
        .component-value {{
            font-size: 0.9em;
            color: #555;
        }}
        
        /* Layer 6: Quality Dimensions Styles */
        .quality-header {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 5px;
        }}

        .overall-grade {{
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
            border: 2px solid;
        }}

        .grade-label {{
            font-size: 0.65em;
            margin-bottom: 2px;
        }}

        .grade-value {{
            font-size: 1.1em;
            font-weight: bold;
            margin: 2px 0;
        }}

        .grade-score {{
            font-size: 0.7em;
            opacity: 0.8;
        }}

        .dimensions-row {{
            display: flex;
            justify-content: space-around;
            margin: 5px 0;
        }}

        .dimension-circle {{
            text-align: center;
        }}

        .circle-score {{
            width: 35px;
            height: 35px;
            border-radius: 50%;
            border: 2px solid;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85em;
            font-weight: bold;
            margin: 0 auto 3px;
        }}

        .circle-label {{
            font-size: 0.65em;
            font-weight: bold;
        }}
        
        .quality-details {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 4px;
            margin-top: 5px;
        }}

        .semantic-analysis, .issues-summary {{
            background: #f8f9fa;
            padding: 4px;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }}

        .semantic-analysis h5, .issues-summary h5 {{
            margin-top: 0;
            color: #495057;
            font-size: 0.7em;
        }}

        .entity-breakdown {{
            margin: 3px 0;
        }}

        .entity-type {{
            margin: 2px 0;
            font-size: 0.65em;
        }}

        .entity-type ul {{
            margin: 2px 0 0 10px;
            padding: 0;
        }}

        .entity-type li {{
            font-size: 0.65em;
            color: #666;
        }}

        .ambiguities {{
            margin-top: 4px;
            font-size: 0.65em;
        }}

        .severity-breakdown {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 3px;
        }}

        .severity-item {{
            padding: 2px 4px;
            background: white;
            border-radius: 3px;
            font-size: 0.65em;
        }}
        
        /* Layer 7: Final Decision Styles */
        .decision-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }}

        .decision-box {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            border-radius: 5px;
            border: 2px solid;
            width: 100%;
        }}

        .decision-symbol {{
            font-size: 1.8em;
            font-weight: bold;
        }}

        .decision-content {{
            flex: 1;
        }}

        .decision-title {{
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 3px;
        }}

        .decision-action {{
            font-size: 0.7em;
            color: #666;
        }}

        .decision-metrics {{
            display: flex;
            gap: 10px;
            background: #f8f9fa;
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            width: 100%;
            justify-content: space-around;
        }}

        .decision-metrics .metric-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }}

        .decision-metrics .metric-item span:first-child {{
            font-size: 0.65em;
            color: #666;
        }}

        .decision-metrics .metric-item span:last-child {{
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .error {{
            color: #e74c3c;
            background: #fadbd8;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e74c3c;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">{title}</div>

        <div class="layer layer-1">
            <div class="layer-title">Layer 1: Raw Inputs</div>
            {layer1_html}
        </div>

        <div class="layer layer-2">
            <div class="layer-title">Layer 2: Preprocessing</div>
            {layer2_html}
        </div>

        <div class="layer layer-3">
            <div class="layer-title">Layer 3: Algorithms</div>
            {layer3_html}
        </div>

        <div class="layer layer-4">
            <div class="layer-title">Layer 4: Combination</div>
            {layer4_html}
        </div>

        <div class="layer layer-5">
            <div class="layer-title">Layer 5: INCOSE Analysis</div>
            {layer5_html}
        </div>

        <div class="layer layer-6">
            <div class="layer-title">Layer 6: Quality</div>
            {layer6_html}
        </div>

        <div class="layer layer-7">
            <div class="layer-title">Layer 7: Final Decision</div>
            {layer7_html}
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
            'semantic': 'Semantic',
            'bm25': 'BM25',
            'domain': 'Domain',
            'query_expansion': 'Query Exp.'
        }

        self.layer_data['algorithms'] = algorithm_scores.copy()
        self.layer_data['algorithms']['explanation'] = explanation

        # Extract explanations from the correct JSON structure
        explanations_obj = explanation.get('explanations', {})

        algo_boxes_html = ""
        for algo, score in algorithm_scores.items():
            color = self.colors[algo]
            score_color = self._get_score_color(score)
            percentage = self._get_score_percentage(score)

            # Get explanation from the 'explanations' object
            algo_explanation = 'No explanation'

            if explanations_obj:
                exp_data = explanations_obj.get(algo, '')

                if algo == 'semantic':
                    # Semantic is usually a string
                    if isinstance(exp_data, str) and exp_data:
                        algo_explanation = exp_data[:100]
                    elif exp_data:
                        algo_explanation = str(exp_data)[:100]

                elif algo == 'bm25':
                    # BM25 might have shared terms
                    shared_terms = explanation.get('shared_terms', [])
                    if shared_terms:
                        algo_explanation = f"Matched: {', '.join(shared_terms[:4])}"
                    elif isinstance(exp_data, str) and exp_data:
                        algo_explanation = exp_data[:100]

                elif algo == 'domain':
                    # Domain usually has aerospace_terms and activity_patterns
                    if isinstance(exp_data, dict):
                        aero_terms = exp_data.get('aerospace_terms', [])
                        if aero_terms:
                            algo_explanation = f"Aero terms: {', '.join(aero_terms[:3])}"
                        else:
                            algo_explanation = str(exp_data)[:100]
                    elif exp_data:
                        algo_explanation = str(exp_data)[:100]

                elif algo == 'query_expansion':
                    # Query expansion has matched_terms
                    if isinstance(exp_data, dict):
                        matched = exp_data.get('matched_terms', [])
                        if matched:
                            algo_explanation = f"Expanded: {', '.join(matched[:3])}"
                        else:
                            algo_explanation = str(exp_data)[:100]
                    elif exp_data:
                        algo_explanation = str(exp_data)[:100]

            # Truncate if needed
            if len(algo_explanation) > 100:
                algo_explanation = algo_explanation[:100] + "..."
            
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
    
    def _layer5_incose_analysis_html(self, req_text: str) -> str:
        """Generate HTML for INCOSE pattern analysis layer"""
        try:
            analysis = self.quality_analyzer.incose_analyzer.analyze_incose_compliance(req_text)
            
            pattern_key = analysis.best_pattern
            pattern_name = pattern_key
            pattern_def = None
            
            if pattern_key in self.quality_analyzer.incose_analyzer.patterns:
                pattern_def = self.quality_analyzer.incose_analyzer.patterns[pattern_key]
                pattern_name = pattern_def.get('name', pattern_key)
            
            components = analysis.components_found
            required_components = set(pattern_def.get('required', [])) if pattern_def else set()
            optional_components = set(pattern_def.get('optional', [])) if pattern_def else set()
            
            present_required = len([c for c in required_components if components.get(c)])
            total_required = len(required_components)
            present_optional = len([c for c in optional_components if components.get(c)])
            total_optional = len(optional_components)
            
            score_color = self._get_score_color(float(analysis.compliance_score) if isinstance(analysis.compliance_score, (int, float)) else 0)
            
            self.layer_data['incose_analysis'] = {
                'pattern_name': pattern_name,
                'compliance_score': analysis.compliance_score,
                'components_found': components,
                'required_present': f"{present_required}/{total_required}",
                'optional_present': f"{present_optional}/{total_optional}"
            }
            
            # Generate component items HTML
            component_items_html = ""
            sorted_components = sorted(components.keys(), key=lambda x: (x not in required_components, not bool(components[x]), x))
            
            for comp in sorted_components:
                val = components[comp]
                is_required = comp in required_components
                is_present = bool(val)
                
                status_class = "required" if is_required else "optional"
                present_class = "present" if is_present else "missing"
                
                comp_display = comp.replace('_', ' ').title()
                value_display = str(val) if is_present else "Not found"
                
                component_items_html += f"""
                <div class="component-item {status_class} {present_class}">
                    <span class="component-name">{comp_display}</span>
                    <span class="component-value">{value_display}</span>
                </div>
                """
            
            return f"""
            <div class="incose-header">
                <div class="pattern-info">
                    <h4>Best Pattern: {pattern_name}</h4>
                    <div class="compliance-score" style="color: {score_color};">
                        Score: {analysis.compliance_score}
                    </div>
                </div>
            </div>
            
            <div class="pattern-template">
                <strong>Template:</strong> {pattern_def.get('template', 'No template available') if pattern_def else 'No template available'}
            </div>
            
            <div class="components-analysis">
                <div class="component-summary">
                    <div class="summary-item required-summary">
                        Required: {present_required}/{total_required}
                    </div>
                    <div class="summary-item optional-summary">
                        Optional: {present_optional}/{total_optional}
                    </div>
                </div>
                
                <div class="components-list">
                    {component_items_html}
                </div>
            </div>
            """
            
        except Exception as e:
            return f"<div class='error'>INCOSE analysis failed: {str(e)}</div>"
    
    def _layer6_quality_dimensions_html(self, req_text: str) -> str:
        """Generate HTML for quality dimensions analysis layer"""
        try:
            issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)
            
            dimensions = [
                ('Clarity', metrics.clarity_score),
                ('Complete', metrics.completeness_score),
                ('Verifiable', metrics.verifiability_score),
                ('Atomic', metrics.atomicity_score),
                ('Consistent', metrics.consistency_score)
            ]
            
            overall_score = metrics.quality_score
            grade = self.quality_analyzer._get_grade(overall_score)
            grade_color = self._get_score_color(overall_score)
            
            self.layer_data['quality_analysis'] = {
                'overall_score': overall_score,
                'grade': grade,
                'dimensions': {dim[0]: dim[1] for dim in dimensions},
                'semantic_quality': metrics.semantic_quality_score,
                'total_issues': metrics.total_issues,
                'severity_breakdown': metrics.severity_breakdown
            }
            
            # Generate dimension circles
            dimension_circles_html = ""
            for dim_name, score in dimensions:
                dim_color = self._get_score_color(score)
                dimension_circles_html += f"""
                <div class="dimension-circle">
                    <div class="circle-score" style="border-color: {dim_color}; color: {dim_color};">
                        {score:.0f}
                    </div>
                    <div class="circle-label">{dim_name}</div>
                </div>
                """
            
            # Generate semantic analysis details
            entities = semantic.entity_completeness
            entity_details_html = ""
            for entity_type, found_items in entities.items():
                count = len(found_items)
                color = '#27ae60' if count > 0 else '#95a5a6'
                entity_details_html += f"""
                <div class="entity-type" style="color: {color};">
                    <strong>{entity_type.title()}:</strong> {count}
                    {('<ul>' + ''.join([f'<li>{item}</li>' for item in found_items[:3]]) + '</ul>') if found_items else ''}
                </div>
                """
            
            # Generate issues summary
            severity_items_html = ""
            for severity, count in metrics.severity_breakdown.items():
                color = {'critical': '#e74c3c', 'high': '#f39c12', 'medium': '#f1c40f', 'low': '#95a5a6'}.get(severity, '#95a5a6')
                severity_items_html += f"""
                <div class="severity-item">
                    <span style="color: {color};">{severity.title()}: {count}</span>
                </div>
                """
            
            return f"""
            <div class="quality-header">
                <div class="overall-grade" style="background-color: {grade_color}20; border-color: {grade_color};">
                    <div class="grade-label">Overall Grade</div>
                    <div class="grade-value" style="color: {grade_color};">{grade}</div>
                    <div class="grade-score">{overall_score:.0f}/100</div>
                </div>
            </div>
            
            <div class="dimensions-row">
                {dimension_circles_html}
            </div>
            
            <div class="quality-details">
                <div class="semantic-analysis">
                    <h5>Semantic Analysis (Score: {metrics.semantic_quality_score:.0f})</h5>
                    <div class="entity-breakdown">
                        {entity_details_html}
                    </div>
                    
                    <div class="ambiguities">
                        <strong>Ambiguities Found:</strong> {len(semantic.contextual_ambiguities)}
                        {('<ul>' + ''.join([f'<li>{amb}</li>' for amb in semantic.contextual_ambiguities[:3]]) + '</ul>') if semantic.contextual_ambiguities else '<span style="color: #27ae60;"> None</span>'}
                    </div>
                </div>
                
                <div class="issues-summary">
                    <h5>Issues Summary (Total: {metrics.total_issues})</h5>
                    <div class="severity-breakdown">
                        {severity_items_html}
                    </div>
                </div>
            </div>
            """
            
        except Exception as e:
            return f"<div class='error'>Quality analysis failed: {str(e)}</div>"
    
    def _layer7_final_decision_html(self, combined_score: float, req_text: str, match: pd.Series) -> str:
        """Generate HTML for final decision layer"""
        try:
            # Get quality results for decision logic
            issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)
            grade = self.quality_analyzer._get_grade(metrics.quality_score)
            
            # Decision logic
            if combined_score >= 0.8 and grade in ['EXCELLENT', 'GOOD']:
                decision = "ACCEPT MATCH"
                action = "High confidence match with good quality"
                color = self.colors['good']
                symbol = "✓"
            elif combined_score >= 0.35:
                decision = "REVIEW NEEDED"
                action = "Moderate confidence - engineer review required"
                color = self.colors['warning']
                symbol = "?"
            else:
                decision = "ORPHAN"
                action = "No suitable match - write bridge requirement"
                color = self.colors['bad']
                symbol = "✗"
            
            self.layer_data['final_decision'] = {
                'decision': decision,
                'action': action,
                'combined_score': combined_score,
                'quality_grade': grade,
                'total_issues': metrics.total_issues
            }
            
            return f"""
            <div class="decision-container">
                <div class="decision-box" style="border-color: {color}; background-color: {color}20;">
                    <div class="decision-symbol" style="color: {color};">{symbol}</div>
                    <div class="decision-content">
                        <div class="decision-title" style="color: {color};">{decision}</div>
                        <div class="decision-action">{action}</div>
                    </div>
                </div>
                
                <div class="decision-metrics">
                    <div class="metric-item">
                        <span>Match Score:</span>
                        <span style="color: {self._get_score_color(combined_score)};">{combined_score:.3f}</span>
                    </div>
                    <div class="metric-item">
                        <span>Quality Grade:</span>
                        <span style="color: {self._get_score_color(metrics.quality_score)};">{grade}</span>
                    </div>
                    <div class="metric-item">
                        <span>Total Issues:</span>
                        <span>{metrics.total_issues}</span>
                    </div>
                </div>
            </div>
            """
            
        except Exception as e:
            return f"<div class='error'>Final decision analysis failed: {str(e)}</div>"
    
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
            padding: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
            padding: 15px;
            aspect-ratio: 16/9;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 10px;
            overflow: hidden;
        }}

        .header {{
            text-align: center;
            border-bottom: 2px solid #667eea;
            padding-bottom: 6px;
        }}

        .title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 4px;
        }}

        .subtitle {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            height: 100%;
        }}

        .metric-section {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 8px;
            border-left: 3px solid;
            display: flex;
            flex-direction: column;
        }}

        .section-title {{
            font-size: 0.95em;
            font-weight: bold;
            margin-bottom: 6px;
            text-align: center;
        }}

        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 2px 0;
            padding: 4px 6px;
            background: white;
            border-radius: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}

        .metric-name {{
            font-weight: 600;
            color: #34495e;
            font-size: 0.8em;
        }}

        .metric-value {{
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .value-excellent {{ color: #2c3e50; }}
        .value-good {{ color: #2c3e50; }}
        .value-poor {{ color: #2c3e50; }}
        
        .f1-section {{ border-left-color: #3498db; }}
        .hit-section {{ border-left-color: #27ae60; }}
        .precision-section {{ border-left-color: #9b59b6; }}
        .recall-section {{ border-left-color: #e67e22; }}

        .f1-section .section-title {{ color: #3498db; }}
        .hit-section .section-title {{ color: #27ae60; }}
        .precision-section .section-title {{ color: #9b59b6; }}
        .recall-section .section-title {{ color: #e67e22; }}
        
        .overall-stats {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            margin-top: 8px;
        }}

        .overall-title {{
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 8px;
        }}

        .overall-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }}

        .overall-metric {{
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            padding: 8px;
        }}

        .overall-metric-name {{
            font-size: 0.75em;
            margin-bottom: 3px;
            opacity: 0.9;
        }}

        .overall-metric-value {{
            font-size: 1.1em;
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
    
    def create_algorithm_contribution_html(self) -> str:
        """
        Create sophisticated HTML visualization of algorithm contribution analysis like V1.
        Returns path to generated HTML file.
        """
        # Load matches data
        matches_df, _, _ = self.load_real_data()
        
        # Calculate comprehensive algorithm statistics like V1
        algorithm_stats = self._calculate_comprehensive_algorithm_stats(matches_df)
        
        # Generate HTML
        html_content = self._generate_sophisticated_algorithm_html(algorithm_stats, matches_df)
        
        # Save HTML file
        output_path = self.output_dir / "algorithm_contribution_analysis.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📈 Saved algorithm contribution analysis: {output_path}")
        return str(output_path)
    
    def _calculate_comprehensive_algorithm_stats(self, matches_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics for algorithm contribution analysis like V1"""
        
        algorithms = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        algo_names = ['Semantic', 'BM25', 'Domain', 'Query Expansion']
        
        stats = {}
        
        # 1. Average contributions (for pie chart)
        avg_scores = {}
        for algo, name in zip(algorithms, algo_names):
            if algo in matches_df.columns:
                avg_scores[name] = matches_df[algo].mean()
        
        # 2. Score distributions (for box plot equivalent)
        distributions = {}
        for algo, name in zip(algorithms, algo_names):
            if algo in matches_df.columns:
                scores = matches_df[algo].dropna()
                distributions[name] = {
                    'min': float(scores.min()),
                    'q25': float(scores.quantile(0.25)),
                    'median': float(scores.median()),
                    'q75': float(scores.quantile(0.75)),
                    'max': float(scores.max()),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
        
        # 3. Correlations (for correlation matrix)
        correlation_matrix = {}
        for i, (algo1, name1) in enumerate(zip(algorithms, algo_names)):
            if algo1 in matches_df.columns:
                correlation_matrix[name1] = {}
                for algo2, name2 in zip(algorithms, algo_names):
                    if algo2 in matches_df.columns:
                        corr = matches_df[algo1].corr(matches_df[algo2])
                        correlation_matrix[name1][name2] = float(corr)
        
        # 4. Success rate by dominant algorithm
        if all(algo in matches_df.columns for algo in algorithms):
            matches_df['Dominant_Algorithm'] = matches_df[algorithms].idxmax(axis=1)
            matches_df['Dominant_Algorithm'] = matches_df['Dominant_Algorithm'].str.replace('_Score', '')
            
            success_rates = {}
            for name in algo_names:
                dominant_matches = matches_df[matches_df['Dominant_Algorithm'] == name.replace(' ', '_')]
                total = len(dominant_matches)
                if 'Combined_Score' in matches_df.columns:
                    success = len(dominant_matches[dominant_matches['Combined_Score'] >= 0.8])
                    success_rates[name] = {
                        'rate': (success / total * 100) if total > 0 else 0,
                        'count': total,
                        'successes': success
                    }
                else:
                    success_rates[name] = {'rate': 0, 'count': total, 'successes': 0}
        else:
            success_rates = {}
        
        # 5. Algorithm performance tiers
        performance_tiers = {}
        for name, avg_score in avg_scores.items():
            if avg_score >= 0.7:
                tier = "Excellent"
                tier_color = "#27ae60"
            elif avg_score >= 0.5:
                tier = "Good"
                tier_color = "#f39c12"
            elif avg_score >= 0.3:
                tier = "Fair"
                tier_color = "#e67e22"
            else:
                tier = "Poor"
                tier_color = "#e74c3c"
            
            performance_tiers[name] = {
                'tier': tier,
                'color': tier_color,
                'score': avg_score
            }
        
        return {
            'avg_scores': avg_scores,
            'distributions': distributions,
            'correlations': correlation_matrix,
            'success_rates': success_rates,
            'performance_tiers': performance_tiers,
            'total_matches': len(matches_df)
        }
    
    def _generate_sophisticated_algorithm_html(self, stats: Dict, matches_df: pd.DataFrame) -> str:
        """Generate sophisticated HTML for algorithm contribution analysis like V1"""
        
        avg_scores = stats['avg_scores']
        distributions = stats['distributions']
        correlations = stats['correlations']
        success_rates = stats['success_rates']
        performance_tiers = stats['performance_tiers']
        
        # Generate pie chart data (using CSS conic-gradient)
        total = sum(avg_scores.values())
        pie_segments = []
        cumulative = 0
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
        
        for i, (name, score) in enumerate(avg_scores.items()):
            percentage = (score / total * 100) if total > 0 else 0
            pie_segments.append({
                'name': name,
                'percentage': percentage,
                'color': colors[i % len(colors)],
                'start': cumulative,
                'end': cumulative + percentage
            })
            cumulative += percentage
        
        # Create conic-gradient for pie chart
        gradient_stops = []
        cumulative = 0
        for segment in pie_segments:
            color = segment['color']
            start_pct = cumulative
            end_pct = cumulative + segment['percentage']
            gradient_stops.append(f"{color} {start_pct}% {end_pct}%")
            cumulative += segment['percentage']
        
        pie_gradient = f"conic-gradient({', '.join(gradient_stops)})"
        
        # Generate correlation matrix HTML
        correlation_matrix_html = ""
        algo_names = list(correlations.keys())
        for i, algo1 in enumerate(algo_names):
            correlation_matrix_html += "<tr>"
            for j, algo2 in enumerate(algo_names):
                if i == j:
                    correlation_matrix_html += f'<td class="corr-diagonal">1.00</td>'
                elif i < j:
                    corr_val = correlations[algo1].get(algo2, 0)
                    corr_class = "corr-high" if abs(corr_val) >= 0.7 else "corr-medium" if abs(corr_val) >= 0.4 else "corr-low"
                    correlation_matrix_html += f'<td class="{corr_class}">{corr_val:.2f}</td>'
                else:
                    correlation_matrix_html += '<td class="corr-empty"></td>'
            correlation_matrix_html += "</tr>"
        
        # Generate distribution box plots (simplified as bar charts)
        distribution_charts_html = ""
        for name, dist in distributions.items():
            color = performance_tiers[name]['color']
            distribution_charts_html += f"""
            <div class="distribution-chart">
                <h4>{name}</h4>
                <div class="box-plot">
                    <div class="box-plot-line" style="left: {dist['min']*100}%; width: {(dist['max']-dist['min'])*100}%;"></div>
                    <div class="box-plot-box" style="left: {dist['q25']*100}%; width: {(dist['q75']-dist['q25'])*100}%; border-color: {color};"></div>
                    <div class="box-plot-median" style="left: {dist['median']*100}%; border-color: {color};"></div>
                </div>
                <div class="dist-stats">
                    <span>Min: {dist['min']:.3f}</span>
                    <span>Med: {dist['median']:.3f}</span>
                    <span>Max: {dist['max']:.3f}</span>
                </div>
            </div>
            """
        
        # Generate success rate bars
        success_bars_html = ""
        for name, success_data in success_rates.items():
            color = performance_tiers[name]['color']
            rate = success_data['rate']
            count = success_data['count']
            successes = success_data['successes']
            
            success_bars_html += f"""
            <div class="success-bar">
                <div class="success-label">{name}</div>
                <div class="success-bar-container">
                    <div class="success-bar-fill" style="width: {rate}%; background-color: {color};"></div>
                </div>
                <div class="success-stats">
                    <span>{rate:.1f}%</span>
                    <span>({successes}/{count})</span>
                </div>
            </div>
            """
        
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
            padding: 8px;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
            padding: 15px;
            aspect-ratio: 16/9;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 10px;
            overflow: hidden;
        }}

        .header {{
            text-align: center;
            border-bottom: 2px solid #74b9ff;
            padding-bottom: 6px;
        }}

        .title {{
            font-size: 1.6em;
            font-weight: bold;
            color: #2d3436;
            margin-bottom: 4px;
        }}

        .subtitle {{
            font-size: 0.9em;
            color: #636e72;
        }}

        .analysis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            height: 100%;
        }}

        .analysis-panel {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid #dee2e6;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .panel-title {{
            font-size: 0.95em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 8px;
            text-align: center;
        }}
        
        /* Pie Chart Styles */
        .pie-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            flex: 1;
        }}

        .pie-chart {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: {pie_gradient};
            position: relative;
            flex-shrink: 0;
        }}

        .pie-legend {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-color {{
            width: 10px;
            height: 10px;
            border-radius: 2px;
            flex-shrink: 0;
        }}

        .legend-text {{
            font-size: 0.75em;
            font-weight: 500;
        }}

        .legend-value {{
            font-size: 0.7em;
            color: #666;
        }}
        
        /* Distribution Chart Styles */
        .distributions-container {{
            flex: 1;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            overflow: hidden;
        }}

        .distribution-chart {{
            background: white;
            padding: 6px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }}

        .distribution-chart h4 {{
            margin: 0 0 4px 0;
            text-align: center;
            color: #495057;
            font-size: 0.8em;
        }}

        .box-plot {{
            position: relative;
            height: 24px;
            margin: 8px 0;
            background: #f1f3f4;
            border-radius: 3px;
        }}

        .box-plot-line {{
            position: absolute;
            top: 11px;
            height: 2px;
            background: #95a5a6;
        }}

        .box-plot-box {{
            position: absolute;
            top: 4px;
            height: 16px;
            background: rgba(255,255,255,0.8);
            border: 2px solid;
            border-radius: 3px;
        }}

        .box-plot-median {{
            position: absolute;
            top: 2px;
            height: 20px;
            width: 2px;
            border-left: 2px solid;
        }}

        .dist-stats {{
            display: flex;
            justify-content: space-between;
            font-size: 0.65em;
            color: #666;
        }}
        
        /* Correlation Matrix Styles */
        .correlation-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: auto;
        }}

        .correlation-matrix {{
            border-collapse: collapse;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .correlation-matrix th {{
            background: #495057;
            color: white;
            padding: 6px 8px;
            font-size: 0.7em;
            text-align: center;
        }}

        .correlation-matrix td {{
            padding: 6px 8px;
            text-align: center;
            font-weight: bold;
            font-size: 0.7em;
            border: 1px solid #dee2e6;
        }}
        
        .corr-diagonal {{
            background: #e9ecef;
            color: #495057;
        }}
        
        .corr-high {{
            background: #d4edda;
            color: #155724;
        }}
        
        .corr-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .corr-low {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .corr-empty {{
            background: #f8f9fa;
        }}
        
        /* Success Rate Styles */
        .success-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 6px;
            overflow: hidden;
        }}

        .success-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: white;
            padding: 6px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }}

        .success-label {{
            font-weight: bold;
            min-width: 70px;
            color: #495057;
            font-size: 0.75em;
        }}

        .success-bar-container {{
            flex: 1;
            height: 14px;
            background: #e9ecef;
            border-radius: 7px;
            overflow: hidden;
        }}

        .success-bar-fill {{
            height: 100%;
            border-radius: 7px;
            transition: width 0.3s ease;
        }}

        .success-stats {{
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 45px;
        }}

        .success-stats span:first-child {{
            font-weight: bold;
            color: #495057;
            font-size: 0.75em;
        }}

        .success-stats span:last-child {{
            font-size: 0.65em;
            color: #6c757d;
        }}

        .summary-note {{
            text-align: center;
            color: #6c757d;
            font-size: 0.7em;
            margin-top: 4px;
            padding: 4px;
            background: rgba(116, 185, 255, 0.1);
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">Algorithm Contribution Analysis</div>
            <div class="subtitle">Comprehensive Performance Breakdown & Interactions</div>
        </div>
        
        <div class="analysis-grid">
            <!-- Top Left: Average Contribution Pie Chart -->
            <div class="analysis-panel">
                <div class="panel-title">Average Algorithm Contribution</div>
                <div class="pie-container">
                    <div class="pie-chart"></div>
                    <div class="pie-legend">
                        {self._generate_pie_legend_html(pie_segments)}
                    </div>
                </div>
                <div class="summary-note">
                    Shows relative contribution of each algorithm across all {stats['total_matches']} matches
                </div>
            </div>
            
            <!-- Top Right: Score Distributions -->
            <div class="analysis-panel">
                <div class="panel-title">Score Distribution by Algorithm</div>
                <div class="distributions-container">
                    {distribution_charts_html}
                </div>
                <div class="summary-note">
                    Box plots showing score ranges and quartiles for each algorithm
                </div>
            </div>
            
            <!-- Bottom Left: Algorithm Correlations -->
            <div class="analysis-panel">
                <div class="panel-title">Algorithm Score Correlations</div>
                <div class="correlation-container">
                    <table class="correlation-matrix">
                        <thead>
                            <tr>
                                <th></th>
                                {self._generate_correlation_headers_html(algo_names)}
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_correlation_rows_html(algo_names, correlations)}
                        </tbody>
                    </table>
                </div>
                <div class="summary-note">
                    Higher correlations indicate algorithms tend to agree on match quality
                </div>
            </div>
            
            <!-- Bottom Right: Success Rates When Dominant -->
            <div class="analysis-panel">
                <div class="panel-title">Success Rate When Algorithm Dominates</div>
                <div class="success-container">
                    {success_bars_html}
                </div>
                <div class="summary-note">
                    Success rate (Combined Score ≥ 0.8) when each algorithm has the highest individual score
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def _generate_pie_legend_html(self, pie_segments: List[Dict]) -> str:
        """Generate HTML for pie chart legend"""
        legend_html = ""
        for segment in pie_segments:
            legend_html += f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {segment['color']};"></div>
                <span class="legend-text">{segment['name']}</span>
                <span class="legend-value">({segment['percentage']:.1f}%)</span>
            </div>
            """
        return legend_html
    
    def _generate_correlation_headers_html(self, algo_names: List[str]) -> str:
        """Generate HTML for correlation matrix headers"""
        headers_html = ""
        for name in algo_names:
            headers_html += f"<th>{name}</th>"
        return headers_html
    
    def _generate_correlation_rows_html(self, algo_names: List[str], correlations: Dict) -> str:
        """Generate HTML for correlation matrix rows"""
        rows_html = ""
        for i, algo1 in enumerate(algo_names):
            rows_html += f"<tr><th>{algo1}</th>"
            for j, algo2 in enumerate(algo_names):
                if i == j:
                    rows_html += '<td class="corr-diagonal">1.00</td>'
                elif i < j:
                    corr_val = correlations[algo1].get(algo2, 0)
                    corr_class = "corr-high" if abs(corr_val) >= 0.7 else "corr-medium" if abs(corr_val) >= 0.4 else "corr-low"
                    rows_html += f'<td class="{corr_class}">{corr_val:.2f}</td>'
                else:
                    rows_html += '<td class="corr-empty"></td>'
            rows_html += "</tr>"
        return rows_html
    
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

        # Extract requirement ID from the journey path for layer data export
        req_id = Path(journey_path).stem.replace('_technical_journey', '')

        # Save layer data exports
        print("\n📄 Saving layer data exports...")
        visualizer.save_layer_data(req_id)

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