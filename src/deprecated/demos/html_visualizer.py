"""
HTML Journey Visualizer
V4's clean HTML rendering + V1's comprehensive data collection

LOCATION: Save as src/visualization/html_visualizer.py
"""

import json
import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer


class HTMLJourneyVisualizer:
    """
    HTML-based visualizer with V1's complete data collection.
    Clean, scalable, and detailed.
    """
    
    def __init__(self):
        """Initialize with all V1 components."""
        try:
            self.repo_manager = RepositoryStructureManager()
            self.file_handler = SafeFileHandler(self.repo_manager)
            self.output_dir = self.repo_manager.structure['visuals_output']
        except Exception as e:
            print(f"⚠️ Repository manager unavailable: {e}")
            raise ValueError("Repository manager required")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (from V1)
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.domain_resources = DomainResources()
        self.req_analyzer = EnhancedRequirementAnalyzer()
        
        # Data storage (from V1)
        self.layer_data = {}
        self.text_output = []
        
        # Colors (from V1)
        self.colors = {
            'semantic': '#3498db',
            'bm25': '#e74c3c',
            'domain': '#27ae60',
            'query_expansion': '#f39c12',
            'good': '#27ae60',
            'warning': '#f39c12',
            'bad': '#e74c3c'
        }
    
    # ============= V1's DATA COLLECTION (EXACT STRUCTURE) =============
    
    def add_text_output(self, layer_name: str, content: Dict):
        """V1's method - unchanged."""
        self.text_output.append({
            'layer': layer_name,
            'timestamp': datetime.now().isoformat(),
            'content': content
        })
        self.layer_data[layer_name] = content
    
    def save_text_outputs(self, req_id: str):
        """V1's method - unchanged."""
        json_path = self.output_dir / f"{req_id}_layer_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_data, f, indent=2)
        
        csv_path = self.output_dir / f"{req_id}_layer_summary.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Component', 'Value', 'Details'])
            
            for layer_name, content in self.layer_data.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                writer.writerow([layer_name, f"{key}.{sub_key}", 
                                               str(sub_value)[:100], str(sub_value)])
                        else:
                            writer.writerow([layer_name, key, str(value)[:100], str(value)])
                else:
                    writer.writerow([layer_name, 'data', str(content)[:100], str(content)])
        
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
                            for item in value[:5]:
                                f.write(f"    - {item}\n")
                        else:
                            wrapped = textwrap.wrap(str(value), width=70)
                            for line in wrapped[:3]:
                                f.write(f"    {line}\n")
                else:
                    f.write(f"  {content}\n")
                f.write("\n")
        
        print(f"📄 Saved text outputs:")
        print(f"   - JSON: {json_path}")
        print(f"   - CSV: {csv_path}")
        print(f"   - TXT: {txt_path}")
    
    def load_real_data(self):
        """V1's method - unchanged."""
        if self.repo_manager:
            matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
            expl_path = self.repo_manager.structure['matching_results'] / "match_explanations.json"
            req_path = self.repo_manager.structure['data_raw'] / "requirements.csv"
        else:
            matches_path = Path("outputs/matching_results/aerospace_matches.csv")
            expl_path = Path("outputs/matching_results/match_explanations.json")
            req_path = Path("requirements.csv")
        
        if matches_path.exists():
            matches_df = pd.read_csv(matches_path)
        else:
            raise FileNotFoundError(f"Matches file not found: {matches_path}")
        
        explanations = {}
        if expl_path.exists():
            with open(expl_path, 'r') as f:
                explanations = json.load(f)
        
        requirements_df = None
        if req_path.exists():
            requirements_df = pd.read_csv(req_path)
        
        return matches_df, explanations, requirements_df
    
    def collect_all_layer_data(self, match, req_text: str, activity_name: str, explanation: Dict):
        """
        Collect ALL layer data matching V1's EXACT structure from the JSON you showed.
        """
        req_id = match['Requirement_ID']
        
        # Clear previous
        self.layer_data = {}
        self.text_output = []
        
        # Layer 1: Raw Inputs
        self.add_text_output("Layer 1: Raw Inputs", {
            'requirement_id': req_id,
            'requirement_text': req_text,
            'activity_name': activity_name,
            'source_files': ['requirements.csv', 'activities.csv']
        })
        
        # Layer 2: Preprocessing (EXACT V1 structure)
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        expanded_act = self.matcher._expand_aerospace_abbreviations(activity_name)
        req_doc = self.matcher.nlp(expanded_req)
        act_doc = self.matcher.nlp(expanded_act)
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        act_terms = self.matcher._preprocess_text_aerospace(activity_name)
        
        tokens = [token.text for token in req_doc]
        pos_counts = {}
        for token in req_doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        aerospace_terms = [t for t in set(req_terms + act_terms) if t in self.matcher.all_aerospace_terms]
        
        # Check for abbreviations
        abbrevs_expanded = []
        if expanded_req != req_text or expanded_act != activity_name:
            for abbr, full in self.domain_resources.abbreviations.items():
                if abbr.lower() in req_text.lower() or abbr.lower() in activity_name.lower():
                    abbrevs_expanded.append(f"{abbr} -> {full}")
        
        self.add_text_output("Layer 2: Preprocessing", {
            'abbreviations_expanded': abbrevs_expanded,
            'tokenization': {
                'tokens': tokens,
                'pos_counts': pos_counts,
                'total_tokens': len(tokens),
                'unique_tokens': len(set(tokens))
            },
            'extracted_terms': {
                'requirement_terms': req_terms,
                'activity_terms': act_terms,
                'aerospace_terms': aerospace_terms
            }
        })
        
        # Layer 3: Algorithm Analysis (EXACT V1 structure)
        scores_obj = explanation.get('scores', {})
        explanations_data = explanation.get('explanations', {})
        
        # Get detailed explanations
        semantic_expl = explanations_data.get('semantic', 'spaCy semantic')
        bm25_expl_data = explanations_data.get('bm25', {})
        domain_expl_data = explanations_data.get('domain', {})
        qe_expl_data = explanations_data.get('query_expansion', {})
        
        layer3 = {
            'semantic': {
                'score': scores_obj.get('semantic', match.get('Semantic_Score', 0)),
                'explanation': semantic_expl if isinstance(semantic_expl, str) else f"spaCy semantic: {scores_obj.get('semantic', 0):.3f}"
            },
            'bm25': {
                'score': scores_obj.get('bm25', match.get('BM25_Score', 0)),
                'explanation': bm25_expl_data.get('explanation', 'BM25 lexical matching') if isinstance(bm25_expl_data, dict) else str(bm25_expl_data),
                'shared_terms': bm25_expl_data.get('shared_terms', []) if isinstance(bm25_expl_data, dict) else []
            },
            'domain': {
                'score': scores_obj.get('domain', match.get('Domain_Score', 0)),
                'aerospace_terms': domain_expl_data.get('aerospace_terms', []) if isinstance(domain_expl_data, dict) else [],
                'indicators': domain_expl_data.get('indicators', {}) if isinstance(domain_expl_data, dict) else {},
                'learned_relationships': domain_expl_data.get('learned_relationships', {}) if isinstance(domain_expl_data, dict) else {},
                'bonus': domain_expl_data.get('multi_evidence_bonus', 0) if isinstance(domain_expl_data, dict) else 0
            },
            'query_expansion': {
                'score': scores_obj.get('query_expansion', match.get('Query_Expansion_Score', 0)),
                'requirement_terms': qe_expl_data.get('requirement_terms', []) if isinstance(qe_expl_data, dict) else [],
                'matched_terms': qe_expl_data.get('matched_terms', []) if isinstance(qe_expl_data, dict) else [],
                'expanded_activity_terms': qe_expl_data.get('expanded_activity_terms', []) if isinstance(qe_expl_data, dict) else [],
                'match_percentage': qe_expl_data.get('match_percentage', 0) if isinstance(qe_expl_data, dict) else 0
            }
        }
        self.add_text_output("Layer 3: Algorithm Analysis", layer3)
        
        # Layer 4: Score Combination (EXACT V1 structure)
        combined = match['Combined_Score']
        sem_score = layer3['semantic']['score']
        bm25_score = layer3['bm25']['score']
        dom_score = layer3['domain']['score']
        qe_score = layer3['query_expansion']['score']
        
        if combined >= 0.8:
            classification = "High Match"
        elif combined >= 0.35:
            classification = "Medium Match"
        else:
            classification = "Orphan"
        
        self.add_text_output("Layer 4: Score Combination", {
            'formula': f"(({sem_score:.3f} × 1.0) + ({bm25_score:.3f} × 1.0) + ({dom_score:.3f} × 1.0) + ({qe_score:.3f} × 1.0)) / 4.0",
            'combined_score': combined,
            'classification': classification,
            'individual_scores': {
                'semantic': sem_score,
                'bm25': bm25_score,
                'domain': dom_score,
                'query_expansion': qe_score
            },
            'contributions': {
                'semantic': f"{sem_score * 0.25:.3f}",
                'bm25': f"{bm25_score * 0.25:.3f}",
                'domain': f"{dom_score * 0.25:.3f}",
                'query_expansion': f"{qe_score * 0.25:.3f}"
            }
        })
        
        # Layer 5: INCOSE Analysis (EXACT V1 structure)
        incose_analysis = self.req_analyzer.incose_analyzer.analyze_incose_compliance(req_text)
        pattern_key = incose_analysis.best_pattern
        pattern_def = self.req_analyzer.incose_analyzer.patterns.get(pattern_key, {})
        
        self.add_text_output("Layer 5: INCOSE Analysis", {
            'best_pattern': pattern_def.get('name', pattern_key),
            'pattern_key': pattern_key,
            'pattern_template': pattern_def.get('template', ''),
            'pattern_required_components': list(pattern_def.get('required', [])),
            'pattern_optional_components': list(pattern_def.get('optional', [])),
            'compliance_score': incose_analysis.compliance_score,
            'components_found': incose_analysis.components_found,
            'required_present': f"{sum(1 for c in pattern_def.get('required', []) if incose_analysis.components_found.get(c))}/{len(pattern_def.get('required', []))}",
            'optional_present': f"{sum(1 for c in pattern_def.get('optional', []) if incose_analysis.components_found.get(c))}/{len(pattern_def.get('optional', []))}",
            'missing_required': [c for c in pattern_def.get('required', []) if not incose_analysis.components_found.get(c)],
            'missing_optional': [c for c in pattern_def.get('optional', []) if not incose_analysis.components_found.get(c)],
            'suggestions': getattr(incose_analysis, 'suggestions', []),
            'template_recommendation': getattr(incose_analysis, 'template_recommendation', '')
        })
        
        # Layer 6: Quality Analysis (EXACT V1 structure)
        issues, metrics, incose, semantic = self.req_analyzer.analyze_requirement(req_text)
        grade = self.req_analyzer._get_grade(metrics.quality_score)
        
        severity_breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            severity_breakdown[issue.get('severity', 'low')] += 1
        
        self.add_text_output("Layer 6: Quality Analysis", {
            'overall_score': metrics.quality_score,
            'grade': grade,
            'clarity': metrics.clarity_score,
            'completeness': metrics.completeness_score,
            'verifiability': metrics.verifiability_score,
            'atomicity': metrics.atomicity_score,
            'consistency': metrics.consistency_score,
            'semantic_quality': getattr(metrics, 'semantic_quality_score', 100),
            'entities_found': {
                'actors': getattr(semantic, 'actors', []),
                'actions': getattr(semantic, 'actions', []),
                'objects': getattr(semantic, 'objects', []),
                'conditions': getattr(semantic, 'conditions', []),
                'standards': getattr(semantic, 'standards', [])
            },
            'ambiguities': [i for i in issues if 'ambiguous' in i.get('description', '').lower()],
            'tone_issues': [i for i in issues if 'tone' in i.get('description', '').lower()],
            'semantic_suggestions': getattr(semantic, 'suggestions', []),
            'total_issues': len(issues),
            'severity_breakdown': severity_breakdown
        })
        
        # Layer 7: Final Decision (EXACT V1 structure)
        if combined >= 0.8:
            decision = "ACCEPT MATCH"
            action = "High confidence - use this trace"
        elif combined >= 0.35:
            decision = "REVIEW NEEDED"
            action = "Moderate confidence - engineer review required"
        else:
            decision = "ORPHAN"
            action = "No suitable match - write bridge requirement"
        
        metrics_text = f"Match Score: {combined:.3f} | Quality: {grade} | Issues: {len(issues)}"
        
        self.add_text_output("Layer 7: Final Decision", {
            'decision': decision,
            'action': action,
            'metrics': metrics_text
        })
    
    # ============= HTML GENERATION WITH V1's DATA =============
    
    def create_html_journey(self, requirement_id: str = None, label: str = None) -> str:
        """
        Create HTML visualization with ALL of V1's data.
        Clean, readable, professional.
        """
        # Load data
        matches_df, explanations, requirements_df = self.load_real_data()
        
        # Select requirement
        if requirement_id:
            match = matches_df[matches_df['Requirement_ID'] == requirement_id]
            if match.empty:
                print(f"⚠️ {requirement_id} not found, selecting best match")
                match = matches_df.nlargest(1, 'Combined_Score')
        else:
            match = matches_df.nlargest(1, 'Combined_Score')
        
        if match.empty:
            raise ValueError("No matches found")
        
        match = match.iloc[0]
        req_id = match['Requirement_ID']
        req_text = match['Requirement_Text']
        activity_name = match['Activity_Name']
        
        print(f"📊 Visualizing journey for: {req_id}")
        print(f"   Activity: {activity_name}")
        print(f"   Score: {match['Combined_Score']:.3f}")
        
        # Get explanation
        explanation = explanations.get((req_id, activity_name), {})
        
        # Collect ALL layer data using V1's structure
        self.collect_all_layer_data(match, req_text, activity_name, explanation)
        
        # Generate HTML
        html = self._generate_html(req_id, label)
        
        # Save HTML
        safe_req_id = req_id.replace('/', '_')
        output_path = self.output_dir / f"{safe_req_id}_journey.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Save text outputs
        self.save_text_outputs(req_id)
        
        print(f"✅ Created HTML journey: {output_path}")
        return str(output_path)
    
    def _generate_html(self, req_id: str, label: str = None) -> str:
        """Generate complete HTML with V1's data."""
        
        # Get all layers
        layer1 = self.layer_data.get("Layer 1: Raw Inputs", {})
        layer2 = self.layer_data.get("Layer 2: Preprocessing", {})
        layer3 = self.layer_data.get("Layer 3: Algorithm Analysis", {})
        layer4 = self.layer_data.get("Layer 4: Score Combination", {})
        layer5 = self.layer_data.get("Layer 5: INCOSE Analysis", {})
        layer6 = self.layer_data.get("Layer 6: Quality Analysis", {})
        layer7 = self.layer_data.get("Layer 7: Final Decision", {})
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Technical Journey: {req_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f7fa;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        .label {{
            background: #e74c3c;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            display: inline-block;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        .layer {{
            margin-bottom: 30px;
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            overflow: hidden;
        }}
        .layer-title {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 20px;
            font-weight: bold;
        }}
        .layer-content {{
            padding: 20px;
        }}
        .subsection {{
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        .subsection-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        .card-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .card-value {{
            color: #212529;
            font-size: 16px;
        }}
        .score {{
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
        }}
        .score-high {{ background: #d4edda; color: #155724; }}
        .score-medium {{ background: #fff3cd; color: #856404; }}
        .score-low {{ background: #f8d7da; color: #721c24; }}
        .list {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .list li {{
            margin: 5px 0;
            line-height: 1.6;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-warning {{ background: #ffc107; color: #212529; }}
        .badge-danger {{ background: #dc3545; color: white; }}
        .badge-info {{ background: #17a2b8; color: white; }}
        .decision-box {{
            text-align: center;
            padding: 30px;
            border-radius: 8px;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .decision-accept {{ background: #d4edda; color: #155724; border: 3px solid #28a745; }}
        .decision-review {{ background: #fff3cd; color: #856404; border: 3px solid #ffc107; }}
        .decision-reject {{ background: #f8d7da; color: #721c24; border: 3px solid #dc3545; }}
        .code {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #6c757d;
            font-size: 14px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Technical Journey: {req_id}</h1>
        {f'<div class="label">{label}</div>' if label else ''}
        
        <!-- Layer 1: Raw Inputs -->
        <div class="layer">
            <div class="layer-title">Layer 1: Raw Inputs</div>
            <div class="layer-content">
                <div class="subsection">
                    <div class="subsection-title">Requirement ID</div>
                    <div>{layer1.get('requirement_id', 'N/A')}</div>
                </div>
                <div class="subsection">
                    <div class="subsection-title">Requirement Text</div>
                    <div>{layer1.get('requirement_text', 'N/A')}</div>
                </div>
                <div class="subsection">
                    <div class="subsection-title">Activity Name</div>
                    <div>{layer1.get('activity_name', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <!-- Layer 2: Preprocessing -->
        <div class="layer">
            <div class="layer-title">Layer 2: Preprocessing</div>
            <div class="layer-content">
                <div class="grid">
                    <div class="card">
                        <div class="card-title">Abbreviations Expanded</div>
                        <div class="card-value">
                            {len(layer2.get('abbreviations_expanded', []))} found
                            {'<ul class="list">' + ''.join(f'<li>{a}</li>' for a in layer2.get('abbreviations_expanded', [])[:5]) + '</ul>' if layer2.get('abbreviations_expanded') else '<div>None</div>'}
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-title">Tokenization</div>
                        <div class="card-value">
                            Total: {layer2.get('tokenization', {}).get('total_tokens', 0)}<br>
                            Unique: {layer2.get('tokenization', {}).get('unique_tokens', 0)}
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-title">Terms Extracted</div>
                        <div class="card-value">
                            Requirement: {len(layer2.get('extracted_terms', {}).get('requirement_terms', []))}<br>
                            Activity: {len(layer2.get('extracted_terms', {}).get('activity_terms', []))}<br>
                            Aerospace: {len(layer2.get('extracted_terms', {}).get('aerospace_terms', []))}
                        </div>
                    </div>
                </div>
                
                {self._render_term_lists(layer2.get('extracted_terms', {}))}
            </div>
        </div>
        
        <!-- Layer 3: Algorithm Analysis -->
        <div class="layer">
            <div class="layer-title">Layer 3: Algorithm Analysis (Parallel Execution)</div>
            <div class="layer-content">
                <div class="grid">
                    {self._render_algorithm_card('Semantic', layer3.get('semantic', {}), '#3498db')}
                    {self._render_algorithm_card('BM25', layer3.get('bm25', {}), '#e74c3c')}
                    {self._render_algorithm_card('Domain', layer3.get('domain', {}), '#27ae60')}
                    {self._render_algorithm_card('Query Expansion', layer3.get('query_expansion', {}), '#f39c12')}
                </div>
            </div>
        </div>
        
        <!-- Layer 4: Score Combination -->
        <div class="layer">
            <div class="layer-title">Layer 4: Score Combination</div>
            <div class="layer-content">
                <div class="subsection">
                    <div class="subsection-title">Formula</div>
                    <div class="code">{layer4.get('formula', 'N/A')}</div>
                </div>
                
                <div class="{self._get_score_class(layer4.get('combined_score', 0))}">
                    {layer4.get('combined_score', 0):.3f}
                </div>
                
                <div style="text-align: center; font-size: 20px; font-weight: bold; margin: 15px 0;">
                    Classification: {layer4.get('classification', 'N/A')}
                </div>
                
                <div class="grid">
                    {self._render_contribution_cards(layer4.get('individual_scores', {}), layer4.get('contributions', {}))}
                </div>
            </div>
        </div>
        
        <!-- Layer 5: INCOSE Analysis -->
        <div class="layer">
            <div class="layer-title">Layer 5: INCOSE Pattern Analysis</div>
            <div class="layer-content">
                <div class="subsection">
                    <div class="subsection-title">Best Pattern Match</div>
                    <div style="font-size: 18px; font-weight: bold; color: #2c3e50;">
                        {layer5.get('best_pattern', 'N/A')} 
                        <span class="badge badge-info">{layer5.get('compliance_score', 0):.0f}% Compliance</span>
                    </div>
                </div>
                
                <div class="subsection">
                    <div class="subsection-title">Pattern Template</div>
                    <div class="code" style="font-size: 12px;">{layer5.get('pattern_template', 'N/A')}</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="card-title">Required Components</div>
                        <div class="card-value">
                            {layer5.get('required_present', '0/0')}
                            {self._render_components_list(layer5.get('components_found', {}), layer5.get('pattern_required_components', []))}
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-title">Optional Components</div>
                        <div class="card-value">
                            {layer5.get('optional_present', '0/0')}
                            {self._render_components_list(layer5.get('components_found', {}), layer5.get('pattern_optional_components', []))}
                        </div>
                    </div>
                </div>
                
                {self._render_missing_components(layer5)}
            </div>
        </div>
        
        <!-- Layer 6: Quality Analysis -->
        <div class="layer">
            <div class="layer-title">Layer 6: Quality Analysis</div>
            <div class="layer-content">
                <div class="grid">
                    <div class="card">
                        <div class="card-title">Overall Grade</div>
                        <div class="{self._get_grade_class(layer6.get('grade', 'N/A'))}">
                            {layer6.get('grade', 'N/A')}
                        </div>
                        <div style="text-align: center; margin-top: 10px;">
                            {layer6.get('overall_score', 0):.0f}/100
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-title">Total Issues</div>
                        <div style="text-align: center; font-size: 32px; font-weight: bold; color: {'#28a745' if layer6.get('total_issues', 0) == 0 else '#ffc107' if layer6.get('total_issues', 0) <= 3 else '#dc3545'};">
                            {layer6.get('total_issues', 0)}
                        </div>
                    </div>
                </div>
                
                <div class="subsection">
                    <div class="subsection-title">Quality Dimensions</div>
                    <div class="grid">
                        <div class="card">
                            <div class="card-title">Clarity</div>
                            <div class="card-value">{layer6.get('clarity', 0):.0f}/100</div>
                        </div>
                        <div class="card">
                            <div class="card-title">Completeness</div>
                            <div class="card-value">{layer6.get('completeness', 0):.0f}/100</div>
                        </div>
                        <div class="card">
                            <div class="card-title">Verifiability</div>
                            <div class="card-value">{layer6.get('verifiability', 0):.0f}/100</div>
                        </div>
                        <div class="card">
                            <div class="card-title">Atomicity</div>
                            <div class="card-value">{layer6.get('atomicity', 0):.0f}/100</div>
                        </div>
                        <div class="card">
                            <div class="card-title">Consistency</div>
                            <div class="card-value">{layer6.get('consistency', 0):.0f}/100</div>
                        </div>
                        <div class="card">
                            <div class="card-title">Semantic Quality</div>
                            <div class="card-value">{layer6.get('semantic_quality', 0):.0f}/100</div>
                        </div>
                    </div>
                </div>
                
                {self._render_entities(layer6.get('entities_found', {}))}
                {self._render_severity_breakdown(layer6.get('severity_breakdown', {}))}
            </div>
        </div>
        
        <!-- Layer 7: Final Decision -->
        <div class="layer">
            <div class="layer-title">Layer 7: Final Decision</div>
            <div class="layer-content">
                <div class="{self._get_decision_class(layer7.get('decision', ''))}">
                    {layer7.get('decision', 'N/A')}
                </div>
                <div style="text-align: center; font-size: 18px; margin: 20px 0;">
                    {layer7.get('action', 'N/A')}
                </div>
                <div style="text-align: center; font-size: 16px; color: #6c757d;">
                    {layer7.get('metrics', 'N/A')}
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    
    def _render_pos_tags(self, pos_counts: Dict) -> str:
        """Render top POS tags."""
        if not pos_counts:
            return 'N/A'
        
        sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        html = ''
        for pos, count in sorted_pos:
            html += f'<div style="margin: 2px 0;">{pos}: {count}</div>'
        return html
    
    def _render_term_lists(self, extracted_terms: Dict) -> str:
        """Render extracted terms lists."""
        req_terms = extracted_terms.get('requirement_terms', [])[:10]
        act_terms = extracted_terms.get('activity_terms', [])[:10]
        aero_terms = extracted_terms.get('aerospace_terms', [])
        
        html = '<div class="subsection">'
        html += '<div class="subsection-title">Extracted Terms (Top 10)</div>'
        html += '<div class="grid">'
        
        if req_terms:
            html += '<div class="card">'
            html += '<div class="card-title">Requirement Terms</div>'
            html += '<ul class="list">'
            for term in req_terms:
                html += f'<li>{term}</li>'
            html += '</ul></div>'
        
        if act_terms:
            html += '<div class="card">'
            html += '<div class="card-title">Activity Terms</div>'
            html += '<ul class="list">'
            for term in act_terms:
                html += f'<li>{term}</li>'
            html += '</ul></div>'
        
        if aero_terms:
            html += '<div class="card">'
            html += '<div class="card-title">Aerospace Terms</div>'
            html += '<ul class="list">'
            for term in aero_terms:
                html += f'<li><span class="badge badge-info">{term}</span></li>'
            html += '</ul></div>'
        
        html += '</div></div>'
        return html
    
    
    def _render_algorithm_card_detailed(self, name: str, algo_data: Dict, color: str) -> str:
        """Render detailed algorithm card with ALL V1 information."""
        score = algo_data.get('score', 0)
        score_class = 'score-high' if score >= 0.7 else 'score-medium' if score >= 0.4 else 'score-low'
        
        html = f'<div class="card" style="border-left: 4px solid {color}; min-height: 300px;">'
        html += f'<div class="card-title" style="color: {color}; font-size: 14px; font-weight: bold;">{name}</div>'
        html += f'<div class="{score_class}" style="font-size: 24px; padding: 5px; margin: 5px 0;">{score:.3f}</div>'
        
        # Explanation
        explanation = algo_data.get('explanation', '')
        if isinstance(explanation, str) and explanation:
            html += f'<div class="detail-text" style="color: #6c757d; margin: 5px 0; font-style: italic;">{explanation}</div>'
        
        # Algorithm-specific details
        if name == 'BM25':
            shared = algo_data.get('shared_terms', [])
            if shared:
                html += '<div class="detail-text" style="margin-top: 8px;">'
                html += '<strong>Shared terms:</strong><br>'
                for term in shared[:6]:
                    html += f'<span class="badge badge-info">{term}</span> '
                html += '</div>'
        
        elif name == 'Domain':
            aero = algo_data.get('aerospace_terms', [])
            if aero:
                html += '<div class="detail-text" style="margin-top: 8px;">'
                html += '<strong>Aerospace terms:</strong><br>'
                for term in aero[:6]:
                    html += f'<span class="badge badge-success">{term}</span> '
                html += '</div>'
            
            # Key indicators
            indicators = algo_data.get('indicators', {})
            if indicators:
                html += '<div class="detail-text" style="margin-top: 8px;">'
                html += '<strong>Key indicators:</strong><br>'
                for term, data in list(indicators.items())[:3]:
                    weight = data.get('weight', 0)
                    html += f'<div style="font-size: 10px;">• {term}: {weight:.2f}</div>'
                html += '</div>'
            
            # Learned relationships
            learned = algo_data.get('learned_relationships', {})
            if learned:
                html += '<div class="detail-text" style="margin-top: 8px;">'
                html += '<strong>Learned patterns:</strong><br>'
                for req_term, acts in list(learned.items())[:2]:
                    acts_str = ', '.join(acts[:2])
                    html += f'<div style="font-size: 10px;">• {req_term} → {acts_str}</div>'
                html += '</div>'
            
            bonus = algo_data.get('bonus', 0)
            if bonus > 0:
                html += f'<div style="font-size: 10px; margin-top: 5px; color: #28a745; font-weight: bold;">Multi-evidence bonus: +{bonus:.2f}</div>'
        
        elif name == 'Query Expansion':
            match_pct = algo_data.get('match_percentage', 0)
            matched = algo_data.get('matched_terms', [])
            req_terms_count = len(algo_data.get('requirement_terms', []))
            expanded_count = len(algo_data.get('expanded_activity_terms', []))
            
            html += f'<div class="detail-text" style="margin-top: 8px;">'
            html += f'<strong>Match rate:</strong> {match_pct:.0f}%<br>'
            html += f'<strong>Req terms:</strong> {req_terms_count}<br>'
            html += f'<strong>Expanded:</strong> {expanded_count}<br>'
            if matched:
                html += '<strong>Matched:</strong><br>'
                for term in matched[:5]:
                    html += f'<span class="badge badge-warning">{term}</span> '
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_algorithm_card(self, name: str, algo_data: Dict, color: str) -> str:
        """Render individual algorithm card."""
        score = algo_data.get('score', 0)
        score_class = 'score-high' if score >= 0.7 else 'score-medium' if score >= 0.4 else 'score-low'
        
        html = f'<div class="card" style="border-left: 4px solid {color};">'
        html += f'<div class="card-title" style="color: {color}; font-size: 18px;">{name}</div>'
        html += f'<div class="{score_class}" style="font-size: 28px;">{score:.3f}</div>'
        
        # Explanation
        explanation = algo_data.get('explanation', '')
        if isinstance(explanation, str) and explanation:
            html += f'<div style="font-size: 13px; color: #6c757d; margin: 10px 0;">{explanation}</div>'
        
        # Special details per algorithm
        if name == 'BM25':
            shared = algo_data.get('shared_terms', [])
            if shared:
                html += '<div style="font-size: 12px; margin-top: 10px;">'
                html += '<strong>Shared terms:</strong> '
                html += ', '.join([f'<span class="badge badge-info">{t}</span>' for t in shared[:5]])
                html += '</div>'
        
        elif name == 'Domain':
            aero = algo_data.get('aerospace_terms', [])
            if aero:
                html += '<div style="font-size: 12px; margin-top: 10px;">'
                html += '<strong>Aerospace terms:</strong><br>'
                html += ', '.join([f'<span class="badge badge-success">{t}</span>' for t in aero[:5]])
                html += '</div>'
            
            bonus = algo_data.get('bonus', 0)
            if bonus > 0:
                html += f'<div style="font-size: 12px; margin-top: 5px; color: #28a745;">Multi-evidence bonus: +{bonus:.2f}</div>'
        
        elif name == 'Query Expansion':
            match_pct = algo_data.get('match_percentage', 0)
            matched = algo_data.get('matched_terms', [])
            html += f'<div style="font-size: 12px; margin-top: 10px;">'
            html += f'<strong>Match rate:</strong> {match_pct:.0f}%<br>'
            if matched:
                html += '<strong>Matched:</strong> '
                html += ', '.join([f'<span class="badge badge-warning">{t}</span>' for t in matched[:5]])
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_contribution_cards(self, scores: Dict, contributions: Dict) -> str:
        """Render contribution breakdown cards."""
        html = ''
        colors = {
            'semantic': '#3498db',
            'bm25': '#e74c3c',
            'domain': '#27ae60',
            'query_expansion': '#f39c12'
        }
        names = {
            'semantic': 'Semantic',
            'bm25': 'BM25',
            'domain': 'Domain',
            'query_expansion': 'Query Expansion'
        }
        
        for key in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = scores.get(key, 0)
            contrib = contributions.get(key, '0.000')
            html += f'<div class="card" style="border-left: 4px solid {colors[key]};">'
            html += f'<div class="card-title">{names[key]}</div>'
            html += f'<div class="card-value">'
            html += f'Score: {score:.3f}<br>'
            html += f'Weight: 0.25 (25%)<br>'
            html += f'<strong>Contribution: {contrib}</strong>'
            html += '</div></div>'
        
        return html
    
    def _render_components_list(self, components: Dict, component_list: List[str]) -> str:
        """Render INCOSE components list."""
        if not component_list:
            return ''
        
        html = '<ul class="list" style="font-size: 13px; margin-top: 8px;">'
        for comp in component_list:
            value = components.get(comp)
            if value:
                html += f'<li><span class="badge badge-success">{comp}</span>: {value}</li>'
            else:
                html += f'<li><span class="badge badge-warning">{comp}</span>: Missing</li>'
        html += '</ul>'
        return html
    
    def _render_missing_components(self, layer5: Dict) -> str:
        """Render missing components section."""
        missing_req = layer5.get('missing_required', [])
        missing_opt = layer5.get('missing_optional', [])
        
        if not missing_req and not missing_opt:
            return ''
        
        html = '<div class="subsection">'
        html += '<div class="subsection-title">Missing Components</div>'
        
        if missing_req:
            html += '<div style="color: #dc3545; margin: 10px 0;">'
            html += '<strong>Required:</strong> '
            html += ', '.join([f'<span class="badge badge-danger">{c}</span>' for c in missing_req])
            html += '</div>'
        
        if missing_opt:
            html += '<div style="color: #ffc107; margin: 10px 0;">'
            html += '<strong>Optional:</strong> '
            html += ', '.join([f'<span class="badge badge-warning">{c}</span>' for c in missing_opt])
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_entities(self, entities: Dict) -> str:
        """Render semantic entities."""
        if not any(entities.values()):
            return ''
        
        html = '<div class="subsection">'
        html += '<div class="subsection-title">Semantic Entities Found</div>'
        html += '<div class="grid">'
        
        for entity_type in ['actors', 'actions', 'objects', 'conditions', 'standards']:
            items = entities.get(entity_type, [])
            if items:
                html += '<div class="card">'
                html += f'<div class="card-title">{entity_type.title()}</div>'
                html += '<div class="card-value">'
                html += ', '.join([f'<span class="badge badge-info">{item}</span>' for item in items])
                html += '</div></div>'
        
        html += '</div></div>'
        return html
    
    def _render_severity_breakdown(self, severity: Dict) -> str:
        """Render issue severity breakdown."""
        if not any(severity.values()):
            return ''
        
        html = '<div class="subsection">'
        html += '<div class="subsection-title">Issue Severity Breakdown</div>'
        html += '<div class="grid" style="grid-template-columns: repeat(4, 1fr);">'
        
        severity_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#6c757d'
        }
        
        for sev in ['critical', 'high', 'medium', 'low']:
            count = severity.get(sev, 0)
            html += '<div class="card">'
            html += f'<div class="card-title" style="color: {severity_colors[sev]};">{sev.title()}</div>'
            html += f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: {severity_colors[sev]};">{count}</div>'
            html += '</div>'
        
        html += '</div></div>'
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score."""
        if score >= 0.8:
            return 'score score-high'
        elif score >= 0.35:
            return 'score score-medium'
        else:
            return 'score score-low'
    
    def _get_grade_class(self, grade: str) -> str:
        """Get CSS class for quality grade."""
        if grade in ['EXCELLENT', 'GOOD']:
            return 'score score-high'
        elif grade == 'FAIR':
            return 'score score-medium'
        else:
            return 'score score-low'
    
    def _get_decision_class(self, decision: str) -> str:
        """Get CSS class for final decision."""
        if 'ACCEPT' in decision:
            return 'decision-box decision-accept'
        elif 'REVIEW' in decision or 'MODERATE' in decision:
            return 'decision-box decision-review'
        else:
            return 'decision-box decision-reject'
    
    # ============= V1's MATPLOTLIB CONTRIBUTION CHART (UNCHANGED) =============
    
    def create_algorithm_contribution_chart(self) -> str:
        """V1's contribution chart - UNCHANGED."""
        matches_df, _, _ = self.load_real_data()
        
        algorithms = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        avg_scores = {algo: matches_df[algo].mean() for algo in algorithms}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Contribution Analysis Across All Matches', 
                     fontsize=20, fontweight='bold')
        
        # 1. Pie chart
        ax1 = axes[0, 0]
        colors_list = [self.colors[a] for a in ['semantic', 'bm25', 'domain', 'query_expansion']]
        labels = ['Semantic', 'BM25', 'Domain', 'Query Exp']
        values = list(avg_scores.values())
        
        ax1.pie(values, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Average Algorithm Contribution', fontsize=16, fontweight='bold')
        
        # 2. Box plots
        ax2 = axes[0, 1]
        ax2.boxplot([matches_df[algo] for algo in algorithms], labels=labels)
        ax2.set_title('Score Distribution by Algorithm', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        ax3 = axes[1, 0]
        correlation_data = matches_df[algorithms].corr()
        im = ax3.imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_yticklabels(labels)
        ax3.set_title('Algorithm Score Correlations', fontsize=16, fontweight='bold')
        
        for i in range(len(algorithms)):
            for j in range(len(algorithms)):
                ax3.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=16)
        
        plt.colorbar(im, ax=ax3)
        
        # 4. Success rate
        ax4 = axes[1, 1]
        matches_df['Dominant_Algorithm'] = matches_df[algorithms].idxmax(axis=1)
        matches_df['Dominant_Algorithm'] = matches_df['Dominant_Algorithm'].str.replace('_Score', '')
        
        success_counts = []
        total_counts = []
        
        for algo in ['Semantic', 'BM25', 'Domain', 'Query_Expansion']:
            dominant_matches = matches_df[matches_df['Dominant_Algorithm'] == algo]
            total = len(dominant_matches)
            success = len(dominant_matches[dominant_matches['Combined_Score'] >= 0.8])
            total_counts.append(total)
            success_counts.append(success / total * 100 if total > 0 else 0)
        
        x_pos = np.arange(len(labels))
        bars = ax4.bar(x_pos, success_counts, color=colors_list, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Success Rate When Algorithm Dominates', fontsize=16, fontweight='bold')
        ax4.set_ylim(0, 100)
        
        for i, (bar, total) in enumerate(zip(bars, total_counts)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={total}', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "algorithm_contribution_analysis.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved algorithm contribution analysis: {output_path}")
        return str(output_path)


def main():
    """Create all visualizations."""
    print("🎨 HTML JOURNEY VISUALIZER")
    print("="*60)
    print("V4's HTML + V1's comprehensive data")
    print("="*60)
    
    viz = HTMLJourneyVisualizer()
    
    # 1. HTML journey with all V1's data
    print("\n1️⃣ Creating HTML technical journey...")
    journey_path = viz.create_html_journey(label="HIGH SCORING")
    
    # 2. Algorithm contribution chart (V1's matplotlib version)
    print("\n2️⃣ Creating algorithm contribution analysis...")
    contrib_path = viz.create_algorithm_contribution_chart()
    
    print("\n✅ All visualizations complete!")
    print(f"\nOutputs saved to: {viz.output_dir}")
    print(f"\nOpen the HTML file in your browser: {journey_path}")


if __name__ == "__main__":
    main()