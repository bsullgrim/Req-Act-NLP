"""
Technical Journey Visualization - Shows actual processing steps through matcher.py and reqGrading.py
Uses real requirements and activities from the pipeline output
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
import pandas as pd
import numpy as np
import json
import textwrap
import re
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import project modules following conventions
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer

# Set up matplotlib for high-quality output
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Arial',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none'
})

class TechnicalJourneyVisualizer:
    """
    Creates technical layer-by-layer visualization of requirement processing
    using actual data from matches and showing real method outputs.
    """
    
    def __init__(self):
        """Initialize with project conventions"""
        self.repo_manager = RepositoryStructureManager("outputs")
        self.file_handler = SafeFileHandler(self.repo_manager)
        
        # Create output directory for visualizations
        self.output_dir = self.repo_manager.structure['matching_results'] / "technical_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.quality_analyzer = EnhancedRequirementAnalyzer(repo_manager=self.repo_manager)
        self.domain = DomainResources()
        
        # Color scheme for algorithms
        self.colors = {
            'semantic': '#3498db',      # Blue
            'bm25': '#e74c3c',          # Red
            'domain': '#2ecc71',        # Green
            'query_expansion': '#f39c12', # Orange
            'good': '#27ae60',           # Dark green
            'warning': '#f39c12',        # Orange
            'bad': '#c0392b',           # Dark red
            'neutral': '#95a5a6',        # Gray
        }
        
        print(f"✅ Technical Journey Visualizer initialized")
    
    def load_real_data(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Load actual matching results, explanations, and requirements"""
        
        # Load matches
        matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
        if not matches_path.exists():
            raise FileNotFoundError(f"Run matcher.py first to generate matches")
        matches_df = pd.read_csv(matches_path)
        
        # Load explanations
        explanations_path = self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json"
        explanations = {}
        if explanations_path.exists():
            with open(explanations_path, 'r', encoding='utf-8') as f:
                explanations_list = json.load(f)
                for exp in explanations_list:
                    key = (exp.get('requirement_id'), exp.get('activity_name'))
                    explanations[key] = exp
        
        # Load original requirements for quality analysis
        req_path = self.repo_manager.structure['data_raw'] / "requirements.csv"
        requirements_df = None
        if req_path.exists():
            requirements_df = self.file_handler.safe_read_csv(str(req_path))
        
        return matches_df, explanations, requirements_df
    
    def create_processing_journey(self, requirement_id: str = None, label: str = None) -> str:
        """
        Create technical visualization showing each processing step.
        Optionally show a label (e.g., "HIGH scoring pair") in the figure title.
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
        
        match_row = match.iloc[0]
        req_id = match_row['Requirement_ID']
        req_text = match_row['Requirement_Text']
        activity_name = match_row['Activity_Name']
        
        print(f"📊 Visualizing journey for: {req_id}")
        print(f"   Requirement: {req_text[:80]}...")
        print(f"   Activity: {activity_name}")
        print(f"   Score: {match_row['Combined_Score']:.3f}")
        
        # Get explanation data - NEW FORMAT
        explanation = explanations.get((req_id, activity_name), {})
        
        # Create the visualization
        fig = plt.figure(figsize=(24, 20))
        
        # Title
        title = f'Technical Processing Journey: {req_id} (Score: {match_row["Combined_Score"]:.3f})'
        if label:
            title = f"{label} | {title}"
        fig.suptitle(title, fontsize=30, fontweight='bold', y=0.93)
        
        # Create grid for layers
        gs = fig.add_gridspec(8, 1, hspace=0.25)  # increased hspace for dense content
        
        # LAYER 1: Raw Inputs
        ax1 = fig.add_subplot(gs[0])
        self._layer1_raw_inputs(ax1, req_text, activity_name, req_id)
        
        # LAYER 2: Preprocessing Steps
        ax2 = fig.add_subplot(gs[1])
        preprocessing_data = self._layer2_preprocessing(ax2, req_text, activity_name)
        
        # LAYER 3: Four Algorithm Processing (use JSON explanations)
        ax3 = fig.add_subplot(gs[2:4])
        algorithm_scores = self._layer3_algorithms(ax3, req_text, activity_name, 
                                                preprocessing_data, explanation, explanation)
        
        # LAYER 4: Score Combination
        ax4 = fig.add_subplot(gs[4])
        combined_score = self._layer4_score_combination(ax4, explanation, algorithm_scores)
        
        # LAYER 5: INCOSE Pattern Analysis
        ax5 = fig.add_subplot(gs[5])
        incose_results = self._layer5_incose_analysis(ax5, req_text)
        
        # LAYER 6: Quality Dimensions
        ax6 = fig.add_subplot(gs[6])
        quality_results = self._layer6_quality_dimensions(ax6, req_text)
        
        # LAYER 7: Final Decision
        ax7 = fig.add_subplot(gs[7])
        self._layer7_final_decision(ax7, combined_score, quality_results, explanation)
        
        # Save figure
        output_path = self.output_dir / f"{req_id}_technical_journey.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved technical journey: {output_path}")
        return str(output_path)
    

    def _layer1_raw_inputs(self, ax, req_text: str, activity_name: str, req_id: str):
        """Layer 1: Show raw inputs with fully scaled boxes and wrapped text."""
        import textwrap

        ax.set_title("Layer 1: Raw Inputs (From CSV Files)", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')

        # --- Wrap texts ---
        max_chars = 60
        req_lines = textwrap.wrap(req_text, width=max_chars)
        act_lines = textwrap.wrap(activity_name, width=max_chars)

        # --- Heights and spacing ---
        line_height = 0.17
        padding_top_bottom = 0.1   # extra space above first line and below last line
        req_height = padding_top_bottom + line_height * len(req_lines)
        act_height = padding_top_bottom + line_height * len(act_lines)

        # --- Anchor boxes from top ---
        top_y = 1
        req_box_y = top_y - req_height - 0.2
        act_box_y = top_y - act_height - 0.2

        # --- Draw boxes ---
        req_box = FancyBboxPatch((0.02, req_box_y), 0.45, req_height,
                                boxstyle="round,pad=0.02",
                                facecolor='#e8f4fd', edgecolor='#3498db', linewidth=2)
        act_box = FancyBboxPatch((0.53, act_box_y), 0.45, act_height,
                                boxstyle="round,pad=0.02",
                                facecolor='#e8f6f3', edgecolor='#27ae60', linewidth=2)
        ax.add_patch(req_box)
        ax.add_patch(act_box)

        # --- Draw titles above boxes ---
        title_padding = 0.07
        ax.text(0.025, req_box_y + req_height + title_padding, f"Requirement {req_id}:", 
                fontweight='bold', fontsize=24, ha='left', va='bottom')
        ax.text(0.535, act_box_y + act_height + title_padding, "Activity:", 
                fontweight='bold', fontsize=24, ha='left', va='bottom')

        # --- Draw wrapped text inside boxes with padding ---
        y_cursor_req = req_box_y + req_height - 0.03  # start below top padding
        for line in req_lines:
            ax.text(0.025, y_cursor_req, line, fontsize=20, ha='left', va='top')
            y_cursor_req -= line_height

        y_cursor_act = act_box_y + act_height - 0.03
        for line in act_lines:
            ax.text(0.535, y_cursor_act, line, fontsize=20, ha='left', va='top')
            y_cursor_act -= line_height

        # --- Metadata ---
        ax.text(0.025, req_box_y-0.15, "Source: requirements.csv", fontsize=16, color='gray')
        ax.text(0.535, act_box_y-0.15, "Source: activities.csv", fontsize=16, color='gray')



    def _layer2_preprocessing(self, ax, req_text: str, activity_name: str) -> Dict:
        """Layer 2: Show actual preprocessing steps"""
        ax.set_title("Layer 2: Preprocessing (matcher.py methods)", fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')
        
        # Run actual preprocessing methods
        # Step 1: Expand abbreviations
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        expanded_act = self.matcher._expand_aerospace_abbreviations(activity_name)
        
        # Step 2: Process with spaCy
        req_doc = self.matcher.nlp(expanded_req)
        act_doc = self.matcher.nlp(expanded_act)
        
        # Step 3: Extract terms
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        act_terms = self.matcher._preprocess_text_aerospace(activity_name)
        
        # Visualization
        y_positions = [0.85, 0.55, 0.25]
        step_names = [
            "1. _expand_aerospace_abbreviations()",
            "2. nlp() tokenization",
            "3. _preprocess_text_aerospace()"
        ]
        
        for i, (y_pos, step) in enumerate(zip(y_positions, step_names)):
            ax.text(0.02, y_pos, step, fontweight='bold', fontsize=16)
            
            if i == 0:  # Abbreviation expansion
                # Show if any abbreviations were expanded
                abbrevs_found = []
                for abbrev in self.domain.abbreviations:
                    if abbrev in req_text.lower():
                        abbrevs_found.append(f"{abbrev} → {self.domain.abbreviations[abbrev]}")
                
                if abbrevs_found:
                    ax.text(0.25, y_pos, f"Expanded: {', '.join(abbrevs_found[:2])}", fontsize=12)
                else:
                    ax.text(0.25, y_pos, "No abbreviations to expand", fontsize=12, style='italic')
            
            elif i == 1:  # Tokenization
                req_tokens = [token.text for token in req_doc][:10]
                ax.text(0.25, y_pos, f"Tokens: {req_tokens}...", fontsize=12)
                ax.text(0.25, y_pos - 0.05, f"POS tags: {[token.pos_ for token in req_doc][:10]}...", 
                       fontsize=7, color='gray')
            
            else:  # Term extraction
                ax.text(0.25, y_pos, f"Requirement terms ({len(req_terms)}): {req_terms[:8]}...", fontsize=12)
                ax.text(0.25, y_pos - 0.07, f"Activity terms ({len(act_terms)}): {act_terms}", fontsize=12)
                
                # Highlight aerospace terms
                aero_req = [t for t in req_terms if t in self.matcher.all_aerospace_terms]
                aero_act = [t for t in act_terms if t in self.matcher.all_aerospace_terms]
                if aero_req or aero_act:
                    ax.text(0.25, y_pos - 0.14, f"Aerospace terms: {aero_req + aero_act}", 
                           fontsize=12, color=self.colors['domain'], fontweight='bold')
        
        return {
            'req_doc': req_doc,
            'act_doc': act_doc,
            'req_terms': req_terms,
            'act_terms': act_terms,
            'expanded_req': expanded_req,
            'expanded_act': expanded_act
        }
    
    def _layer3_algorithms(self, ax, req_text: str, activity_name: str, 
                        preprocessing_data: Dict, match, explanation: Dict) -> Dict:
        """Layer 3: Show four algorithm processing with actual scores from JSON explanations."""
        ax.set_title("Layer 3: Four Algorithm Analysis (compute_* methods)", 
                    fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')

        # NEW FORMAT: Get scores from explanation JSON structure
        scores = explanation.get('scores', {})
        exp_details = explanation.get('explanations', {})

        x_positions = [0.02, 0.26, 0.50, 0.74]
        width = 0.22

        for i, (algo, x_pos) in enumerate(zip(['semantic', 'bm25', 'domain', 'query_expansion'], x_positions)):
            # Algorithm box
            algo_box = Rectangle((x_pos, 0.1), width, 0.8,
                                facecolor=self.colors[algo], alpha=0.1,
                                edgecolor=self.colors[algo], linewidth=2)
            ax.add_patch(algo_box)

            # Algorithm name
            algo_names = {
                'semantic': 'Semantic',
                'bm25': 'BM25',
                'domain': 'Domain',
                'query_expansion': 'Query Expansion'
            }
            ax.text(x_pos + width/2, 0.85, algo_names[algo], ha='center', fontweight='bold', fontsize=20)

            # Method name
            method_names = {
                'semantic': 'compute_semantic_similarity()',
                'bm25': 'compute_bm25_score()',
                'domain': 'compute_domain_similarity()',
                'query_expansion': 'expand_query_aerospace()'
            }
            ax.text(x_pos + width/2, 0.75, method_names[algo], ha='center', fontsize=7, style='italic')

            # Scores & explanations
            score = scores.get(algo, 0)
            ax.text(x_pos + width/2, 0.60, f"Score: {score:.3f}", ha='center', fontsize=16, fontweight='bold', color=self.colors[algo])

            if algo == 'semantic':
                detail = exp_details.get('semantic', 'Vector similarity computed')
                ax.text(x_pos + 0.01, 0.45, detail[:80], fontsize=7, wrap=True)

            elif algo == 'bm25':
                bm25_detail = exp_details.get('bm25', '')
                shared_terms = explanation.get('shared_terms', [])
                ax.text(x_pos + 0.01, 0.45, bm25_detail[:60] if bm25_detail else f"Matched terms: {len(shared_terms)}", fontsize=7)
                if shared_terms:
                    ax.text(x_pos + 0.01, 0.35, f"{', '.join(shared_terms[:3])}", fontsize=6)

            elif algo == 'domain':
                # NEW FORMAT: Handle structured domain explanation
                domain_exp = exp_details.get('domain', {})
                y = 0.50

                if isinstance(domain_exp, dict):
                    # Aerospace terms
                    aero_terms = domain_exp.get('aerospace_terms', [])
                    if aero_terms:
                        ax.text(x_pos + 0.01, y, f"Aerospace: {', '.join(aero_terms[:3])}", fontsize=6)
                        y -= 0.06

                    # Key indicators with weights
                    key_indicators = domain_exp.get('key_indicators', {})
                    if key_indicators:
                        lines = [f"{k}({v.get('weight', 0):.2f})" for k,v in list(key_indicators.items())[:2]]
                        ax.text(x_pos + 0.01, y, f"Indicators: {', '.join(lines)}", fontsize=6)
                        y -= 0.06

                    # Learned relationships (top 2)
                    learned = domain_exp.get('learned_relationships', {})
                    if learned:
                        rel_count = len(learned)
                        ax.text(x_pos + 0.01, y, f"Learned relations: {rel_count}", fontsize=6)
                        y -= 0.06
                        # Show one example
                        if learned:
                            first_key = list(learned.keys())[0]
                            first_val = learned[first_key][:2]
                            ax.text(x_pos + 0.01, y, f"{first_key}→{','.join(first_val)}", fontsize=5, color='gray')
                            y -= 0.05

                    # Multi-evidence bonus
                    bonus = domain_exp.get('multi_evidence_bonus', 0)
                    if bonus > 0:
                        ax.text(x_pos + 0.01, y, f"Bonus: +{bonus:.2f}", fontsize=6, color=self.colors['good'])
                        y -= 0.05

                    # Final score confirmation
                    final_score = domain_exp.get('final_score', score)
                    if abs(final_score - score) > 0.001:
                        ax.text(x_pos + 0.01, y, f"Final: {final_score:.3f}", fontsize=6, fontweight='bold')
                else:
                    # Fallback for string format
                    ax.text(x_pos + 0.01, y, str(domain_exp)[:60], fontsize=6)

            elif algo == 'query_expansion':
                qe_exp = exp_details.get('query_expansion', 'N/A')
                # Parse the new format: "Activity expansion: 6/11 req terms matched (expanded 19 synonyms) via [condition, range, temperature]"
                lines = qe_exp.split(' via ')
                y = 0.50
                for line in lines:
                    ax.text(x_pos + 0.01, y, line[:70], fontsize=7)
                    y -= 0.08

        return scores    

    def _layer4_score_combination(self, ax, match, algorithm_scores: Dict) -> float:
        """Layer 4: Show actual score combination calculation"""
        ax.set_title("Layer 4: Score Combination (Weighted Average)", 
                    fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')
        
        # Show the actual calculation
        weights = {'semantic': 1.0, 'bm25': 1.0, 'domain': 1.0, 'query_expansion': 1.0}
        
        # Calculation visualization
        y_pos = 0.7
        ax.text(0.02, y_pos, "Calculation:", fontweight='bold', fontsize=20)
        
        calc_text = "Combined = ("
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = algorithm_scores[algo]
            weight = weights[algo]
            calc_text += f"{score:.3f}×{weight:.1f} + "
        
        calc_text = calc_text[:-3] + f") / {sum(weights.values()):.1f}"
        ax.text(0.15, y_pos, calc_text, fontsize=16, family='monospace')
        
        # Show result (actual from data)
        combined = match['Combined_Score']
        ax.text(0.15, y_pos - 0.15, f"= {combined:.3f}", fontsize=18, fontweight='bold')
        
        # Visual bar showing contribution
        bar_y = 0.3
        bar_height = 0.15
        x_start = 0.15
        total_width = 0.7
        
        # Background
        ax.add_patch(Rectangle((x_start, bar_y), total_width, bar_height,
                              facecolor='lightgray', edgecolor='black'))
        
        # Contributions
        x_current = x_start
        for algo, color in zip(['semantic', 'bm25', 'domain', 'query_expansion'],
                              [self.colors[a] for a in ['semantic', 'bm25', 'domain', 'query_expansion']]):
            contribution = (algorithm_scores[algo] * weights[algo]) / sum(weights.values())
            width = contribution * total_width / combined if combined > 0 else 0
            
            ax.add_patch(Rectangle((x_current, bar_y), width, bar_height,
                                  facecolor=color, alpha=0.7))
            x_current += width
        
        # Threshold indicators
        ax.text(0.15, 0.1, "Thresholds:", fontsize=16, fontweight='bold')
        ax.text(0.30, 0.1, "≥0.8 High", fontsize=12, color=self.colors['good'])
        ax.text(0.45, 0.1, "0.35-0.8 Medium", fontsize=12, color=self.colors['warning'])
        ax.text(0.65, 0.1, "<0.35 Orphan", fontsize=12, color=self.colors['bad'])
        
        return combined
    
    def _layer5_incose_analysis(self, ax, req_text: str) -> Dict:
        """Layer 5: Show INCOSE pattern extraction"""
        ax.set_title("Layer 5: INCOSE Pattern Analysis (INCOSEPatternAnalyzer.analyze_incose_compliance)", 
                    fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')
        
        # Run actual INCOSE analysis
        incose_analysis = self.quality_analyzer.incose_analyzer.analyze_incose_compliance(req_text)
        
        # Show extracted components
        components = incose_analysis.components_found
        
        y_pos = 0.8
        ax.text(0.02, y_pos, "Pattern Components Extracted:", fontweight='bold', fontsize=20)
        
        # Component list
        component_display = [
            ('AGENT', components.get('AGENT', 'Not found')),
            ('FUNCTION', components.get('FUNCTION', 'Not found')),
            ('PERFORMANCE', components.get('PERFORMANCE', 'Not found')),
            ('CONDITION', components.get('CONDITION', 'Not found'))
        ]
        
        x_positions = [0.15, 0.35, 0.55, 0.75]
        for (comp_name, comp_value), x_pos in zip(component_display, x_positions):
            color = self.colors['good'] if comp_value != 'Not found' else self.colors['bad']
            marker = "✓" if comp_value != 'Not found' else "✗"
            
            ax.text(x_pos, 0.5, f"{comp_name}:", fontweight='bold', fontsize=16)
            comp_value = components.get(comp_name, 'Not found') or 'Not found'
            ax.text(x_pos, 0.35, f"{marker} {comp_value[:20]}", fontsize=12, color=color)
        
        # Compliance score
        ax.text(0.02, 0.15, f"INCOSE Compliance Score: {incose_analysis.compliance_score:.0f}%",
               fontsize=20, fontweight='bold',
               color=self.colors['good'] if incose_analysis.compliance_score >= 75 else self.colors['warning'])
        
        # Best pattern
        ax.text(0.5, 0.15, f"Best Pattern: {incose_analysis.best_pattern}",
               fontsize=16, style='italic')
        
        return {'compliance_score': incose_analysis.compliance_score}
    
    def _layer6_quality_dimensions(self, ax, req_text: str) -> Dict:
        """Layer 6: Show quality dimension analysis"""
        ax.set_title("Layer 6: Quality Dimensions (analyze_requirement methods)", 
                    fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')
        
        # Run actual quality analysis
        issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)
        
        # Show five quality dimensions
        dimensions = [
            ('Clarity', metrics.clarity_score, '_analyze_clarity()'),
            ('Completeness', metrics.completeness_score, '_analyze_completeness()'),
            ('Verifiability', metrics.verifiability_score, '_analyze_verifiability()'),
            ('Atomicity', metrics.atomicity_score, '_analyze_atomicity()'),
            ('Consistency', metrics.consistency_score, '_analyze_consistency()')
        ]
        
        x_positions = np.linspace(0.1, 0.9, 5)
        
        for i, ((dim_name, score, method), x_pos) in enumerate(zip(dimensions, x_positions)):
            # Determine color based on score
            if score >= 80:
                color = self.colors['good']
            elif score >= 60:
                color = self.colors['warning']
            else:
                color = self.colors['bad']
            
            # Draw score circle
            circle = Circle((x_pos, 0.5), 0.06, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            
            # Score text
            ax.text(x_pos, 0.5, f"{score:.0f}", ha='center', va='center',
                   fontsize=20, fontweight='bold', color=color)
            
            # Dimension name
            ax.text(x_pos, 0.3, dim_name, ha='center', fontsize=16, fontweight='bold')
            
            # Method name
            ax.text(x_pos, 0.2, method, ha='center', fontsize=6, style='italic', color='gray')
        
        # Overall grade
        overall_score = (metrics.clarity_score + metrics.completeness_score + 
                        metrics.verifiability_score + metrics.atomicity_score + 
                        metrics.consistency_score) / 5
        grade = self.quality_analyzer._get_grade(overall_score)
        
        grade_color = self.colors['good'] if grade in ['EXCELLENT', 'GOOD'] else \
                     self.colors['warning'] if grade == 'FAIR' else self.colors['bad']
        
        ax.text(0.5, 0.05, f"Quality Grade: {grade} ({overall_score:.0f}%)",
               ha='center', fontsize=16, fontweight='bold', color=grade_color)
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'issues_count': len(issues)
        }
    
    def _layer7_final_decision(self, ax, combined_score: float, quality_results: Dict, match):
        """Layer 7: Show final decision based on both pipelines"""
        ax.set_title("Layer 7: Final Decision & Action", 
                    fontsize=16, fontweight='bold', loc='left')
        ax.axis('off')
        
        # Decision logic (actual from code)
        if combined_score >= 0.8 and quality_results['grade'] in ['EXCELLENT', 'GOOD']:
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
        
        # Decision box
        decision_box = FancyBboxPatch((0.25, 0.25), 0.5, 0.5,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, alpha=0.2,
                                     edgecolor=color, linewidth=3)
        ax.add_patch(decision_box)
        
        # Symbol
        ax.text(0.35, 0.5, symbol, ha='center', va='center',
               fontsize=36, color=color, fontweight='bold')
        
        # Decision text
        ax.text(0.55, 0.55, decision, ha='center', va='center',
               fontsize=18, fontweight='bold')
        ax.text(0.55, 0.40, action, ha='center', va='center',
               fontsize=16, color='gray')
        
        # Summary metrics
        metrics_text = (f"Match Score: {combined_score:.3f} | "
                       f"Quality: {quality_results['grade']} | "
                       f"Issues: {quality_results['issues_count']}")
        ax.text(0.5, 0.1, metrics_text, ha='center', fontsize=16, color='gray')
       
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value"""
        if score >= 0.7:
            return self.colors['good']
        elif score >= 0.4:
            return self.colors['warning']
        else:
            return self.colors['bad']
    
    def create_algorithm_contribution_chart(self) -> str:
        """Create chart showing how each algorithm contributes across all matches"""
        
        # Load real data
        matches_df, _, _ = self.load_real_data()
        
        # Calculate average contributions
        algorithms = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        avg_scores = {algo: matches_df[algo].mean() for algo in algorithms}
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Contribution Analysis Across All Matches', fontsize=20, fontweight='bold')
        
        # 1. Average contribution pie chart
        ax1 = axes[0, 0]
        colors_list = [self.colors[a] for a in ['semantic', 'bm25', 'domain', 'query_expansion']]
        labels = ['Semantic', 'BM25', 'Domain', 'Query Exp']
        values = list(avg_scores.values())
        
        ax1.pie(values, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Average Algorithm Contribution', fontsize=16, fontweight='bold')
        
        # 2. Score distribution by algorithm
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
        
        # Add correlation values
        for i in range(len(algorithms)):
            for j in range(len(algorithms)):
                text = ax3.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=16)
        
        plt.colorbar(im, ax=ax3)
        
        # 4. Success rate by dominant algorithm
        ax4 = axes[1, 1]
        
        # Find dominant algorithm for each match
        matches_df['Dominant_Algorithm'] = matches_df[algorithms].idxmax(axis=1)
        matches_df['Dominant_Algorithm'] = matches_df['Dominant_Algorithm'].str.replace('_Score', '')
        
        # Count successes (score >= 0.8) by dominant algorithm
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
        
        # Add count labels
        for i, (bar, total) in enumerate(zip(bars, total_counts)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={total}', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "algorithm_contribution_analysis.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved algorithm contribution analysis: {output_path}")
        return str(output_path)
    
    def create_all_visualizations(self) -> List[str]:
        """Create all technical visualizations for highest and lowest scoring pairs"""
        
        print("\n" + "="*70)
        print("🎨 CREATING TECHNICAL JOURNEY VISUALIZATIONS (HIGHEST & LOWEST)")
        print("="*70)
        
        paths = []
        
        # Load matches
        matches_df, explanations, _ = self.load_real_data()
        if matches_df.empty:
            print("❌ No matches found")
            return paths
        
        # Determine highest and lowest scoring pairs
        highest_match = matches_df.nlargest(1, 'Combined_Score').iloc[0]
        lowest_candidates = matches_df[matches_df['Combined_Score'] > 0]
        if not lowest_candidates.empty:
            lowest_match = lowest_candidates.nsmallest(1, 'Combined_Score').iloc[0]
        else:
            lowest_match = matches_df.nsmallest(1, 'Combined_Score').iloc[0]
        
        for label, match in [('high', highest_match), ('low', lowest_match)]:
            req_id = match['Requirement_ID']
            print(f"\n📊 Creating technical journey for {label.upper()} scoring pair: {req_id}")
            try:
                path = self.create_processing_journey(requirement_id=req_id, label=label)
                paths.append(path)
            except Exception as e:
                print(f"❌ Error creating journey for {req_id}: {e}")
        
        print("\n" + "="*70)
        print(f"✅ Created {len(paths)} visualizations")
        print("Files saved to: outputs/matching_results/technical_visualizations/")
        print("="*70)
        
        return paths

def main():
    """Main function to create technical visualizations"""
    
    try:
        visualizer = TechnicalJourneyVisualizer()
        
        # Create all visualizations
        paths = visualizer.create_all_visualizations()
        
        # Print summary
        print("\n📁 Generated Technical Visualizations:")
        for path in paths:
            print(f"   • {Path(path).name}")
        
        return paths
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run matcher.py first to generate matching results")
        return []
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()