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
        Enhanced version of create_processing_journey with better layout and error handling
        """
        
        # Load real data
        try:
            matches_df, explanations, requirements_df = self.load_real_data()
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please run matcher.py first to generate matching results")
            return ""
        
        # Select requirement to visualize
        if requirement_id:
            match = matches_df[matches_df['Requirement_ID'] == requirement_id]
            if match.empty:
                print(f"⚠️ {requirement_id} not found, selecting best match")
                match = matches_df.nlargest(1, 'Combined_Score')
        else:
            # Get the highest scoring match for best demonstration
            match = matches_df.nlargest(1, 'Combined_Score')
        
        if match.empty:
            print("❌ No matches found in aerospace_matches.csv")
            return ""
        
        match = match.iloc[0]
        req_id = str(match.get('Requirement_ID', 'Unknown'))
        
        # Safely extract text fields and convert to strings
        req_text = match.get('Requirement_Text', '')
        if pd.isna(req_text) or req_text is None:
            req_text = ''
        else:
            req_text = str(req_text)
        
        activity_name = match.get('Activity_Name', '')
        if pd.isna(activity_name) or activity_name is None:
            activity_name = ''
        else:
            activity_name = str(activity_name)
        
        # Validate we have enough data to proceed
        if not req_text.strip() or not activity_name.strip():
            print(f"❌ Missing text data for {req_id}: req_text='{req_text}', activity='{activity_name}'")
            return ""
        
        print(f"📊 Creating enhanced visualization for: {req_id}")
        print(f"   Requirement: {req_text[:80]}...")
        print(f"   Activity: {activity_name}")
        
        # Safely get combined score
        try:
            combined_score = float(match.get('Combined_Score', 0))
        except (ValueError, TypeError):
            combined_score = 0.0
        
        print(f"   Score: {combined_score:.3f}")
        
        # Get explanation data with safety check
        explanation_key = (req_id, activity_name)
        explanation = explanations.get(explanation_key, {})
        if not isinstance(explanation, dict):
            explanation = {}
        
        # Create the visualization with better spacing
        fig = plt.figure(figsize=(24, 30))  # Taller figure for more content
        
        # Enhanced title with status badge
        if combined_score >= 0.8:
            status_badge = "🟢 HIGH CONFIDENCE"
            title_color = '#27ae60'
        elif combined_score >= 0.35:
            status_badge = "🟡 NEEDS REVIEW" 
            title_color = '#f39c12'
        else:
            status_badge = "🔴 ORPHAN"
            title_color = '#c0392b'
        
        main_title = f'Technical Processing Journey: {req_id}'
        subtitle = f'{status_badge} | Score: {combined_score:.3f}'
        if label:
            subtitle = f"{label.upper()} SCORING PAIR | {subtitle}"
        
        fig.suptitle(main_title, fontsize=32, fontweight='bold', y=0.97)
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=20, color=title_color, fontweight='bold')
        
        # Create grid with better spacing for content-heavy layers
        gs = fig.add_gridspec(9, 1, height_ratios=[1, 1, 2, 1, 1.5, 1.5, 1, 0.3, 0.3], hspace=0.3)
        
        layer_results = {}
        
        try:
            # LAYER 1: Raw Inputs
            print("   Creating Layer 1: Raw Inputs...")
            ax1 = fig.add_subplot(gs[0])
            self._layer1_raw_inputs(ax1, req_text, activity_name, req_id)
            
            # LAYER 2: Preprocessing Steps  
            print("   Creating Layer 2: Preprocessing...")
            ax2 = fig.add_subplot(gs[1])
            preprocessing_data = self._layer2_preprocessing(ax2, req_text, activity_name)
            layer_results['preprocessing'] = preprocessing_data
            
            # LAYER 3: Four Algorithm Processing (larger space)
            print("   Creating Layer 3: Algorithm Analysis...")
            ax3 = fig.add_subplot(gs[2])
            algorithm_scores = self._layer3_algorithms(ax3, req_text, activity_name, 
                                                    preprocessing_data, match, explanation)
            layer_results['algorithms'] = algorithm_scores
            
            # LAYER 4: Score Combination
            print("   Creating Layer 4: Score Combination...")
            ax4 = fig.add_subplot(gs[3])
            final_combined_score = self._layer4_score_combination(ax4, match, algorithm_scores)
            layer_results['combined_score'] = final_combined_score
            
            # LAYER 5: INCOSE Pattern Analysis
            print("   Creating Layer 5: INCOSE Analysis...")
            ax5 = fig.add_subplot(gs[4])
            incose_results = self._layer5_incose_analysis(ax5, req_text)
            layer_results['incose'] = incose_results
            
            # LAYER 6: Quality Dimensions
            print("   Creating Layer 6: Quality Analysis...")
            ax6 = fig.add_subplot(gs[5])
            quality_results = self._layer6_quality_dimensions(ax6, req_text)
            layer_results['quality'] = quality_results
            
            # LAYER 7: Final Decision
            print("   Creating Layer 7: Final Decision...")
            ax7 = fig.add_subplot(gs[6])
            self._layer7_final_decision(ax7, final_combined_score, quality_results, match)
            
            # Add footer with metadata
            footer_ax = fig.add_subplot(gs[7])
            footer_ax.axis('off')
            footer_text = (f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"Matcher: AerospaceMatcher | Quality: EnhancedRequirementAnalyzer | "
                        f"Domain Terms: {len(self.matcher.all_aerospace_terms)}")
            footer_ax.text(0.5, 0.5, footer_text, ha='center', va='center', 
                        fontsize=12, color='gray', style='italic')
            
            print("   All layers created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating layer: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error message to the figure
            error_ax = fig.add_subplot(gs[7])
            error_ax.axis('off')
            error_ax.text(0.5, 0.5, f"⚠️ Visualization partially failed: {str(e)[:100]}...", 
                        ha='center', va='center', fontsize=16, color='red', fontweight='bold')
        
        # Save figure with enhanced metadata
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        if label:
            filename = f"{req_id}_{label}_technical_journey_{timestamp}.png"
        else:
            filename = f"{req_id}_technical_journey_{timestamp}.png"
        
        # Clean filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        output_path = self.output_dir / filename
        
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            print(f"✅ Saved enhanced technical journey: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"❌ Error saving figure: {e}")
            plt.close()
            return ""    

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
        """Layer 2: Show actual preprocessing steps with visual flow"""
        ax.set_title("Layer 2: Text Preprocessing Pipeline", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Run actual preprocessing methods
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        expanded_act = self.matcher._expand_aerospace_abbreviations(activity_name)
        
        req_doc = self.matcher.nlp(expanded_req)
        act_doc = self.matcher.nlp(expanded_act)
        
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        act_terms = self.matcher._preprocess_text_aerospace(activity_name)
        
        # Visual layout: 3 processing steps from left to right
        step_width = 0.3
        step_height = 0.8
        step_spacing = 0.35
        start_x = 0.02
        
        steps = [
            {
                'title': 'Step 1: Abbreviation Expansion',
                'method': '_expand_aerospace_abbreviations()',
                'color': '#e8f4fd',
                'edge_color': '#3498db'
            },
            {
                'title': 'Step 2: NLP Processing', 
                'method': 'nlp() + spaCy pipeline',
                'color': '#fff3cd',
                'edge_color': '#ffc107'
            },
            {
                'title': 'Step 3: Term Extraction',
                'method': '_preprocess_text_aerospace()',
                'color': '#e8f6f3',
                'edge_color': '#27ae60'
            }
        ]
        
        for i, step in enumerate(steps):
            x_pos = start_x + i * step_spacing
            
            # Step box
            step_box = FancyBboxPatch((x_pos, 0.1), step_width, step_height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=step['color'], 
                                    edgecolor=step['edge_color'], linewidth=2)
            ax.add_patch(step_box)
            
            # Title and method
            ax.text(x_pos + step_width/2, 0.85, step['title'], 
                ha='center', fontweight='bold', fontsize=16)
            ax.text(x_pos + step_width/2, 0.78, step['method'], 
                ha='center', fontsize=10, style='italic', color='gray')
            
            # Content based on step
            if i == 0:  # Abbreviation expansion
                abbrevs_found = []
                for abbrev in self.domain.abbreviations:
                    if abbrev in req_text.lower() or abbrev in activity_name.lower():
                        abbrevs_found.append(f"{abbrev} → {self.domain.abbreviations[abbrev]}")
                
                if abbrevs_found:
                    y_pos = 0.65
                    ax.text(x_pos + 0.01, y_pos, "Found:", fontweight='bold', fontsize=12)
                    for abbrev in abbrevs_found[:3]:  # Show max 3
                        y_pos -= 0.08
                        ax.text(x_pos + 0.01, y_pos, f"• {abbrev}", fontsize=10, 
                            color=step['edge_color'])
                else:
                    ax.text(x_pos + 0.01, 0.6, "No abbreviations found", 
                        fontsize=12, style='italic', color='gray')
            
            elif i == 1:  # NLP processing
                # Show tokenization results
                req_tokens = [token.text for token in req_doc if not token.is_punct][:12]
                act_tokens = [token.text for token in act_doc if not token.is_punct][:12]
                
                ax.text(x_pos + 0.01, 0.65, "Requirement tokens:", fontweight='bold', fontsize=11)
                ax.text(x_pos + 0.01, 0.58, " | ".join(req_tokens), fontsize=9, 
                    family='monospace')
                
                ax.text(x_pos + 0.01, 0.45, "Activity tokens:", fontweight='bold', fontsize=11)
                ax.text(x_pos + 0.01, 0.38, " | ".join(act_tokens), fontsize=9, 
                    family='monospace')
                
                # Show POS tags for first few tokens
                pos_tags = [f"{token.text}({token.pos_})" for token in req_doc][:8]
                ax.text(x_pos + 0.01, 0.25, "POS tags:", fontweight='bold', fontsize=10)
                ax.text(x_pos + 0.01, 0.18, " | ".join(pos_tags), fontsize=8, 
                    color='gray', family='monospace')
            
            else:  # Term extraction
                ax.text(x_pos + 0.01, 0.65, f"Req terms ({len(req_terms)}):", 
                    fontweight='bold', fontsize=11)
                req_display = ", ".join(req_terms[:8])
                if len(req_terms) > 6:
                    req_display += "..."
                ax.text(x_pos + 0.01, 0.58, req_display, fontsize=9, wrap=True)
                
                ax.text(x_pos + 0.01, 0.45, f"Act terms ({len(act_terms)}):", 
                    fontweight='bold', fontsize=11)
                act_display = ", ".join(act_terms[:8])
                if len(act_terms) > 6:
                    act_display += "..."
                ax.text(x_pos + 0.01, 0.38, act_display, fontsize=9, wrap=True)
                
                # Highlight aerospace terms
                aero_req = [t for t in req_terms if t in self.matcher.all_aerospace_terms]
                aero_act = [t for t in act_terms if t in self.matcher.all_aerospace_terms]
                if aero_req or aero_act:
                    ax.text(x_pos + 0.01, 0.25, "Aerospace terms:", 
                        fontweight='bold', fontsize=10, color=step['edge_color'])
                    aero_display = ", ".join((aero_req + aero_act)[:4])
                    ax.text(x_pos + 0.01, 0.18, aero_display, fontsize=9, 
                        color=step['edge_color'], fontweight='bold')
            
            # Arrow to next step
            if i < len(steps) - 1:
                arrow = FancyArrowPatch((x_pos + step_width, 0.5), 
                                    (x_pos + step_spacing - 0.02, 0.5),
                                    arrowstyle='->', mutation_scale=20, 
                                    color='black', linewidth=2)
                ax.add_patch(arrow)
        
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
        """Layer 3: Four algorithm processing with detailed visual breakdown"""
        ax.set_title("Layer 3: Four Algorithm Analysis", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Safely get actual scores from match data with defaults
        scores = {
            'semantic': float(match.get('Semantic_Score', 0)) if match.get('Semantic_Score') is not None else 0.0,
            'bm25': float(match.get('BM25_Score', 0)) if match.get('BM25_Score') is not None else 0.0,
            'domain': float(match.get('Domain_Score', 0)) if match.get('Domain_Score') is not None else 0.0,
            'query_expansion': float(match.get('Query_Expansion_Score', 0)) if match.get('Query_Expansion_Score') is not None else 0.0
        }
        
        # Safely get explanation details
        exp_details = explanation.get('explanations', {}) if isinstance(explanation, dict) else {}
        
        # Layout: 2x2 grid of algorithms
        algo_width = 0.45
        algo_height = 0.4
        left_x, right_x = 0.02, 0.53
        top_y, bottom_y = 0.5, 0.05
        
        algorithms = [
            ('semantic', 'Semantic Similarity', 'compute_semantic_similarity()', left_x, top_y),
            ('bm25', 'BM25 Keyword Matching', 'compute_bm25_score()', right_x, top_y),
            ('domain', 'Domain Knowledge', 'compute_domain_similarity()', left_x, bottom_y),
            ('query_expansion', 'Query Expansion', 'expand_query_aerospace()', right_x, bottom_y)
        ]
        
        for algo_key, algo_name, method_name, x_pos, y_pos in algorithms:
            try:
                # Algorithm box with color coding based on score
                score = scores.get(algo_key, 0.0)
                if score >= 0.7:
                    bg_color = '#d4edda'  # Light green
                    border_color = '#28a745'
                elif score >= 0.4:
                    bg_color = '#fff3cd'  # Light yellow
                    border_color = '#ffc107'
                else:
                    bg_color = '#f8d7da'  # Light red
                    border_color = '#dc3545'
                
                algo_box = FancyBboxPatch((x_pos, y_pos), algo_width, algo_height,
                                         boxstyle="round,pad=0.02",
                                         facecolor=bg_color, 
                                         edgecolor=border_color, linewidth=3)
                ax.add_patch(algo_box)
                
                # Algorithm header
                ax.text(x_pos + algo_width/2, y_pos + algo_height - 0.05, algo_name, 
                       ha='center', fontweight='bold', fontsize=18)
                ax.text(x_pos + algo_width/2, y_pos + algo_height - 0.1, method_name, 
                       ha='center', fontsize=10, style='italic', color='gray')
                
                # Score display with visual indicator
                score_y = y_pos + algo_height - 0.18
                ax.text(x_pos + 0.02, score_y, f"Score: {score:.3f}", 
                       fontweight='bold', fontsize=16, color=border_color)
                
                # Score bar
                bar_width = 0.3
                bar_height = 0.03
                bar_x = x_pos + algo_width - bar_width - 0.02
                
                # Background bar
                ax.add_patch(Rectangle((bar_x, score_y - 0.01), bar_width, bar_height,
                                      facecolor='lightgray', edgecolor='gray'))
                # Score bar
                ax.add_patch(Rectangle((bar_x, score_y - 0.01), bar_width * min(score, 1.0), bar_height,
                                      facecolor=border_color))
                
                # Algorithm-specific details with error handling
                detail_y = y_pos + algo_height - 0.28
                
                if algo_key == 'semantic':
                    semantic_exp = str(exp_details.get('semantic', 'Vector similarity computed'))
                    if 'Enhanced semantic' in semantic_exp:
                        ax.text(x_pos + 0.02, detail_y, "✓ Enhanced transformer model", 
                               fontsize=12, color='green')
                    elif 'spaCy semantic' in semantic_exp:
                        ax.text(x_pos + 0.02, detail_y, "✓ spaCy word vectors", 
                               fontsize=12, color='blue')
                    else:
                        ax.text(x_pos + 0.02, detail_y, "✓ Text-based fallback", 
                               fontsize=12, color='orange')
                
                elif algo_key == 'bm25':
                    # Safely get shared terms
                    shared_terms = explanation.get('shared_terms', []) if isinstance(explanation, dict) else []
                    if isinstance(shared_terms, list):
                        shared_terms = [str(term) for term in shared_terms if term]
                    else:
                        shared_terms = []
                    
                    ax.text(x_pos + 0.02, detail_y, f"Shared terms: {len(shared_terms)}", 
                           fontweight='bold', fontsize=12)
                    if shared_terms:
                        terms_display = ", ".join(shared_terms[:5])
                        if len(shared_terms) > 5:
                            terms_display += "..."
                        ax.text(x_pos + 0.02, detail_y - 0.06, f"[{terms_display}]", 
                               fontsize=10, color='darkred')
                        
                        # Show aerospace term boost
                        aero_terms = [str(t) for t in shared_terms if str(t) in self.matcher.all_aerospace_terms]
                        if aero_terms:
                            ax.text(x_pos + 0.02, detail_y - 0.12, f"Aerospace boost: {len(aero_terms)} terms", 
                                   fontsize=10, color='green', fontweight='bold')
                
                elif algo_key == 'domain':
                    domain_exp = exp_details.get('domain', {})
                    
                    # Handle both old string format and new dictionary format
                    if isinstance(domain_exp, dict):
                        # New structured format
                        detail_y_current = detail_y
                        
                        # Show aerospace terms
                        aerospace_terms = domain_exp.get('aerospace_terms', [])
                        if aerospace_terms:
                            ax.text(x_pos + 0.02, detail_y_current, f"✓ Aerospace: {', '.join(aerospace_terms[:3])}", 
                                   fontsize=10, color='green', fontweight='bold')
                            detail_y_current -= 0.05
                        
                        # Show key indicators
                        key_indicators = domain_exp.get('key_indicators', {})
                        if key_indicators:
                            indicator_names = list(key_indicators.keys())[:2]  # Show first 2
                            ax.text(x_pos + 0.02, detail_y_current, f"✓ Indicators: {', '.join(indicator_names)}", 
                                   fontsize=10, color='blue')
                            detail_y_current -= 0.05
                        
                        # Show learned relationships
                        learned_relationships = domain_exp.get('learned_relationships', {})
                        if learned_relationships:
                            rel_count = len(learned_relationships)
                            ax.text(x_pos + 0.02, detail_y_current, f"✓ Learned relations: {rel_count} patterns", 
                                   fontsize=10, color='purple')
                            detail_y_current -= 0.05
                        
                        # Show multi-evidence bonus
                        bonus = domain_exp.get('multi_evidence_bonus', 0)
                        if bonus > 0:
                            ax.text(x_pos + 0.02, detail_y_current, f"✓ Multi-evidence: +{bonus:.2f}", 
                                   fontsize=10, color='orange', fontweight='bold')
                            detail_y_current -= 0.05
                        
                        # Show final score
                        final_score = domain_exp.get('final_score', score)
                        ax.text(x_pos + 0.02, detail_y_current, f"Final: {final_score:.3f}", 
                               fontsize=9, color='black', style='italic')
                        
                        # If no specific elements found, show minimal overlap
                        if not aerospace_terms and not key_indicators and not learned_relationships:
                            ax.text(x_pos + 0.02, detail_y, "Minimal domain overlap", 
                                   fontsize=12, color='gray', style='italic')
                    
                    else:
                        # Old string format (fallback)
                        domain_exp_str = str(domain_exp)
                        
                        # Parse domain explanation for key information
                        if 'aerospace terms' in domain_exp_str.lower():
                            ax.text(x_pos + 0.02, detail_y, "✓ Aerospace vocabulary match", 
                                   fontsize=12, color='green')
                        
                        if 'indicator' in domain_exp_str.lower():
                            ax.text(x_pos + 0.02, detail_y - 0.06, "✓ Key indicators found", 
                                   fontsize=12, color='blue')
                        
                        if 'evidence' in domain_exp_str.lower():
                            ax.text(x_pos + 0.02, detail_y - 0.12, "✓ Multi-evidence bonus", 
                                   fontsize=12, color='purple')
                        
                        if score < 0.2:
                            ax.text(x_pos + 0.02, detail_y, "Minimal domain overlap", 
                                   fontsize=12, color='gray', style='italic')
                
                else:  # query_expansion
                    qe_exp = str(exp_details.get('query_expansion', 'Query expansion performed'))
                    
                    # Extract expansion information
                    if 'matched' in qe_exp.lower():
                        # Try to extract the number from expressions like "3/5 req terms matched"
                        import re
                        match_info = re.search(r'(\d+)/(\d+)', qe_exp)
                        if match_info:
                            matched, total = match_info.groups()
                            ax.text(x_pos + 0.02, detail_y, f"Matched: {matched}/{total} req terms", 
                                   fontweight='bold', fontsize=12)
                            
                            coverage = int(matched) / int(total) if int(total) > 0 else 0
                            if coverage >= 0.5:
                                ax.text(x_pos + 0.02, detail_y - 0.06, "✓ Good coverage", 
                                       fontsize=11, color='green')
                            else:
                                ax.text(x_pos + 0.02, detail_y - 0.06, "! Sparse coverage", 
                                       fontsize=11, color='orange')
                    
                    if 'expanded' in qe_exp.lower():
                        ax.text(x_pos + 0.02, detail_y - 0.12, "✓ Synonyms expanded", 
                               fontsize=11, color='blue')
            
            except Exception as e:
                print(f"⚠️ Error processing algorithm {algo_key}: {e}")
                # Fallback display
                ax.text(x_pos + 0.02, detail_y, f"Score: {scores.get(algo_key, 0.0):.3f}", 
                       fontsize=12, color='gray')
        
        return scores

    def _extract_domain_explanation_info(self, domain_exp) -> Dict[str, Any]:
        """
        Helper method to extract domain explanation information from both old and new formats.
        
        Args:
            domain_exp: Either a string (old format) or dict (new format)
        
        Returns:
            Dictionary with standardized domain explanation info
        """
        info = {
            'aerospace_terms': [],
            'key_indicators': {},
            'learned_relationships': {},
            'multi_evidence_bonus': 0.0,
            'has_aerospace_match': False,
            'has_indicators': False,
            'has_relationships': False,
            'summary_text': ''
        }
        
        if isinstance(domain_exp, dict):
            # New structured format
            info['aerospace_terms'] = domain_exp.get('aerospace_terms', [])
            info['key_indicators'] = domain_exp.get('key_indicators', {})
            info['learned_relationships'] = domain_exp.get('learned_relationships', {})
            info['multi_evidence_bonus'] = domain_exp.get('multi_evidence_bonus', 0.0)
            
            # Set flags
            info['has_aerospace_match'] = len(info['aerospace_terms']) > 0
            info['has_indicators'] = len(info['key_indicators']) > 0
            info['has_relationships'] = len(info['learned_relationships']) > 0
            
            # Create summary
            components = []
            if info['has_aerospace_match']:
                components.append(f"{len(info['aerospace_terms'])} aerospace terms")
            if info['has_indicators']:
                components.append(f"{len(info['key_indicators'])} indicators")
            if info['has_relationships']:
                components.append(f"{len(info['learned_relationships'])} relationships")
            if info['multi_evidence_bonus'] > 0:
                components.append(f"bonus: +{info['multi_evidence_bonus']:.2f}")
            
            info['summary_text'] = ', '.join(components) if components else 'minimal overlap'
            
        else:
            # Old string format
            domain_str = str(domain_exp).lower()
            info['has_aerospace_match'] = 'aerospace' in domain_str
            info['has_indicators'] = 'indicator' in domain_str
            info['has_relationships'] = 'evidence' in domain_str or 'relationship' in domain_str
            info['summary_text'] = str(domain_exp)[:50] + '...' if len(str(domain_exp)) > 50 else str(domain_exp)
        
        return info

    def _layer4_score_combination(self, ax, match, algorithm_scores: Dict) -> float:
        """Layer 4: Score combination with detailed calculation visualization"""
        ax.set_title("Layer 4: Score Combination & Weighting", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Weights (from actual implementation)
        weights = {'semantic': 1.0, 'bm25': 1.0, 'domain': 1.0, 'query_expansion': 1.0}
        
        # Calculation section
        calc_y = 0.7
        ax.text(0.02, calc_y, "Weighted Average Calculation:", fontweight='bold', fontsize=20)
        
        # Show detailed calculation
        calc_parts = []
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = algorithm_scores[algo]
            weight = weights[algo]
            calc_parts.append(f"{score:.3f}×{weight:.1f}")
        
        calc_text = f"({' + '.join(calc_parts)}) ÷ {sum(weights.values()):.1f}"
        ax.text(0.02, calc_y - 0.1, calc_text, fontsize=16, family='monospace', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # Show result
        combined = match['Combined_Score']
        ax.text(0.02, calc_y - 0.25, f"= {combined:.3f}", fontsize=24, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.5))
        
        # Visual contribution breakdown
        contrib_y = 0.3
        ax.text(0.02, contrib_y, "Algorithm Contributions:", fontweight='bold', fontsize=18)
        
        # Horizontal stacked bar
        bar_x = 0.25
        bar_y = contrib_y - 0.1
        bar_width = 0.6
        bar_height = 0.08
        
        # Background
        ax.add_patch(Rectangle((bar_x, bar_y), bar_width, bar_height,
                            facecolor='lightgray', edgecolor='black', linewidth=2))
        
        # Individual contributions
        current_x = bar_x
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # semantic, bm25, domain, qe
        algo_names = ['Semantic', 'BM25', 'Domain', 'Query Exp']
        
        for i, (algo, color, name) in enumerate(zip(['semantic', 'bm25', 'domain', 'query_expansion'], 
                                                colors, algo_names)):
            contribution = algorithm_scores[algo] * weights[algo] / sum(weights.values())
            segment_width = (contribution / combined) * bar_width if combined > 0 else 0
            
            # Segment
            segment = Rectangle((current_x, bar_y), segment_width, bar_height,
                            facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
            ax.add_patch(segment)
            
            # Label if segment is wide enough
            if segment_width > 0.05:
                ax.text(current_x + segment_width/2, bar_y + bar_height/2, 
                    f"{name}\n{contribution:.2f}", ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
            
            current_x += segment_width
        
        # Threshold indicators
        threshold_y = 0.05
        ax.text(0.02, threshold_y, "Decision Thresholds:", fontweight='bold', fontsize=16)
        
        thresholds = [
            (0.8, "High Confidence", '#27ae60'),
            (0.35, "Review Needed", '#f39c12'),
            (0.0, "Orphan", '#c0392b')
        ]
        
        for i, (threshold, label, color) in enumerate(thresholds):
            x_pos = 0.25 + i * 0.2
            
            # Threshold indicator
            if combined >= threshold:
                marker = "●"
                text_color = color
            else:
                marker = "○"
                text_color = 'lightgray'
            
            ax.text(x_pos, threshold_y, f"{marker} ≥{threshold:.2f}", fontweight='bold', 
                fontsize=14, color=text_color)
            ax.text(x_pos, threshold_y - 0.05, label, fontsize=12, color=text_color)
        
        return combined

    def _layer5_incose_analysis(self, ax, req_text: str) -> Dict:
        """Layer 5: INCOSE pattern analysis with component visualization"""
        ax.set_title("Layer 5: INCOSE Requirements Pattern Analysis", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Run actual INCOSE analysis
        incose_analysis = self.quality_analyzer.incose_analyzer.analyze_incose_compliance(req_text)
        
        # Pattern info section
        pattern_y = 0.85
        ax.text(0.02, pattern_y, f"Best Matching Pattern: {incose_analysis.best_pattern}", 
            fontweight='bold', fontsize=18)
        
        # Compliance score with visual indicator
        score = incose_analysis.compliance_score
        score_color = '#27ae60' if score >= 75 else '#f39c12' if score >= 50 else '#c0392b'
        
        # Score circle
        circle = Circle((0.85, pattern_y - 0.05), 0.06, facecolor=score_color, alpha=0.3, 
                    edgecolor=score_color, linewidth=3)
        ax.add_patch(circle)
        ax.text(0.85, pattern_y - 0.05, f"{score:.0f}%", ha='center', va='center',
            fontsize=18, fontweight='bold', color=score_color)
        
        # Component analysis
        components_y = 0.65
        ax.text(0.02, components_y, "INCOSE Component Analysis:", fontweight='bold', fontsize=16)
        
        # Required components
        components = incose_analysis.components_found
        required_components = [
            ('AGENT', 'Who/What system'),
            ('FUNCTION', 'What action'),
            ('PERFORMANCE', 'How well'),
            ('CONDITION', 'When/Where')
        ]
        
        # Component grid
        comp_width = 0.22
        comp_height = 0.35
        start_x = 0.02
        
        for i, (comp_name, comp_desc) in enumerate(required_components):
            x_pos = start_x + i * 0.24
            comp_value = components.get(comp_name, None)
            
            # Component box
            if comp_value and comp_value != 'Not found':
                box_color = '#d4edda'
                border_color = '#28a745'
                status_symbol = "✓"
                status_color = '#28a745'
            else:
                box_color = '#f8d7da'
                border_color = '#dc3545'
                status_symbol = "✗"
                status_color = '#dc3545'
            
            comp_box = FancyBboxPatch((x_pos, 0.15), comp_width, comp_height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=box_color, 
                                    edgecolor=border_color, linewidth=2)
            ax.add_patch(comp_box)
            
            # Component header
            ax.text(x_pos + comp_width/2, 0.45, comp_name, ha='center', 
                fontweight='bold', fontsize=14)
            ax.text(x_pos + comp_width/2, 0.40, comp_desc, ha='center', 
                fontsize=10, style='italic', color='gray')
            
            # Status
            ax.text(x_pos + 0.02, 0.35, status_symbol, fontsize=20, color=status_color, 
                fontweight='bold')
            
            # Value
            if comp_value and comp_value != 'Not found':
                display_value = comp_value[:15] + "..." if len(comp_value) > 15 else comp_value
                ax.text(x_pos + 0.02, 0.25, f'"{display_value}"', fontsize=10, 
                    color=status_color, fontweight='bold')
            else:
                ax.text(x_pos + 0.02, 0.25, "Missing", fontsize=10, 
                    color=status_color, style='italic')
        
        # Suggestions if score is low
        if score < 75 and incose_analysis.suggestions:
            ax.text(0.02, 0.05, f"Improvement needed: {incose_analysis.suggestions[0][:80]}...", 
                fontsize=12, color='#c0392b', style='italic')
        
        return {'compliance_score': score, 'components_found': len([c for c in components.values() if c and c != 'Not found'])}

    def _layer6_quality_dimensions(self, ax, req_text: str) -> Dict:
        """Layer 6: Quality dimensions with radar chart style visualization"""
        ax.set_title("Layer 6: Quality Dimension Analysis", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Run actual quality analysis
        issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)
        
        # Quality dimensions with scores
        dimensions = [
            ('Clarity', metrics.clarity_score, '_analyze_clarity()'),
            ('Completeness', metrics.completeness_score, '_analyze_completeness()'),
            ('Verifiability', metrics.verifiability_score, '_analyze_verifiability()'),
            ('Atomicity', metrics.atomicity_score, '_analyze_atomicity()'),
            ('Consistency', metrics.consistency_score, '_analyze_consistency()')
        ]
        
        # Calculate overall score and grade
        overall_score = sum([score for _, score, _ in dimensions]) / len(dimensions)
        grade = self.quality_analyzer._get_grade(overall_score)
        
        # Grade display
        grade_y = 0.85
        grade_colors = {
            'EXCELLENT': '#27ae60', 'GOOD': '#2ecc71', 'FAIR': '#f39c12', 
            'POOR': '#e74c3c', 'CRITICAL': '#c0392b'
        }
        grade_color = grade_colors.get(grade, '#6c757d')
        
        ax.text(0.02, grade_y, f"Overall Quality Grade: {grade} ({overall_score:.0f}%)", 
            fontweight='bold', fontsize=20, color=grade_color)
        
        # Dimension visualization - pentagon radar style
        center_x, center_y = 0.5, 0.4
        radius = 0.25
        
        # Draw background pentagon and score pentagon
        import numpy as np
        angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
        
        # Background pentagon (100% score)
        bg_points = [(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)) 
                    for angle in angles]
        bg_polygon = plt.Polygon(bg_points, fill=False, edgecolor='lightgray', linewidth=2, linestyle='--')
        ax.add_patch(bg_polygon)
        
        # Score pentagon
        score_points = [(center_x + (radius * score/100) * np.cos(angle), 
                        center_y + (radius * score/100) * np.sin(angle)) 
                    for (_, score, _), angle in zip(dimensions, angles)]
        score_polygon = plt.Polygon(score_points, fill=True, facecolor=grade_color, alpha=0.3, 
                                edgecolor=grade_color, linewidth=3)
        ax.add_patch(score_polygon)
        
        # Dimension labels and scores
        for i, ((dim_name, score, method), angle) in enumerate(zip(dimensions, angles)):
            # Label position (outside pentagon)
            label_radius = radius + 0.1
            label_x = center_x + label_radius * np.cos(angle)
            label_y = center_y + label_radius * np.sin(angle)
            
            # Dimension name and score
            ax.text(label_x, label_y, f"{dim_name}\n{score:.0f}%", ha='center', va='center',
                fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Line from center to dimension point
            dim_x = center_x + (radius * score/100) * np.cos(angle)
            dim_y = center_y + (radius * score/100) * np.sin(angle)
            ax.plot([center_x, dim_x], [center_y, dim_y], color=grade_color, linewidth=2)
            
            # Score point
            circle = Circle((dim_x, dim_y), 0.02, facecolor=grade_color, edgecolor='white', linewidth=2)
            ax.add_patch(circle)
        
        # Issues summary
        issues_y = 0.05
        severity_counts = metrics.severity_breakdown
        issues_text = f"Issues Found: {metrics.total_issues} total"
        if metrics.total_issues > 0:
            issue_details = []
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    issue_details.append(f"{count} {severity}")
            issues_text += f" ({', '.join(issue_details)})"
        
        ax.text(0.02, issues_y, issues_text, fontsize=14, 
            color='#c0392b' if metrics.total_issues > 0 else '#27ae60')
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'issues_count': metrics.total_issues,
            'dimensions': {dim: score for dim, score, _ in dimensions}
        }

    def _layer7_final_decision(self, ax, combined_score: float, quality_results: Dict, match):
        """Layer 7: Final decision with clear action items"""
        ax.set_title("Layer 7: Final Decision & Recommended Actions", fontsize=28, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Decision logic matrix
        match_quality = quality_results['grade']
        issues_count = quality_results['issues_count']
        
        # Determine decision based on both pipelines
        if combined_score >= 0.8 and match_quality in ['EXCELLENT', 'GOOD']:
            decision = "✅ ACCEPT MATCH"
            action = "High confidence match with good quality requirement"
            next_steps = "• Add to traceability matrix\n• Proceed with V&V planning"
            color = '#27ae60'
            confidence = "HIGH"
        elif combined_score >= 0.6 and match_quality in ['EXCELLENT', 'GOOD', 'FAIR']:
            decision = "⚠️ CONDITIONAL ACCEPT"
            action = "Good match but requires engineering review"
            next_steps = "• Engineer review recommended\n• Consider requirement refinement"
            color = '#f39c12'
            confidence = "MEDIUM"
        elif combined_score >= 0.35:
            decision = "🔍 REQUIRES REVIEW"
            action = "Moderate match - detailed analysis needed"
            next_steps = "• Manual review required\n• Consider alternative activities\n• May need bridge requirement"
            color = '#e67e22'
            confidence = "LOW"
        else:
            decision = "❌ ORPHAN REQUIREMENT"
            action = "No suitable activity found"
            next_steps = "• Create bridge requirement\n• Design new V&V activity\n• Gap analysis needed"
            color = '#c0392b'
            confidence = "VERY LOW"
        
        # Main decision box
        decision_width = 0.6
        decision_height = 0.4
        decision_x = 0.2
        decision_y = 0.45
        
        decision_box = FancyBboxPatch((decision_x, decision_y), decision_width, decision_height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, alpha=0.2,
                                    edgecolor=color, linewidth=4)
        ax.add_patch(decision_box)
        
        # Decision text
        ax.text(decision_x + decision_width/2, decision_y + decision_height - 0.08, decision, 
            ha='center', va='center', fontsize=24, fontweight='bold', color=color)
        
        ax.text(decision_x + decision_width/2, decision_y + decision_height - 0.18, action, 
            ha='center', va='center', fontsize=16, color='black')
        
        # Confidence indicator
        ax.text(decision_x + decision_width/2, decision_y + decision_height - 0.28, 
            f"Confidence: {confidence}", ha='center', va='center', 
            fontsize=14, fontweight='bold', color=color)
        
        # Next steps
        ax.text(decision_x + 0.02, decision_y + 0.15, "Next Steps:", 
            fontweight='bold', fontsize=14)
        ax.text(decision_x + 0.02, decision_y + 0.05, next_steps, 
            fontsize=12, color='black')
        
        # Supporting metrics display
        metrics_y = 0.25
        ax.text(0.02, metrics_y, "Decision Factors:", fontweight='bold', fontsize=16)
        
        # Match score with visual bar
        bar_width = 0.3
        bar_height = 0.03
        bar_x = 0.25
        
        # Match score bar
        ax.text(0.02, metrics_y - 0.08, f"Match Score: {combined_score:.3f}", fontsize=14, fontweight='bold')
        ax.add_patch(Rectangle((bar_x, metrics_y - 0.1), bar_width, bar_height,
                            facecolor='lightgray', edgecolor='black'))
        ax.add_patch(Rectangle((bar_x, metrics_y - 0.1), bar_width * combined_score, bar_height,
                            facecolor=color))
        
        # Quality grade
        grade_color = '#27ae60' if match_quality in ['EXCELLENT', 'GOOD'] else '#f39c12' if match_quality == 'FAIR' else '#c0392b'
        ax.text(0.02, metrics_y - 0.16, f"Quality Grade: {match_quality}", fontsize=14, fontweight='bold', color=grade_color)
        
        # Issues count
        issues_color = '#27ae60' if issues_count == 0 else '#f39c12' if issues_count <= 2 else '#c0392b'
        ax.text(0.02, metrics_y - 0.24, f"Quality Issues: {issues_count}", fontsize=14, fontweight='bold', color=issues_color)
        
        # Process outcome
        outcome_y = 0.05
        if decision.startswith("✅"):
            outcome_text = "🎯 TRACEABILITY ESTABLISHED - Requirement successfully linked to V&V activity"
            outcome_color = '#27ae60'
        elif decision.startswith("⚠️") or decision.startswith("🔍"):
            outcome_text = "🔄 HUMAN REVIEW REQUIRED - Algorithm provides guidance for engineer decision"
            outcome_color = '#f39c12'
        else:
            outcome_text = "🔧 GAP IDENTIFIED - New V&V activity may be needed"
            outcome_color = '#c0392b'
        
        ax.text(0.5, outcome_y, outcome_text, ha='center', fontsize=14, fontweight='bold', 
            color=outcome_color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=outcome_color, alpha=0.1))       
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value"""
        if score >= 0.7:
            return self.colors['good']
        elif score >= 0.4:
            return self.colors['warning']
        else:
            return self.colors['bad']
    def create_algorithm_deep_dive(self, requirement_id: str = None) -> str:
        """Create detailed algorithm analysis for a specific requirement with enhanced error handling"""
        
        try:
            # Load data
            matches_df, explanations, _ = self.load_real_data()
            
            # Select requirement
            if requirement_id:
                matches = matches_df[matches_df['Requirement_ID'] == requirement_id]
            else:
                matches = matches_df.nlargest(5, 'Combined_Score')
            
            if matches.empty:
                print("❌ No matches found for deep dive analysis")
                return ""
            
            # Get first match for detailed analysis
            match = matches.iloc[0]
            req_id = str(match.get('Requirement_ID', 'Unknown'))
            req_text = str(match.get('Requirement_Text', ''))
            activity_name = str(match.get('Activity_Name', ''))
            
            if not req_text.strip() or not activity_name.strip():
                print(f"❌ Invalid text data for deep dive: {req_id}")
                return ""
            
            explanation_key = (req_id, activity_name)
            explanation = explanations.get(explanation_key, {})
            if not isinstance(explanation, dict):
                explanation = {}
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Algorithm Deep Dive Analysis: {req_id}', 
                        fontsize=24, fontweight='bold')
            
            # Subplot 1: Score breakdown with details
            ax1 = axes[0, 0]
            try:
                scores = [
                    float(match.get('Semantic_Score', 0)),
                    float(match.get('BM25_Score', 0)), 
                    float(match.get('Domain_Score', 0)),
                    float(match.get('Query_Expansion_Score', 0))
                ]
            except (ValueError, TypeError):
                scores = [0.0, 0.0, 0.0, 0.0]
            
            labels = ['Semantic', 'BM25', 'Domain', 'Query Exp']
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            bars = ax1.bar(labels, scores, color=colors, alpha=0.7)
            ax1.set_title('Algorithm Score Breakdown', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', fontweight='bold')
            
            # Subplot 2: Term analysis with enhanced explanation parsing
            ax2 = axes[0, 1]
            ax2.axis('off')
            ax2.set_title('Enhanced Term & Domain Analysis', fontsize=16, fontweight='bold')
            
            try:
                # Run preprocessing to get terms with error handling
                req_terms = self.matcher._preprocess_text_aerospace(req_text)
                act_terms = self.matcher._preprocess_text_aerospace(activity_name)
                
                # Ensure terms are strings
                req_terms = [str(term) for term in req_terms if term] if req_terms else []
                act_terms = [str(term) for term in act_terms if term] if act_terms else []
                
                shared_terms = list(set(req_terms) & set(act_terms))
                
                y_pos = 0.95
                ax2.text(0.02, y_pos, f"Requirement terms ({len(req_terms)}):", fontweight='bold', fontsize=11)
                req_display = ", ".join(req_terms[:8]) + ("..." if len(req_terms) > 8 else "")
                ax2.text(0.02, y_pos - 0.08, req_display, fontsize=9, wrap=True)
                
                y_pos -= 0.18
                ax2.text(0.02, y_pos, f"Activity terms ({len(act_terms)}):", fontweight='bold', fontsize=11)
                act_display = ", ".join(act_terms[:8]) + ("..." if len(act_terms) > 8 else "")
                ax2.text(0.02, y_pos - 0.08, act_display, fontsize=9, wrap=True)
                
                y_pos -= 0.18
                ax2.text(0.02, y_pos, f"Shared terms ({len(shared_terms)}):", fontweight='bold', fontsize=11, color='green')
                shared_display = ", ".join(shared_terms[:6]) + ("..." if len(shared_terms) > 6 else "")
                ax2.text(0.02, y_pos - 0.08, shared_display, fontsize=9, wrap=True, color='green')
                
                # Enhanced domain analysis from structured explanations
                y_pos -= 0.18
                ax2.text(0.02, y_pos, "Domain Analysis:", fontweight='bold', fontsize=11, color='blue')
                
                # Get domain explanation info
                domain_exp = explanation.get('explanations', {}).get('domain', {})
                domain_info = self._extract_domain_explanation_info(domain_exp)
                
                if domain_info['has_aerospace_match']:
                    aero_terms = domain_info['aerospace_terms'][:4]  # Show first 4
                    ax2.text(0.02, y_pos - 0.08, f"• Aerospace: {', '.join(aero_terms)}", fontsize=9, color='blue')
                    y_pos -= 0.08
                
                if domain_info['has_indicators']:
                    indicators = list(domain_info['key_indicators'].keys())[:3]
                    ax2.text(0.02, y_pos - 0.08, f"• Indicators: {', '.join(indicators)}", fontsize=9, color='purple')
                    y_pos -= 0.08
                
                if domain_info['has_relationships']:
                    rel_count = len(domain_info['learned_relationships'])
                    ax2.text(0.02, y_pos - 0.08, f"• Learned patterns: {rel_count}", fontsize=9, color='orange')
                    y_pos -= 0.08
                
                if domain_info['multi_evidence_bonus'] > 0:
                    ax2.text(0.02, y_pos - 0.08, f"• Multi-evidence bonus: +{domain_info['multi_evidence_bonus']:.2f}", 
                            fontsize=9, color='red', fontweight='bold')
                
            except Exception as e:
                print(f"⚠️ Enhanced term analysis error: {e}")
                ax2.text(0.5, 0.5, f"Term analysis failed: {str(e)[:50]}...", 
                        ha='center', va='center', fontsize=12, color='red')
            
            # Subplot 3: Quality analysis
            ax3 = axes[1, 0]
            ax3.axis('off')
            ax3.set_title('Quality Analysis', fontsize=16, fontweight='bold')
            
            try:
                # Run quality analysis
                issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)
                
                quality_scores = [
                    ('Clarity', float(metrics.clarity_score) if metrics.clarity_score is not None else 0.0),
                    ('Completeness', float(metrics.completeness_score) if metrics.completeness_score is not None else 0.0),
                    ('Verifiability', float(metrics.verifiability_score) if metrics.verifiability_score is not None else 0.0),
                    ('Atomicity', float(metrics.atomicity_score) if metrics.atomicity_score is not None else 0.0),
                    ('Consistency', float(metrics.consistency_score) if metrics.consistency_score is not None else 0.0)
                ]
                
                y_pos = 0.9
                for dim, score in quality_scores:
                    color = '#27ae60' if score >= 80 else '#f39c12' if score >= 60 else '#c0392b'
                    ax3.text(0.02, y_pos, f"{dim}: {score:.0f}%", fontsize=12, fontweight='bold', color=color)
                    y_pos -= 0.15
                
                # Show issues
                y_pos -= 0.1
                issues_count = len(issues) if issues else 0
                ax3.text(0.02, y_pos, f"Issues found: {issues_count}", fontsize=12, fontweight='bold',
                        color='#c0392b' if issues_count > 0 else '#27ae60')
                
            except Exception as e:
                print(f"⚠️ Quality analysis error: {e}")
                ax3.text(0.5, 0.5, f"Quality analysis failed: {str(e)[:50]}...", 
                        ha='center', va='center', fontsize=12, color='red')
            
            # Subplot 4: Decision matrix
            ax4 = axes[1, 1]
            ax4.axis('off')
            ax4.set_title('Decision Matrix', fontsize=16, fontweight='bold')
            
            try:
                # Create decision matrix visualization
                combined_score = float(match.get('Combined_Score', 0))
                overall_quality = sum([score for _, score in quality_scores]) / len(quality_scores) if quality_scores else 0.0
                
                # Plot point on decision matrix
                ax4.scatter(combined_score, overall_quality, s=200, c='red', alpha=0.7, edgecolors='black', linewidth=2)
                
                # Add decision regions
                ax4.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High Quality')
                ax4.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Medium Quality')
                ax4.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='High Match')
                ax4.axvline(x=0.35, color='orange', linestyle='--', alpha=0.5, label='Min Match')
                
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 100)
                ax4.set_xlabel('Match Score')
                ax4.set_ylabel('Quality Score')
                ax4.grid(True, alpha=0.3)
                
                # Add decision labels
                ax4.text(0.9, 85, 'ACCEPT', ha='center', fontweight='bold', color='green', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
                ax4.text(0.6, 85, 'REVIEW', ha='center', fontweight='bold', color='orange', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
                ax4.text(0.2, 30, 'ORPHAN', ha='center', fontweight='bold', color='red', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
                
            except Exception as e:
                print(f"⚠️ Decision matrix error: {e}")
                ax4.text(0.5, 0.5, f"Decision matrix failed: {str(e)[:50]}...", 
                        ha='center', va='center', fontsize=12, color='red')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{req_id}_algorithm_deep_dive_{timestamp}.png"
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            output_path = self.output_dir / filename
            
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            print(f"✅ Saved algorithm deep dive: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Algorithm deep dive failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

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
        """Create enhanced visualizations with error handling and progress tracking"""
        
        print("\n" + "="*80)
        print("🎨 CREATING ENHANCED TECHNICAL JOURNEY VISUALIZATIONS")
        print("="*80)
        
        paths = []
        
        try:
            # Load matches  
            matches_df, explanations, _ = self.load_real_data()
            if matches_df.empty:
                print("❌ No matches found - run matcher.py first")
                return paths
            
            print(f"📊 Found {len(matches_df)} matches to analyze")
            
            # Validate data integrity
            print("🔍 Validating data integrity...")
            valid_matches = []
            for idx, match in matches_df.iterrows():
                req_id = match.get('Requirement_ID', '')
                req_text = match.get('Requirement_Text', '')
                activity_name = match.get('Activity_Name', '')
                combined_score = match.get('Combined_Score', 0)
                
                # Convert to strings and check for valid data
                req_id = str(req_id) if req_id is not None else f"REQ_{idx}"
                req_text = str(req_text) if req_text is not None and not pd.isna(req_text) else ""
                activity_name = str(activity_name) if activity_name is not None and not pd.isna(activity_name) else ""
                
                try:
                    combined_score = float(combined_score) if combined_score is not None else 0.0
                except (ValueError, TypeError):
                    combined_score = 0.0
                
                # Only include matches with valid text data
                if req_text.strip() and activity_name.strip() and len(req_text) > 10:
                    valid_matches.append({
                        'index': idx,
                        'req_id': req_id,
                        'req_text': req_text,
                        'activity_name': activity_name,
                        'combined_score': combined_score,
                        'match': match
                    })
            
            if not valid_matches:
                print("❌ No valid matches found with sufficient text data")
                return paths
            
            print(f"✅ Found {len(valid_matches)} valid matches")
            
            # Sort by score for selection
            valid_matches.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Create visualizations for different score ranges
            visualization_targets = []
            
            # Highest scoring pair
            highest_match = valid_matches[0]
            visualization_targets.append(('HIGHEST', highest_match))
            
            # Medium scoring pair (if exists)
            medium_candidates = [m for m in valid_matches if 0.4 <= m['combined_score'] < 0.8]
            if medium_candidates:
                medium_match = medium_candidates[len(medium_candidates)//2]  # Pick middle one
                visualization_targets.append(('MEDIUM', medium_match))
            
            # Lowest scoring pair (above threshold)
            low_candidates = [m for m in valid_matches if m['combined_score'] >= 0.35]
            if low_candidates and len(low_candidates) > 1:
                lowest_match = low_candidates[-1]  # Last one (lowest score)
                visualization_targets.append(('LOWEST', lowest_match))
            
            # Create visualizations
            for i, (label, match_data) in enumerate(visualization_targets, 1):
                req_id = match_data['req_id']
                print(f"\n📊 [{i}/{len(visualization_targets)}] Creating {label} scoring visualization: {req_id}")
                print(f"   Score: {match_data['combined_score']:.3f}")
                print(f"   Requirement: {match_data['req_text'][:60]}...")
                print(f"   Activity: {match_data['activity_name'][:60]}...")
                
                try:
                    path = self.create_processing_journey(requirement_id=req_id, label=label)
                    if path:
                        paths.append(path)
                        print(f"   ✅ Success: {Path(path).name}")
                    else:
                        print(f"   ❌ Failed to create visualization")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Create algorithm deep dive for best match (if we have successful visualizations)
            if paths and valid_matches:
                print(f"\n📊 Creating algorithm deep dive analysis...")
                try:
                    deep_dive_path = self.create_algorithm_deep_dive(highest_match['req_id'])
                    if deep_dive_path:
                        paths.append(deep_dive_path)
                        print(f"   ✅ Success: {Path(deep_dive_path).name}")
                except Exception as e:
                    print(f"   ❌ Deep dive failed: {e}")
            self.create_algorithm_contribution_chart()
        except Exception as e:
            print(f"❌ Critical error in visualization creation: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print(f"✅ VISUALIZATION COMPLETE")
        print(f"📁 Created {len(paths)} visualizations")
        print(f"💾 Saved to: {self.output_dir}")
        
        if paths:
            print(f"\n📋 Generated Files:")
            for path in paths:
                try:
                    file_size = Path(path).stat().st_size / (1024*1024)  # MB
                    print(f"   • {Path(path).name} ({file_size:.1f} MB)")
                except:
                    print(f"   • {Path(path).name}")
        else:
            print(f"\n⚠️ No visualizations were created successfully")
            print(f"💡 Troubleshooting steps:")
            print(f"   1. Check that matcher.py has been run and produced results")
            print(f"   2. Verify aerospace_matches.csv exists in outputs/matching_results/")
            print(f"   3. Check that requirements have valid text content")
            print(f"   4. Review error messages above for specific issues")
        
        print("="*80)
        
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