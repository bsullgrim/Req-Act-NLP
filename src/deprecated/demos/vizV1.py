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
            'domain': '#2ecc71',        # Green
            'query_expansion': '#f39c12', # Orange
            'good': '#27ae60',           # Dark green
            'warning': '#f39c12',        # Orange
            'bad': '#c0392b',           # Dark red
            'neutral': '#95a5a6',        # Gray
        }
        
        print(f"✅ Technical Journey Visualizer initialized")
    
    def load_real_data(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Load real matching results, explanations, and quality analysis"""
        
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
        
        # Load quality analysis results (from reqGrading output)
        quality_path = self.repo_manager.structure['quality_analysis'] / "requirements_quality_report.csv"
        quality_df = None
        if quality_path.exists():
            quality_df = pd.read_csv(quality_path)
            print(f"✅ Loaded quality analysis from {quality_path}")
        else:
            print(f"⚠️  Quality analysis not found at {quality_path}")
            print("   Run reqGrading.py first, or quality will be calculated on-demand")
        
        # Merge quality data into matches if available
        if quality_df is not None:
            # Merge quality metrics into matches by Requirement_ID
            quality_subset = quality_df[['ID', 'Total_Issues', 'Quality_Score', 'Quality_Grade', 
                                        'INCOSE_Best_Pattern', 'INCOSE_Compliance_Score']]
            quality_subset = quality_subset.rename(columns={'ID': 'Requirement_ID'})
            matches_df = matches_df.merge(quality_subset, on='Requirement_ID', how='left')
        
        # Load original requirements for fallback
        req_path = self.repo_manager.structure['data_raw'] / "requirements.csv"
        requirements_df = None
        if req_path.exists():
            requirements_df = self.file_handler.safe_read_csv(str(req_path))
        
        return matches_df, explanations, requirements_df    
    
    def add_text_output(self, layer_name: str, content: Dict):
        """Add text output for a layer"""
        self.text_output.append({
            'layer': layer_name,
            'timestamp': datetime.now().isoformat(),
            'content': content
        })
        self.layer_data[layer_name] = content
    
    def save_text_outputs(self, req_id: str):
        """Save text outputs to CSV and JSON for PowerPoint creation"""
        # Save as JSON for structured data
        json_path = self.output_dir / f"{req_id}_layer_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_data, f, indent=2)
        
        # Save as CSV for tabular view
        csv_path = self.output_dir / f"{req_id}_layer_summary.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Component', 'Value', 'Details'])
            
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
        
        print(f"📄 Saved text outputs:")
        print(f"   - JSON: {json_path}")
        print(f"   - CSV: {csv_path}")
        print(f"   - TXT: {txt_path}")

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
        
        match = match.iloc[0]
        req_id = match['Requirement_ID']
        req_text = match['Requirement_Text']
        activity_name = match['Activity_Name']
        
        print(f"📊 Visualizing journey for: {req_id}")
        print(f"   Requirement: {req_text[:80]}...")
        print(f"   Activity: {activity_name}")
        print(f"   Score: {match['Combined_Score']:.3f}")

        # Clear previous text output
        self.text_output = []
        self.layer_data = {}

        # Get explanation data
        explanation = explanations.get((req_id, activity_name), {})
        
        # Create the visualization
        fig = plt.figure(figsize=(24, 24))
        
        # Title
        title = f'Technical Processing Journey: {req_id} (Score: {match["Combined_Score"]:.3f})'
        if label:
            title = f"{label} | {title}"
        fig.suptitle(title, fontsize=30, fontweight='bold', y=0.93)
        
        # Create grid for layers
        gs = fig.add_gridspec(8, 1, hspace=0.25) 
        
        # LAYER 1: Raw Inputs
        ax1 = fig.add_subplot(gs[0])
        self._layer1_raw_inputs(ax1, req_text, activity_name, req_id)
        
        # LAYER 2: Preprocessing Steps
        ax2 = fig.add_subplot(gs[1])
        preprocessing_data = self._layer2_preprocessing(ax2, req_text, activity_name)
        
        # LAYER 3: Four Algorithm Processing (use JSON explanations)
        ax3 = fig.add_subplot(gs[2:4])
        algorithm_scores = self._layer3_algorithms(ax3, req_text, activity_name, 
                                                preprocessing_data, match, explanation)
        
        # LAYER 4: Score Combination
        ax4 = fig.add_subplot(gs[4])
        combined_score = self._layer4_score_combination(ax4, match, algorithm_scores)
        
        # LAYER 5: INCOSE Pattern Analysis
        ax5 = fig.add_subplot(gs[5])
        incose_results = self._layer5_incose_analysis(ax5, req_text)
        
        # LAYER 6: Quality Dimensions
        ax6 = fig.add_subplot(gs[6])
        quality_results = self._layer6_quality_dimensions(ax6, req_text)
        
        # LAYER 7: Final Decision
        ax7 = fig.add_subplot(gs[7])
        self._layer7_final_decision(ax7, combined_score, quality_results, match)
        
        # Save figure
        output_path = self.output_dir / f"{req_id}_technical_journey.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save text outputs
        self.save_text_outputs(req_id)

        print(f"✅ Saved technical journey: {output_path}")
        return str(output_path)
    
    def _layer1_raw_inputs(self, ax, req_text: str, activity_name: str, req_id: str):
        """Layer 1: Show raw inputs with fully scaled boxes and wrapped text."""


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
        
        # Add to text output
        self.add_text_output("Layer 1: Raw Inputs", {
            'requirement_id': req_id,
            'requirement_text': req_text,
            'activity_name': activity_name,
            'source_files': ['requirements.csv', 'activities.csv']
        })

    def _layer2_preprocessing(self, ax, req_text: str, activity_name: str) -> Dict:
        """Layer 2: Show actual preprocessing steps with improved horizontal layout"""
        ax.set_title("Layer 2: Preprocessing Pipeline", fontsize=24, fontweight='bold', loc='left', pad=20)
        ax.axis('off')
        
        # Run actual preprocessing methods
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        expanded_act = self.matcher._expand_aerospace_abbreviations(activity_name)
        
        req_doc = self.matcher.nlp(expanded_req)
        act_doc = self.matcher.nlp(expanded_act)
        
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        act_terms = self.matcher._preprocess_text_aerospace(activity_name)

        # Improved layout - wider boxes with better spacing
        box_width = 0.24  # Increased from 0.24
        box_height = 0.8
        box_y = 0.13
        x_positions = [0.03, 0.35, 0.67]  # Better spacing
        
        preprocessing_steps = [
            {
                'title': 'Abbreviation Expansion',
                'method': '_expand_aerospace_abbreviations()',
                'color': '#e8f4fd',
                'border': '#3498db'
            },
            {
                'title': 'NLP Tokenization',
                'method': 'nlp() + POS tagging',
                'color': '#fef5e7',
                'border': '#f39c12'
            },
            {
                'title': 'Term Extraction',
                'method': '_preprocess_text_aerospace()',
                'color': '#e8f6f3',
                'border': '#27ae60'
            }
        ]
        
        layer2_data = {}
        
        for i, (x_pos, step) in enumerate(zip(x_positions, preprocessing_steps)):
            # Draw box
            box = FancyBboxPatch((x_pos, box_y), box_width, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor=step['color'], 
                                edgecolor=step['border'], 
                                linewidth=2)
            ax.add_patch(box)
            
            # Title and method name
            ax.text(x_pos + box_width/2, box_y + box_height + 0.05, 
                step['title'], ha='center', fontsize=20, fontweight='bold')
            ax.text(x_pos + box_width/2, box_y + box_height - 0.08, 
                step['method'], ha='center', fontsize=14, style='italic', color='#555')
            
            # Content based on step
            content_y = box_y + box_height - 0.18
            
            if i == 0:  # Abbreviations - Two column layout
                abbrevs_found = []
                for abbrev in self.domain.abbreviations:
                    if abbrev in req_text.lower() or abbrev in activity_name.lower():
                        expansion = self.domain.abbreviations[abbrev]
                        abbrevs_found.append((abbrev, expansion))
                
                if abbrevs_found:
                    ax.text(x_pos + 0.01, content_y, "Found:", fontsize=18, fontweight='bold')
                    
                    # Two column layout for better space usage
                    col_width = (box_width - 0.02) / 2
                    max_items = min(6, len(abbrevs_found))
                    
                    for j, (abbrev, expansion) in enumerate(abbrevs_found[:max_items]):
                        col = j % 2
                        row = j // 2
                        x_offset = x_pos + 0.01 + col * col_width
                        y_offset = content_y - 0.16 - row * 0.09
                        
                        # Truncate long expansions
                        short_exp = expansion[:28] + '...' if len(expansion) > 28 else expansion
                        ax.text(x_offset, y_offset, f"• {abbrev.upper()}", 
                            fontsize=18, fontweight='bold', color=step['border'])
                        ax.text(x_offset, y_offset - 0.04, f"  → {short_exp}", 
                            fontsize=8, color='#333')
                    
                    if len(abbrevs_found) > max_items:
                        ax.text(x_pos + box_width/2, box_y + 0.03, 
                            f"+ {len(abbrevs_found) - max_items} more", 
                            ha='center', fontsize=16, style='italic', color='gray')
                    
                    layer2_data['abbreviations_expanded'] = abbrevs_found
                else:
                    ax.text(x_pos + box_width/2, content_y - 0.15, "No abbreviations found", 
                        ha='center', fontsize=18, style='italic', color='#888')
                    layer2_data['abbreviations_expanded'] = []
            
            elif i == 1:  # Tokenization - Show stats with sample
                tokens = [token.text for token in req_doc]
                pos_counts = {}
                for token in req_doc:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                
                # Show summary stats
                ax.text(x_pos + 0.01, content_y, "Statistics:", fontsize=18, fontweight='bold')
                ax.text(x_pos + 0.01, content_y - 0.12, f"• Total tokens: {len(tokens)}", 
                    fontsize=14)
                ax.text(x_pos + 0.01, content_y - 0.24, f"• Unique: {len(set(tokens))}", 
                    fontsize=14)
                
                # Show top POS tags
                sorted_pos = sorted(pos_counts.items(), key=lambda x: x[-1], reverse=True)[:4]
                ax.text(x_pos + 0.12, content_y, "Top POS tags:", 
                    fontsize=18, fontweight='bold')
                for j, (pos, count) in enumerate(sorted_pos):
                    ax.text(x_pos + 0.01, content_y - 0.34 - j*0.07, 
                        f"• {pos}: {count}", fontsize=14)
                
                # Sample tokens at bottom
                sample_tokens = tokens[:12]
                sample_text = ', '.join(sample_tokens)
                if len(sample_text) > 35:
                    sample_text = sample_text[:32] + '...'
                ax.text(x_pos + box_width/2, box_y + 0.05, 
                    f"Sample: {sample_text}", 
                    ha='center', fontsize=8, style='italic', color='#555',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                layer2_data['tokenization'] = {
                    'tokens': tokens,
                    'pos_counts': pos_counts,
                    'total_tokens': len(tokens),
                    'unique_tokens': len(set(tokens))
                }
            
            else:  # Term extraction - Side by side comparison
                # Left side: Requirement terms
                left_x = x_pos + 0.01
                right_x = x_pos + box_width/2 + 0.01
                
                ax.text(left_x, content_y, f"Requirement ({len(req_terms)}):", 
                    fontsize=10, fontweight='bold', color=step['border'])
                for j, term in enumerate(req_terms[:5]):
                    ax.text(left_x, content_y - 0.08 - j*0.07, 
                        f"• {term}", fontsize=8)
                if len(req_terms) > 5:
                    ax.text(left_x, content_y - 0.08 - 5*0.07, 
                        f"  + {len(req_terms)-5} more", fontsize=7, style='italic')
                
                # Right side: Activity terms
                ax.text(right_x, content_y, f"Activity ({len(act_terms)}):", 
                    fontsize=10, fontweight='bold', color=step['border'])
                for j, term in enumerate(act_terms[:5]):
                    ax.text(right_x, content_y - 0.08 - j*0.07, 
                        f"• {term}", fontsize=8)
                if len(act_terms) > 5:
                    ax.text(right_x, content_y - 0.08 - 5*0.07, 
                        f"  + {len(act_terms)-5} more", fontsize=7, style='italic')
                
                # Highlight aerospace terms at bottom
                aero_terms = [t for t in set(req_terms + act_terms) if t in self.matcher.all_aerospace_terms]
                if aero_terms:
                    ax.text(x_pos + box_width/2, box_y + 0.05, 
                        f"🚀 {len(aero_terms)} domain-specific terms", 
                        ha='center', fontsize=9, color=step['border'], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                layer2_data['extracted_terms'] = {
                    'requirement_terms': req_terms,
                    'activity_terms': act_terms,
                    'aerospace_terms': aero_terms
                }
        
        # Add to text output
        self.add_text_output("Layer 2: Preprocessing", layer2_data)
        
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
        """Layer 3: Four Algorithm Analysis with meta explanations and improved query expansion layout"""
        ax.set_title("Layer 3: Four Algorithm Analysis", 
                    fontsize=24, fontweight='bold', loc='left', pad=15)
        ax.axis('off')

        # Get scores from the new JSON structure
        scores_obj = explanation.get('scores', {})
        scores = {
            'semantic': scores_obj.get('semantic', 0),
            'bm25': scores_obj.get('bm25', 0),
            'domain': scores_obj.get('domain', 0),
            'query_expansion': scores_obj.get('query_expansion', 0)
        }
        
        explanations = explanation.get('explanations', {})
        
        # Improved layout - better use of horizontal space
        x_positions = [0.015, 0.255, 0.495, 0.735]
        box_width = 0.22
        box_height = 0.82
        box_y = 0.08
        
        # Meta explanations for each algorithm
        algo_meta = {
            'semantic': 'Uses sentence embeddings to measure conceptual similarity between texts',
            'bm25': 'Term-frequency ranking algorithm that finds shared keywords',
            'domain': 'Leverages aerospace vocabulary and learned requirement-activity patterns',
            'query_expansion': 'Expands activity terms with synonyms to find matches with requirement terms'
        }
        
        layer3_data = {}
        
        for algo, x_pos in zip(['semantic', 'bm25', 'domain', 'query_expansion'], x_positions):
            # Algorithm box
            algo_box = FancyBboxPatch((x_pos, box_y), box_width, box_height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=self.colors[algo], 
                                    alpha=0.15,
                                    edgecolor=self.colors[algo], 
                                    linewidth=2)
            ax.add_patch(algo_box)
            
            # Algorithm header
            algo_names = {
                'semantic': 'Semantic',
                'bm25': 'BM25',
                'domain': 'Domain',
                'query_expansion': 'Query Exp.'
            }
            
            # Title and score (more compact)
            ax.text(x_pos + box_width/2, box_y + box_height - 0.03, 
                    algo_names[algo], ha='center', fontsize=16, fontweight='bold')
            
            score = scores[algo]
            score_color = self._get_score_color(score)
            ax.text(x_pos + box_width/2, box_y + box_height - 0.12, 
                    f"{score:.3f}", ha='center', fontsize=22, 
                    fontweight='bold', color=score_color)
            
            # Score bar (more compact)
            bar_y = box_y + box_height - 0.19
            bar_height = 0.025
            ax.add_patch(Rectangle((x_pos + 0.01, bar_y), box_width - 0.02, bar_height,
                                facecolor='lightgray', edgecolor='gray', linewidth=0.5))
            ax.add_patch(Rectangle((x_pos + 0.01, bar_y), (box_width - 0.02) * score, bar_height,
                                facecolor=score_color, alpha=0.7))
            
            # Meta explanation box (light background)
            meta_y = bar_y - 0.04
            meta_height = 0.09
            meta_box = FancyBboxPatch((x_pos + 0.01, meta_y - meta_height), 
                                    box_width - 0.02, meta_height,
                                    boxstyle="round,pad=0.005",
                                    facecolor='white', 
                                    alpha=0.6,
                                    edgecolor=self.colors[algo], 
                                    linewidth=0.5,
                                    linestyle='dashed')
            ax.add_patch(meta_box)
            
            # Wrap meta text
            meta_text = algo_meta[algo]
            wrapped_meta = textwrap.wrap(meta_text, width=28)
            meta_y_text = meta_y - 0.015
            for i, line in enumerate(wrapped_meta[:3]):
                ax.text(x_pos + box_width/2, meta_y_text - i*0.028, line, 
                    ha='center', fontsize=7, style='italic', color='#444')
            
            # Algorithm-specific details (properly contained)
            detail_y = meta_y - meta_height - 0.04
            content_x = x_pos + 0.01
            content_width = box_width - 0.02
            
            if algo == 'semantic':
                exp_text = explanations.get('semantic', '')
                if isinstance(exp_text, str):
                    # Show key insight
                    ax.text(content_x, detail_y, "Analysis:", fontsize=9, fontweight='bold')
                    detail_y -= 0.05
                    
                    # Wrap text to fit width
                    wrapped_lines = []
                    for line in exp_text.split('\n'):
                        wrapped_lines.extend(textwrap.wrap(line, width=28))
                    
                    # Show as many lines as fit
                    max_lines = 9
                    for i, line in enumerate(wrapped_lines[:max_lines]):
                        ax.text(content_x, detail_y - i*0.045, line, fontsize=8)
                    
                    if len(wrapped_lines) > max_lines:
                        ax.text(x_pos + box_width/2, box_y + 0.02, 
                            f"[+{len(wrapped_lines)-max_lines} more lines]", 
                            ha='center', fontsize=7, style='italic', color='gray')
                
                layer3_data['semantic'] = {'score': score, 'explanation': exp_text}
                
            elif algo == 'bm25':
                exp_text = explanations.get('bm25', '')
                shared_terms = explanation.get('shared_terms', [])
                
                # Show matched terms in compact format
                if shared_terms:
                    ax.text(content_x, detail_y, f"Matched terms ({len(shared_terms)}):", 
                        fontsize=9, fontweight='bold')
                    detail_y -= 0.05
                    
                    # Show terms in compact rows (2 per row if short)
                    row_terms = []
                    current_row = []
                    for term in shared_terms[:10]:
                        if len(term) > 12:  # Long term gets own row
                            if current_row:
                                row_terms.append(', '.join(current_row))
                                current_row = []
                            row_terms.append(term)
                        else:
                            current_row.append(term)
                            if len(', '.join(current_row)) > 24:
                                row_terms.append(', '.join(current_row[:-1]))
                                current_row = [term]
                    
                    if current_row:
                        row_terms.append(', '.join(current_row))
                    
                    for i, row in enumerate(row_terms[:7]):
                        ax.text(content_x, detail_y - i*0.045, f"• {row}", fontsize=7.5)
                    
                    detail_y -= len(row_terms[:7]) * 0.045 + 0.03
                
                # BM25 explanation (compact)
                if exp_text:
                    ax.text(content_x, detail_y, "Why it matters:", fontsize=8, fontweight='bold')
                    detail_y -= 0.045
                    wrapped = textwrap.wrap(exp_text, width=28)
                    for line in wrapped[:3]:
                        ax.text(content_x, detail_y, line, fontsize=7)
                        detail_y -= 0.04
                
                layer3_data['bm25'] = {
                    'score': score,
                    'explanation': exp_text,
                    'shared_terms': shared_terms
                }
                
            elif algo == 'domain':
                domain_exp = explanations.get('domain', {})
                if isinstance(domain_exp, dict):
                    # Aerospace terms (compact, 2 columns if many)
                    aero_terms = domain_exp.get('aerospace_terms', [])
                    if aero_terms:
                        ax.text(content_x, detail_y, f"Aerospace terms ({len(aero_terms)}):", 
                            fontsize=9, fontweight='bold')
                        detail_y -= 0.05
                        
                        # Show in 2 columns if more than 4
                        if len(aero_terms) <= 4:
                            for term in aero_terms[:4]:
                                ax.text(content_x, detail_y, f"• {term}", fontsize=7.5)
                                detail_y -= 0.04
                        else:
                            col_width = content_width / 2
                            for i, term in enumerate(aero_terms[:8]):
                                col = i % 2
                                row = i // 2
                                x_off = content_x + col * col_width
                                y_off = detail_y - row * 0.04
                                ax.text(x_off, y_off, f"• {term[:11]}", fontsize=7)
                            detail_y -= (min(4, (len(aero_terms[:8]) + 1) // 2)) * 0.04
                        
                        detail_y -= 0.03
                    
                    # Key indicators (compact)
                    indicators = domain_exp.get('key_indicators', {})
                    if indicators:
                        ax.text(content_x, detail_y, "Key indicators:", fontsize=8, fontweight='bold')
                        detail_y -= 0.045
                        for term, data in list(indicators.items())[:3]:
                            weight = data.get('weight', 0)
                            short_term = term[:14] + '...' if len(term) > 14 else term
                            ax.text(content_x, detail_y, f"• {short_term} (w={weight:.2f})", fontsize=7)
                            detail_y -= 0.04
                        detail_y -= 0.02
                    
                    # Learned relationships (very compact)
                    learned = domain_exp.get('learned_relationships', {})
                    if learned and detail_y > box_y + 0.06:
                        ax.text(content_x, detail_y, "Learned patterns:", fontsize=8, fontweight='bold')
                        detail_y -= 0.04
                        for req_term, acts in list(learned.items())[:2]:
                            short_req = req_term[:10]
                            short_acts = ', '.join(acts[:2])[:14]
                            ax.text(content_x, detail_y, f"• {short_req}→{short_acts}", fontsize=6.5)
                            detail_y -= 0.035
                    
                    # Multi-evidence bonus
                    bonus = domain_exp.get('multi_evidence_bonus', 0)
                    if bonus > 0:
                        ax.text(x_pos + box_width/2, box_y + 0.02, 
                            f"Multi-evidence: +{bonus:.2f}", 
                            ha='center', fontsize=7.5, color=self.colors['good'],
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    
                    layer3_data['domain'] = {
                        'score': score,
                        'aerospace_terms': aero_terms,
                        'indicators': indicators,
                        'learned_relationships': learned,
                        'bonus': bonus
                    }
                    
            elif algo == 'query_expansion':
                qe = explanations.get('query_expansion', '')
                if isinstance(qe, dict):
                    req_terms = qe.get('requirement_terms', [])
                    matched = qe.get('matched_terms', [])
                    expanded = qe.get('expanded_activity_terms', [])

                    # Three column layout: Req | Matched | Activity
                    col1_x = content_x
                    col2_x = content_x + content_width * 0.33
                    col3_x = content_x + content_width * 0.66
                    col_width = content_width * 0.32
                    
                    # Column headers
                    ax.text(col1_x + col_width/2, detail_y, "Req", 
                        ha='center', fontsize=8, fontweight='bold', color=self.colors[algo])
                    ax.text(col2_x + col_width/2, detail_y, "Match", 
                        ha='center', fontsize=8, fontweight='bold', color='green')
                    ax.text(col3_x + col_width/2, detail_y, "Activity", 
                        ha='center', fontsize=8, fontweight='bold', color=self.colors[algo])
                    
                    # Draw separator lines
                    detail_y -= 0.04
                    sep_y = detail_y
                    ax.plot([col2_x - 0.005, col2_x - 0.005], [sep_y, box_y + 0.03], 
                        color='lightgray', linewidth=0.5, linestyle='--')
                    ax.plot([col3_x - 0.005, col3_x - 0.005], [sep_y, box_y + 0.03], 
                        color='lightgray', linewidth=0.5, linestyle='--')
                    
                    detail_y -= 0.02
                    
                    # Determine max rows we can show
                    max_rows = 10
                    
                    # Column 1: Requirement terms
                    for i, term in enumerate(req_terms[:max_rows]):
                        short_term = term[:9] + '..' if len(term) > 9 else term
                        ax.text(col1_x, detail_y - i*0.04, short_term, fontsize=7)
                    
                    if len(req_terms) > max_rows:
                        ax.text(col1_x, detail_y - max_rows*0.04, f"+{len(req_terms)-max_rows}", 
                            fontsize=6, style='italic', color='gray')
                    
                    # Column 2: Matched terms (with checkmarks)
                    for i, term in enumerate(matched[:max_rows]):
                        short_term = term[:9] + '..' if len(term) > 9 else term
                        ax.text(col2_x, detail_y - i*0.04, f"✓ {short_term}", 
                            fontsize=7, color='green', fontweight='bold')
                    
                    if len(matched) > max_rows:
                        ax.text(col2_x, detail_y - max_rows*0.04, f"+{len(matched)-max_rows}", 
                            fontsize=6, style='italic', color='gray')
                    
                    # Column 3: Expanded activity terms
                    for i, term in enumerate(expanded[:max_rows]):
                        short_term = term[:9] + '..' if len(term) > 9 else term
                        ax.text(col3_x, detail_y - i*0.04, short_term, fontsize=7)
                    
                    if len(expanded) > max_rows:
                        ax.text(col3_x, detail_y - max_rows*0.04, f"+{len(expanded)-max_rows}", 
                            fontsize=6, style='italic', color='gray')
                    
                    # Summary at bottom
                    match_pct = (len(matched) / len(req_terms) * 100) if req_terms else 0
                    ax.text(x_pos + box_width/2, box_y + 0.02, 
                        f"{len(matched)}/{len(req_terms)} matched ({match_pct:.0f}%)", 
                        ha='center', fontsize=7.5, 
                        color='green' if match_pct > 50 else 'orange',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    
                    layer3_data['query_expansion'] = {
                        'score': score,
                        'requirement_terms': req_terms,
                        'matched_terms': matched,
                        'expanded_activity_terms': expanded,
                        'match_percentage': match_pct
                    }

        # Add to text output
        self.add_text_output("Layer 3: Algorithm Analysis", layer3_data)
        
        return scores  
  
    def _layer4_score_combination(self, ax, match, algorithm_scores: Dict) -> float:
        """Layer 4: Score combination - compact and clear visualization"""
        ax.set_title("Layer 4: Score Combination", fontsize=24, fontweight='bold', loc='left', pad=15)
        ax.axis('off')

        weights = {'semantic': 1.0, 'bm25': 1.0, 'domain': 1.0, 'query_expansion': 1.0}
        total_weight = sum(weights.values())
        
        # Combined score from match
        combined = float(match.get('Combined_Score', 0.0))

        # Top section: Formula explanation (left) and result (right)
        formula_y = 0.78
        
        # Formula box
        formula_box = FancyBboxPatch((0.02, formula_y - 0.15), 0.63, 0.13,
                                    boxstyle="round,pad=0.01",
                                    facecolor='#f8f9fa',
                                    edgecolor='#95a5a6',
                                    linewidth=1.5)
        ax.add_patch(formula_box)
        
        ax.text(0.03, formula_y - 0.02, "Weighted Average Formula:", fontsize=11, fontweight='bold')
        
        # Build calculation string in a clearer format
        calc_parts = []
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = float(algorithm_scores.get(algo, 0.0))
            weight = float(weights[algo])
            calc_parts.append(f"({score:.3f} × {weight:.1f})")
        
        calc_line1 = f"Combined = [{calc_parts[0]} + {calc_parts[1]} +"
        calc_line2 = f"            {calc_parts[2]} + {calc_parts[3]}] / {total_weight:.1f}"
        
        ax.text(0.03, formula_y - 0.07, calc_line1, fontsize=9, family='monospace', color='#555')
        ax.text(0.03, formula_y - 0.11, calc_line2, fontsize=9, family='monospace', color='#555')
        
        # Result box (right side)
        result_box = FancyBboxPatch((0.68, formula_y - 0.15), 0.30, 0.13,
                                    boxstyle="round,pad=0.01",
                                    facecolor='#e8f4fd',
                                    edgecolor='#3498db',
                                    linewidth=2)
        ax.add_patch(result_box)
        
        ax.text(0.83, formula_y - 0.03, "Combined Score", ha='center', fontsize=11, fontweight='bold')
        
        # Score with color
        score_color = self._get_score_color(combined)
        ax.text(0.83, formula_y - 0.10, f"{combined:.3f}", ha='center', 
            fontsize=18, fontweight='bold', color=score_color)
        
        # Middle section: Contribution bar with labels
        bar_y = 0.40
        ax.text(0.02, bar_y + 0.16, "Individual Contributions to Combined Score:", 
            fontsize=11, fontweight='bold')
        
        # Legend for the bar (show what each color means)
        legend_y = bar_y + 0.11
        legend_x = 0.02
        algo_display = {
            'semantic': 'Semantic',
            'bm25': 'BM25',
            'domain': 'Domain',
            'query_expansion': 'Query Exp'
        }
        
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            color = self.colors[algo]
            # Small color square
            ax.add_patch(Rectangle((legend_x, legend_y - 0.01), 0.02, 0.02, 
                                facecolor=color, edgecolor=color))
            ax.text(legend_x + 0.03, legend_y, algo_display[algo], 
                fontsize=9, va='center')
            legend_x += 0.13
        
        # Draw main bar
        bar_height = 0.06
        x_start = 0.02
        total_width = 0.96
        
        # Background bar
        ax.add_patch(Rectangle((x_start, bar_y), total_width, bar_height, 
                            facecolor='#ecf0f1', edgecolor='#95a5a6', linewidth=1))
        
        # Individual algorithm contributions
        x_current = x_start
        
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            score = float(algorithm_scores.get(algo, 0.0))
            contribution = (score * weights[algo]) / total_weight
            width = contribution * total_width
            
            # Draw segment
            color = self.colors[algo]
            ax.add_patch(Rectangle((x_current, bar_y), width, bar_height, 
                                facecolor=color, alpha=0.7, edgecolor=color, linewidth=1))
            
            # Label below bar (score and contribution percentage)
            if width > 0.06:  # Only show label if segment is wide enough
                percentage = (contribution / combined * 100) if combined > 0 else 0
                ax.text(x_current + width/2, bar_y - 0.03, 
                    f"{score:.3f}", ha='center', fontsize=9, color=color, fontweight='bold')
                ax.text(x_current + width/2, bar_y - 0.06, 
                    f"({percentage:.0f}%)", ha='center', fontsize=8, color='#666')
            
            x_current += width
        
        # Scale reference below bar
        scale_y = bar_y - 0.11
        ax.text(x_start, scale_y, "0.0", fontsize=8, color='#999')
        ax.text(x_start + total_width/4, scale_y, "0.25", ha='center', fontsize=8, color='#999')
        ax.text(x_start + total_width/2, scale_y, "0.5", ha='center', fontsize=8, color='#999')
        ax.text(x_start + 3*total_width/4, scale_y, "0.75", ha='center', fontsize=8, color='#999')
        ax.text(x_start + total_width, scale_y, "1.0", ha='right', fontsize=8, color='#999')
        
        # Bottom section: Threshold indicators
        threshold_y = 0.15
        
        # Threshold box
        threshold_box = FancyBboxPatch((0.02, threshold_y - 0.05), 0.96, 0.09,
                                    boxstyle="round,pad=0.01",
                                    facecolor='#fffef0',
                                    edgecolor='#f39c12',
                                    linewidth=1.5,
                                    linestyle='--')
        ax.add_patch(threshold_box)
        
        ax.text(0.03, threshold_y + 0.025, "Classification Thresholds:", 
            fontsize=11, fontweight='bold')
        
        # Threshold badges in a row
        threshold_x = 0.30
        thresholds = [
            ("High Match", "≥ 0.80", self.colors['good']),
            ("Medium Match", "0.35 - 0.80", self.colors['warning']),
            ("Orphan", "< 0.35", self.colors['bad'])
        ]
        
        for label, range_text, color in thresholds:
            ax.text(threshold_x, threshold_y + 0.025, "●", 
                fontsize=12, color=color, va='center')
            ax.text(threshold_x + 0.02, threshold_y + 0.025, f"{label}: {range_text}", 
                fontsize=9, fontweight='bold', va='center')
            threshold_x += 0.22
        
        # Indicate which threshold this score meets
        if combined >= 0.80:
            classification = "High Match"
            class_color = self.colors['good']
        elif combined >= 0.35:
            classification = "Medium Match"
            class_color = self.colors['warning']
        else:
            classification = "Orphan"
            class_color = self.colors['bad']
        
        # Classification indicator at bottom
        class_y = threshold_y - 0.04
        ax.text(0.50, class_y, f"→ This match classified as: {classification}", 
            ha='center', fontsize=10, fontweight='bold', color=class_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=class_color, linewidth=2))
        
        # Add to text output
        self.add_text_output("Layer 4: Score Combination", {
            "formula": f"({calc_parts[0]} + {calc_parts[1]} + {calc_parts[2]} + {calc_parts[3]}) / {total_weight}",
            "combined_score": combined,
            "classification": classification,
            "individual_scores": {
                algo: float(algorithm_scores.get(algo, 0.0)) 
                for algo in ['semantic', 'bm25', 'domain', 'query_expansion']
            },
            "contributions": {
                algo: f"{(float(algorithm_scores.get(algo, 0.0)) * weights[algo]) / total_weight:.3f}"
                for algo in ['semantic', 'bm25', 'domain', 'query_expansion']
            }
        })
        
        return combined
    
    def _layer5_incose_analysis(self, ax, req_text: str) -> Dict:
        """Layer 5: INCOSE Component Analysis with pattern template and improved readability."""
        ax.set_title("Layer 5: INCOSE Component Analysis", fontsize=24, fontweight='bold', loc='left', pad=15)
        ax.axis('off')

        analysis = self.quality_analyzer.incose_analyzer.analyze_incose_compliance(req_text)

        # FIXED: best_pattern returns the KEY, not the name
        pattern_key = analysis.best_pattern  # This is 'design', 'functional_performance', etc.
        pattern_def = None
        pattern_name = pattern_key  # Default to key if not found
        
        # Get the pattern definition directly using the key
        if pattern_key in self.quality_analyzer.incose_analyzer.patterns:
            pattern_def = self.quality_analyzer.incose_analyzer.patterns[pattern_key]
            pattern_name = pattern_def.get('name', pattern_key)  # Get the display name
        
        # Top section: Pattern display name and score
        ax.text(0.02, 0.96, f"Pattern: {pattern_name}", 
            fontsize=16, fontweight='bold', color='#2c3e50')
        
        # Draw score box in FAR top right corner
        score_txt = f"{analysis.compliance_score:.2f}" if isinstance(analysis.compliance_score, (int, float)) else str(analysis.compliance_score)
        score_box = FancyBboxPatch((0.88, 0.94), 0.10, 0.1,
                                boxstyle="round,pad=0.005",
                                facecolor='#e8f4fd',
                                edgecolor='#3498db',
                                linewidth=2)
        ax.add_patch(score_box)
        ax.text(0.93, 0.98, "Score", ha='center', fontsize=10, fontweight='bold')
        ax.text(0.93, 0.92, score_txt, ha='center', fontsize=15, fontweight='bold', 
            color=self._get_score_color(float(score_txt) if score_txt.replace('.','').isdigit() else 0))
        
        # Pattern template with placeholders
        template_y = 0.90
        if pattern_def and 'template' in pattern_def:
            template_text = pattern_def['template']
            
            # Calculate height needed
            wrapped_template = textwrap.wrap(template_text, width=105)
            template_height = 0.04 + len(wrapped_template[:3]) * 0.028
            
            # Show template in a box
            template_box = FancyBboxPatch((0.02, template_y - template_height), 0.84, template_height,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#fffef0',
                                        edgecolor='#f39c12',
                                        linewidth=1.5,
                                        linestyle='--')
            ax.add_patch(template_box)
            
            ax.text(0.03, template_y - 0.01, "Pattern Template:", 
                fontsize=11, fontweight='bold', color='#d68910')
            
            # Display template with placeholders
            for i, line in enumerate(wrapped_template[:3]):
                ax.text(0.03, template_y - 0.038 - i*0.028, line, 
                    fontsize=9, family='monospace', color='#555')
        
        # Create legend
        legend_y = 0.79
        ax.text(0.02, legend_y, "Legend:", fontsize=11, fontweight='bold')
        
        # Simple colored circles
        ax.text(0.12, legend_y, "●", fontsize=12, color='#e74c3c', va='center')
        ax.text(0.14, legend_y, "Required", fontsize=10, va='center')
        
        ax.text(0.26, legend_y, "●", fontsize=12, color='#3498db', va='center')
        ax.text(0.28, legend_y, "Optional", fontsize=10, va='center')
        
        # Draw components
        components = analysis.components_found
        all_components = list(components.keys())
        
        # Use pattern definition's required/optional lists EXACTLY
        required_components = set()
        optional_components = set()
        
        if pattern_def:
            # Use the exact lists from the pattern definition
            required_components = set(pattern_def.get('required', []))
            optional_components = set(pattern_def.get('optional', []))
            
            print(f"\nDEBUG Layer 5:")
            print(f"  Pattern key from analysis: '{pattern_key}'")
            print(f"  Pattern name: '{pattern_name}'")
            print(f"  Required from pattern def: {required_components}")
            print(f"  Optional from pattern def: {optional_components}")
        else:
            print(f"ERROR: No pattern definition found for key '{pattern_key}'")
        
        # Layout: 3 columns for better space usage
        cols_per_row = 3
        col_width = 0.326
        start_x = 0.02
        start_y = 0.73
        row_height = 0.063  # Bigger rows
        
        # Sort: required first, then optional; present before missing
        def sort_key(comp):
            val = components[comp]
            is_required = comp in required_components
            is_present = bool(val)
            return (not is_required, not is_present, comp)
        
        sorted_components = sorted(all_components, key=sort_key)
        
        for i, comp in enumerate(sorted_components):
            val = components[comp]
            row = i // cols_per_row
            col = i % cols_per_row
            
            x_pos = start_x + col * col_width
            y_pos = start_y - row * row_height
            
            # Determine component status
            is_required = comp in required_components
            is_present = bool(val)
            
            # Draw component box - BIGGER
            box_width = col_width - 0.008
            box_height = 0.056  # Increased
            
            # Color based on status
            if is_present:
                box_color = '#d5f4e6' if is_required else '#e8f4fd'
                border_color = '#27ae60' if is_required else '#3498db'
            else:
                box_color = '#fadbd8' if is_required else '#f0f0f0'
                border_color = '#e74c3c' if is_required else '#95a5a6'
            
            comp_box = FancyBboxPatch((x_pos, y_pos - box_height), box_width, box_height,
                                    boxstyle="round,pad=0.006",
                                    facecolor=box_color,
                                    edgecolor=border_color,
                                    linewidth=1.5 if is_required else 1)
            ax.add_patch(comp_box)
            
            # Inline format: DOT COMPONENT_NAME: value
            left_x = x_pos + 0.01
            center_y = y_pos - box_height/2
            
            # Colored dot indicator
            dot_color = '#e74c3c' if is_required else '#3498db'
            ax.text(left_x, center_y, "●", fontsize=10, color=dot_color, va='center')
            
            # Component name with colon - BIGGER TEXT
            comp_display = comp.replace('_', ' ').title()
            ax.text(left_x + 0.02, center_y, f"{comp_display}:", 
                fontsize=11, fontweight='bold', va='center', ha='left')
            
            # Component value on same line - BIGGER TEXT
            value_x = left_x + 0.13  # Adjusted for 3 columns
            if is_present:
                val_str = str(val)
                # Truncate if too long for column width
                max_len = 22
                if len(val_str) > max_len:
                    val_str = val_str[:max_len] + "..."
                ax.text(value_x, center_y, val_str, 
                    fontsize=10, color='#2c3e50', va='center', ha='left')
            else:
                ax.text(value_x, center_y, "-- Not found --", 
                    fontsize=9, color='#95a5a6', va='center', ha='left', style='italic')
        
        # Summary statistics at bottom
        summary_y = start_y - ((len(sorted_components) + cols_per_row - 1) // cols_per_row) * row_height - 0.04
        
        # Count based on pattern definition
        total_required = len(required_components)
        present_required = len([c for c in required_components if components.get(c)])
        total_optional = len(optional_components)
        present_optional = len([c for c in optional_components if components.get(c)])
        
        # Summary boxes - BIGGER
        summary_box_y = max(0.02, summary_y)
        
        # Required summary
        req_box = FancyBboxPatch((0.02, summary_box_y), 0.40, 0.07,
                                boxstyle="round,pad=0.01",
                                facecolor='#fadbd8' if present_required < total_required else '#d5f4e6',
                                edgecolor='#e74c3c',
                                linewidth=2)
        ax.add_patch(req_box)
        ax.text(0.22, summary_box_y + 0.035, f"Required: {present_required}/{total_required}", 
            ha='center', fontsize=14, fontweight='bold', va='center')
        
        # Optional summary
        opt_box = FancyBboxPatch((0.44, summary_box_y), 0.40, 0.07,
                                boxstyle="round,pad=0.01",
                                facecolor='#e8f4fd',
                                edgecolor='#3498db',
                                linewidth=2)
        ax.add_patch(opt_box)
        ax.text(0.64, summary_box_y + 0.035, f"Optional: {present_optional}/{total_optional}", 
            ha='center', fontsize=14, fontweight='bold', va='center')

        # Add to text output
        self.add_text_output("Layer 5: INCOSE Analysis", {
            'best_pattern': pattern_name,
            'pattern_key': pattern_key,
            'pattern_template': pattern_def['template'] if pattern_def else None,
            'pattern_required_components': list(required_components),
            'pattern_optional_components': list(optional_components),
            'compliance_score': analysis.compliance_score,
            'components_found': analysis.components_found,
            'required_present': f"{present_required}/{total_required}",
            'optional_present': f"{present_optional}/{total_optional}",
            'missing_required': [c for c in required_components if not components.get(c)],
            'missing_optional': [c for c in optional_components if not components.get(c)],
            'suggestions': analysis.suggestions if hasattr(analysis, 'suggestions') else [],
            'template_recommendation': analysis.template_recommendation if hasattr(analysis, 'template_recommendation') else None
        })
        
        return analysis
    
    def _layer6_quality_dimensions(self, ax, req_text: str) -> Dict:
        """Layer 6: Quality Dimensions with substeps and semantic analysis"""
        ax.set_title("Layer 6: Quality Analysis", fontsize=24, fontweight='bold', loc='left', pad=15)
        ax.axis('off')

        # Run full analysis
        issues, metrics, incose, semantic = self.quality_analyzer.analyze_requirement(req_text)

        # Section 1: Five Core Quality Dimensions (Top)
        dim_y = 0.92
        ax.text(0.02, dim_y, "Core Quality Dimensions:", fontsize=14, fontweight='bold')
        
        dimensions = [
            ('Clarity', metrics.clarity_score, '_analyze_clarity()'),
            ('Complete', metrics.completeness_score, '_analyze_completeness()'),
            ('Verifiable', metrics.verifiability_score, '_analyze_verifiability()'),
            ('Atomic', metrics.atomicity_score, '_analyze_atomicity()'),
            ('Consistent', metrics.consistency_score, '_analyze_consistency()')
        ]

        x_positions = [0.10, 0.28, 0.46, 0.64, 0.82]
        
        for (dim_name, score, method), x_pos in zip(dimensions, x_positions):
            # Color based on score
            color = (self.colors['good'] if score >= 80 
                    else self.colors['warning'] if score >= 60 
                    else self.colors['bad'])
            
            # Draw circle
            circle = Circle((x_pos, dim_y - 0.10), 0.05, facecolor=color, alpha=0.3, 
                        edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            
            # Score in circle
            ax.text(x_pos, dim_y - 0.10, f"{score:.0f}", ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
            
            # Dimension name below
            ax.text(x_pos, dim_y - 0.17, dim_name, ha='center', fontsize=11, fontweight='bold')
            
            # Method name (smaller)
            ax.text(x_pos, dim_y - 0.21, method, ha='center', fontsize=7, 
                style='italic', color='#666')
        
        # Overall quality grade
        overall_score = metrics.quality_score
        grade = self.quality_analyzer._get_grade(overall_score)
        grade_color = (self.colors['good'] if grade in ['EXCELLENT', 'GOOD'] 
                    else self.colors['warning'] if grade == 'FAIR' 
                    else self.colors['bad'])
        
        # Grade box (top right)
        grade_box = FancyBboxPatch((0.75, dim_y - 0.01), 0.23, 0.06,
                                boxstyle="round,pad=0.008",
                                facecolor=grade_color,
                                alpha=0.2,
                                edgecolor=grade_color,
                                linewidth=2)
        ax.add_patch(grade_box)
        ax.text(0.865, dim_y + 0.02, f"Overall: {grade}", ha='center', 
            fontsize=12, fontweight='bold', color=grade_color, va='center')
        
        # Section 2: Semantic Analysis (Expanded middle section)
        sem_y = 0.68
        ax.text(0.02, sem_y, "Semantic Analysis:", fontsize=14, fontweight='bold')
        ax.text(0.25, sem_y, "analyze_semantic_quality()", fontsize=9, 
            style='italic', color='#666')
        
        # Semantic score
        sem_score_box = FancyBboxPatch((0.85, sem_y - 0.015), 0.12, 0.05,
                                    boxstyle="round,pad=0.005",
                                    facecolor='#e8f4fd',
                                    edgecolor='#3498db',
                                    linewidth=1.5)
        ax.add_patch(sem_score_box)
        ax.text(0.91, sem_y + 0.01, f"{metrics.semantic_quality_score:.0f}", 
            ha='center', fontsize=14, fontweight='bold', 
            color=self._get_score_color(metrics.semantic_quality_score), va='center')
        
        # Semantic substeps in 3 columns with more space
        sem_start_y = sem_y - 0.08
        col_width = 0.32
        
        # Column 1: Entity Extraction
        col1_x = 0.02
        
        # Entity extraction box
        entity_box = FancyBboxPatch((col1_x, sem_start_y - 0.30), col_width - 0.01, 0.28,
                                boxstyle="round,pad=0.008",
                                facecolor='#f0f8ff',
                                edgecolor='#3498db',
                                linewidth=1)
        ax.add_patch(entity_box)
        
        ax.text(col1_x + 0.01, sem_start_y - 0.02, "Entity Extraction", 
            fontsize=11, fontweight='bold')
        ax.text(col1_x + 0.01, sem_start_y - 0.055, "extract_entities()", 
            fontsize=8, style='italic', color='#666')
        
        entities = semantic.entity_completeness
        entity_y = sem_start_y - 0.09
        for entity_type in ['actors', 'actions', 'objects', 'conditions']:
            found = entities.get(entity_type, [])
            count = len(found)
            color = '#27ae60' if count > 0 else '#95a5a6'
            
            # Entity type with count
            ax.text(col1_x + 0.015, entity_y, f"{entity_type.title()}: {count}", 
                fontsize=10, color=color, fontweight='bold')
            entity_y -= 0.04
            
            # Show first 2 entities
            for item in found[:2]:
                item_text = item[:23] + "..." if len(item) > 23 else item
                ax.text(col1_x + 0.025, entity_y, f"• {item_text}", 
                    fontsize=8, color='#555')
                entity_y -= 0.03
            
            entity_y -= 0.005  # Extra spacing between types
        
        # Column 2: Ambiguities
        col2_x = 0.35
        
        # Ambiguity box
        amb_box = FancyBboxPatch((col2_x, sem_start_y - 0.30), col_width - 0.01, 0.28,
                                boxstyle="round,pad=0.008",
                                facecolor='#fff5f5',
                                edgecolor='#e74c3c',
                                linewidth=1)
        ax.add_patch(amb_box)
        
        ax.text(col2_x + 0.01, sem_start_y - 0.02, "Ambiguities", 
            fontsize=11, fontweight='bold')
        ax.text(col2_x + 0.01, sem_start_y - 0.055, "find_contextual_ambiguities()", 
            fontsize=8, style='italic', color='#666')
        
        ambiguities = semantic.contextual_ambiguities
        amb_y = sem_start_y - 0.09
        if ambiguities:
            for amb in ambiguities[:6]:
                # Truncate long ambiguities
                amb_text = amb[:27] + "..." if len(amb) > 27 else amb
                ax.text(col2_x + 0.015, amb_y, f"• {amb_text}", 
                    fontsize=8, color='#e74c3c')
                amb_y -= 0.04
            
            if len(ambiguities) > 6:
                ax.text(col2_x + 0.015, amb_y, f"+ {len(ambiguities)-6} more", 
                    fontsize=7, style='italic', color='#999')
        else:
            ax.text(col2_x + col_width/2, sem_start_y - 0.15, "✓ None found", 
                ha='center', fontsize=10, color='#27ae60', fontweight='bold')
        
        # Column 3: Tone Issues & Suggestions
        col3_x = 0.68
        
        # Tone/suggestions box
        tone_box = FancyBboxPatch((col3_x, sem_start_y - 0.30), col_width - 0.02, 0.28,
                                boxstyle="round,pad=0.008",
                                facecolor='#fffef0',
                                edgecolor='#f39c12',
                                linewidth=1)
        ax.add_patch(tone_box)
        
        ax.text(col3_x + 0.01, sem_start_y - 0.02, "Tone & Style", 
            fontsize=11, fontweight='bold')
        ax.text(col3_x + 0.01, sem_start_y - 0.055, "analyze_tone_and_subjectivity()", 
            fontsize=8, style='italic', color='#666')
        
        tone_y = sem_start_y - 0.09
        tone_issues = semantic.tone_issues
        if tone_issues:
            for issue in tone_issues[:3]:
                issue_text = issue[:26] + "..." if len(issue) > 26 else issue
                ax.text(col3_x + 0.015, tone_y, f"• {issue_text}", 
                    fontsize=8, color='#f39c12')
                tone_y -= 0.04
        else:
            ax.text(col3_x + 0.015, tone_y, "✓ No tone issues", 
                fontsize=9, color='#27ae60')
            tone_y -= 0.04
        
        # Suggestions below
        tone_y -= 0.01
        ax.text(col3_x + 0.01, tone_y, "Suggestions:", 
            fontsize=9, fontweight='bold', color='#3498db')
        tone_y -= 0.035
        suggestions = semantic.improvement_suggestions
        if suggestions:
            for sugg in suggestions[:3]:
                sugg_text = sugg[:26] + "..." if len(sugg) > 26 else sugg
                ax.text(col3_x + 0.015, tone_y, f"→ {sugg_text}", 
                    fontsize=8, color='#3498db')
                tone_y -= 0.04
        else:
            ax.text(col3_x + 0.015, tone_y, "None", fontsize=8, color='#999')
        
        # Section 3: Issues Summary (Bottom)
        issues_y = 0.20
        
        # Issues header with box
        issues_header_box = FancyBboxPatch((0.02, issues_y - 0.01), 0.96, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#f8f9fa',
                                        edgecolor='#95a5a6',
                                        linewidth=1.5)
        ax.add_patch(issues_header_box)
        
        ax.text(0.04, issues_y + 0.045, "Issue Summary by Severity:", 
            fontsize=13, fontweight='bold', va='center')
        
        # Issue counts by severity
        severity = metrics.severity_breakdown
        issue_x = 0.35
        for sev_type, color in [('Critical', '#e74c3c'), ('High', '#f39c12'), 
                                ('Medium', '#f1c40f'), ('Low', '#95a5a6')]:
            count = severity.get(sev_type.lower(), 0)
            text_color = color if count > 0 else '#bdc3c7'
            
            # Count badge
            ax.text(issue_x, issues_y + 0.045, f"{sev_type}:", 
                fontsize=10, fontweight='bold', va='center')
            ax.text(issue_x + 0.085, issues_y + 0.045, f"{count}", 
                fontsize=14, fontweight='bold', color=text_color, va='center')
            
            issue_x += 0.16
        
        # Total issues (bottom right)
        total_box = FancyBboxPatch((0.75, issues_y + 0.015), 0.21, 0.06,
                                boxstyle="round,pad=0.008",
                                facecolor='#ecf0f1',
                                edgecolor='#7f8c8d',
                                linewidth=2)
        ax.add_patch(total_box)
        ax.text(0.855, issues_y + 0.045, f"Total: {metrics.total_issues}", 
            ha='center', fontsize=12, fontweight='bold', va='center')
        
        # Show a few sample issues below if any exist
        sample_y = 0.10
        if issues and len(issues) > 0:
            ax.text(0.02, sample_y, "Sample Issues:", fontsize=11, fontweight='bold')
            sample_y -= 0.04
            for issue in issues[:3]:
                issue_text = issue[:100] + "..." if len(issue) > 100 else issue
                ax.text(0.03, sample_y, f"• {issue_text}", fontsize=8, color='#555')
                sample_y -= 0.03

        # Compile output
        quality_dict = {
            'overall_score': overall_score,
            'grade': grade,
            'clarity': metrics.clarity_score,
            'completeness': metrics.completeness_score,
            'verifiability': metrics.verifiability_score,
            'atomicity': metrics.atomicity_score,
            'consistency': metrics.consistency_score,
            'semantic_quality': metrics.semantic_quality_score,
            'entities_found': semantic.entity_completeness,
            'ambiguities': semantic.contextual_ambiguities,
            'tone_issues': semantic.tone_issues,
            'semantic_suggestions': semantic.improvement_suggestions,
            'total_issues': metrics.total_issues,
            'severity_breakdown': metrics.severity_breakdown
        }
        
        self.add_text_output("Layer 6: Quality Analysis", quality_dict)
        return quality_dict    
    
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
            symbol = "✔"

        elif combined_score >= 0.35:
            decision = "REVIEW NEEDED"
            action = "Moderate confidence - engineer review required"
            color = self.colors['warning']
            symbol = "?"

        else:
            decision = "ORPHAN"
            action = "No suitable match - write bridge requirement"
            color = self.colors['bad']
            symbol = "✘"

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
                       f"Issues: {quality_results['total_issues']}")
        ax.text(0.5, 0.1, metrics_text, ha='center', fontsize=16, color='gray')
        self.add_text_output("Layer 7: Final Decision", {'decision': decision, 'action': action, 'metrics': metrics_text})

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
        ax2.boxplot([matches_df[algo] for algo in algorithms], tick_labels=labels)
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
        ax4.set_ylim(0, 10)
        
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
        
        # Load matches (now includes quality data if available)
        matches_df, explanations, _ = self.load_real_data()
        if matches_df.empty:
            print("❌ No matches found")
            return paths
        
        # Check if quality data is available
        has_quality_data = 'Total_Issues' in matches_df.columns and 'INCOSE_Best_Pattern' in matches_df.columns
        
        if not has_quality_data:
            print("\n⚠️  Quality analysis data not found in matches.")
            print("   Please run: python src/quality/reqGrading.py")
            print("   Falling back to simple highest/lowest match selection...\n")
        
        # === HIGH SCORING MATCH ===
        print("\n🔍 Finding high scoring match...")
        
        if has_quality_data:
            # Filter for functional_performance pattern
            functional_perf = matches_df[matches_df['INCOSE_Best_Pattern'] == 'functional_performance']
            
            if not functional_perf.empty:
                highest_match = functional_perf.nlargest(1, 'Combined_Score').iloc[0]
                print(f"   ✓ Found functional_performance: {highest_match['Requirement_ID']} " +
                    f"(Score: {highest_match['Combined_Score']:.3f})")
            else:
                highest_match = matches_df.nlargest(1, 'Combined_Score').iloc[0]
                print(f"   ⚠️  No functional_performance found, using highest: {highest_match['Requirement_ID']}")
        else:
            # Fallback: just use highest score
            highest_match = matches_df.nlargest(1, 'Combined_Score').iloc[0]
            print(f"   Selected: {highest_match['Requirement_ID']} (Score: {highest_match['Combined_Score']:.3f})")
        
        # === LOW SCORING MATCH ===
        print("\n🔍 Finding low scoring match...")
        
        if has_quality_data:
            # Filter for low match scores (bottom 30%)
            match_threshold = matches_df['Combined_Score'].quantile(0.30)
            low_candidates = matches_df[matches_df['Combined_Score'] <= match_threshold]
            
            # From those, find the one with most issues
            if not low_candidates.empty:
                lowest_match = low_candidates.nlargest(1, 'Total_Issues').iloc[0]
                print(f"   ✓ Found low match with most issues: {lowest_match['Requirement_ID']}")
                print(f"      Match Score: {lowest_match['Combined_Score']:.3f}")
                print(f"      Total Issues: {lowest_match['Total_Issues']}")
                print(f"      Quality Score: {lowest_match['Quality_Score']:.1f}")
            else:
                lowest_match = matches_df.nsmallest(1, 'Combined_Score').iloc[0]
                print(f"   ⚠️  Using lowest match score: {lowest_match['Requirement_ID']}")
        else:
            # Fallback: just use lowest score
            lowest_match = matches_df.nsmallest(1, 'Combined_Score').iloc[0]
            print(f"   Selected: {lowest_match['Requirement_ID']} (Score: {lowest_match['Combined_Score']:.3f})")
        
        # Create visualizations for both
        for label, match in [('high', highest_match), ('low', lowest_match)]:
            req_id = match['Requirement_ID']
            print(f"\n📊 Creating technical journey for {label.upper()} scoring pair: {req_id}")
            try:
                path = self.create_processing_journey(requirement_id=req_id, label=label)
                paths.append(path)
            except Exception as e:
                print(f"❌ Error creating journey for {req_id}: {e}")
                import traceback
                traceback.print_exc()
        
        self.create_algorithm_contribution_chart()

        print("\n" + "="*70)
        print(f"✅ Created {len(paths)} visualizations")
        print("Files saved to: outputs/visuals/")
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