"""
Technical Journey Visualization - Graphviz Implementation  
READABLE, DETAILED, and showing parallel algorithm execution.
Preserves all V1 data export methods (JSON, CSV, TXT).

LOCATION: Save this file as: src/visualization/journey_visualizer.py
"""

import json
import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from graphviz import Digraph
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import sys
import os

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer


class JourneyVisualizer:
    """
    Creates technical journey visualizations with READABLE fonts and DETAILED information.
    Shows parallel algorithm execution, not linear flow.
    """
    
    def __init__(self):
        """Initialize visualizer with project structure."""
        try:
            self.repo_manager = RepositoryStructureManager()
            self.file_handler = SafeFileHandler(self.repo_manager)
            self.output_dir = self.repo_manager.structure['visuals_output']
        except Exception as e:
            print(f"⚠️ Repository manager unavailable: {e}")
            raise ValueError("Repository manager is required for JourneyVisualizer")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.domain_resources = DomainResources()
        self.req_analyzer = EnhancedRequirementAnalyzer()
        
        # Data storage for exports
        self.layer_data = {}
        self.text_output = []
        
        # Colors matching V1
        self.colors = {
            'semantic': '#3498db',
            'bm25': '#e74c3c',
            'domain': '#27ae60',
            'query_expansion': '#f39c12',
            'good': '#27ae60',
            'warning': '#f39c12',
            'bad': '#e74c3c'
        }
    
    def add_text_output(self, layer_name: str, content: Dict):
        """Add text output for a layer (preserves V1 method)."""
        self.text_output.append({
            'layer': layer_name,
            'timestamp': datetime.now().isoformat(),
            'content': content
        })
        self.layer_data[layer_name] = content
    
    def save_text_outputs(self, req_id: str):
        """Save text outputs to CSV and JSON (preserves V1 method)."""
        # Save as JSON
        json_path = self.output_dir / f"{req_id}_layer_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_data, f, indent=2)
        
        # Save as CSV
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
        
        # Save as text
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
        """Load matches and requirements data."""
        if self.repo_manager:
            matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
            expl_path = self.repo_manager.structure['matching_results'] / "match_explanations.json"
            req_path = self.repo_manager.structure['data_raw'] / "requirements.csv"
        else:
            matches_path = Path("outputs/matching_results/aerospace_matches.csv")
            expl_path = Path("outputs/matching_results/match_explanations.json")
            req_path = Path("requirements.csv")
        
        # Load matches
        if matches_path.exists():
            matches_df = pd.read_csv(matches_path)
        else:
            raise FileNotFoundError(f"Matches file not found: {matches_path}")
        
        # Load explanations
        explanations = {}
        if expl_path.exists():
            with open(expl_path, 'r') as f:
                explanations = json.load(f)
        
        # Load requirements
        requirements_df = None
        if req_path.exists():
            requirements_df = pd.read_csv(req_path)
        
        return matches_df, explanations, requirements_df
    
    def create_technical_journey_matplotlib(self, requirement_id: str = None, 
                                           label: str = None) -> str:
        """
        Create READABLE matplotlib visualization showing parallel algorithm execution.
        Much larger fonts, clearer layout than before.
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
        
        # Clear previous data
        self.layer_data = {}
        self.text_output = []
        
        # Get explanation
        explanation = explanations.get((req_id, activity_name), {})
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(20, 14))
        
        title = f'Technical Journey: {req_id} → {activity_name}\nCombined Score: {match["Combined_Score"]:.3f}'
        if label:
            title = f'{label}\n{title}'
        fig.suptitle(title, fontsize=24, fontweight='bold')
        
        # Create sections
        # Top: Input + Preprocessing
        # Middle: 4 parallel algorithms
        # Bottom: Combination + Quality
        
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3,
                             top=0.92, bottom=0.05, left=0.05, right=0.95)
        
        # Layer 1: Input (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._draw_input_layer(ax1, req_id, req_text, activity_name)
        
        # Layer 2: Preprocessing (full width)
        ax2 = fig.add_subplot(gs[1, :])
        self._draw_preprocessing_layer(ax2, req_text, activity_name)
        
        # Layer 3: Four algorithms in PARALLEL (one per column)
        scores = explanation.get('scores', {})
        explanations_data = explanation.get('explanations', {})
        
        ax_sem = fig.add_subplot(gs[2, 0])
        self._draw_algorithm_box(ax_sem, 'Semantic', scores.get('semantic', 0),
                                 explanations_data.get('semantic', ''), match)
        
        ax_bm25 = fig.add_subplot(gs[2, 1])
        self._draw_algorithm_box(ax_bm25, 'BM25', scores.get('bm25', 0),
                                explanations_data.get('bm25', ''), match)
        
        ax_domain = fig.add_subplot(gs[2, 2])
        self._draw_algorithm_box(ax_domain, 'Domain', scores.get('domain', 0),
                                 explanations_data.get('domain', ''), match)
        
        ax_qe = fig.add_subplot(gs[2, 3])
        self._draw_algorithm_box(ax_qe, 'Query Exp', scores.get('query_expansion', 0),
                                explanations_data.get('query_expansion', ''), match)
        
        # Layer 4: Score combination (full width)
        ax4 = fig.add_subplot(gs[3, :])
        self._draw_combination_layer(ax4, match, scores)
        
        # Layer 5: Quality (full width)
        ax5 = fig.add_subplot(gs[4, :])
        self._draw_quality_layer(ax5, req_text)
        
        # Save
        safe_req_id = req_id.replace('/', '_')
        output_path = self.output_dir / f"{safe_req_id}_journey.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save text outputs
        self.save_text_outputs(req_id)
        
        print(f"✅ Created journey visualization: {output_path}")
        return str(output_path)
    
    def _draw_input_layer(self, ax, req_id, req_text, activity_name):
        """Draw input layer with large readable text."""
        ax.axis('off')
        ax.set_title('INPUT DATA', fontsize=20, fontweight='bold', loc='left', pad=10)
        
        # Box
        box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor='#e8f5e9',
                             edgecolor='#27ae60',
                             linewidth=2)
        ax.add_patch(box)
        
        # Content with LARGE fonts
        ax.text(0.5, 0.7, f'Requirement: {req_id}', ha='center', fontsize=16, fontweight='bold')
        
        # Wrap requirement text
        wrapped_req = textwrap.wrap(req_text, width=100)
        y_pos = 0.55
        for line in wrapped_req[:3]:
            ax.text(0.5, y_pos, line, ha='center', fontsize=12)
            y_pos -= 0.1
        
        ax.text(0.5, 0.25, f'Activity: {activity_name}', ha='center', fontsize=14, fontweight='bold')
        
        self.add_text_output("Layer 1: Input", {
            'requirement_id': req_id,
            'requirement_text': req_text,
            'activity_name': activity_name
        })
    
    def _draw_preprocessing_layer(self, ax, req_text, activity_name):
        """Draw preprocessing layer."""
        ax.axis('off')
        ax.set_title('PREPROCESSING PIPELINE', fontsize=20, fontweight='bold', loc='left', pad=10)
        
        # Process data
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        req_doc = self.matcher.nlp(expanded_req)
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        
        tokens = [token.text for token in req_doc]
        aerospace_terms = [t for t in req_terms if t in self.matcher.all_aerospace_terms]
        
        # Box
        box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor='#fff3e0',
                             edgecolor='#f39c12',
                             linewidth=2)
        ax.add_patch(box)
        
        # Content with LARGE fonts
        steps = [
            f'1. Abbreviation Expansion: {"Yes" if expanded_req != req_text else "No"}',
            f'2. Tokenization: {len(tokens)} total tokens, {len(set(tokens))} unique',
            f'3. Term Extraction: {len(req_terms)} terms',
            f'4. Aerospace Terms Found: {len(aerospace_terms)}'
        ]
        
        y_pos = 0.7
        for step in steps:
            ax.text(0.1, y_pos, step, fontsize=14, va='top')
            y_pos -= 0.15
        
        if aerospace_terms:
            terms_str = ', '.join(aerospace_terms[:5])
            ax.text(0.1, 0.25, f'Top terms: {terms_str}', fontsize=12, style='italic')
        
        self.add_text_output("Layer 2: Preprocessing", {
            'tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'terms_extracted': len(req_terms),
            'aerospace_terms': aerospace_terms
        })
    
    def _draw_algorithm_box(self, ax, algo_name, score, explanation_text, match):
        """Draw individual algorithm box with LARGE readable fonts."""
        ax.axis('off')
        
        # Color by score
        if score >= 0.7:
            color = self.colors['good']
        elif score >= 0.4:
            color = self.colors['warning']
        else:
            color = self.colors['bad']
        
        # Box
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                             boxstyle="round,pad=0.02",
                             facecolor=color,
                             alpha=0.15,
                             edgecolor=color,
                             linewidth=3)
        ax.add_patch(box)
        
        # Title with LARGE font
        ax.text(0.5, 0.85, algo_name.upper(), ha='center', fontsize=18, fontweight='bold')
        
        # Score with HUGE font
        ax.text(0.5, 0.55, f'{score:.3f}', ha='center', fontsize=36, fontweight='bold', color=color)
        
        # Method
        methods = {
            'Semantic': 'Sentence-BERT\nCosine Similarity',
            'BM25': 'Okapi BM25\nk1=1.5, b=0.75',
            'Domain': 'Aerospace Terms\nJaccard Overlap',
            'Query Exp': 'Synonym Expansion\nBroadened Matching'
        }
        ax.text(0.5, 0.25, methods.get(algo_name, ''), ha='center', fontsize=11, style='italic')
        
        # Store data
        algo_key = algo_name.lower().replace(' ', '_')
        self.add_text_output(f"Algorithm: {algo_name}", {
            'score': score,
            'method': methods.get(algo_name, '')
        })
    
    def _draw_combination_layer(self, ax, match, scores):
        """Draw score combination layer."""
        ax.axis('off')
        ax.set_title('SCORE COMBINATION (Equal Weights)', fontsize=20, fontweight='bold', loc='left', pad=10)
        
        combined = match['Combined_Score']
        
        # Box
        if combined >= 0.8:
            color = self.colors['good']
            label = 'HIGH MATCH'
        elif combined >= 0.35:
            color = self.colors['warning']
            label = 'MEDIUM MATCH'
        else:
            color = self.colors['bad']
            label = 'ORPHAN'
        
        box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=color,
                             alpha=0.15,
                             edgecolor=color,
                             linewidth=3)
        ax.add_patch(box)
        
        # Combined score HUGE
        ax.text(0.5, 0.6, f'{combined:.3f}', ha='center', fontsize=48, fontweight='bold', color=color)
        ax.text(0.5, 0.4, label, ha='center', fontsize=20, fontweight='bold')
        
        # Formula
        formula = f'({scores.get("semantic", 0):.3f} + {scores.get("bm25", 0):.3f} + {scores.get("domain", 0):.3f} + {scores.get("query_expansion", 0):.3f}) ÷ 4'
        ax.text(0.5, 0.25, formula, ha='center', fontsize=12)
        
        self.add_text_output("Layer 4: Combination", {
            'combined_score': combined,
            'classification': label,
            'formula': formula
        })
    
    def _draw_quality_layer(self, ax, req_text):
        """Draw quality analysis layer."""
        ax.axis('off')
        ax.set_title('QUALITY ANALYSIS', fontsize=20, fontweight='bold', loc='left', pad=10)
        
        # Analyze
        issues, metrics, incose, semantic = self.req_analyzer.analyze_requirement(req_text)
        grade = self.req_analyzer._get_grade(metrics.quality_score)
        
        # Box
        if grade in ['EXCELLENT', 'GOOD']:
            color = self.colors['good']
        elif grade == 'FAIR':
            color = self.colors['warning']
        else:
            color = self.colors['bad']
        
        box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=color,
                             alpha=0.15,
                             edgecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        
        # Content
        ax.text(0.15, 0.7, f'Grade: {grade}', fontsize=18, fontweight='bold')
        ax.text(0.15, 0.55, f'Quality Score: {metrics.quality_score:.0f}/100', fontsize=14)
        ax.text(0.15, 0.4, f'Total Issues: {len(issues)}', fontsize=14)
        
        ax.text(0.6, 0.7, f'Clarity: {metrics.clarity_score:.0f}', fontsize=14)
        ax.text(0.6, 0.55, f'Completeness: {metrics.completeness_score:.0f}', fontsize=14)
        ax.text(0.6, 0.4, f'Verifiability: {metrics.verifiability_score:.0f}', fontsize=14)
        
        ax.text(0.15, 0.25, f'INCOSE Pattern: {incose.best_pattern}', fontsize=12, style='italic')
        
        self.add_text_output("Layer 5: Quality", {
            'grade': grade,
            'score': metrics.quality_score,
            'issues': len(issues),
            'incose_pattern': incose.best_pattern
        })
    
    def create_algorithm_contribution_analysis(self) -> str:
        """
        Create comprehensive algorithm contribution analysis matching V1 style.
        Shows distributions, correlations, and contributions.
        """
        print("\n📊 Creating algorithm contribution analysis...")
        
        # Load data
        matches_df, _, _ = self.load_real_data()
        
        algo_cols = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        algo_names = ['Semantic', 'BM25', 'Domain', 'Query Exp']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Algorithm Contribution Analysis Across All Matches', 
                     fontsize=24, fontweight='bold')
        
        # 1. Average contribution pie chart
        ax1 = axes[0, 0]
        avg_scores = [matches_df[col].mean() for col in algo_cols]
        colors_list = [self.colors[a] for a in ['semantic', 'bm25', 'domain', 'query_expansion']]
        
        ax1.pie(avg_scores, labels=algo_names, colors=colors_list, 
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
        ax1.set_title('Average Algorithm Contribution', fontsize=18, fontweight='bold', pad=15)
        
        # 2. Score distribution box plots
        ax2 = axes[0, 1]
        bp = ax2.boxplot([matches_df[col] for col in algo_cols], 
                         labels=algo_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        ax2.set_title('Score Distribution by Algorithm', fontsize=18, fontweight='bold', pad=15)
        ax2.set_ylabel('Score', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        ax3 = axes[1, 0]
        correlation_data = matches_df[algo_cols].corr()
        im = ax3.imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(algo_names)))
        ax3.set_yticks(range(len(algo_names)))
        ax3.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=12)
        ax3.set_yticklabels(algo_names, fontsize=12)
        ax3.set_title('Algorithm Score Correlations', fontsize=18, fontweight='bold', pad=15)
        
        # Add correlation values
        for i in range(len(algo_cols)):
            for j in range(len(algo_cols)):
                text = ax3.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Score ranges histogram
        ax4 = axes[1, 1]
        for col, name, color in zip(algo_cols, algo_names, colors_list):
            ax4.hist(matches_df[col], bins=20, alpha=0.5, label=name, color=color)
        ax4.set_title('Score Distributions', fontsize=18, fontweight='bold', pad=15)
        ax4.set_xlabel('Score', fontsize=14)
        ax4.set_ylabel('Frequency', fontsize=14)
        ax4.legend(fontsize=12)
        ax4.tick_params(axis='both', labelsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "algorithm_contribution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Created contribution analysis: {output_path}")
        return str(output_path)


def main():
    """Demonstrate all visualization capabilities."""
    print("🎨 JOURNEY VISUALIZER - READABLE & DETAILED")
    print("="*60)
    
    viz = JourneyVisualizer()
    
    # 1. Create technical journey (now with parallel algorithms!)
    print("\n1️⃣ Creating technical journey visualization...")
    journey_path = viz.create_technical_journey_matplotlib(label="HIGH SCORING")
    
    # 2. Create algorithm contribution analysis (matching V1 style)
    print("\n2️⃣ Creating algorithm contribution analysis...")
    contrib_path = viz.create_algorithm_contribution_analysis()
    
    print("\n✅ All visualizations complete!")
    print(f"\nOutputs saved to: {viz.output_dir}")


if __name__ == "__main__":
    main()