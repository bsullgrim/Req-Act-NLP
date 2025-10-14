"""
Hybrid Journey Visualizer
Combines V1's data collection with Graphviz's auto-layout and V1's contribution charts.

LOCATION: Save as src/visualization/hybrid_visualizer.py
"""

import json
import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from graphviz import Digraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer


class HybridJourneyVisualizer:
    """
    Hybrid visualizer combining:
    - V1's comprehensive data collection
    - Graphviz's automatic layout (no text overlap!)
    - V1's algorithm contribution charts
    - Bigger, readable fonts
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
    
    # ============= V1's DATA COLLECTION METHODS (UNCHANGED) =============
    
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
    
    # ============= V1's LAYER DATA EXTRACTION (UNCHANGED LOGIC) =============
    
    def collect_layer_data(self, match, req_text: str, activity_name: str, explanation: Dict) -> Dict:
        """
        Collect ALL layer data using V1's logic.
        Returns structured data for visualization.
        """
        layers = {}
        
        # Clear previous
        self.layer_data = {}
        self.text_output = []
        
        # Layer 1: Raw Inputs
        layers['layer1'] = {
            'requirement_id': match['Requirement_ID'],
            'requirement_text': req_text,
            'activity_name': activity_name
        }
        self.add_text_output("Layer 1: Raw Inputs", layers['layer1'])
        
        # Layer 2: Preprocessing
        expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
        expanded_act = self.matcher._expand_aerospace_abbreviations(activity_name)
        req_doc = self.matcher.nlp(expanded_req)
        act_doc = self.matcher.nlp(expanded_act)
        req_terms = self.matcher._preprocess_text_aerospace(req_text)
        act_terms = self.matcher._preprocess_text_aerospace(activity_name)
        
        tokens = [token.text for token in req_doc]
        aerospace_terms = [t for t in req_terms if t in self.matcher.all_aerospace_terms]
        
        layers['layer2'] = {
            'expanded_req': expanded_req,
            'expanded_act': expanded_act,
            'total_tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'req_terms': req_terms[:10],
            'act_terms': act_terms[:10],
            'aerospace_terms': aerospace_terms[:10],
            'abbreviations_expanded': expanded_req != req_text
        }
        self.add_text_output("Layer 2: Preprocessing", layers['layer2'])
        
        # Layer 3: Algorithm Analysis (ALL 4 in parallel)
        scores_obj = explanation.get('scores', {})
        explanations_data = explanation.get('explanations', {})
        
        layers['layer3'] = {
            'semantic': {
                'score': scores_obj.get('semantic', match.get('Semantic_Score', 0)),
                'explanation': explanations_data.get('semantic', 'Sentence-BERT embeddings')
            },
            'bm25': {
                'score': scores_obj.get('bm25', match.get('BM25_Score', 0)),
                'explanation': explanations_data.get('bm25', 'Term frequency ranking')
            },
            'domain': {
                'score': scores_obj.get('domain', match.get('Domain_Score', 0)),
                'explanation': explanations_data.get('domain', {}),
                'aerospace_terms': explanations_data.get('domain', {}).get('aerospace_terms', [])[:5] if isinstance(explanations_data.get('domain'), dict) else []
            },
            'query_expansion': {
                'score': scores_obj.get('query_expansion', match.get('Query_Expansion_Score', 0)),
                'explanation': explanations_data.get('query_expansion', 'Synonym expansion')
            }
        }
        self.add_text_output("Layer 3: Algorithm Analysis", layers['layer3'])
        
        # Layer 4: Score Combination
        combined = match['Combined_Score']
        if combined >= 0.8:
            classification = "HIGH MATCH"
        elif combined >= 0.35:
            classification = "MEDIUM MATCH"
        else:
            classification = "ORPHAN"
        
        layers['layer4'] = {
            'combined_score': combined,
            'classification': classification,
            'semantic_contribution': layers['layer3']['semantic']['score'] * 0.25,
            'bm25_contribution': layers['layer3']['bm25']['score'] * 0.25,
            'domain_contribution': layers['layer3']['domain']['score'] * 0.25,
            'query_exp_contribution': layers['layer3']['query_expansion']['score'] * 0.25
        }
        self.add_text_output("Layer 4: Score Combination", layers['layer4'])
        
        # Layer 5: Quality Analysis
        issues, metrics, incose, semantic = self.req_analyzer.analyze_requirement(req_text)
        grade = self.req_analyzer._get_grade(metrics.quality_score)
        
        layers['layer5'] = {
            'grade': grade,
            'quality_score': metrics.quality_score,
            'total_issues': len(issues),
            'clarity_score': metrics.clarity_score,
            'completeness_score': metrics.completeness_score,
            'verifiability_score': metrics.verifiability_score,
            'incose_pattern': incose.best_pattern,
            'incose_compliance': incose.compliance_score
        }
        self.add_text_output("Layer 5: Quality Analysis", layers['layer5'])
        
        return layers
    
    # ============= NEW: GRAPHVIZ VISUALIZATION WITH V1'S DATA =============
    
    def create_graphviz_journey(self, requirement_id: str = None, label: str = None) -> str:
        """
        Create Graphviz visualization using ALL of V1's data.
        BIG FONTS, clear layout, parallel algorithms.
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
        
        # Collect ALL layer data using V1's logic
        layers = self.collect_layer_data(match, req_text, activity_name, explanation)
        
        # Create Graphviz diagram with PARALLEL layout
        g = Digraph(name=f'journey_{req_id}', engine='dot')
        g.attr(rankdir='TB', dpi='300', nodesep='1.2', ranksep='1.0',
               bgcolor='white', pad='0.5', fontname='Arial')
        
        # Title
        title = f'{req_id} → {activity_name[:50]}\nCombined Score: {match["Combined_Score"]:.3f}'
        if label:
            title = f'{label}\n{title}'
        g.node('title', title, shape='plaintext', fontsize='22', fontname='Arial Bold')
        
        # Layer 1: Input
        input_text = f"INPUT\n\nRequirement: {req_id}\n{req_text[:80]}...\n\nActivity: {activity_name}"
        g.node('input', input_text, shape='box', style='rounded,filled', 
               fillcolor='#e8f5e9', fontsize='12', width='8', height='2')
        
        # Layer 2: Preprocessing
        l2 = layers['layer2']
        preprocess_text = f"PREPROCESSING\n\nTokens: {l2['total_tokens']} total, {l2['unique_tokens']} unique\nAbbreviations: {'Expanded' if l2['abbreviations_expanded'] else 'None'}\nAerospace terms: {len(l2['aerospace_terms'])}\nTop terms: {', '.join(l2['aerospace_terms'][:3])}"
        g.node('preprocess', preprocess_text, shape='box', style='filled',
               fillcolor='#fff3e0', fontsize='11', width='8', height='2')
        
        # Layer 3: Four algorithms IN PARALLEL (using subgraph for horizontal layout)
        with g.subgraph(name='cluster_algorithms') as c:
            c.attr(label='ALGORITHM ANALYSIS (Parallel Execution)', fontsize='16', 
                   fontname='Arial Bold', style='dashed', color='#666')
            c.attr(rank='same')
            
            l3 = layers['layer3']
            
            # Semantic
            sem_score = l3['semantic']['score']
            sem_color = '#c8e6c9' if sem_score >= 0.7 else '#fff9c4' if sem_score >= 0.4 else '#ffcdd2'
            sem_text = f"SEMANTIC\n\nScore: {sem_score:.3f}\n\nSentence-BERT\nCosine Similarity"
            c.node('semantic', sem_text, shape='box', style='filled',
                   fillcolor=sem_color, fontsize='12', width='2.5', height='1.8')
            
            # BM25
            bm25_score = l3['bm25']['score']
            bm25_color = '#c8e6c9' if bm25_score >= 0.7 else '#fff9c4' if bm25_score >= 0.4 else '#ffcdd2'
            bm25_text = f"BM25\n\nScore: {bm25_score:.3f}\n\nOkapi BM25\nk1=1.5, b=0.75"
            c.node('bm25', bm25_text, shape='box', style='filled',
                   fillcolor=bm25_color, fontsize='12', width='2.5', height='1.8')
            
            # Domain
            dom_score = l3['domain']['score']
            dom_color = '#c8e6c9' if dom_score >= 0.7 else '#fff9c4' if dom_score >= 0.4 else '#ffcdd2'
            dom_terms = ', '.join(l3['domain']['aerospace_terms'][:3]) if l3['domain']['aerospace_terms'] else 'None'
            dom_text = f"DOMAIN\n\nScore: {dom_score:.3f}\n\nAerospace Terms:\n{dom_terms}"
            c.node('domain', dom_text, shape='box', style='filled',
                   fillcolor=dom_color, fontsize='12', width='2.5', height='1.8')
            
            # Query Expansion
            qe_score = l3['query_expansion']['score']
            qe_color = '#c8e6c9' if qe_score >= 0.7 else '#fff9c4' if qe_score >= 0.4 else '#ffcdd2'
            qe_text = f"QUERY EXP\n\nScore: {qe_score:.3f}\n\nSynonym\nExpansion"
            c.node('query_exp', qe_text, shape='box', style='filled',
                   fillcolor=qe_color, fontsize='12', width='2.5', height='1.8')
        
        # Layer 4: Combination
        l4 = layers['layer4']
        comb_color = '#c8e6c9' if l4['combined_score'] >= 0.8 else '#fff9c4' if l4['combined_score'] >= 0.35 else '#ffcdd2'
        comb_text = f"SCORE COMBINATION\n\nCombined: {l4['combined_score']:.3f}\nClassification: {l4['classification']}\n\nFormula: (Sem + BM25 + Domain + QE) / 4\nContributions:\n  Semantic: {l4['semantic_contribution']:.3f}\n  BM25: {l4['bm25_contribution']:.3f}\n  Domain: {l4['domain_contribution']:.3f}\n  Query Exp: {l4['query_exp_contribution']:.3f}"
        g.node('combination', comb_text, shape='box', style='filled',
               fillcolor=comb_color, fontsize='11', width='8', height='2.5')
        
        # Layer 5: Quality
        l5 = layers['layer5']
        qual_color = '#c8e6c9' if l5['grade'] in ['EXCELLENT', 'GOOD'] else '#fff9c4' if l5['grade'] == 'FAIR' else '#ffcdd2'
        qual_text = f"QUALITY ANALYSIS\n\nGrade: {l5['grade']} ({l5['quality_score']:.0f}/100)\nIssues: {l5['total_issues']}\n\nDimensions:\n  Clarity: {l5['clarity_score']:.0f}\n  Completeness: {l5['completeness_score']:.0f}\n  Verifiability: {l5['verifiability_score']:.0f}\n\nINCOSE: {l5['incose_pattern']} ({l5['incose_compliance']:.0f}%)"
        g.node('quality', qual_text, shape='box', style='filled',
               fillcolor=qual_color, fontsize='11', width='8', height='2.5')
        
        # Edges
        g.edge('title', 'input', style='invis')
        g.edge('input', 'preprocess', penwidth='3', label='Process', fontsize='12')
        
        # Preprocess fans out to all 4 algorithms
        g.edge('preprocess', 'semantic', penwidth='2', label='Analyze', fontsize='10')
        g.edge('preprocess', 'bm25', penwidth='2', label='Analyze', fontsize='10')
        g.edge('preprocess', 'domain', penwidth='2', label='Analyze', fontsize='10')
        g.edge('preprocess', 'query_exp', penwidth='2', label='Analyze', fontsize='10')
        
        # All 4 algorithms feed into combination
        g.edge('semantic', 'combination', penwidth='2', label='0.25×', fontsize='10')
        g.edge('bm25', 'combination', penwidth='2', label='0.25×', fontsize='10')
        g.edge('domain', 'combination', penwidth='2', label='0.25×', fontsize='10')
        g.edge('query_exp', 'combination', penwidth='2', label='0.25×', fontsize='10')
        
        g.edge('combination', 'quality', penwidth='3', label='Validate', fontsize='12')
        
        # Render
        safe_req_id = req_id.replace('/', '_')
        output_path = self.output_dir / f"{safe_req_id}_journey"
        g.render(output_path, format='png', cleanup=True)
        
        # Save text outputs
        self.save_text_outputs(req_id)
        
        print(f"✅ Created journey visualization: {output_path}.png")
        return f"{output_path}.png"
    
    # ============= V1's ALGORITHM CONTRIBUTION CHART (UNCHANGED) =============
    
    def create_algorithm_contribution_chart(self) -> str:
        """
        V1's contribution chart - UNCHANGED.
        Pie chart, box plots, correlation heatmap, success rates.
        """
        matches_df, _, _ = self.load_real_data()
        
        algorithms = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        avg_scores = {algo: matches_df[algo].mean() for algo in algorithms}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Contribution Analysis Across All Matches', 
                     fontsize=20, fontweight='bold')
        
        # 1. Average contribution pie chart
        ax1 = axes[0, 0]
        colors_list = [self.colors[a] for a in ['semantic', 'bm25', 'domain', 'query_expansion']]
        labels = ['Semantic', 'BM25', 'Domain', 'Query Exp']
        values = list(avg_scores.values())
        
        ax1.pie(values, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Average Algorithm Contribution', fontsize=16, fontweight='bold')
        
        # 2. Score distribution box plots
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
        
        # 4. Success rate by dominant algorithm
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
    print("🎨 HYBRID JOURNEY VISUALIZER")
    print("="*60)
    print("Combining V1's data + Graphviz's layout + V1's charts")
    print("="*60)
    
    viz = HybridJourneyVisualizer()
    
    # 1. Technical journey with Graphviz (V1's data, bigger fonts)
    print("\n1️⃣ Creating technical journey...")
    journey_path = viz.create_graphviz_journey(label="HIGH SCORING")
    
    # 2. Algorithm contribution chart (V1's matplotlib version)
    print("\n2️⃣ Creating algorithm contribution analysis...")
    contrib_path = viz.create_algorithm_contribution_chart()
    
    print("\n✅ All visualizations complete!")
    print(f"\nOutputs saved to: {viz.output_dir}")


if __name__ == "__main__":
    main()
