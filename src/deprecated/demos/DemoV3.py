# clean_displacy_visualizer.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import json
import re
from pathlib import Path
import sys, os

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.repository_setup import RepositoryStructureManager
from src.matching.matcher import AerospaceMatcher

class CleanDisplacyVisualizer:
    """Create clean displaCy-style PNGs using matplotlib and actual algorithm data."""
    
    def __init__(self):
        self.repo_manager = RepositoryStructureManager()
        self.OUTPUT_DIR = self.repo_manager.base_dir / "displacy_visuals"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load matcher for domain knowledge access
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        
        # Set up clean matplotlib styling
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'DejaVu Sans',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        print(f"✨ Clean displaCy visualizations will be saved to: {self.OUTPUT_DIR}")

    def load_algorithm_data(self):
        """Load matches and explanations from the pipeline."""
        
        # Load matches
        matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
        if not matches_path.exists():
            raise FileNotFoundError(f"❌ No matches found! Run matching first: python src/matching/matcher.py")
        
        matches_df = pd.read_csv(matches_path)
        print(f"📂 Loaded {len(matches_df)} matches")
        
        # Load explanations
        explanations_path = self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json"
        if not explanations_path.exists():
            raise FileNotFoundError(f"❌ No explanations found! Run matching with save_explanations=True")
        
        with open(explanations_path, 'r', encoding='utf-8') as f:
            explanations = json.load(f)
        
        print(f"📊 Loaded {len(explanations)} detailed explanations")
        return matches_df, explanations

    def extract_algorithm_terms(self, explanation_data: dict):
        """Extract terms from actual algorithm explanations."""
        
        explanations = explanation_data.get('explanations', {})
        shared_terms = explanation_data.get('shared_terms', [])
        
        # Extract BM25 terms from explanation like "BM25: 3 matches [term1, term2]"
        bm25_terms = self._parse_bracketed_terms(explanations.get('bm25', ''))
        
        # Extract domain terms from explanation like "Domain: aerospace terms [nav, system]"
        domain_terms = self._parse_bracketed_terms(explanations.get('domain', ''))
        
        # Extract query expansion terms like "via [synonym1, synonym2]"
        expansion_terms = self._parse_expansion_terms(explanations.get('query_expansion', ''))
        
        return {
            'BM25_MATCH': bm25_terms,
            'DOMAIN_KNOWLEDGE': domain_terms,
            'SHARED_SEMANTIC': shared_terms,
            'QUERY_EXPANSION': expansion_terms
        }

    def _parse_bracketed_terms(self, explanation: str) -> list:
        """Extract terms from bracketed patterns like [term1, term2]."""
        terms = []
        bracket_matches = re.findall(r'\[([^\]]+)\]', explanation)
        for match in bracket_matches:
            terms.extend([t.strip() for t in match.split(',') if t.strip() and len(t.strip()) > 1])
        return terms

    def _parse_expansion_terms(self, explanation: str) -> list:
        """Extract query expansion terms from explanations."""
        terms = []
        
        # Look for "via [term1, term2]" pattern
        via_match = re.search(r'via\s+\[([^\]]+)\]', explanation, re.IGNORECASE)
        if via_match:
            terms.extend([t.strip() for t in via_match.group(1).split(',') if t.strip()])
        
        # Look for other expansion patterns
        terms.extend(self._parse_bracketed_terms(explanation))
        
        return [t for t in terms if t and len(t) > 1]

    def create_text_highlight_plot(self, text: str, term_annotations: dict, title: str, subtitle: str, filename: str):
        """Create matplotlib plot highlighting algorithm terms in text."""
        
        # Color scheme for different algorithm components
        colors = {
            'BM25_MATCH': '#e74c3c',         # Red - exact matches
            'DOMAIN_KNOWLEDGE': '#2ecc71',   # Green - aerospace knowledge
            'SHARED_SEMANTIC': '#3498db',    # Blue - semantic similarity
            'QUERY_EXPANSION': '#f39c12',    # Orange - synonym expansion
            'MATCHED_TERMS': '#e74c3c',      # Red - matched terms (for activities)
            'AEROSPACE_TERMS': '#3498db'     # Blue - aerospace terms (for activities)
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Add title and subtitle
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.88, subtitle, ha='center', va='top', fontsize=12, color='#7f8c8d', transform=ax.transAxes)
        
        # Split text into words for positioning
        words = text.split()
        if not words:
            ax.text(0.5, 0.5, "No text to analyze", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(self.OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return
        
        # Find which words should be highlighted
        word_highlights = {}
        for annotation_type, terms in term_annotations.items():
            color = colors.get(annotation_type, '#95a5a6')
            
            for term in terms:
                if not term or len(term.strip()) < 2:
                    continue
                
                term_lower = term.lower().strip()
                
                # Find term in text (simple word matching)
                for i, word in enumerate(words):
                    word_clean = re.sub(r'[^\w\s]', '', word.lower())
                    if word_clean == term_lower or term_lower in word_clean:
                        if i not in word_highlights or len(term) > len(word_highlights[i][1]):
                            word_highlights[i] = (annotation_type, term, color)
        
        # Layout words with highlights
        x_start = 0.05
        x_current = x_start
        y_pos = 0.65
        line_height = 0.08
        max_width = 0.9
        
        for i, word in enumerate(words):
            # Calculate approximate word width
            word_width = len(word) * 0.012 + 0.015
            
            # Check if we need to wrap to next line
            if x_current + word_width > max_width:
                y_pos -= line_height
                x_current = x_start
            
            # Add highlight if this word is annotated
            if i in word_highlights:
                annotation_type, term, color = word_highlights[i]
                
                # Add colored background rectangle
                rect = Rectangle((x_current - 0.005, y_pos - 0.02), 
                               word_width, 0.04, 
                               facecolor=color, alpha=0.4, 
                               transform=ax.transAxes)
                ax.add_patch(rect)
                
                # Add label above the word
                label_text = annotation_type.replace('_', ' ')
                ax.text(x_current + word_width/2, y_pos + 0.03, label_text, 
                       ha='center', va='bottom', fontsize=9, color=color, 
                       fontweight='bold', transform=ax.transAxes)
            
            # Add the word text
            ax.text(x_current, y_pos, word, ha='left', va='center', 
                   fontsize=12, transform=ax.transAxes)
            
            x_current += word_width + 0.01  # Space between words
        
        # Create legend for highlighted terms
        legend_elements = []
        used_annotations = set()
        
        for annotation_type, _, color in word_highlights.values():
            if annotation_type not in used_annotations:
                used_annotations.add(annotation_type)
                label = annotation_type.replace('_', ' ').title()
                legend_elements.append(mpatches.Patch(color=color, alpha=0.4, label=label))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower center', ncol=min(len(legend_elements), 4),
                     bbox_to_anchor=(0.5, -0.05))
        
        # Clean up plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save with high quality
        output_path = self.OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {output_path}")

    def create_requirement_visualization(self, explanation_data: dict):
        """Create visualization showing algorithm analysis of requirement."""
        
        req_id = explanation_data.get('requirement_id', 'REQ_UNKNOWN')
        req_text = explanation_data.get('requirement_text', '')
        
        print(f"\n🔍 Analyzing requirement {req_id}")
        print(f"   Text: {req_text[:80]}...")
        
        # Extract terms from algorithm explanations
        algorithm_terms = self.extract_algorithm_terms(explanation_data)
        
        # Count terms found
        term_counts = {k: len(v) for k, v in algorithm_terms.items() if v}
        print(f"   Found terms: {term_counts}")
        
        # Create title with algorithm scores
        scores = explanation_data.get('scores', {})
        title = f"Algorithm Analysis: Requirement {req_id}"
        subtitle = f"Semantic: {scores.get('semantic', 0):.3f} | BM25: {scores.get('bm25', 0):.3f} | Domain: {scores.get('domain', 0):.3f} | Query Expansion: {scores.get('query_expansion', 0):.3f}"
        
        # Create visualization
        filename = f"{req_id}_requirement_analysis.png"
        self.create_text_highlight_plot(req_text, algorithm_terms, title, subtitle, filename)

    def create_activity_visualization(self, explanation_data: dict):
        """Create visualization showing why activity matched."""
        
        req_id = explanation_data.get('requirement_id', 'REQ_UNKNOWN')
        activity_name = explanation_data.get('activity_name', '')
        
        print(f"🎯 Analyzing activity for {req_id}")
        print(f"   Activity: {activity_name}")
        
        # Get shared terms and find aerospace terms in activity
        shared_terms = explanation_data.get('shared_terms', [])
        
        # Use actual matcher to find aerospace terms in activity
        activity_terms = self.matcher._preprocess_text_aerospace(activity_name)
        aerospace_in_activity = [t for t in activity_terms if t in self.matcher.all_aerospace_terms]
        
        print(f"   Shared: {len(shared_terms)} terms, Aerospace: {len(aerospace_in_activity)} terms")
        
        # Create annotations for activity
        activity_annotations = {
            'MATCHED_TERMS': shared_terms,
            'AEROSPACE_TERMS': aerospace_in_activity
        }
        
        # Create title
        combined_score = explanation_data.get('combined_score', 0)
        title = f"Activity Match Analysis: {req_id}"
        subtitle = f"Combined Score: {combined_score:.3f} | Shared Terms: {len(shared_terms)} | Aerospace Terms: {len(aerospace_in_activity)}"
        
        # Create visualization
        filename = f"{req_id}_activity_analysis.png"
        self.create_text_highlight_plot(activity_name, activity_annotations, title, subtitle, filename)

    def visualize_top_matches(self, n_matches: int = 5):
        """Create visualizations for top N matches using actual algorithm data."""
        
        try:
            matches_df, explanations = self.load_algorithm_data()
            
            # Get top matches by score
            top_matches = matches_df.nlargest(n_matches, 'Combined_Score')
            
            print(f"\n🎯 Creating visualizations for top {n_matches} matches:")
            
            for idx, (_, match) in enumerate(top_matches.iterrows()):
                req_id = str(match['Requirement_ID'])
                score = match['Combined_Score']
                
                print(f"\n📊 Match {idx+1}: {req_id} (Score: {score:.3f})")
                
                # Find corresponding explanation data
                explanation = None
                for exp in explanations:
                    if str(exp.get('requirement_id', '')) == req_id:
                        explanation = exp
                        break
                
                if not explanation:
                    print(f"⚠️ No explanation found for {req_id}, skipping")
                    continue
                
                # Create both visualizations
                self.create_requirement_visualization(explanation)
                self.create_activity_visualization(explanation)
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False

    def create_summary_statistics(self, matches_df, explanations):
        """Create a summary statistics visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Algorithm Analysis Summary', fontsize=16, fontweight='bold')
        
        # Top left: Score distribution
        ax1 = axes[0, 0]
        scores = matches_df['Combined_Score']
        ax1.hist(scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {scores.mean():.3f}')
        ax1.set_title('Match Score Distribution')
        ax1.set_xlabel('Combined Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Top right: Algorithm components
        ax2 = axes[0, 1]
        if all(col in matches_df.columns for col in ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']):
            components = ['Semantic', 'BM25', 'Domain', 'Query Exp']
            scores = [
                matches_df['Semantic_Score'].mean(),
                matches_df['BM25_Score'].mean(), 
                matches_df['Domain_Score'].mean(),
                matches_df['Query_Expansion_Score'].mean()
            ]
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            bars = ax2.bar(components, scores, color=colors, alpha=0.8)
            ax2.set_title('Average Algorithm Performance')
            ax2.set_ylabel('Average Score')
            ax2.set_ylim(0, 1)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Bottom left: Quality distribution
        ax3 = axes[1, 0]
        high = len(matches_df[matches_df['Combined_Score'] >= 0.6])
        medium = len(matches_df[(matches_df['Combined_Score'] >= 0.4) & (matches_df['Combined_Score'] < 0.6)])
        low = len(matches_df[matches_df['Combined_Score'] < 0.4])
        
        if high + medium + low > 0:
            sizes = [high, medium, low]
            labels = ['High (≥0.6)', 'Medium (0.4-0.6)', 'Low (<0.4)']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            
            filtered = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if filtered:
                sizes, labels, colors = zip(*filtered)
                ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Match Quality Distribution')
        
        # Bottom right: Dataset overview  
        ax4 = axes[1, 1]
        stats = {
            'Total Matches': len(matches_df),
            'Unique Requirements': matches_df['Requirement_ID'].nunique(),
            'High Quality': high,
            'With Explanations': len(explanations)
        }
        
        bars = ax4.bar(stats.keys(), stats.values(), color=['lightblue', 'lightgreen', 'gold', 'orange'])
        ax4.set_title('Dataset Overview')
        ax4.set_ylabel('Count')
        
        for bar, count in zip(bars, stats.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats.values()) * 0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.OUTPUT_DIR / "algorithm_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Summary saved: {output_path}")

def main():
    """Generate clean displaCy-style visualizations from actual algorithm data."""
    
    print("🎨 Clean displaCy Visualizer")
    print("Using real algorithm analysis data")
    print("=" * 40)
    
    viz = CleanDisplacyVisualizer()
    
    try:
        # Create visualizations for top matches
        success = viz.visualize_top_matches(n_matches=3)
        
        if success:
            # Create summary
            matches_df, explanations = viz.load_algorithm_data()
            viz.create_summary_statistics(matches_df, explanations)
            
            print(f"\n🎉 All visualizations complete!")
            print(f"📂 Location: {viz.OUTPUT_DIR}")
            print(f"\n📋 Files created:")
            print(f"   • REQ_*_requirement_analysis.png - Algorithm analysis on requirements")
            print(f"   • REQ_*_activity_analysis.png - Why activities matched")
            print(f"   • algorithm_summary.png - Overall project statistics")
            print(f"\n🎯 Algorithm components shown:")
            print(f"   🔴 BM25 exact matches")
            print(f"   🟢 Domain knowledge terms") 
            print(f"   🔵 Semantic similarity terms")
            print(f"   🟠 Query expansion results")
            print(f"\n💡 Perfect for PowerPoint presentations!")
            print(f"💡 Shows actual NLP intelligence from your pipeline!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n🔧 To fix this:")
        print("   1. Run: python src/matching/matcher.py")
        print("   2. Ensure save_explanations=True in run_matching()")
        print("   3. Then run this visualizer")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()