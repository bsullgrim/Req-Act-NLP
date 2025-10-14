"""
Technical Journey Visualization - Plotly Version
Creates PowerPoint-ready visualizations with minimal fuss
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from scipy import stats

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer

"""
Technical Journey Visualization - Plotly Version
Creates PowerPoint-ready visualizations with minimal fuss
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from scipy import stats

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.matching.domain_resources import DomainResources
from src.quality.reqGrading import EnhancedRequirementAnalyzer


class TechnicalJourneyVisualizer:
    """
    Creates technical journey visualizations using Plotly for PowerPoint presentations.
    Much simpler than matplotlib - no manual positioning or font size fussing!
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
        
        print(f"✅ Initialized TechnicalJourneyVisualizer")
        print(f"   Output directory: {self.output_dir}")
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score threshold"""
        if score >= 0.8:
            return self.colors['good']
        elif score >= 0.6:
            return self.colors['warning']
        else:
            return self.colors['bad']
    
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
                for exp in explanations_list:
                    key = (exp.get('requirement_id'), exp.get('activity_name'))
                    explanations[key] = exp
        
        # Load quality analysis results (from reqGrading output)
        quality_path = self.repo_manager.structure['quality_analysis'] / "requirements_quality_report.csv"
        quality_df = None
        if quality_path.exists():
            quality_df = pd.read_csv(quality_path)
            print(f"✅ Loaded quality analysis from {quality_path}")
            
            # Merge quality data into matches if available
            quality_subset = quality_df[['ID', 'Total_Issues', 'Quality_Score', 'Quality_Grade', 
                                         'INCOSE_Best_Pattern', 'INCOSE_Compliance_Score']]
            quality_subset = quality_subset.rename(columns={'ID': 'Requirement_ID'})
            matches_df = matches_df.merge(quality_subset, on='Requirement_ID', how='left')
        else:
            print(f"⚠️  Quality analysis not found at {quality_path}")
            print("   Run reqGrading.py first for enhanced selection")
        
        # Load original requirements for fallback
        req_path = self.repo_manager.structure['data_raw'] / "requirements.csv"
        requirements_df = None
        if req_path.exists():
            requirements_df = self.file_handler.safe_read_csv(str(req_path))
        
        return matches_df, explanations, requirements_df
    
    def create_journey_html(self, req_id: str, label: str = '') -> str:
        """Dead simple: just display the JSON in a nice HTML format"""
        
        print(f"📊 Creating journey HTML for {req_id}...")
        
        # Load the JSON data (it's already being generated)
        json_path = self.output_dir / f"{req_id}_layer_data.json"
        
        if not json_path.exists():
            print(f"❌ JSON not found at {json_path}")
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            journey_data = json.load(f)
        
        # Create simple, clean HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Technical Journey: {req_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .layer {{
                    background: white;
                    margin: 20px 0;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .layer-title {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 15px;
                    border-left: 5px solid #3498db;
                    padding-left: 15px;
                }}
                .content {{
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .key {{
                    font-weight: bold;
                    color: #34495e;
                    display: inline-block;
                    min-width: 200px;
                }}
                .value {{
                    color: #555;
                }}
                .score {{
                    font-size: 32px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 0;
                }}
                .score.high {{ background: #d5f4e6; color: #27ae60; }}
                .score.medium {{ background: #fff3cd; color: #f39c12; }}
                .score.low {{ background: #fadbd8; color: #e74c3c; }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                    margin: 10px 0;
                }}
                .card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }}
                ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                li {{
                    margin: 5px 0;
                }}
                .decision {{
                    font-size: 28px;
                    font-weight: bold;
                    text-align: center;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .decision.accept {{ background: #d5f4e6; color: #27ae60; }}
                .decision.review {{ background: #fff3cd; color: #f39c12; }}
                .decision.reject {{ background: #fadbd8; color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Technical Journey: {req_id} ({label.upper()})</h1>
        """
        
        # Layer 1
        layer1 = journey_data.get('Layer 1: Raw Inputs', {})
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 1: Raw Inputs</div>
                <div class="content">
                    <div class="grid">
                        <div class="card">
                            <div class="key">Requirement ID:</div>
                            <div class="value">{layer1.get('requirement_id', 'N/A')}</div>
                        </div>
                        <div class="card">
                            <div class="key">Activity:</div>
                            <div class="value">{layer1.get('activity_name', 'N/A')}</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div class="key">Requirement Text:</div>
                        <div class="value">{layer1.get('requirement_text', 'N/A')}</div>
                    </div>
                </div>
            </div>
        """
        
        # Layer 2
        layer2 = journey_data.get('Layer 2: Preprocessing', {})
        tokenization = layer2.get('tokenization', {})
        extracted = layer2.get('extracted_terms', {})
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 2: Preprocessing</div>
                <div class="content">
                    <div class="grid">
                        <div class="card">
                            <div class="key">Tokenization</div>
                            <div class="value">
                                Total: {tokenization.get('total_tokens', 0)}<br>
                                Unique: {tokenization.get('unique_tokens', 0)}
                            </div>
                        </div>
                        <div class="card">
                            <div class="key">Extracted Terms</div>
                            <div class="value">
                                Requirement: {len(extracted.get('requirement_terms', []))}<br>
                                Activity: {len(extracted.get('activity_terms', []))}<br>
                                Aerospace: {len(extracted.get('aerospace_terms', []))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """
        
        # Layer 3
        layer3 = journey_data.get('Layer 3: Algorithm Analysis', {})
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 3: Algorithm Scores</div>
                <div class="content">
                    <div class="grid">
        """
        for algo in ['semantic', 'bm25', 'domain', 'query_expansion']:
            algo_data = layer3.get(algo, {})
            score = algo_data.get('score', 0)
            score_class = 'high' if score >= 0.8 else 'medium' if score >= 0.6 else 'low'
            html += f"""
                        <div class="card">
                            <div class="key">{algo.replace('_', ' ').title()}</div>
                            <div class="score {score_class}">{score:.3f}</div>
                        </div>
            """
        html += """
                    </div>
                </div>
            </div>
        """
        
        # Layer 4
        layer4 = journey_data.get('Layer 4: Score Combination', {})
        combined = layer4.get('combined_score', 0)
        combined_class = 'high' if combined >= 0.8 else 'medium' if combined >= 0.35 else 'low'
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 4: Combined Score</div>
                <div class="content">
                    <div style="text-align: center;">
                        <div class="score {combined_class}" style="font-size: 48px;">{combined:.3f}</div>
                        <div style="margin-top: 10px; font-size: 18px;">
                            Classification: {layer4.get('classification', 'N/A')}
                        </div>
                    </div>
                </div>
            </div>
        """
        
        # Layer 5
        layer5 = journey_data.get('Layer 5: INCOSE Analysis', {})
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 5: INCOSE Analysis</div>
                <div class="content">
                    <div class="key">Pattern:</div>
                    <div class="value">{layer5.get('best_pattern', 'N/A')}</div>
                    <div class="key">Compliance Score:</div>
                    <div class="value">{layer5.get('compliance_score', 0)}%</div>
                    <div class="key">Required Components:</div>
                    <div class="value">{layer5.get('required_present', 'N/A')}</div>
                    <div class="key">Optional Components:</div>
                    <div class="value">{layer5.get('optional_present', 'N/A')}</div>
                </div>
            </div>
        """
        
        # Layer 6
        layer6 = journey_data.get('Layer 6: Quality Analysis', {})
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 6: Quality Analysis</div>
                <div class="content">
                    <div class="grid">
                        <div class="card">
                            <div class="key">Overall Score</div>
                            <div class="score medium">{layer6.get('overall_score', 0):.0f}%</div>
                        </div>
                        <div class="card">
                            <div class="key">Grade</div>
                            <div class="value" style="font-size: 24px; font-weight: bold;">{layer6.get('grade', 'N/A')}</div>
                        </div>
                        <div class="card">
                            <div class="key">Total Issues</div>
                            <div class="value" style="font-size: 24px;">{layer6.get('total_issues', 0)}</div>
                        </div>
                    </div>
                </div>
            </div>
        """
        
        # Layer 7
        layer7 = journey_data.get('Layer 7: Final Decision', {})
        decision = layer7.get('decision', 'N/A')
        decision_class = 'accept' if 'ACCEPT' in decision else 'review' if 'REVIEW' in decision else 'reject'
        html += f"""
            <div class="layer">
                <div class="layer-title">Layer 7: Final Decision</div>
                <div class="content">
                    <div class="decision {decision_class}">
                        {decision}
                    </div>
                    <div style="text-align: center; font-size: 16px;">
                        {layer7.get('metrics', 'N/A')}
                    </div>
                </div>
            </div>
        """
        
        html += """
        </body>
        </html>
        """
        
        # Save HTML
        output_path = self.output_dir / f"{req_id}_technical_journey_{label}.html"
        output_path.write_text(html, encoding='utf-8')
        
        print(f"✅ Saved journey HTML: {output_path}")
        return str(output_path)
    
    def create_algorithm_contribution_chart_plotly(self):
        """Create algorithm contribution analysis using Plotly"""
        
        print("\n📊 Creating algorithm contribution chart...")
        
        # Load matches
        matches_df, _, _ = self.load_real_data()
        
        if matches_df.empty:
            print("❌ No matches to analyze")
            return
        
        # Calculate average contribution of each algorithm
        algo_names = ['Semantic', 'BM25', 'Domain', 'Query Expansion']
        algo_cols = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        algo_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Average Algorithm Scores</b>',
                '<b>Score Distribution</b>',
                '<b>Algorithm Correlation</b>',
                '<b>Combined Score vs Algorithms</b>'
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Average scores bar chart
        avg_scores = [matches_df[col].mean() for col in algo_cols]
        
        fig.add_trace(
            go.Bar(
                x=algo_names,
                y=avg_scores,
                text=[f'{s:.3f}' for s in avg_scores],
                textposition='auto',
                marker_color=algo_colors,
                hovertemplate='%{x}<br>Average: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Box plots for distribution
        for i, (name, col, color) in enumerate(zip(algo_names, algo_cols, algo_colors)):
            fig.add_trace(
                go.Box(
                    y=matches_df[col],
                    name=name,
                    marker_color=color,
                    boxmean='sd',
                    hovertemplate='%{y:.4f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Correlation heatmap (as scatter for simplicity)
        # Show correlation between semantic and domain as example
        fig.add_trace(
            go.Scatter(
                x=matches_df['Semantic_Score'],
                y=matches_df['Domain_Score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=matches_df['Combined_Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Combined<br>Score", x=0.46)
                ),
                text=matches_df['Requirement_ID'],
                hovertemplate='Semantic: %{x:.3f}<br>Domain: %{y:.3f}<br>%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add trend line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            matches_df['Semantic_Score'], matches_df['Domain_Score']
        )
        line_x = [matches_df['Semantic_Score'].min(), matches_df['Semantic_Score'].max()]
        line_y = [slope * x + intercept for x in line_x]
        
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'Trend (R²={r_value**2:.3f})',
                showlegend=False,
                hovertemplate='Trendline<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Combined score vs individual algorithms
        for name, col, color in zip(algo_names, algo_cols, algo_colors):
            fig.add_trace(
                go.Scatter(
                    x=matches_df['Combined_Score'],
                    y=matches_df[col],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=6, opacity=0.6),
                    hovertemplate='Combined: %{x:.3f}<br>'+name+': %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update axes labels
        fig.update_xaxes(title_text="Algorithm", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Algorithm", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Semantic Score", row=2, col=1)
        fig.update_yaxes(title_text="Domain Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Combined Score", row=2, col=2)
        fig.update_yaxes(title_text="Individual Algorithm Score", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1400,
            width=2000,
            title_text="<b>Algorithm Contribution Analysis</b>",
            title_font_size=28,
            title_x=0.5,
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            showlegend=True,
            legend=dict(x=1.02, y=0.5),
            margin=dict(l=80, r=150, t=120, b=80)
        )
        
        fig.update_yaxes(gridcolor='lightgray')
        
        # Save as high-res PNG
        # output_path = self.output_dir / "algorithm_contribution_analysis.png"
        # fig.write_image(str(output_path), width=2000, height=1400, scale=2)
        output_path_html = self.output_dir / "algorithm_contribution_analysis.html"
        fig.write_html(str(output_path_html))
        print(f"✅ Saved contribution analysis: {output_path_html}")
        return str(output_path_html)

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
            print(f"\n📊 Creating journey for {label.upper()} scoring pair: {req_id}")
            try:
                # # First create the matplotlib version to generate JSON
                # path = self.create_journey_html(requirement_id=req_id, label=label)
                
                # Then create simple HTML from that JSON
                path = self.create_journey_html(req_id=req_id, label=label)
                
                # paths.append(path)
                # if html_path:
                #     paths.append(html_path)
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
    
        # Create algorithm contribution chart
        try:
            contrib_path = self.create_algorithm_contribution_chart_plotly()
            if contrib_path:
                paths.append(contrib_path)
        except Exception as e:
            print(f"❌ Error creating contribution chart: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*70)
        print(f"✅ Created {len(paths)} visualizations")
        print("Files saved to: outputs/visuals/")
        print("="*70)
        
        return paths


def main():
    """Main function to create technical visualizations"""
    
    try:
        print("\n" + "="*70)
        print("🎨 PLOTLY TECHNICAL JOURNEY VISUALIZER")
        print("="*70)
        
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
        print("Optionally run reqGrading.py for enhanced selection")
        return []
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()