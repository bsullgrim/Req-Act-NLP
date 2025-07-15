"""
Simple Dashboard Generator - Replaces complex modular dashboard
Creates a clean, maintainable HTML report with essential functionality
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SimpleDashboard:
    """Generate a clean, simple HTML dashboard without complex modals or JavaScript."""
    
    def __init__(self, repo_manager=None):
        if repo_manager is None:
            from src.utils.repository_setup import RepositoryStructureManager
            self.repo_manager = RepositoryStructureManager("outputs")
        else:
            self.repo_manager = repo_manager
    
    def create_dashboard(self, enhanced_df: pd.DataFrame, 
                        evaluation_results: Optional[Dict] = None,
                        output_name: str = "simple_dashboard") -> str:
        """
        Create a simple, clean HTML dashboard.
        
        Args:
            enhanced_df: DataFrame with predictions and quality analysis
            evaluation_results: Optional evaluation results
            output_name: Name for output file
            
        Returns:
            Path to generated HTML file
        """
        
        # Calculate summary stats
        stats = self._calculate_summary_stats(enhanced_df, evaluation_results)
        
        # Create HTML content
        html_content = self._build_html(enhanced_df, stats, evaluation_results)
        
        # Save dashboard
        output_path = self.repo_manager.structure['evaluation_dashboards'] / f"{output_name}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Simple dashboard created: {output_path}")
        return str(output_path)
    
    def _calculate_summary_stats(self, df: pd.DataFrame, eval_results: Optional[Dict]) -> Dict:
        """Calculate key summary statistics - ENHANCED COLUMN DETECTION."""
        
        # Flexible column detection
        score_col = None
        id_col = None
        
        # Find score column (multiple possible names)
        for col in ['Combined Score', 'Combined_Score', 'combined_score', 'score']:
            if col in df.columns:
                score_col = col
                break
        
        # Find ID column (multiple possible names)
        for col in ['ID', 'Requirement_ID', 'requirement_id', 'req_id']:
            if col in df.columns:
                id_col = col
                break
        
        stats = {
            'total_matches': len(df),
            'unique_requirements': df[id_col].nunique() if id_col else 0,
            'avg_score': df[score_col].mean() if score_col else 0,
            'high_confidence': len(df[df[score_col] >= 0.8]) if score_col else 0,
            'medium_confidence': len(df[(df[score_col] >= 0.5) & (df[score_col] < 0.8)]) if score_col else 0,
            'low_confidence': len(df[df[score_col] < 0.5]) if score_col else 0
        }
        
        # Quality stats (if available) - flexible column detection
        quality_grade_col = None
        quality_score_col = None
        
        for col in ['Quality_Grade', 'quality_grade', 'Quality Grade']:
            if col in df.columns:
                quality_grade_col = col
                break
        
        for col in ['Quality_Score', 'quality_score', 'Quality Score']:
            if col in df.columns:
                quality_score_col = col
                break
        
        if quality_grade_col:
            # Get unique requirements for quality (not matches)
            unique_req_quality = df.groupby(id_col).first() if id_col else df
            stats['quality_distribution'] = unique_req_quality[quality_grade_col].value_counts().to_dict()
            stats['avg_quality'] = unique_req_quality[quality_score_col].mean() if quality_score_col else 0
        else:
            stats['quality_distribution'] = {}
            stats['avg_quality'] = 0
        
        # Evaluation stats (compatible with simple evaluator format)
        if eval_results:
            # Handle both direct metrics and nested format
            if 'aggregate_metrics' in eval_results:
                stats['f1_at_5'] = eval_results['aggregate_metrics'].get('f1_at_5', {}).get('mean', 0)
                stats['coverage'] = eval_results.get('coverage', 0)
            else:
                stats['f1_at_5'] = eval_results.get('f1_at_5', 0)
                stats['coverage'] = eval_results.get('coverage', 0)
            
            stats['discoveries'] = 0  # Simple evaluator doesn't have discovery analysis
        else:
            stats['f1_at_5'] = 0
            stats['coverage'] = 0
            stats['discoveries'] = 0
        
        return stats        
    
    def _build_html(self, df: pd.DataFrame, stats: Dict, eval_results: Optional[Dict]) -> str:
        """Build complete HTML dashboard."""
        
        # Get table previews
        predictions_table = self._create_predictions_table(df.head(50))  # First 50 rows
        quality_summary = self._create_quality_summary(stats)
        
        # Discovery table (if available)
        discovery_table = ""
        if eval_results and 'discovery_analysis' in eval_results:
            discovery_data = eval_results['discovery_analysis'].get('high_scoring_misses', [])
            if discovery_data:
                discovery_df = pd.DataFrame(discovery_data[:20])  # Top 20
                discovery_table = self._create_discovery_table(discovery_df)
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Requirements Traceability Dashboard</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Requirements Traceability Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="summary-grid">
            {self._create_summary_cards(stats)}
        </div>
        
        <div class="section">
            <h2>📊 Quality Analysis</h2>
            {quality_summary}
        </div>
        
        <div class="section">
            <h2>🎯 Top Predictions</h2>
            <p class="description">Top 50 algorithm predictions sorted by confidence score</p>
            {predictions_table}
        </div>
        
        {f'''
        <div class="section">
            <h2>🔍 Discovery Analysis</h2>
            <p class="description">Novel connections found by algorithm (not in manual traces)</p>
            {discovery_table}
        </div>
        ''' if discovery_table else ''}
        
        <div class="section">
            <h2>📈 Performance Summary</h2>
            {self._create_performance_summary(stats, eval_results)}
        </div>
        
        <footer class="footer">
            <p>Requirements Traceability Analysis System | 
               Total: {stats['total_matches']} predictions | 
               Avg Score: {stats['avg_score']:.3f}</p>
        </footer>
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
    
    def _create_summary_cards(self, stats: Dict) -> str:
        """Create summary statistic cards."""
        
        cards = []
        
        # Main metrics
        cards.append(f"""
            <div class="card">
                <div class="card-value">{stats['total_matches']}</div>
                <div class="card-label">Total Predictions</div>
            </div>
        """)
        
        cards.append(f"""
            <div class="card">
                <div class="card-value">{stats['unique_requirements']}</div>
                <div class="card-label">Requirements</div>
            </div>
        """)
        
        cards.append(f"""
            <div class="card">
                <div class="card-value">{stats['avg_score']:.3f}</div>
                <div class="card-label">Average Score</div>
            </div>
        """)
        
        # Confidence distribution
        cards.append(f"""
            <div class="card high-conf">
                <div class="card-value">{stats['high_confidence']}</div>
                <div class="card-label">High Confidence</div>
            </div>
        """)
        
        cards.append(f"""
            <div class="card medium-conf">
                <div class="card-value">{stats['medium_confidence']}</div>
                <div class="card-label">Medium Confidence</div>
            </div>
        """)
        
        cards.append(f"""
            <div class="card low-conf">
                <div class="card-value">{stats['low_confidence']}</div>
                <div class="card-label">Low Confidence</div>
            </div>
        """)
        
        # Quality and discovery
        if stats['avg_quality'] > 0:
            cards.append(f"""
                <div class="card">
                    <div class="card-value">{stats['avg_quality']:.1f}</div>
                    <div class="card-label">Avg Quality</div>
                </div>
            """)
        
        if stats['discoveries'] > 0:
            cards.append(f"""
                <div class="card discovery">
                    <div class="card-value">{stats['discoveries']}</div>
                    <div class="card-label">Discoveries</div>
                </div>
            """)
        
        return '\n'.join(cards)
    
    def _create_predictions_table(self, df: pd.DataFrame) -> str:
        """Create simple predictions table with basic sorting."""
        
        # Select key columns
        display_cols = ['ID', 'Requirement Name', 'Activity Name', 'Combined Score']
        if 'Quality_Grade' in df.columns:
            display_cols.append('Quality_Grade')
        
        available_cols = [col for col in display_cols if col in df.columns]
        table_df = df[available_cols].copy()
        
        # Sort by score descending
        if 'Combined Score' in table_df.columns:
            table_df = table_df.sort_values('Combined Score', ascending=False)
        
        # Build table HTML
        html = f'''
        <div class="table-container">
            <table class="data-table" id="predictions-table">
                <thead>
                    <tr>
        '''
        
        # Headers
        for col in table_df.columns:
            html += f'<th onclick="sortTable(0, {list(table_df.columns).index(col)})">{col} ↕</th>'
        
        html += '''
                    </tr>
                </thead>
                <tbody>
        '''
        
        # Rows
        for _, row in table_df.iterrows():
            html += '<tr>'
            for col in table_df.columns:
                value = row[col]
                css_class = ""
                
                # Apply styling based on content
                if col == 'Combined Score':
                    if value >= 0.8:
                        css_class = "high-score"
                    elif value >= 0.5:
                        css_class = "medium-score"
                    else:
                        css_class = "low-score"
                    value = f"{value:.3f}"
                elif col == 'Quality_Grade':
                    css_class = f"quality-{value.lower()}" if isinstance(value, str) else ""
                elif isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                
                html += f'<td class="{css_class}" title="{row[col] if isinstance(row[col], str) else value}">{value}</td>'
            html += '</tr>'
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
    
    def _create_discovery_table(self, discovery_df: pd.DataFrame) -> str:
        """Create discovery analysis table."""
        
        if discovery_df.empty:
            return "<p>No discovery data available.</p>"
        
        # Select key columns
        display_cols = ['requirement_id', 'activity_name', 'score']
        if 'manual_matches_count' in discovery_df.columns:
            display_cols.append('manual_matches_count')
        
        available_cols = [col for col in display_cols if col in discovery_df.columns]
        table_df = discovery_df[available_cols].copy()
        
        # Sort by score
        if 'score' in table_df.columns:
            table_df = table_df.sort_values('score', ascending=False)
        
        html = '''
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Requirement ID</th>
                        <th>Discovered Activity</th>
                        <th>Discovery Score</th>
        '''
        
        if 'manual_matches_count' in table_df.columns:
            html += '<th>Manual Matches</th>'
        
        html += '''
                    </tr>
                </thead>
                <tbody>
        '''
        
        for _, row in table_df.iterrows():
            activity_name = row['activity_name']
            if len(activity_name) > 60:
                activity_name = activity_name[:57] + "..."
            
            score_class = "high-score" if row['score'] > 1.0 else "medium-score"
            
            html += f'''
                <tr>
                    <td><strong>{row['requirement_id']}</strong></td>
                    <td title="{row['activity_name']}">{activity_name}</td>
                    <td class="{score_class}">{row['score']:.3f}</td>
            '''
            
            if 'manual_matches_count' in table_df.columns:
                html += f'<td>{row["manual_matches_count"]}</td>'
            
            html += '</tr>'
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
    
    def _create_quality_summary(self, stats: Dict) -> str:
        """Create quality analysis summary."""
        
        quality_dist = stats.get('quality_distribution', {})
        avg_quality = stats.get('avg_quality', 0)
        
        if not quality_dist:
            return "<p>Quality analysis not available.</p>"
        
        html = f'''
        <div class="quality-summary">
            <div class="quality-stats">
                <div class="quality-stat">
                    <span class="stat-value">{avg_quality:.1f}/100</span>
                    <span class="stat-label">Average Quality Score</span>
                </div>
        '''
        
        # Quality grade distribution
        grade_colors = {
            'EXCELLENT': '#28a745',
            'GOOD': '#17a2b8', 
            'FAIR': '#ffc107',
            'POOR': '#fd7e14',
            'CRITICAL': '#dc3545'
        }
        
        for grade, count in quality_dist.items():
            color = grade_colors.get(grade, '#6c757d')
            html += f'''
                <div class="quality-stat">
                    <span class="stat-value" style="color: {color}">{count}</span>
                    <span class="stat-label">{grade}</span>
                </div>
            '''
        
        html += '''
            </div>
        </div>
        '''
        
        return html
    
    def _create_performance_summary(self, stats: Dict, eval_results: Optional[Dict]) -> str:
        """Create performance metrics summary."""
        
        if not eval_results:
            return '''
            <div class="performance-summary">
                <p><strong>Exploration Mode:</strong> No ground truth available for performance validation.</p>
                <p>Focus on high-confidence matches (≥0.8) for manual review and approval.</p>
            </div>
            '''
        
        coverage = stats.get('coverage', 0)
        f1_score = stats.get('f1_at_5', 0)
        discoveries = stats.get('discoveries', 0)
        
        html = f'''
        <div class="performance-summary">
            <div class="perf-grid">
                <div class="perf-item">
                    <span class="perf-value">{coverage:.1%}</span>
                    <span class="perf-label">Coverage</span>
                </div>
                <div class="perf-item">
                    <span class="perf-value">{f1_score:.3f}</span>
                    <span class="perf-label">F1@5 Score</span>
                </div>
                <div class="perf-item">
                    <span class="perf-value">{discoveries}</span>
                    <span class="perf-label">Novel Discoveries</span>
                </div>
            </div>
            <p><strong>Validation Mode:</strong> Algorithm performance validated against manual traces.</p>
        </div>
        '''
        
        return html
    
    def _get_css(self) -> str:
        """Return CSS styles."""
        return '''
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            font-weight: 300;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            padding: 2rem;
            background: #f8f9fa;
        }

        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
        }

        .card.high-conf { border-left-color: #28a745; }
        .card.medium-conf { border-left-color: #ffc107; }
        .card.low-conf { border-left-color: #dc3545; }
        .card.discovery { border-left-color: #6f42c1; }

        .card-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }

        .card-label {
            font-size: 0.9rem;
            color: #666;
        }

        .section {
            padding: 2rem;
            border-bottom: 1px solid #e9ecef;
        }

        .section h2 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.8rem;
            font-weight: 400;
        }

        .description {
            color: #666;
            margin-bottom: 1.5rem;
            font-style: italic;
        }

        .table-container {
            overflow-x: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }

        .data-table th {
            background: #343a40;
            color: white;
            padding: 1rem 0.75rem;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }

        .data-table th:hover {
            background: #495057;
        }

        .data-table td {
            padding: 0.75rem;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9rem;
        }

        .data-table tr:hover {
            background: #f8f9fa;
        }

        .high-score {
            background: #d4edda !important;
            color: #155724;
            font-weight: bold;
            text-align: center;
        }

        .medium-score {
            background: #fff3cd !important;
            color: #856404;
            font-weight: bold;
            text-align: center;
        }

        .low-score {
            background: #f8d7da !important;
            color: #721c24;
            font-weight: bold;
            text-align: center;
        }

        .quality-excellent { background: #d4edda; color: #155724; font-weight: bold; text-align: center; }
        .quality-good { background: #d1ecf1; color: #0c5460; font-weight: bold; text-align: center; }
        .quality-fair { background: #fff3cd; color: #856404; font-weight: bold; text-align: center; }
        .quality-poor { background: #f8d7da; color: #721c24; font-weight: bold; text-align: center; }
        .quality-critical { background: #f5c6cb; color: #721c24; font-weight: bold; text-align: center; }

        .quality-summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
        }

        .quality-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }

        .quality-stat {
            text-align: center;
            background: white;
            padding: 1rem;
            border-radius: 6px;
        }

        .stat-value {
            display: block;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            font-size: 0.85rem;
            color: #666;
        }

        .performance-summary {
            background: #e7f3ff;
            border-radius: 8px;
            padding: 1.5rem;
        }

        .perf-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .perf-item {
            text-align: center;
            background: white;
            padding: 1rem;
            border-radius: 6px;
        }

        .perf-value {
            display: block;
            font-size: 1.5rem;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 0.25rem;
        }

        .perf-label {
            font-size: 0.85rem;
            color: #666;
        }

        .footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 1.5rem;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .summary-grid {
                grid-template-columns: repeat(2, 1fr);
                padding: 1rem;
            }
            
            .section {
                padding: 1rem;
            }
            
            .data-table {
                font-size: 0.8rem;
            }
            
            .perf-grid {
                grid-template-columns: 1fr;
            }
        }
        '''
    
    def _get_javascript(self) -> str:
        """Return minimal JavaScript for table sorting."""
        return '''
        function sortTable(tableIndex, columnIndex) {
            const table = document.getElementsByClassName('data-table')[tableIndex];
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Determine sort direction
            const header = table.querySelectorAll('th')[columnIndex];
            const isAscending = !header.classList.contains('desc');
            
            // Reset all headers
            table.querySelectorAll('th').forEach(h => {
                h.classList.remove('asc', 'desc');
                h.innerHTML = h.innerHTML.replace(/[↑↓]/g, '↕');
            });
            
            // Sort rows
            rows.sort((a, b) => {
                const aVal = a.cells[columnIndex].textContent.trim();
                const bVal = b.cells[columnIndex].textContent.trim();
                
                // Try numeric comparison first
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return isAscending ? aNum - bNum : bNum - aNum;
                }
                
                // String comparison
                return isAscending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            });
            
            // Update table
            rows.forEach(row => tbody.appendChild(row));
            
            // Update header
            header.classList.add(isAscending ? 'asc' : 'desc');
            header.innerHTML = header.innerHTML.replace('↕', isAscending ? '↑' : '↓');
        }
        '''


def create_simple_dashboard(enhanced_df: pd.DataFrame, 
                          evaluation_results: Optional[Dict] = None,
                          repo_manager=None) -> str:
    """
    Convenience function to create simple dashboard.
    
    Args:
        enhanced_df: Enhanced predictions DataFrame
        evaluation_results: Optional evaluation results  
        repo_manager: Repository manager
        
    Returns:
        Path to created dashboard
    """
    dashboard = SimpleDashboard(repo_manager)
    return dashboard.create_dashboard(enhanced_df, evaluation_results)