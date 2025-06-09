"""
HTML Templates - Main HTML structure and styling
FIXED VERSION: JavaScript syntax errors corrected
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

class HTMLTemplateGenerator:
    """Generates the main HTML structure and integrates all components."""
    
    def build_dashboard_html(self, processed_data: Dict[str, Any], 
                        charts: Dict[str, str], 
                        tables: Dict[str, str],
                        capabilities: Optional[Dict[str, bool]] = None) -> str:
        """Build the complete HTML dashboard."""
        
        self.capabilities = capabilities or {}  # Store capabilities for use in other methods
        summary_stats = processed_data.get('summary_stats', {})
        
        return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Requirements Traceability Evaluation Dashboard - Modular</title>
        <style>
            {self._get_css_styles()}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            {self._build_header(summary_stats)}
            {self._build_navigation()}
            {self._build_sections(charts, tables)}
            {self._build_modal()}
            {self._build_footer(summary_stats)}
        </div>
        <script>
            {self._get_javascript()}
        </script>
    </body>
    </html>
        """
        
    def _build_header(self, summary_stats: Dict) -> str:
        """Build dashboard header with summary statistics."""
        return f"""
        <header class="dashboard-header">
            <h1>Requirements Traceability Evaluation Dashboard</h1>
            <div class="header-stats">
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('total_requirements', 0)}</div>
                    <div class="stat-label">Total Requirements</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('total_predictions', 0)}</div>
                    <div class="stat-label">Total Predictions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('coverage_percentage', 0):.1f}%</div>
                    <div class="stat-label">Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('f1_at_5', 0):.3f}</div>
                    <div class="stat-label">F1@5 Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('discoveries_count', 0)}</div>
                    <div class="stat-label">Discoveries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary_stats.get('score_gaps_count', 0)}</div>
                    <div class="stat-label">Score Gaps</div>
                </div>
            </div>
            <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>
        """
    
    def _build_navigation(self) -> str:
        """Build adaptive navigation based on capabilities."""
        caps = getattr(self, 'capabilities', {})
        
        tabs = [
            ('exploration', '📋 Match Exploration', True),  # Always shown
            ('quality', '🎯 Quality Intelligence', caps.get('quality_analysis', True)),  # Always shown  
            ('validation', '📊 Algorithm Validation', caps.get('validation_mode', False)),  # Only if ground truth
            ('coverage', '📈 Coverage Analysis', True),  # Always shown
            ('discoveries', '🔍 Discovery Insights', True),  # Always shown but adaptive content
        ]
        
        nav_html = '<nav class="dashboard-nav">'
        first_tab = True
        for tab_id, tab_name, show_tab in tabs:
            if show_tab:
                active_class = ' active' if first_tab else ''
                nav_html += f'<button class="nav-btn{active_class}" onclick="showSection(\'{tab_id}\')">{tab_name}</button>'
                first_tab = False
        nav_html += '</nav>'
        
        return nav_html
    
    def _build_sections(self, charts: Dict[str, str], tables: Dict[str, str]) -> str:
        """Build all dashboard sections with adaptive content."""
        caps = getattr(self, 'capabilities', {})
        
        sections_html = ""
        
        # Tab 1: Match Exploration (Always Available)
        sections_html += f"""
        <section id="exploration-section" class="dashboard-section active">
            <h2>Match Exploration</h2>
            <div class="alert alert-info">
                <strong>Match Exploration:</strong> Browse and analyze all algorithm predictions with 
                searchable/sortable interface and detailed score breakdowns.
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('score_distribution', '')}" 
                    alt="Score Distribution Chart" class="chart-image">
            </div>
            
            <div class="data-table-container">
                <h3>All Predictions</h3>
                {tables.get('all_predictions', '<p>No predictions data available.</p>')}
            </div>
        </section>
        """
        
        # Tab 2: Quality Intelligence (Always Available)
        if caps.get('quality_analysis', True):
            sections_html += f"""
            <section id="quality-section" class="dashboard-section">
                <h2>Quality Intelligence</h2>
                <div class="alert alert-warning">
                    <strong>Quality Analysis:</strong> Requirement quality metrics and correlation with matching performance.
                    Poor quality requirements often lead to poor matches.
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts.get('quality_analysis', '')}" 
                        alt="Quality Analysis Chart" class="chart-image">
                </div>
                
                <div class="table-container">
                    <h3>Quality Analysis Summary</h3>
                    {tables.get('quality_analysis', '<p>Quality analysis data is not available. Ensure your predictions contain quality columns like Quality_Grade, Quality_Score, etc.</p>')}
                </div>
            </section>
            """
        
        # Tab 3: Algorithm Validation (Only if manual_matches exist)
        if caps.get('validation_mode', False):
            sections_html += f"""
            <section id="validation-section" class="dashboard-section">
                <h2>Algorithm Validation</h2>
                <div class="alert alert-success">
                    <strong>Validation Mode:</strong> Performance metrics against ground truth for algorithm assessment.
                    Use these metrics to determine deployment readiness.
                </div>
                
                <div class="chart-grid">
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts.get('performance_metrics', '')}" 
                            alt="Performance Metrics Chart" class="chart-image">
                    </div>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts.get('mrr_ndcg', '')}" 
                            alt="MRR and NDCG Chart" class="chart-image">
                    </div>
                </div>
                
                <div class="table-container">
                    <h3>Performance Summary</h3>
                    {tables.get('performance_summary', '<p>No performance data available.</p>')}
                </div>
            </section>
            """
        
        # Tab 4: Coverage Analysis (Always Available)
        sections_html += f"""
        <section id="coverage-section" class="dashboard-section">
            <h2>Coverage Analysis</h2>
            <div class="alert alert-info">
                <strong>Coverage Analysis:</strong> Review which requirements have strong/weak matches 
                and prioritize review efforts.
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('coverage_analysis', '')}" 
                    alt="Coverage Analysis Chart" class="chart-image">
            </div>
            
            <div class="table-container">
                <h3>Coverage Analysis Summary</h3>
                {tables.get('coverage_analysis', '<p>Coverage analysis data is not available. This usually means predictions data could not be processed.</p>')}
            </div>
        </section>
        """
        
        # Tab 5: Discovery Insights (Adaptive)
        sections_html += f"""
        <section id="discoveries-section" class="dashboard-section">
            <h2>Discovery Insights</h2>
            """
        
        if caps.get('validation_mode', False):
            sections_html += f"""
            <div class="alert alert-success">
                <strong>Discovery Analysis:</strong> High-scoring algorithm suggestions not found in manual traces.
                These represent potential missed connections worth reviewing.
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('discovery_overview', '')}" 
                    alt="Discovery Analysis Chart" class="chart-image">
            </div>
            
            <div class="table-container">
                <h3>Discovery Examples</h3>
                {tables.get('discovery_results', '<p>No discovery examples available.</p>')}
            </div>
            
            <div class="table-container">
                <h3>Score Gaps Analysis</h3>
                {tables.get('score_gaps', '<p>No score gaps found.</p>')}
            </div>
            """
        else:
            sections_html += f"""
            <div class="alert alert-info">
                <strong>High-Confidence Discoveries:</strong> Novel connections identified by the algorithm.
                Without ground truth, focus on highest-scoring suggestions for manual review.
            </div>
            
            <div class="table-container">
                <h3>High-Confidence Novel Connections</h3>
                {tables.get('high_confidence_matches', '<p>High-confidence analysis coming soon.</p>')}
            </div>
            """
        
        sections_html += "</section>"
        
        return sections_html
    
    def _build_modal(self) -> str:
        """Build modal for detailed row views."""
        return """
        <div id="row-detail-modal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <div id="modal-body">
                    <!-- Dynamic content -->
                </div>
            </div>
        </div>
        """
    
    def _build_footer(self, summary_stats: Dict) -> str:
        """Build dashboard footer."""
        return f"""
        <footer class="dashboard-footer">
            <p>Generated by Modular Requirements Traceability Evaluation System | 
               Dataset: {summary_stats.get('total_requirements', 0)} requirements, {summary_stats.get('total_predictions', 0)} predictions | 
               F1@5 = {summary_stats.get('f1_at_5', 0):.3f} | 
               Discoveries: {summary_stats.get('discoveries_count', 0)} | 
               Score Gaps: {summary_stats.get('score_gaps_count', 0)}</p>
        </footer>
        """
    
    def _get_css_styles(self) -> str:
        """Return CSS styles for the dashboard."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .dashboard-container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        .dashboard-header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .dashboard-header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 300;
        }

        .header-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: scale(1.05);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .timestamp {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 1rem;
        }

        .dashboard-nav {
            background: #f8f9fa;
            padding: 1rem 2rem;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .nav-btn {
            background: none;
            border: none;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        .nav-btn:hover {
            background: #e9ecef;
        }

        .nav-btn.active {
            background: #007bff;
            color: white;
        }

        .dashboard-section {
            display: none;
            padding: 2rem;
        }

        .dashboard-section.active {
            display: block;
        }

        .dashboard-section h2 {
            color: #2c3e50;
            margin-bottom: 1.5rem;
            font-size: 2rem;
            font-weight: 300;
        }

        .section-description {
            color: #666;
            margin-bottom: 2rem;
            font-style: italic;
        }

        .alert {
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .alert-info {
            background: #e7f3ff;
            border-color: #007bff;
            color: #0056b3;
        }

        .alert-success {
            background: #e8f5e8;
            border-color: #28a745;
            color: #155724;
        }

        .alert-warning {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .chart-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem;
        }

        .chart-container.full-width {
            grid-column: 1 / -1;
        }

        .chart-image {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }

        .table-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .table-container h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.5rem;
            font-weight: 500;
        }

        .table-description {
            color: #666;
            margin-bottom: 1rem;
            font-style: italic;
        }

        .data-table-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .table-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .search-box {
            position: relative;
            flex: 1;
            min-width: 250px;
        }

        .search-input {
            width: 100%;
            padding: 0.5rem 1rem 0.5rem 2.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .search-icon {
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
        }

        .filter-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .filter-select {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .export-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s ease;
        }

        .export-btn:hover {
            background: #218838;
        }

        .table-wrapper {
            max-height: 70vh;
            overflow: auto;
        }

        .sortable-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .sortable-table th {
            background: #f8f9fa;
            padding: 1rem 0.5rem;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 10;
            font-weight: 600;
            color: #495057;
        }

        .sortable-table th.sortable {
            cursor: pointer;
            user-select: none;
        }

        .sortable-table th.sortable:hover {
            background: #e9ecef;
        }

        .sort-arrow {
            margin-left: 0.5rem;
            opacity: 0.5;
        }

        .sortable-table td {
            padding: 0.75rem 0.5rem;
            border-bottom: 1px solid #dee2e6;
            vertical-align: top;
        }

        .data-row {
            transition: background-color 0.2s ease;
            cursor: pointer;
        }

        .data-row:hover {
            background-color: #f8f9fa;
        }

        .req-id {
            font-weight: bold;
            color: #007bff;
        }

        .req-name, .activity-name {
            max-width: 200px;
            word-wrap: break-word;
        }

        .score-cell {
            text-align: center;
            font-weight: bold;
        }

        .high {
            background-color: #d4edda !important;
        }

        .medium {
            background-color: #fff3cd !important;
        }

        .low {
            background-color: #f8d7da !important;
        }

        .table-info {
            padding: 1rem;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #666;
            font-size: 14px;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            animation: fadeIn 0.3s ease;
        }

        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 8px;
            width: 90%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
            animation: slideIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .close {
            position: absolute;
            right: 1rem;
            top: 1rem;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: #aaa;
        }

        .close:hover {
            color: #000;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .analysis-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1.5rem;
        }

        .analysis-card h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.3rem;
            font-weight: 500;
        }

        .quality-metrics {
            space-y: 0.75rem;
        }

        .quality-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #f1f1f1;
        }

        .quality-label {
            font-weight: 500;
            color: #495057;
        }

        .quality-value {
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }

        .recommendations {
            list-style: none;
        }

        .recommendations li {
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }

        .recommendations li::before {
            content: "•";
            position: absolute;
            left: 0;
            font-weight: bold;
        }

        .rec-high::before { color: #dc3545; }
        .rec-medium::before { color: #ffc107; }
        .rec-low::before { color: #28a745; }
        .rec-info::before { color: #007bff; }

        .dashboard-footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 1.5rem;
            font-size: 0.9rem;
        }

        /* Standard table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        .performance-table .highlight-cell {
            background: #e7f3ff;
            font-weight: bold;
        }

        .center-cell {
            text-align: center;
            font-weight: bold;
        }

        .high-score {
            background: #d4edda;
            color: #155724;
            font-weight: bold;
            text-align: center;
        }

        .medium-score {
            background: #fff3cd;
            color: #856404;
            font-weight: bold;
            text-align: center;
        }

        .activity-name {
            font-size: 0.9rem;
            max-width: 300px;
            word-wrap: break-word;
        }

        .manual-count {
            text-align: center;
            font-weight: bold;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .dashboard-header {
                padding: 1rem;
            }

            .dashboard-header h1 {
                font-size: 1.8rem;
            }

            .header-stats {
                grid-template-columns: repeat(2, 1fr);
            }

            .chart-grid {
                grid-template-columns: 1fr;
            }

            .dashboard-nav {
                padding: 1rem;
                flex-direction: column;
            }

            .nav-btn {
                width: 100%;
                margin: 0.25rem 0;
            }

            .dashboard-section {
                padding: 1rem;
            }

            .table-controls {
                flex-direction: column;
                align-items: stretch;
            }

            .search-box {
                min-width: auto;
            }

            .modal-content {
                width: 95%;
                margin: 2% auto;
                padding: 1rem;
            }
        
        .coverage-analysis-container, .quality-analysis-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }

        .coverage-section, .quality-section {
            margin-bottom: 2rem;
            padding: 1.5rem;
            border-bottom: 1px solid #dee2e6;
        }

        .coverage-section:last-child, .quality-section:last-child {
            border-bottom: none;
        }

        .coverage-section h4, .quality-section h4 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
        }

        .coverage-table, .quality-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .coverage-table th, .quality-table th,
        .coverage-table td, .quality-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        .coverage-table th, .quality-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        .coverage-summary, .effort-summary {
            margin-top: 1rem;
            padding: 1rem;
            background: #e7f3ff;
            border-radius: 4px;
            text-align: center;
            color: #0056b3;
        }

        .recommendations-list {
            margin-top: 1rem;
        }

        .recommendation-item {
            margin: 0.75rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .priority-badge {
            background-color: #007bff;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            min-width: 60px;
            text-align: center;
        }

        .rec-text {
            flex: 1;
            font-weight: 500;
        }
                .modal-header {
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        .modal-header h3 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.5rem;
        }

        .modal-section {
            margin-bottom: 2rem;
        }

        .modal-section h4 {
            color: #495057;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            border-left: 4px solid #007bff;
            padding-left: 1rem;
        }

        .modal-field {
            margin-bottom: 0.75rem;
        }

        .req-text-box, .activity-text-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.5;
            color: #495057;
            max-height: 150px;
            overflow-y: auto;
        }

        .score-breakdown {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
        }

        .score-item {
            margin-bottom: 1rem;
        }

        .score-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }

        .score-bar {
            position: relative;
            background: #e9ecef;
            border-radius: 20px;
            height: 30px;
            overflow: hidden;
        }

        .score-fill {
            background: linear-gradient(90deg, #28a745 0%, #20c997 50%, #17a2b8 100%);
            height: 100%;
            border-radius: 20px;
            transition: width 0.3s ease;
            position: relative;
        }

        .score-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            color: #2c3e50;
            font-size: 0.9rem;
            z-index: 1;
        }

        .confidence-indicator {
            margin-top: 1.5rem;
            padding: 1rem;
            background: white;
            border-radius: 6px;
            text-align: center;
        }

        .confidence-badge {
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .confidence-badge.high-conf {
            background: #d4edda;
            color: #155724;
        }

        .confidence-badge.medium-conf {
            background: #fff3cd;
            color: #856404;
        }

        .confidence-badge.low-conf {
            background: #f8d7da;
            color: #721c24;
        }
        
        .match-summary {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .overall-score {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .score-description {
            display: block;
            font-size: 0.85rem;
            color: #666;
            font-weight: normal;
            margin-top: 0.25rem;
        }

        .contribution {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.25rem;
            font-style: italic;
        }

        .semantic-fill {
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        }

        .bm25-fill {
            background: linear-gradient(90deg, #007bff 0%, #17a2b8 100%);
        }

        .other-fill {
            background: linear-gradient(90deg, #6f42c1 0%, #e83e8c 100%);
        }

        .insights-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }

        .insight-item {
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }

        .manual-matches-box {
            background: #e7f3ff;
            border: 1px solid #b3d7ff;
            border-radius: 6px;
            padding: 1rem;
            margin-top: 0.5rem;
        }

        .manual-match-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid #d1ecf1;
            color: #0c5460;
        }

        .manual-match-item:last-child {
            border-bottom: none;
        }

        .score-highlight {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #fff3cd;
            border-radius: 4px;
            font-weight: bold;
            color: #856404;
        }

        @media (max-width: 768px) {
            .insights-grid {
                grid-template-columns: 1fr;
            }
            
            .match-summary {
                flex-direction: column;
                gap: 1rem;
            }
        }
        .shared-terms-container {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
            margin-top: 0.5rem;
        }

        .shared-terms-list {
            margin-bottom: 0.75rem;
        }

        .shared-term-badge {
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            padding: 0.25rem 0.5rem;
            margin: 0.125rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .evidence-summary {
            font-size: 0.9rem;
            color: #495057;
            font-style: italic;
        }

        /* Match Explanation */
        .match-explanation {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
        }

        .explanation-item {
            margin-bottom: 1rem;
        }

        .explanation-item ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }

        .explanation-item li {
            margin: 0.25rem 0;
            line-height: 1.4;
            font-size: 0.9rem;
        }

        .matcher-summary {
            background: #e7f3ff;
            border-radius: 4px;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }

        .matcher-summary p {
            margin: 0.25rem 0;
            font-size: 0.9rem;
        }

        .recommendation-box {
            margin-top: 1rem;
            padding: 0.75rem;
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }

        /* Discovery Modal Styles */
        .discovery-header-summary {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .discovery-badge {
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.85rem;
        }

        .discovery-badge.high-disc {
            background: #d4edda;
            color: #155724;
        }

        .discovery-badge.medium-disc {
            background: #fff3cd;
            color: #856404;
        }

        .discovery-badge.low-disc {
            background: #f8d7da;
            color: #721c24;
        }

        .discovery-score {
            font-weight: bold;
            color: #2c3e50;
        }

        .discovery-score-details {
            margin-top: 1rem;
        }

        .discovery-explanation {
            margin-top: 0.5rem;
            padding: 0.75rem;
            background: #e7f3ff;
            border-radius: 4px;
            font-size: 0.9rem;
            line-height: 1.4;
        }

        /* Manual Analysis Containers */
        .manual-analysis-container {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
        }

        .manual-count-display {
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            font-weight: 500;
        }

        .manual-score-display {
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            font-weight: 500;
        }

        .manual-match-item {
            display: flex;
            align-items: flex-start;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
        }

        .manual-match-item:last-child {
            border-bottom: none;
        }

        .match-number {
            color: #007bff;
            font-weight: bold;
            margin-right: 0.5rem;
            min-width: 20px;
        }

        .match-text {
            flex: 1;
            line-height: 1.4;
        }

        /* Analysis Comparison */
        .analysis-comparison {
            margin-top: 1rem;
            padding: 1rem;
            background: #fff;
            border-radius: 4px;
            border-left: 4px solid #17a2b8;
        }

        .comparison-insight {
            margin-bottom: 0.75rem;
            color: #495057;
        }

        .discovery-value {
            font-size: 0.9rem;
            color: #6c757d;
            font-style: italic;
        }

        /* Gap Analysis Styles */
        .gap-header-summary {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .gap-badge {
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.85rem;
        }

        .gap-badge.large-gap {
            background: #f8d7da;
            color: #721c24;
        }

        .gap-badge.medium-gap {
            background: #fff3cd;
            color: #856404;
        }

        .gap-badge.small-gap {
            background: #d4edda;
            color: #155724;
        }

        .gap-score {
            font-weight: bold;
            color: #2c3e50;
        }

        .algorithm-score-details {
            margin-top: 1rem;
        }

        .gap-explanation {
            margin-top: 0.5rem;
            padding: 0.75rem;
            background: #fff3cd;
            border-radius: 4px;
            font-size: 0.9rem;
            line-height: 1.4;
        }

        .gap-comparison {
            margin-top: 1rem;
            padding: 1rem;
            background: #fff;
            border-radius: 4px;
            border-left: 4px solid #dc3545;
        }

        .gap-significance {
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: #6c757d;
            font-style: italic;
        }

        /* Error and Warning States */
        .no-manual-matches {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 1rem;
        }

        .no-manual-matches ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }

        .no-manual-data {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 1rem;
            color: #721c24;
        }

        /* Insights Grid */
        .insights-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .insight-item {
            background: white;
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }

        .insight-item.full-width {
            grid-column: 1 / -1;
            border-left-color: #28a745;
        }

        .metric-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0.25rem;
        }

        .metric-value.confidence-high {
            color: #28a745;
        }

        .metric-value.confidence-medium {
            color: #ffc107;
        }

        .metric-value.confidence-low {
            color: #dc3545;
        }

        .metric-value.gap-large {
            color: #dc3545;
        }

        .metric-value.gap-medium {
            color: #ffc107;
        }

        .metric-value.gap-small {
            color: #28a745;
        }

        .metric-description {
            font-size: 0.85rem;
            color: #6c757d;
            line-height: 1.3;
        }

        .recommendation-content {
            background: #e8f5e8;
            border-radius: 4px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            line-height: 1.4;
        }

        .score-description {
            display: block;
            font-size: 0.8rem;
            color: #666;
            font-weight: normal;
            margin-top: 0.25rem;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .insights-grid {
                grid-template-columns: 1fr;
            }
            
            .discovery-header-summary, .gap-header-summary {
                flex-direction: column;
                gap: 0.5rem;
                text-align: center;
            }
        }
        """
    
    def _get_javascript(self) -> str:
        """Return JavaScript for dashboard interactivity - FIXED VERSION with integrated modals."""
        return """
        // Global variables for sorting
        let predictionsSort = { column: -1, ascending: true };

        // Section management
        function showSection(sectionName) {
            // Hide all sections
            document.querySelectorAll('.dashboard-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from all nav buttons
            document.querySelectorAll('.nav-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected section
            const selectedSection = document.getElementById(sectionName + '-section');
            if (selectedSection) {
                selectedSection.classList.add('active');
            }
            
            // Activate corresponding nav button
            if (event && event.target) {
                event.target.classList.add('active');
            }
        }

        // SIMPLIFIED search functions - NO FILTERING
        function searchPredictionsTable() {
            const searchTerm = document.getElementById('predictions-search').value.toLowerCase();
            const rows = document.querySelectorAll('#predictions-table .data-row');
            let visibleCount = 0;

            rows.forEach(row => {
                const reqId = row.querySelector('.req-id').textContent.toLowerCase();
                const reqName = row.querySelector('.req-name').textContent.toLowerCase();
                const activityName = row.querySelector('.activity-name').textContent.toLowerCase();

                const searchMatch = !searchTerm || 
                                reqId.includes(searchTerm) || 
                                reqName.includes(searchTerm) || 
                                activityName.includes(searchTerm);

                if (searchMatch) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            document.getElementById('predictions-count').textContent = `Showing ${visibleCount} results`;
        }

        function searchDiscoveryTable() {
            const searchTerm = document.getElementById('discovery-search').value.toLowerCase();
            const rows = document.querySelectorAll('#discovery-table .data-row');
            let visibleCount = 0;

            rows.forEach(row => {
                const reqId = row.querySelector('.req-id').textContent.toLowerCase();
                const activityName = row.querySelector('.activity-name').textContent.toLowerCase();

                const searchMatch = !searchTerm || 
                                reqId.includes(searchTerm) || 
                                activityName.includes(searchTerm);

                if (searchMatch) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            document.getElementById('discovery-count').textContent = `Showing ${visibleCount} results`;
        }

        function searchGapsTable() {
            const searchTerm = document.getElementById('gaps-search').value.toLowerCase();
            const rows = document.querySelectorAll('#gaps-table .data-row');
            let visibleCount = 0;

            rows.forEach(row => {
                const reqId = row.querySelector('.req-id').textContent.toLowerCase();
                const activityName = row.querySelector('.activity-name').textContent.toLowerCase();

                const searchMatch = !searchTerm || 
                                reqId.includes(searchTerm) || 
                                activityName.includes(searchTerm);

                if (searchMatch) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            document.getElementById('gaps-count').textContent = `Showing ${visibleCount} results`;
        }

        // Simple show all function
        function showAllRows() {
            const rows = document.querySelectorAll('#predictions-table .data-row');
            rows.forEach(row => {
                row.style.display = '';
            });
            document.getElementById('predictions-count').textContent = `Showing ${rows.length} results`;
            document.getElementById('predictions-search').value = '';
        }

        // Table sorting functions
        function sortPredictionsTable(columnIndex) {
            const table = document.getElementById('predictions-table');
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('.data-row'));

            if (predictionsSort.column === columnIndex) {
                predictionsSort.ascending = !predictionsSort.ascending;
            } else {
                predictionsSort.column = columnIndex;
                predictionsSort.ascending = true;
            }

            document.querySelectorAll('#predictions-table .sort-arrow').forEach(arrow => {
                arrow.textContent = '↕';
            });
            const currentArrow = document.querySelectorAll('#predictions-table .sort-arrow')[columnIndex];
            if (currentArrow) {
                currentArrow.textContent = predictionsSort.ascending ? '↑' : '↓';
            }

            rows.sort((a, b) => {
                let aVal, bVal;

                switch(columnIndex) {
                    case 0: // Req ID
                        aVal = a.querySelector('.req-id').textContent;
                        bVal = b.querySelector('.req-id').textContent;
                        break;
                    case 1: // Req Name
                        aVal = a.querySelector('.req-name').textContent;
                        bVal = b.querySelector('.req-name').textContent;
                        break;
                    case 2: // Activity Name
                        aVal = a.querySelector('.activity-name').textContent;
                        bVal = b.querySelector('.activity-name').textContent;
                        break;
                    default: // Score columns
                        const scoreCells = a.querySelectorAll('.score-cell');
                        const scoreIndex = columnIndex - 3;
                        aVal = parseFloat(scoreCells[scoreIndex].textContent);
                        bVal = parseFloat(b.querySelectorAll('.score-cell')[scoreIndex].textContent);
                        break;
                }

                if (typeof aVal === 'string') {
                    return predictionsSort.ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                } else {
                    return predictionsSort.ascending ? aVal - bVal : bVal - aVal;
                }
            });

            rows.forEach(row => tbody.appendChild(row));
        }

        function sortDiscoveryTable(columnIndex) {
            const table = document.getElementById('discovery-table');
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('.data-row'));

            document.querySelectorAll('#discovery-table .sort-arrow').forEach(arrow => {
                arrow.textContent = '↕';
            });
            const currentArrow = document.querySelectorAll('#discovery-table .sort-arrow')[columnIndex];
            const ascending = currentArrow && currentArrow.textContent !== '↑';
            if (currentArrow) {
                currentArrow.textContent = ascending ? '↑' : '↓';
            }

            rows.sort((a, b) => {
                let aVal, bVal;

                switch(columnIndex) {
                    case 0: // Req ID
                        aVal = a.querySelector('.req-id').textContent;
                        bVal = b.querySelector('.req-id').textContent;
                        break;
                    case 1: // Activity Name
                        aVal = a.querySelector('.activity-name').textContent;
                        bVal = b.querySelector('.activity-name').textContent;
                        break;
                    case 2: // Discovery Score
                        aVal = parseFloat(a.getAttribute('data-score'));
                        bVal = parseFloat(b.getAttribute('data-score'));
                        break;
                    case 3: // Manual Matches
                        aVal = parseInt(a.querySelector('.manual-count').textContent);
                        bVal = parseInt(b.querySelector('.manual-count').textContent);
                        break;
                }

                if (typeof aVal === 'string') {
                    return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                } else {
                    return ascending ? aVal - bVal : bVal - aVal;
                }
            });

            rows.forEach(row => tbody.appendChild(row));
        }

        function sortGapsTable(columnIndex) {
            const table = document.getElementById('gaps-table');
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('.data-row'));

            document.querySelectorAll('#gaps-table .sort-arrow').forEach(arrow => {
                arrow.textContent = '↕';
            });
            const currentArrow = document.querySelectorAll('#gaps-table .sort-arrow')[columnIndex];
            const ascending = currentArrow && currentArrow.textContent !== '↑';
            if (currentArrow) {
                currentArrow.textContent = ascending ? '↑' : '↓';
            }

            rows.sort((a, b) => {
                let aVal, bVal;

                switch(columnIndex) {
                    case 0: // Req ID
                        aVal = a.querySelector('.req-id').textContent;
                        bVal = b.querySelector('.req-id').textContent;
                        break;
                    case 1: // Activity Name
                        aVal = a.querySelector('.activity-name').textContent;
                        bVal = b.querySelector('.activity-name').textContent;
                        break;
                    case 2: // Algorithm Score
                        const scoreCells = a.querySelectorAll('.score-cell');
                        aVal = parseFloat(scoreCells[0].textContent);
                        bVal = parseFloat(b.querySelectorAll('.score-cell')[0].textContent);
                        break;
                    case 3: // Manual Score
                        const scoreCell1 = a.querySelectorAll('.score-cell');
                        aVal = parseFloat(scoreCell1[1].textContent);
                        bVal = parseFloat(b.querySelectorAll('.score-cell')[1].textContent);
                        break;
                    case 4: // Gap
                        aVal = parseFloat(a.getAttribute('data-gap'));
                        bVal = parseFloat(b.getAttribute('data-gap'));
                        break;
                }

                if (typeof aVal === 'string') {
                    return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                } else {
                    return ascending ? aVal - bVal : bVal - aVal;
                }
            });

            rows.forEach(row => tbody.appendChild(row));
        }

        // Export functions
        function exportPredictionsTable() {
            const rows = document.querySelectorAll('#predictions-table .data-row');
            const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
            
            let csv = 'Requirement ID,Requirement Name,Activity Name,Combined Score,Semantic Score,BM25 Score\\n';
            
            visibleRows.forEach(row => {
                const reqId = row.getAttribute('data-req-id');
                const reqName = JSON.parse(row.getAttribute('data-req-name-full') || '"N/A"');
                const activityName = JSON.parse(row.getAttribute('data-activity-full') || '"N/A"');
                const combinedScore = row.getAttribute('data-combined-score');
                const scoreCells = row.querySelectorAll('.score-cell');
                
                const rowData = [
                    reqId,
                    `"${reqName.replace(/"/g, '""')}"`,
                    `"${activityName.replace(/"/g, '""')}"`,
                    combinedScore,
                    scoreCells[1] ? scoreCells[1].textContent : '',
                    scoreCells[2] ? scoreCells[2].textContent : ''
                ];
                
                csv += rowData.join(',') + '\\n';
            });
            
            downloadCSV(csv, 'predictions_export.csv');
        }
        
        function exportDiscoveryTable() {
            const rows = document.querySelectorAll('#discovery-table .data-row');
            const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
            
            let csv = 'Requirement ID,Activity Name,Discovery Score,Manual Matches Count\\n';
            
            visibleRows.forEach(row => {
                const reqId = row.getAttribute('data-req-id');
                const activityName = JSON.parse(row.getAttribute('data-activity-full') || '"N/A"');
                const score = row.getAttribute('data-score');
                const manualCount = row.querySelector('.manual-count').textContent;
                
                const rowData = [
                    reqId,
                    `"${activityName.replace(/"/g, '""')}"`,
                    score,
                    manualCount
                ];
                
                csv += rowData.join(',') + '\\n';
            });
            
            downloadCSV(csv, 'discovery_export.csv');
        }

        function exportGapsTable() {
            const rows = document.querySelectorAll('#gaps-table .data-row');
            const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
            
            let csv = 'Requirement ID,Best Algorithm Activity,Algorithm Score,Manual Score,Gap\\n';
            
            visibleRows.forEach(row => {
                const reqId = row.getAttribute('data-req-id');
                const activityName = JSON.parse(row.getAttribute('data-activity-full') || '"N/A"');
                const gap = row.getAttribute('data-gap');
                const scoreCells = row.querySelectorAll('.score-cell');
                
                const rowData = [
                    reqId,
                    `"${activityName.replace(/"/g, '""')}"`,
                    scoreCells[0] ? scoreCells[0].textContent : '',
                    scoreCells[1] ? scoreCells[1].textContent : '',
                    gap
                ];
                
                csv += rowData.join(',') + '\\n';
            });
            
            downloadCSV(csv, 'score_gaps_export.csv');
        }

        function downloadCSV(csv, filename) {
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }

        // =================================================================
        // COMPLETE MODAL SYSTEM - INTEGRATED (NO calculations, pure display)
        // =================================================================
        
        // Helper function to get requirement text from multiple sources
        function getRequirementText(reqId) {
            const predictionsRows = document.querySelectorAll('#predictions-table .data-row');
            for (let row of predictionsRows) {
                if (row.getAttribute('data-req-id') === reqId) {
                    const reqText = row.getAttribute('data-req-text');
                    if (reqText && reqText !== 'Requirement text not available' && !reqText.includes('not available')) {
                        return reqText;
                    }
                }
            }
            return `Detailed requirement text for ${reqId} is not available in the current dataset.`;
        }

        // PREDICTIONS MODAL GENERATOR
        function createPredictionsModal(row, reqId) {
            const reqName = row.getAttribute('data-req-name-full') || 'N/A';
            const activityName = row.getAttribute('data-activity-full') || 'N/A';
            const combinedScore = parseFloat(row.getAttribute('data-combined-score')) || 0;
            const reqText = row.getAttribute('data-req-text') || 'Requirement text not available';
            
            // Get all available score components from table cells
            const scoreCells = row.querySelectorAll('.score-cell');
            const semanticScore = scoreCells[1] ? parseFloat(scoreCells[1].textContent) : null;
            const bm25Score = scoreCells[2] ? parseFloat(scoreCells[2].textContent) : null;
            const syntacticScore = scoreCells[3] ? parseFloat(scoreCells[3].textContent) : null;
            const domainScore = scoreCells[4] ? parseFloat(scoreCells[4].textContent) : null;
            const queryExpansionScore = scoreCells[5] ? parseFloat(scoreCells[5].textContent) : null;
            
            // Get explainability data from all available attributes
            const sharedTerms = row.getAttribute('data-shared-terms') || '';
            const semanticLevel = row.getAttribute('data-semantic-level') || row.getAttribute('data-match-quality') || 'Not Available';
            const matchQuality = row.getAttribute('data-match-quality') || 'Pending Analysis';
            const semanticExplanation = row.getAttribute('data-semantic-explanation') || 'Semantic analysis not available';
            const bm25Explanation = row.getAttribute('data-bm25-explanation') || 'BM25 analysis not available';
            const syntacticExplanation = row.getAttribute('data-syntactic-explanation') || 'Syntactic analysis not available';
            const domainExplanation = row.getAttribute('data-domain-explanation') || 'Domain analysis not available';
            const queryExpansionExplanation = row.getAttribute('data-query-expansion-explanation') || 'Query expansion analysis not available';
            
            // Parse shared terms
            let sharedTermsList = [];
            if (sharedTerms) {
                try {
                    sharedTermsList = JSON.parse(sharedTerms);
                } catch (e) {
                    sharedTermsList = sharedTerms.split(',').map(term => term.trim()).filter(term => term);
                }
            }

            // Get enhanced requirement text
            let enhancedReqText = reqText;
            if (reqText === 'Requirement text not available' || reqText.includes('not available')) {
                enhancedReqText = getRequirementText(reqId);
            }

            return `
                <div class="modal-header">
                    <h3>Match Analysis: ${reqId}</h3>
                    <div class="modal-field"><strong>Requirement:</strong> ${reqName}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Requirement Specification</h4>
                    <div class="req-text-box">${enhancedReqText}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Matched Activity</h4>
                    <div class="activity-text-box">${activityName}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Score Analysis</h4>
                    <div class="score-breakdown">
                        <div class="score-item">
                            <div class="score-label">
                                <strong>Combined Score:</strong> ${combinedScore.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${Math.min(combinedScore * 100, 100)}%"></div>
                            </div>
                            <div class="score-description">Final weighted score from all components</div>
                        </div>
                        
                        ${semanticScore !== null ? `
                        <div class="score-item">
                            <div class="score-label">
                                <strong>Semantic Score:</strong> ${semanticScore.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill semantic-fill" style="width: ${Math.min(semanticScore * 100, 100)}%"></div>
                            </div>
                            <div class="score-description">${semanticExplanation}</div>
                        </div>
                        ` : ''}
                        
                        ${bm25Score !== null ? `
                        <div class="score-item">
                            <div class="score-label">
                                <strong>BM25 Score:</strong> ${bm25Score.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill bm25-fill" style="width: ${Math.min(bm25Score * 20, 100)}%"></div>
                            </div>
                            <div class="score-description">${bm25Explanation}</div>
                        </div>
                        ` : ''}
                        
                        ${syntacticScore !== null ? `
                        <div class="score-item">
                            <div class="score-label">
                                <strong>Syntactic Score:</strong> ${syntacticScore.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill other-fill" style="width: ${Math.min(syntacticScore * 100, 100)}%"></div>
                            </div>
                            <div class="score-description">${syntacticExplanation}</div>
                        </div>
                        ` : ''}
                        
                        ${domainScore !== null ? `
                        <div class="score-item">
                            <div class="score-label">
                                <strong>Domain Score:</strong> ${domainScore.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill other-fill" style="width: ${Math.min(domainScore * 100, 100)}%"></div>
                            </div>
                            <div class="score-description">${domainExplanation}</div>
                        </div>
                        ` : ''}
                        
                        ${queryExpansionScore !== null ? `
                        <div class="score-item">
                            <div class="score-label">
                                <strong>Query Expansion Score:</strong> ${queryExpansionScore.toFixed(3)}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill other-fill" style="width: ${Math.min(queryExpansionScore * 100, 100)}%"></div>
                            </div>
                            <div class="score-description">${queryExpansionExplanation}</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
                
                ${sharedTermsList.length > 0 ? `
                <div class="modal-section">
                    <h4>Shared Terms Analysis</h4>
                    <div class="shared-terms-container">
                        <div class="shared-terms-list">
                            <strong>Identified Terms:</strong> ${sharedTermsList.join(', ')}
                        </div>
                        <div class="evidence-summary">
                            Found ${sharedTermsList.length} common terms indicating textual overlap
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="modal-section">
                    <h4>Match Assessment</h4>
                    <table class="assessment-table">
                        <tr>
                            <td><strong>Match Quality:</strong></td>
                            <td>${matchQuality}</td>
                        </tr>
                        <tr>
                            <td><strong>Semantic Level:</strong></td>
                            <td>${semanticLevel}</td>
                        </tr>
                        <tr>
                            <td><strong>Confidence Level:</strong></td>
                            <td>${combinedScore >= 0.8 ? 'High' : combinedScore >= 0.5 ? 'Medium' : 'Low'}</td>
                        </tr>
                        <tr>
                            <td><strong>Recommendation:</strong></td>
                            <td>${combinedScore >= 0.8 ? 'Accept for use' : combinedScore >= 0.5 ? 'Review before use' : 'Requires validation'}</td>
                        </tr>
                    </table>
                </div>
            `;
        }

        // DISCOVERY MODAL GENERATOR - FIXED VERSION
        function createDiscoveryModal(row, reqId) {
            const reqName = row.getAttribute('data-req-name-full') || 'N/A';
            const activityName = row.getAttribute('data-activity-full') || 'N/A';
            const reqText = row.getAttribute('data-req-text') || 'Requirement text not available';
            const score = parseFloat(row.getAttribute('data-score')) || 0;
            const manualCountElement = row.querySelector('.manual-count');
            const manualCount = manualCountElement ? parseInt(manualCountElement.textContent) || 0 : 0;
            
            // Parse manual data - FIXED VERSION
            let manualMatches = [];
            let manualActivityNames = []; // Will be set from manualMatches
            try {
                const manualMatchesData = row.getAttribute('data-manual-matches');
                
                if (manualMatchesData && manualMatchesData !== 'null' && manualMatchesData !== '[]') {
                    manualMatches = JSON.parse(manualMatchesData);
                    manualActivityNames = manualMatches; // FIX: Use the actual data we have
                }
            } catch (e) {
                console.error('Could not parse manual data for discovery:', e);
            }
            
            // Get enhanced requirement text
            let enhancedReqText = reqText;
            if (reqText === 'Requirement text not available' || reqText.includes('not available')) {
                enhancedReqText = getRequirementText(reqId);
            }

            return `
                <div class="modal-header">
                    <h3>Discovery Analysis: ${reqId}</h3>
                    <div class="modal-field"><strong>Requirement:</strong> ${reqName}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Requirement Specification</h4>
                    <div class="req-text-box">${enhancedReqText}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Algorithm Discovery</h4>
                    <div class="activity-text-box">
                        <strong>Discovered Activity:</strong> ${activityName}
                    </div>
                    <div class="discovery-details">
                        <p><strong>Discovery Score:</strong> ${score.toFixed(3)}</p>
                        <p><strong>Priority Level:</strong> ${score > 1.0 ? 'High' : score > 0.7 ? 'Medium' : 'Low'}</p>
                        <p><strong>Status:</strong> Algorithm found this connection but it was not in manual analysis</p>
                    </div>
                </div>
                
                <div class="modal-section">
                    <h4>Manual Analysis Comparison</h4>
                    <table class="comparison-table">
                        <tr>
                            <td><strong>Manual Matches Found:</strong></td>
                            <td>${manualCount} activities</td>
                        </tr>
                        <tr>
                            <td><strong>Algorithm Discovery:</strong></td>
                            <td>1 additional activity</td>
                        </tr>
                        <tr>
                            <td><strong>Discovery Type:</strong></td>
                            <td>${manualCount === 0 ? 'New connection (no manual matches)' : 'Additional match (supplements manual analysis)'}</td>
                        </tr>
                    </table>
                    
                    ${manualActivityNames && manualActivityNames.length > 0 ? `
                        <div class="manual-activities-section">
                            <h5>Manual Activities Identified:</h5>
                            <ul class="manual-activities-list">
                                ${manualActivityNames.map(activity => `<li>${activity}</li>`).join('')}
                            </ul>
                        </div>
                    ` : manualCount > 0 ? `
                        <div class="manual-activities-section">
                            <p><strong>Note:</strong> Manual analysis found ${manualCount} activities, but detailed activity names are not available in current dataset.</p>
                        </div>
                    ` : `
                        <div class="manual-activities-section">
                            <p><strong>Pure Discovery:</strong> No manual matches were found, making this a completely new connection.</p>
                        </div>
                    `}
                </div>
                
                <div class="modal-section">
                    <h4>Recommendation</h4>
                    <div class="recommendation-content">
                        <p><strong>Action Required:</strong> ${score > 1.0 ? 'High priority review recommended' : score > 0.7 ? 'Include in next review cycle' : 'Lower priority, review when resources allow'}</p>
                        <p><strong>Review Focus:</strong> Validate whether this algorithmic discovery represents a genuine traceability connection.</p>
                        ${manualCount > 0 ? '<p><strong>Context:</strong> Consider this discovery alongside the existing manual matches for complete coverage.</p>' : ''}
                    </div>
                </div>
            `;
        }

        // GAPS MODAL GENERATOR - FIXED VERSION  
        function createGapsModal(row, reqId) {
            const reqName = row.getAttribute('data-req-name-full') || 'N/A';
            const activityName = row.getAttribute('data-activity-full') || 'N/A';
            const reqText = row.getAttribute('data-req-text') || 'Requirement text not available';
            const gap = parseFloat(row.getAttribute('data-gap')) || 0;
            
            // Get scores from table cells
            const scoreCells = row.querySelectorAll('.score-cell');
            const algorithmScore = scoreCells[0] ? parseFloat(scoreCells[0].textContent) : 0;
            const manualScore = scoreCells[1] ? parseFloat(scoreCells[1].textContent) : 0;
            
            // Parse manual data - FIXED VERSION
            let manualActivityNames = [];
            try {
                const manualMatchesData = row.getAttribute('data-manual-matches');
                if (manualMatchesData && manualMatchesData !== 'null' && manualMatchesData !== '[]') {
                    manualActivityNames = JSON.parse(manualMatchesData); // FIX: Use data-manual-matches directly
                }
            } catch (e) {
                console.error('Could not parse manual data for gaps:', e);
            }
            
            // Get enhanced requirement text
            let enhancedReqText = reqText;
            if (reqText === 'Requirement text not available' || reqText.includes('not available')) {
                enhancedReqText = getRequirementText(reqId);
            }

            return `
                <div class="modal-header">
                    <h3>Score Gap Analysis: ${reqId}</h3>
                    <div class="modal-field"><strong>Requirement:</strong> ${reqName}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Requirement Specification</h4>
                    <div class="req-text-box">${enhancedReqText}</div>
                </div>
                
                <div class="modal-section">
                    <h4>Activity Comparison</h4>
                    <div class="activity-text-box">
                        <strong>Activity:</strong> ${activityName}
                    </div>
                </div>
                
                <div class="modal-section">
                    <h4>Score Comparison</h4>
                    <table class="score-comparison-table">
                        <tr>
                            <td><strong>Algorithm Score:</strong></td>
                            <td>${algorithmScore.toFixed(3)}</td>
                            <td>${algorithmScore > 1.5 ? 'Very High' : algorithmScore > 0.8 ? 'High' : algorithmScore > 0.4 ? 'Medium' : 'Low'}</td>
                        </tr>
                        <tr>
                            <td><strong>Manual Score:</strong></td>
                            <td>${manualScore.toFixed(3)}</td>
                            <td>${manualScore > 1.5 ? 'Very Strong' : manualScore > 0.8 ? 'Strong' : manualScore > 0.4 ? 'Moderate' : 'Weak'}</td>
                        </tr>
                        <tr>
                            <td><strong>Gap (Algorithm - Manual):</strong></td>
                            <td><strong>${gap >= 0 ? '+' : ''}${gap.toFixed(3)}</strong></td>
                            <td><strong>${gap > 0.5 ? 'Large Positive Gap' : gap > 0.2 ? 'Medium Positive Gap' : gap > -0.2 ? 'Small Gap' : gap > -0.5 ? 'Medium Negative Gap' : 'Large Negative Gap'}</strong></td>
                        </tr>
                    </table>
                </div>
                
                <div class="modal-section">
                    <h4>Manual Analysis Context</h4>
                    ${manualActivityNames && manualActivityNames.length > 0 ? `
                        <div class="manual-context">
                            <p><strong>Manual Activities Identified (${manualActivityNames.length}):</strong></p>
                            <ul class="manual-activities-list">
                                ${manualActivityNames.map(activity => `<li>${activity}</li>`).join('')}
                            </ul>
                            <p><strong>Combined Manual Assessment:</strong> ${manualScore.toFixed(3)} (${manualScore > 0.8 ? 'Strong matches found' : manualScore > 0.5 ? 'Moderate matches found' : manualScore > 0.2 ? 'Weak matches found' : 'No strong matches'})</p>
                        </div>
                    ` : `
                        <div class="manual-context">
                            <p><strong>Manual Score:</strong> ${manualScore.toFixed(3)}</p>
                            <p><strong>Note:</strong> Manual activity details are not available in the current dataset.</p>
                        </div>
                    `}
                </div>
                
                <div class="modal-section">
                    <h4>Gap Analysis</h4>
                    <div class="gap-analysis">
                        <p><strong>Gap Interpretation:</strong></p>
                        <p>${gap > 0.5 ? 
                            `Algorithm significantly overestimated this match (${gap.toFixed(3)} points higher). This suggests the algorithm may be detecting surface-level patterns that don't represent true semantic relationships.` :
                        gap > 0.2 ? 
                            `Algorithm moderately overestimated this match (${gap.toFixed(3)} points higher). This warrants investigation to understand the disagreement.` :
                        gap > -0.2 ? 
                            `Algorithm and manual assessments are well-aligned (gap: ${gap.toFixed(3)}). This indicates good algorithm calibration.` :
                        gap > -0.5 ? 
                            `Algorithm moderately underestimated this match (${Math.abs(gap).toFixed(3)} points lower). The algorithm may be missing semantic nuances.` :
                            `Algorithm significantly underestimated this match (${Math.abs(gap).toFixed(3)} points lower). This indicates potential algorithm blind spots.`
                        }</p>
                        
                        <p><strong>Implications for Algorithm Performance:</strong></p>
                        <p>${gap > 0.3 ? 
                            'Focus on reducing false positives - algorithm may need better semantic understanding training.' :
                        gap < -0.3 ? 
                            'Focus on improving recall - algorithm may need exposure to more diverse matching patterns.' :
                            'Performance is well-calibrated - continue monitoring for consistency across requirements.'
                        }</p>
                    </div>
                </div>
            `;
        }
        // UNIFIED MODAL SYSTEM
        function showRowDetails(row, modalType = 'predictions') {
            const reqId = row.getAttribute('data-req-id');
            if (!reqId) {
                console.error('No requirement ID found for row');
                return;
            }

            let modalContent = '';
            
            try {
                switch(modalType) {
                    case 'predictions':
                        modalContent = createPredictionsModal(row, reqId);
                        break;
                    case 'discovery':
                        modalContent = createDiscoveryModal(row, reqId);
                        break;
                    case 'gaps':
                        modalContent = createGapsModal(row, reqId);
                        break;
                    default:
                        modalContent = createPredictionsModal(row, reqId);
                        break;
                }
            } catch (error) {
                console.error('Error creating modal content:', error);
                modalContent = `
                    <div class="modal-header">
                        <h3>Error Loading Details</h3>
                    </div>
                    <div class="modal-section">
                        <p>Unable to load detailed information for ${reqId}.</p>
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p>Please check the console for more details.</p>
                    </div>
                `;
            }

            document.getElementById('modal-body').innerHTML = modalContent;
            document.getElementById('row-detail-modal').style.display = 'block';
        }

        function closeModal() {
            document.getElementById('row-detail-modal').style.display = 'none';
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('row-detail-modal');
            if (event.target === modal) {
                closeModal();
            }
        }

        // Initialize dashboard when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Requirements Traceability Dashboard loaded successfully');
            
            // Attach click handlers to all data rows
            document.querySelectorAll('#predictions-table .data-row').forEach(row => {
                row.addEventListener('click', () => showRowDetails(row, 'predictions'));
            });
            
            document.querySelectorAll('#discovery-table .data-row').forEach(row => {
                row.addEventListener('click', () => showRowDetails(row, 'discovery'));
            });
            
            document.querySelectorAll('#gaps-table .data-row').forEach(row => {
                row.addEventListener('click', () => showRowDetails(row, 'gaps'));
            });
            
            // Add keyboard support for modal
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeModal();
                }
            });
        });        """