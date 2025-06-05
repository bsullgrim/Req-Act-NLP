"""
HTML Templates - Main HTML structure and styling
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
        """
    
    def _get_javascript(self) -> str:
        """Return JavaScript for dashboard interactivity."""
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
            document.getElementById(sectionName + '-section').classList.add('active');
            
            // Activate corresponding nav button
            event.target.classList.add('active');
        }

        // Table filtering
        function filterPredictionsTable() {
            const searchTerm = document.getElementById('predictions-search').value.toLowerCase();
            const scoreFilter = document.getElementById('score-filter').value;
            const rows = document.querySelectorAll('#predictions-table .data-row');
            let visibleCount = 0;

            rows.forEach(row => {
                const reqId = row.querySelector('.req-id').textContent.toLowerCase();
                const reqName = row.getAttribute('data-req-name-full').toLowerCase();
                const activityName = row.getAttribute('data-activity-full').toLowerCase();
                const score = parseFloat(row.getAttribute('data-combined-score'));

                // Search filter
                const searchMatch = reqId.includes(searchTerm) || 
                                  reqName.includes(searchTerm) || 
                                  activityName.includes(searchTerm);

                // Score filter
                let scoreMatch = true;
                if (scoreFilter === 'high') {
                    scoreMatch = score >= 1.0;
                } else if (scoreFilter === 'medium') {
                    scoreMatch = score >= 0.6 && score < 1.0;
                } else if (scoreFilter === 'low') {
                    scoreMatch = score < 0.6;
                }

                if (searchMatch && scoreMatch) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            document.getElementById('predictions-count').textContent = `Showing ${visibleCount} results`;
        }

        // Table sorting
        function sortPredictionsTable(columnIndex) {
            const table = document.getElementById('predictions-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('.data-row'));

            // Update sort direction
            if (predictionsSort.column === columnIndex) {
                predictionsSort.ascending = !predictionsSort.ascending;
            } else {
                predictionsSort.column = columnIndex;
                predictionsSort.ascending = true;
            }

            // Update sort arrows
            document.querySelectorAll('#predictions-table .sort-arrow').forEach(arrow => {
                arrow.textContent = '↕';
            });
            const currentArrow = document.querySelectorAll('#predictions-table .sort-arrow')[columnIndex];
            currentArrow.textContent = predictionsSort.ascending ? '↑' : '↓';

            // Sort rows
            rows.sort((a, b) => {
                let aVal, bVal;

                switch(columnIndex) {
                    case 0: // Req ID
                        aVal = a.querySelector('.req-id').textContent;
                        bVal = b.querySelector('.req-id').textContent;
                        break;
                    case 1: // Req Name
                        aVal = a.getAttribute('data-req-name-full');
                        bVal = b.getAttribute('data-req-name-full');
                        break;
                    case 2: // Activity Name
                        aVal = a.getAttribute('data-activity-full');
                        bVal = b.getAttribute('data-activity-full');
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

            // Reappend sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }

        // Modal functionality
        function showRowDetails(row, tableType) {
            const modal = document.getElementById('row-detail-modal');
            const modalBody = document.getElementById('modal-body');

            const reqId = row.getAttribute('data-req-id');
            const reqName = row.getAttribute('data-req-name-full');
            const activityName = row.getAttribute('data-activity-full');

            let modalContent = `
                <div class="modal-header">
                    <h3>Details: ${reqId}</h3>
                </div>
                
                <div class="modal-section">
                    <h4>📋 Requirement Name</h4>
                    <div class="modal-text">${reqName}</div>
                </div>
                
                <div class="modal-section">
                    <h4>🎯 Matched Activity</h4>
                    <div class="modal-text">${activityName}</div>
                </div>
            `;

            if (tableType === 'predictions') {
                const combinedScore = row.getAttribute('data-combined-score');
                const scoreCells = row.querySelectorAll('.score-cell');
                
                modalContent += `
                    <div class="modal-section">
                        <h4>📊 Score Breakdown</h4>
                        <div class="modal-scores">
                            <div class="score-card">
                                <div class="score-value">${parseFloat(combinedScore).toFixed(3)}</div>
                                <div class="score-label">Combined Score</div>
                            </div>
                            <div class="score-card">
                                <div class="score-value">${parseFloat(scoreCells[1].textContent).toFixed(3)}</div>
                                <div class="score-label">Semantic</div>
                            </div>
                            <div class="score-card">
                                <div class="score-value">${parseFloat(scoreCells[2].textContent).toFixed(3)}</div>
                                <div class="score-label">BM25</div>
                            </div>
                        </div>
                    </div>
                `;
            }

            modalBody.innerHTML = modalContent;
            modal.style.display = 'block';
        }

        function closeModal() {
            document.getElementById('row-detail-modal').style.display = 'none';
        }

        // Export functions
        function exportPredictionsTable() {
            const rows = document.querySelectorAll('#predictions-table .data-row');
            const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
            
            let csv = 'Requirement ID,Requirement Name,Activity Name,Combined Score,Semantic Score,BM25 Score\\n';
            
            visibleRows.forEach(row => {
                const reqId = row.getAttribute('data-req-id');
                const reqName = row.getAttribute('data-req-name-full');
                const activityName = row.getAttribute('data-activity-full');
                const combinedScore = row.getAttribute('data-combined-score');
                const scoreCells = row.querySelectorAll('.score-cell');
                
                const rowData = [
                    reqId,
                    `"${reqName.replace(/"/g, '""')}"`,
                    `"${activityName.replace(/"/g, '""')}"`,
                    combinedScore,
                    scoreCells[1].textContent,
                    scoreCells[2].textContent
                ];
                
                csv += rowData.join(',') + '\\n';
            });
            
            downloadCSV(csv, 'predictions_export.csv');
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

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            // Close modal when clicking outside
            window.addEventListener('click', function(event) {
                const modal = document.getElementById('row-detail-modal');
                if (event.target === modal) {
                    closeModal();
                }
            });

            // Initialize table counts
            if (document.getElementById('predictions-count')) {
                const predictionsRows = document.querySelectorAll('#predictions-table .data-row').length;
                document.getElementById('predictions-count').textContent = `Showing ${predictionsRows} results`;
            }

            // Keyboard shortcuts
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeModal();
                }
            });
        });
        """