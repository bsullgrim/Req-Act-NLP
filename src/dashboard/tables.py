"""
Table Generation - Create HTML tables for dashboard
"""

from typing import Dict, List, Any
import html
import json

class TableGenerator:
    """Generates all dashboard tables - simplified version."""
    
    def create_all_tables(self, processed_data: Dict[str, Any]) -> Dict[str, str]:
        """Create all tables and return as HTML strings."""
        tables = {}
        
        performance_data = processed_data.get('performance_metrics', {})
        discovery_data = processed_data.get('discovery_data', {})
        predictions_data = processed_data.get('predictions_data', [])
        quality_data = processed_data.get('quality_data', {})
        coverage_data = processed_data.get('coverage_data', {})
        
        print(f"🔍 TableGenerator Debug:")
        print(f"  - predictions_data length: {len(predictions_data)}")
        
        if performance_data:
            tables['performance_summary'] = self._create_performance_table(performance_data)
        
        if discovery_data.get('top_examples'):
            tables['top_discoveries'] = self._create_discovery_table(discovery_data['top_examples'])
        
        if predictions_data:
            tables['all_predictions'] = self._create_predictions_table(predictions_data)
            print(f"✅ Created predictions table with {len(predictions_data)} rows")
        
        if discovery_data.get('high_scoring_misses'):
            tables['discovery_results'] = self._create_discovery_results_table(discovery_data['high_scoring_misses'])
        
        if discovery_data.get('score_gaps'):
            tables['score_gaps'] = self._create_score_gaps_table(discovery_data['score_gaps'])
        
        if quality_data.get('has_quality_data', False):
            tables['quality_analysis'] = self._create_quality_analysis_table(quality_data)
        
        if coverage_data.get('has_coverage_data', False):
            tables['coverage_analysis'] = self._create_coverage_analysis_table(coverage_data)
        
        return tables
    
    def _escape_for_html_attr(self, text: str) -> str:
        """Safely escape text for HTML attributes."""
        if not text:
            return ""
        return html.escape(str(text), quote=True)
    
    def _create_predictions_table(self, predictions_data: List[Dict]) -> str:
        """Create simplified predictions table without complex click handlers."""
        
        print(f"🔧 Creating predictions table with {len(predictions_data)} predictions")
        
        html_content = f"""
        <div class="table-controls">
            <div class="search-box">
                <input type="text" id="predictions-search" placeholder="Search requirements, activities..." 
                    onkeyup="searchPredictionsTable()" class="search-input">
                <span class="search-icon">🔍</span>
            </div>
            <div class="filter-controls">
                <button onclick="exportPredictionsTable()" class="export-btn">📊 Export CSV</button>
                <button onclick="showAllRows()" class="export-btn">👁️ Show All</button>
            </div>
        </div>
        
        <div class="table-wrapper">
            <table id="predictions-table" class="sortable-table">
                <thead>
                    <tr>
                        <th onclick="sortPredictionsTable(0)" class="sortable">Req ID <span class="sort-arrow">↕</span></th>
                        <th onclick="sortPredictionsTable(1)" class="sortable">Requirement Name <span class="sort-arrow">↕</span></th>
                        <th onclick="sortPredictionsTable(2)" class="sortable">Activity Name <span class="sort-arrow">↕</span></th>
                        <th onclick="sortPredictionsTable(3)" class="sortable">Combined Score <span class="sort-arrow">↕</span></th>
                        <th onclick="sortPredictionsTable(4)" class="sortable">Semantic <span class="sort-arrow">↕</span></th>
                        <th onclick="sortPredictionsTable(5)" class="sortable">BM25 <span class="sort-arrow">↕</span></th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Generate table rows with simplified data attributes
        for item in predictions_data:
            req_name_display = item['requirement_name'][:50] + "..." if len(item['requirement_name']) > 50 else item['requirement_name']
            activity_display = item['activity_name'][:80] + "..." if len(item['activity_name']) > 80 else item['activity_name']
            
            html_content += f"""
                <tr class="data-row {item['score_class']}" 
                    data-req-id="{item['requirement_id']}" 
                    data-combined-score="{item['combined_score']}"
                    title="Click for details, Export for full data">
                    <td class="req-id">{item['requirement_id']}</td>
                    <td class="req-name" title="{item['requirement_name']}">{req_name_display}</td>
                    <td class="activity-name" title="{item['activity_name']}">{activity_display}</td>
                    <td class="score-cell"><strong>{item['combined_score']:.3f}</strong></td>
                    <td class="score-cell">{item['semantic_score']:.3f}</td>
                    <td class="score-cell">{item['bm25_score']:.3f}</td>
                </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        <div class="table-info">
            <span id="predictions-count">Showing {len(predictions_data)} results</span>
        </div>
        """
        
        return html_content
    
    def _create_discovery_results_table(self, discovery_results: List[Dict]) -> str:
        """Create discovery results table with proper click handlers."""
        
        html_content = f"""
        <div class="table-controls">
            <div class="search-box">
                <input type="text" id="discovery-search" placeholder="Search discovery results..." 
                    onkeyup="searchDiscoveryTable()" class="search-input">
                <span class="search-icon">🔍</span>
            </div>
            <div class="filter-controls">
                <button onclick="exportDiscoveryTable()" class="export-btn">📊 Export CSV</button>
            </div>
        </div>
        
        <div class="table-wrapper">
            <table id="discovery-table" class="sortable-table">
                <thead>
                    <tr>
                        <th onclick="sortDiscoveryTable(0)" class="sortable">Req ID <span class="sort-arrow">↕</span></th>
                        <th onclick="sortDiscoveryTable(1)" class="sortable">Activity Name <span class="sort-arrow">↕</span></th>
                        <th onclick="sortDiscoveryTable(2)" class="sortable">Discovery Score <span class="sort-arrow">↕</span></th>
                        <th onclick="sortDiscoveryTable(3)" class="sortable">Manual Matches Count <span class="sort-arrow">↕</span></th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in discovery_results:
            activity_display = item['activity_name'][:80] + "..." if len(item['activity_name']) > 80 else item['activity_name']
            score_class = "high-score" if item['score'] > 1.0 else "medium-score"
            
            # Get manual matches count and list
            manual_matches_count = item.get('manual_matches_count', 0)
            manual_matches_list = self._get_manual_matches_list(item)
            
            # Get requirement info
            req_name = self._get_requirement_name(item)
            req_text = self._get_requirement_text(item)
            
            # Escape for HTML attributes
            req_id_escaped = self._escape_for_html_attr(item['requirement_id'])
            req_name_escaped = self._escape_for_html_attr(req_name)
            activity_escaped = self._escape_for_html_attr(item['activity_name'])
            req_text_escaped = self._escape_for_html_attr(req_text)
            manual_matches_json = self._escape_for_html_attr(json.dumps(manual_matches_list))
            
            # Title attributes for hover
            activity_title = self._escape_for_html_attr(item['activity_name'])
            
            html_content += f"""
                <tr class="data-row" 
                    data-req-id="{req_id_escaped}" 
                    data-score="{item['score']}"
                    data-req-name-full="{req_name_escaped}"
                    data-activity-full="{activity_escaped}"
                    data-manual-matches="{manual_matches_json}"
                    data-req-text="{req_text_escaped}"
                    onclick="showRowDetails(this, 'discovery')"
                    style="cursor: pointer;">
                    <td class="req-id">{item['requirement_id']}</td>
                    <td class="activity-name" title="{activity_title}">{activity_display}</td>
                    <td class="score-cell {score_class}"><strong>{item['score']:.3f}</strong></td>
                    <td class="manual-count">{manual_matches_count}</td>
                </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        <div class="table-info">
            <span id="discovery-count">Showing {len(discovery_results)} results</span>
        </div>
        """
        
        return html_content
    
    def _create_score_gaps_table(self, score_gaps: List[Dict]) -> str:
        """Create score gaps table with proper click handlers."""
        
        html_content = f"""
        <div class="table-controls">
            <div class="search-box">
                <input type="text" id="gaps-search" placeholder="Search score gaps..." 
                    onkeyup="searchGapsTable()" class="search-input">
                <span class="search-icon">🔍</span>
            </div>
            <div class="filter-controls">
                <button onclick="exportGapsTable()" class="export-btn">📊 Export CSV</button>
            </div>
        </div>
        
        <div class="table-wrapper">
            <table id="gaps-table" class="sortable-table">
                <thead>
                    <tr>
                        <th onclick="sortGapsTable(0)" class="sortable">Req ID <span class="sort-arrow">↕</span></th>
                        <th onclick="sortGapsTable(1)" class="sortable">Best Algorithm Activity <span class="sort-arrow">↕</span></th>
                        <th onclick="sortGapsTable(2)" class="sortable">Algorithm Score <span class="sort-arrow">↕</span></th>
                        <th onclick="sortGapsTable(3)" class="sortable">Manual Score <span class="sort-arrow">↕</span></th>
                        <th onclick="sortGapsTable(4)" class="sortable">Gap <span class="sort-arrow">↕</span></th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in score_gaps:
            activity_display = item['max_miss_activity'][:80] + "..." if len(item['max_miss_activity']) > 80 else item['max_miss_activity']
            gap = item['gap']
            gap_class = "high-score" if gap > 0.5 else "medium-score" if gap > 0.2 else ""
            
            # Get manual matches and requirement info
            manual_matches_list = self._get_manual_matches_list(item)
            req_name = self._get_requirement_name(item)
            req_text = self._get_requirement_text(item)
            
            # Escape for HTML attributes
            req_id_escaped = self._escape_for_html_attr(item['requirement_id'])
            req_name_escaped = self._escape_for_html_attr(req_name)
            activity_escaped = self._escape_for_html_attr(item['max_miss_activity'])
            req_text_escaped = self._escape_for_html_attr(req_text)
            manual_matches_json = self._escape_for_html_attr(json.dumps(manual_matches_list))
            
            # Title attributes
            activity_title = self._escape_for_html_attr(item['max_miss_activity'])
            
            html_content += f"""
                <tr class="data-row" 
                    data-req-id="{req_id_escaped}" 
                    data-gap="{gap}"
                    data-req-name-full="{req_name_escaped}"
                    data-activity-full="{activity_escaped}"
                    data-manual-matches="{manual_matches_json}"
                    data-req-text="{req_text_escaped}"
                    onclick="showRowDetails(this, 'gaps')"
                    style="cursor: pointer;">
                    <td class="req-id">{item['requirement_id']}</td>
                    <td class="activity-name" title="{activity_title}">{activity_display}</td>
                    <td class="score-cell"><strong>{item['max_miss_score']:.3f}</strong></td>
                    <td class="score-cell">{item['min_manual_score']:.3f}</td>
                    <td class="score-cell {gap_class}"><strong>+{gap:.3f}</strong></td>
                </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        <div class="table-info">
            <span id="gaps-count">Showing {len(score_gaps)} results</span>
        </div>
        """
        
        return html_content
    
    def _get_requirement_name(self, item: Dict) -> str:
        """Get requirement name from various possible sources."""
        for key in ['requirement_name', 'Requirement Name', 'req_name', 'name']:
            if key in item and item[key] and str(item[key]).strip() != 'N/A':
                return str(item[key])
        
        req_id = item.get('requirement_id', 'Unknown')
        return f"Requirement {req_id}"
    
    def _get_requirement_text(self, item: Dict) -> str:
        """Get requirement text from various possible sources."""
        for key in ['requirement_text', 'Requirement Text', 'req_text', 'text', 'description']:
            if key in item and item[key] and str(item[key]).strip():
                text = str(item[key]).strip()
                if text != 'N/A' and text != 'Requirement text not available':
                    return text
        
        return f"Requirement text not available for {item.get('requirement_id', 'Unknown')}"
    
    def _get_manual_matches_list(self, item: Dict) -> List[str]:
        """Extract actual manual matches list from discovery/gap item."""
        manual_matches = []
        
        # Try various possible keys for manual matches
        for key in ['manual_matches', 'ground_truth_activities', 'manual_activities']:
            if key in item and isinstance(item[key], list):
                manual_matches = item[key]
                break
            elif key in item and isinstance(item[key], str):
                manual_matches = [act.strip() for act in item[key].split(',') if act.strip()]
                break
        
        # Return as list of strings, filtering out empty values
        return [str(match).strip() for match in manual_matches if match and str(match).strip()]
    # Keep other table methods unchanged...
    def _create_performance_table(self, performance_data: Dict) -> str:
        """Create performance summary table."""
        html = """
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>@1</th>
                    <th>@3</th>
                    <th>@5</th>
                    <th>@10</th>
                </tr>
            </thead>
            <tbody>
        """
        
        metrics = ['precision', 'recall', 'f1', 'ndcg']
        metric_names = ['Precision', 'Recall', 'F1 Score', 'NDCG']
        
        for metric, name in zip(metrics, metric_names):
            html += f"<tr><td><strong>{name}</strong></td>"
            for k in [1, 3, 5, 10]:
                value = performance_data.get(f'{metric}_at_{k}', {}).get('mean', 0)
                cell_class = 'highlight-cell' if metric == 'f1' and k == 5 else ''
                html += f'<td class="{cell_class}">{value:.3f}</td>'
            html += "</tr>"
        
        mrr_value = performance_data.get('MRR', {}).get('mean', 0)
        html += f"""
            <tr>
                <td><strong>MRR</strong></td>
                <td colspan="4" class="center-cell">{mrr_value:.3f}</td>
            </tr>
        """
        
        html += "</tbody></table>"
        return html

    def _create_discovery_table(self, discovery_examples: List[Dict]) -> str:
        """Create top discovery examples table."""
        html = """
        <table class="discovery-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Requirement ID</th>
                    <th>Activity Name</th>
                    <th>Discovery Score</th>
                    <th>Manual Traces</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, example in enumerate(discovery_examples[:15], 1):
            activity_name = example['activity_name']
            if len(activity_name) > 60:
                activity_name = activity_name[:57] + "..."
            
            score_class = "high-score" if example['score'] > 2.0 else "medium-score"
            manual_count = example.get('manual_matches_count', 0)
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{example['requirement_id']}</strong></td>
                    <td class="activity-name" title="{example['activity_name']}">{activity_name}</td>
                    <td class="{score_class}"><strong>{example['score']:.3f}</strong></td>
                    <td class="manual-count">{manual_count}</td>
                </tr>
            """
        
        html += "</tbody></table>"
        return html
    
    def _create_quality_analysis_table(self, quality_data: Dict) -> str:
        """Create quality analysis summary table."""
        html = """
        <div class="quality-analysis-container">
            <div class="quality-overview">
        """
        
        # Quality distribution
        quality_dist = quality_data.get('quality_distribution', {})
        if quality_dist:
            html += """
                <div class="quality-section">
                    <h4>📊 Quality Grade Distribution</h4>
                    <table class="quality-table">
                        <thead>
                            <tr>
                                <th>Quality Grade</th>
                                <th>Count</th>
                                <th>Percentage</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            total_reqs = sum(quality_dist.values())
            grade_colors = {
                'EXCELLENT': '#28a745',
                'GOOD': '#17a2b8', 
                'FAIR': '#ffc107',
                'POOR': '#fd7e14',
                'CRITICAL': '#dc3545'
            }
            
            for grade, count in quality_dist.items():
                percentage = (count / total_reqs * 100) if total_reqs > 0 else 0
                color = grade_colors.get(grade, '#6c757d')
                
                if grade in ['EXCELLENT', 'GOOD']:
                    status = '✅ Ready'
                elif grade == 'FAIR':
                    status = '⚠️ Review'
                else:
                    status = '❌ Needs Work'
                
                html += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">■</span> {grade}</td>
                        <td><strong>{count}</strong></td>
                        <td>{percentage:.1f}%</td>
                        <td>{status}</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        # Quality statistics
        quality_stats = quality_data.get('quality_stats', {})
        if quality_stats:
            html += """
                <div class="quality-section">
                    <h4>📈 Quality Score Statistics</h4>
                    <table class="quality-table">
                        <tbody>
            """
            
            stats_items = [
                ('Average Score', quality_stats.get('mean', 0), '%.1f'),
                ('Median Score', quality_stats.get('median', 0), '%.1f'),
                ('Standard Deviation', quality_stats.get('std', 0), '%.2f'),
                ('Minimum Score', quality_stats.get('min', 0), '%.1f'),
                ('Maximum Score', quality_stats.get('max', 0), '%.1f')
            ]
            
            for label, value, fmt in stats_items:
                formatted_value = fmt % value
                html += f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{formatted_value}</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        # Quality-Score correlation
        correlation_data = quality_data.get('quality_score_correlation', {})
        if correlation_data:
            html += """
                <div class="quality-section">
                    <h4>🔗 Quality vs Match Score Analysis</h4>
                    <table class="quality-table">
                        <thead>
                            <tr>
                                <th>Quality Level</th>
                                <th>Requirements</th>
                                <th>Avg Match Score</th>
                                <th>Good Matches</th>
                                <th>Success Rate</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            levels = [
                ('High Quality (≥70)', correlation_data.get('high_quality_matches', {}), '#28a745'),
                ('Medium Quality (50-69)', correlation_data.get('medium_quality_matches', {}), '#ffc107'),
                ('Low Quality (<50)', correlation_data.get('low_quality_matches', {}), '#dc3545')
            ]
            
            for level_name, level_data, color in levels:
                count = level_data.get('count', 0)
                avg_score = level_data.get('avg_match_score', 0)
                good_matches = level_data.get('good_matches', 0)
                success_rate = (good_matches / count * 100) if count > 0 else 0
                
                html += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">■</span> {level_name}</td>
                        <td><strong>{count}</strong></td>
                        <td>{avg_score:.3f}</td>
                        <td>{good_matches}</td>
                        <td>{success_rate:.1f}%</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        # Recommendations
        recommendations = quality_data.get('quality_recommendations', [])
        if recommendations:
            html += """
                <div class="quality-section">
                    <h4>💡 Quality Improvement Recommendations</h4>
                    <div class="recommendations-list">
            """
            
            priority_colors = {
                'high': '#dc3545',
                'medium': '#ffc107', 
                'low': '#28a745',
                'info': '#17a2b8'
            }
            
            for rec in recommendations:
                priority = rec.get('priority', 'info')
                text = rec.get('text', '')
                color = priority_colors.get(priority, '#6c757d')
                
                html += f"""
                    <div class="recommendation-item" style="border-left: 4px solid {color};">
                        <span class="priority-badge" style="background-color: {color};">{priority.upper()}</span>
                        <span class="rec-text">{text}</span>
                    </div>
                """
            
            html += """
                    </div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _create_coverage_analysis_table(self, coverage_data: Dict) -> str:
        """Create coverage analysis summary table."""
        html = """
        <div class="coverage-analysis-container">
            <div class="coverage-overview">
        """
        
        # Score distribution
        score_dist = coverage_data.get('score_distribution', {})
        if score_dist:
            html += """
                <div class="coverage-section">
                    <h4>📊 Match Score Distribution</h4>
                    <table class="coverage-table">
                        <thead>
                            <tr>
                                <th>Score Range</th>
                                <th>Matches</th>
                                <th>Percentage</th>
                                <th>Confidence Level</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            score_ranges = [
                ('High (≥1.0)', score_dist.get('high', {}), 'High Confidence', '#28a745'),
                ('Medium (0.6-0.99)', score_dist.get('medium', {}), 'Medium Confidence', '#ffc107'),
                ('Low (<0.6)', score_dist.get('low', {}), 'Low Confidence', '#dc3545')
            ]
            
            for range_name, range_data, confidence, color in score_ranges:
                count = range_data.get('count', 0)
                percentage = range_data.get('percentage', 0)
                
                html += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">■</span> {range_name}</td>
                        <td><strong>{count}</strong></td>
                        <td>{percentage:.1f}%</td>
                        <td>{confidence}</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        # Requirements coverage
        req_coverage = coverage_data.get('requirements_coverage', {})
        if req_coverage:
            html += """
                <div class="coverage-section">
                    <h4>📋 Requirements Coverage Analysis</h4>
                    <table class="coverage-table">
                        <thead>
                            <tr>
                                <th>Confidence Level</th>
                                <th>Requirements</th>
                                <th>Percentage</th>
                                <th>Review Priority</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            total_reqs = req_coverage.get('total_requirements', 0)
            coverage_levels = [
                ('High Confidence', req_coverage.get('high_confidence', {}), 'Quick Review', '#28a745'),
                ('Medium Confidence', req_coverage.get('medium_confidence', {}), 'Detailed Review', '#ffc107'),
                ('Low Confidence', req_coverage.get('low_confidence', {}), 'Manual Analysis', '#dc3545')
            ]
            
            for level_name, level_data, priority, color in coverage_levels:
                count = level_data.get('count', 0)
                percentage = level_data.get('percentage', 0)
                
                html += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">■</span> {level_name}</td>
                        <td><strong>{count}</strong></td>
                        <td>{percentage:.1f}%</td>
                        <td>{priority}</td>
                    </tr>
                """
            
            html += f"""
                        </tbody>
                    </table>
                    <div class="coverage-summary">
                        <strong>Total Requirements Analyzed: {total_reqs}</strong>
                    </div>
                </div>
            """
        
        # Effort estimation
        effort_data = coverage_data.get('effort_estimation', {})
        if effort_data:
            total_hours = effort_data.get('total_hours', 0)
            
            html += """
                <div class="coverage-section">
                    <h4>⏱️ Review Effort Estimation</h4>
                    <table class="coverage-table">
                        <thead>
                            <tr>
                                <th>Review Type</th>
                                <th>Matches</th>
                                <th>Est. Hours</th>
                                <th>Hours per Match</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            effort_types = [
                ('Auto Approve', effort_data.get('auto_approve', {}), 0.25, '#28a745'),
                ('Quick Review', effort_data.get('quick_review', {}), 0.5, '#17a2b8'),
                ('Detailed Review', effort_data.get('detailed_review', {}), 2.0, '#ffc107'),
                ('Manual Analysis', effort_data.get('manual_analysis', {}), 4.0, '#dc3545')
            ]
            
            for type_name, type_data, hours_per, color in effort_types:
                count = type_data.get('count', 0)
                hours = type_data.get('hours', 0)
                
                html += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">■</span> {type_name}</td>
                        <td><strong>{count}</strong></td>
                        <td>{hours:.1f}h</td>
                        <td>{hours_per}h</td>
                    </tr>
                """
            
            html += f"""
                        </tbody>
                    </table>
                    <div class="effort-summary">
                        <strong>Total Estimated Effort: {total_hours:.1f} hours</strong>
                    </div>
                </div>
            """
        
        # Coverage statistics
        coverage_stats = coverage_data.get('coverage_stats', {})
        if coverage_stats:
            html += """
                <div class="coverage-section">
                    <h4>📈 Coverage Statistics</h4>
                    <table class="coverage-table">
                        <tbody>
            """
            
            stats_items = [
                ('Total Matches', coverage_stats.get('total_matches', 0), '%d'),
                ('Average Score', coverage_stats.get('avg_score', 0), '%.3f'),
                ('Median Score', coverage_stats.get('median_score', 0), '%.3f'),
                ('Standard Deviation', coverage_stats.get('std_score', 0), '%.3f'),
                ('Score Range', f"{coverage_stats.get('min_score', 0):.3f} - {coverage_stats.get('max_score', 0):.3f}", '%s')
            ]
            
            for label, value, fmt in stats_items:
                if fmt == '%s':
                    formatted_value = value
                else:
                    formatted_value = fmt % value
                    
                html += f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{formatted_value}</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html