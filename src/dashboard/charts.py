"""
Chart Generation - Create all visualizations as base64-encoded images
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import io
from typing import Dict, Any

class ChartGenerator:
    """Generates all dashboard charts."""
    
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_all_charts(self, processed_data: Dict[str, Any]) -> Dict[str, str]:
        """Create charts based on available data."""
        charts = {}
        capabilities = processed_data.get('capabilities', {})
        
        # Always available charts
        try:
            charts.update(self._create_coverage_charts(processed_data))
        except Exception as e:
            print(f"⚠️ Coverage charts failed: {e}")
        
        # Quality analysis charts
        quality_data = processed_data.get('quality_data', {})
        if quality_data.get('has_quality_data', False):
            try:
                charts.update(self._create_quality_charts(quality_data))
            except Exception as e:
                print(f"⚠️ Quality charts failed: {e}")
        
        # Coverage analysis charts
        coverage_data = processed_data.get('coverage_data', {})
        if coverage_data.get('has_coverage_data', False):
            try:
                charts.update(self._create_coverage_analysis_charts(coverage_data))
            except Exception as e:
                print(f"⚠️ Coverage analysis charts failed: {e}")
        
        performance_data = processed_data.get('performance_metrics', {})
        discovery_data = processed_data.get('discovery_data', {})
        
        if performance_data:
            try:
                charts.update(self._create_performance_charts(performance_data))
            except Exception as e:
                print(f"⚠️ Performance charts failed: {e}")
        
        if discovery_data:
            try:
                charts.update(self._create_discovery_charts(discovery_data, processed_data.get('score_distributions', {})))
            except Exception as e:
                print(f"⚠️ Discovery charts failed: {e}")
        
        return charts

    def _create_exploration_charts(self, processed_data: Dict) -> Dict[str, str]:
        """Charts for basic match exploration."""
        charts = {}
        
        # Score distribution chart (always useful)
        fig, ax = plt.subplots(figsize=(10, 6))
        predictions = processed_data['predictions_data']
        scores = [p['combined_score'] for p in predictions]
        
        ax.hist(scores, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax.set_title('Match Score Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Combined Score')
        ax.set_ylabel('Number of Matches')
        ax.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            value = np.percentile(scores, p)
            ax.axvline(x=value, color='red', linestyle='--', alpha=0.7,
                    label=f'{p}th percentile: {value:.3f}')
        
        ax.legend()
        plt.tight_layout()
        charts['score_distribution'] = self._fig_to_base64(fig)
        plt.close()
        
        return charts
    
    def _create_coverage_charts(self, processed_data: Dict) -> Dict[str, str]:
        """Create coverage analysis charts."""
        charts = {}
        
        # Score distribution chart (always useful)
        fig, ax = plt.subplots(figsize=(10, 6))
        predictions = processed_data.get('predictions_data', [])
        
        if predictions:
            scores = [p['combined_score'] for p in predictions]
            
            ax.hist(scores, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            ax.set_title('Match Score Distribution', fontweight='bold', fontsize=14)
            ax.set_xlabel('Combined Score')
            ax.set_ylabel('Number of Matches')
            ax.grid(True, alpha=0.3)
            
            # Add percentile lines
            percentiles = [50, 75, 90, 95]
            for p in percentiles:
                value = np.percentile(scores, p)
                ax.axvline(x=value, color='red', linestyle='--', alpha=0.7,
                        label=f'{p}th percentile: {value:.3f}')
            
            ax.legend()
            plt.tight_layout()
            charts['score_distribution'] = self._fig_to_base64(fig)
            plt.close()
        
        return charts
    
    def _create_performance_charts(self, performance_data: Dict) -> Dict[str, str]:
        """Create performance metric charts."""
        charts = {}
        
        # Precision/Recall/F1 Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        k_values = [1, 3, 5, 10]
        
        precision_vals = [performance_data.get(f'precision_at_{k}', {}).get('mean', 0) for k in k_values]
        recall_vals = [performance_data.get(f'recall_at_{k}', {}).get('mean', 0) for k in k_values]
        f1_vals = [performance_data.get(f'f1_at_{k}', {}).get('mean', 0) for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        ax.bar(x - width, precision_vals, width, label='Precision@k', alpha=0.8, color='#2E86AB')
        ax.bar(x, recall_vals, width, label='Recall@k', alpha=0.8, color='#A23B72')
        ax.bar(x + width, f1_vals, width, label='F1@k', alpha=0.8, color='#F18F01')
        
        ax.set_xlabel('k')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by k')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (p, r, f1) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        charts['performance_metrics'] = self._fig_to_base64(fig)
        plt.close()
        
        # MRR and NDCG Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MRR
        mrr_data = performance_data.get('MRR', {})
        mrr_mean = mrr_data.get('mean', 0)
        mrr_std = mrr_data.get('std', 0)
        
        ax1.bar(['MRR'], [mrr_mean], yerr=[mrr_std], capsize=5, 
                color='#C73E1D', alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Mean Reciprocal Rank')
        ax1.text(0, mrr_mean + mrr_std + 0.02, f'{mrr_mean:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        y_max = max(1.2, mrr_mean + mrr_std * 1.5)
        ax1.set_ylim(0, y_max)
        
        # NDCG@k
        ndcg_vals = [performance_data.get(f'ndcg_at_{k}', {}).get('mean', 0) for k in k_values]
        ax2.plot(k_values, ndcg_vals, marker='o', linewidth=2, markersize=8, 
                color='#3E92CC', label='NDCG@k')
        ax2.set_xlabel('k')
        ax2.set_ylabel('NDCG Score')
        ax2.set_title('NDCG by k')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (k, ndcg) in enumerate(zip(k_values, ndcg_vals)):
            ax2.text(k, ndcg + 0.01, f'{ndcg:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        charts['mrr_ndcg'] = self._fig_to_base64(fig)
        plt.close()
        
        return charts
    
    def _create_discovery_charts(self, discovery_data: Dict, score_dist_data: Dict) -> Dict[str, str]:
        """Create discovery analysis charts."""
        charts = {}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Discovery Summary
        summary = discovery_data.get('summary', {})
        categories = ['High-Scoring\nDiscoveries', 'Requirements\nwith Discoveries', 'Score\nGaps Found']
        values = [
            summary.get('total_high_scoring_misses', 0),
            summary.get('requirements_with_high_misses', 0),
            summary.get('score_gaps_count', 0)
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_title('Discovery Analysis Summary', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Count')
        max_value = max(values) if values else 10
        ax1.set_ylim(0, max_value * 1.15)
        
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Score Distribution Comparison
        if score_dist_data:
            manual = score_dist_data.get('manual_scores', {})
            non_manual = score_dist_data.get('non_manual_scores', {})
            
            if manual.get('scores') and non_manual.get('scores'):
                score_data = [manual['scores'], non_manual['scores']]
                labels = [f"Manual Matches\n(n={manual['count']})", 
                         f"Algorithm Suggestions\n(n={non_manual['count']})"]
                
                bp = ax2.boxplot(score_data, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('#4ECDC4')
                bp['boxes'][1].set_facecolor('#FF6B6B')
                
                ax2.set_title('Score Distribution: Manual vs Algorithm', fontweight='bold', fontsize=12)
                ax2.set_ylabel('Combined Score')
                ax2.grid(True, alpha=0.3)
                
                # Add mean lines
                ax2.axhline(y=manual['mean'], color='#4ECDC4', linestyle='--', alpha=0.7,
                           label=f"Manual Mean: {manual['mean']:.3f}")
                ax2.axhline(y=non_manual['mean'], color='#FF6B6B', linestyle='--', alpha=0.7,
                           label=f"Algorithm Mean: {non_manual['mean']:.3f}")
                ax2.legend(fontsize=9)
        
        # 3. Discovery Rate Analysis
        discovery_rate = summary.get('discovery_rate', 0)
        # Placeholder for coverage rate - you'd get this from processed_data
        coverage_rate = 0.8  # This should come from your data

        metrics = ['Discovery Rate', 'Requirement\nCoverage']
        rates = [discovery_rate, coverage_rate]
        colors_rates = ['#FFD93D', '#45B7D1']

        bars = ax3.bar(metrics, rates, color=colors_rates, alpha=0.8)
        ax3.set_title('Algorithm Coverage Analysis', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Rate')

        # Dynamic y-axis scaling based on actual data
        max_rate = max(rates) if rates else 1.0
        y_limit = max(1.0, max_rate * 1.1)  # At least 1.0, or 110% of max value
        ax3.set_ylim(0, y_limit)
        
        for bar, rate in zip(bars, rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Score Quality Histogram
        high_scoring_misses = discovery_data.get('high_scoring_misses', [])
        if high_scoring_misses:
            scores = [miss['score'] for miss in high_scoring_misses[:100]]
            if scores:
                ax4.hist(scores, bins=20, alpha=0.7, color='#FFD93D', edgecolor='black')
                ax4.set_title('Discovery Score Distribution', fontweight='bold', fontsize=12)
                ax4.set_xlabel('Discovery Score')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(bottom=0)
                
                mean_score = np.mean(scores)
                ax4.axvline(x=mean_score, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_score:.3f}')
                ax4.legend()
        
        plt.tight_layout()
        charts['discovery_overview'] = self._fig_to_base64(fig)
        plt.close()
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    def _create_quality_charts(self, quality_data: Dict) -> Dict[str, str]:
        """Create quality analysis charts."""
        charts = {}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Quality Grade Distribution
        quality_dist = quality_data.get('quality_distribution', {})
        if quality_dist:
            grades = list(quality_dist.keys())
            counts = list(quality_dist.values())
            colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#8E44AD'][:len(grades)]
            
            bars = ax1.bar(grades, counts, color=colors, alpha=0.8)
            ax1.set_title('Requirement Quality Distribution', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Number of Requirements')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Quality Score vs Match Score Correlation
        correlation_data = quality_data.get('quality_score_correlation', {})
        if correlation_data:
            categories = ['High Quality\n(≥70)', 'Medium Quality\n(50-69)', 'Low Quality\n(<50)']
            avg_scores = [
                correlation_data.get('high_quality_matches', {}).get('avg_match_score', 0),
                correlation_data.get('medium_quality_matches', {}).get('avg_match_score', 0),
                correlation_data.get('low_quality_matches', {}).get('avg_match_score', 0)
            ]
            
            bars = ax2.bar(categories, avg_scores, color=['#2ECC71', '#F39C12', '#E74C3C'], alpha=0.8)
            ax2.set_title('Quality vs Match Score Correlation', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Average Match Score')
            ax2.set_ylim(0, max(avg_scores) * 1.2 if avg_scores else 1)
            
            # Add value labels
            for bar, score in zip(bars, avg_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Quality Statistics Overview
        quality_stats = quality_data.get('quality_stats', {})
        if quality_stats:
            metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
            values = [
                quality_stats.get('mean', 0),
                quality_stats.get('median', 0), 
                quality_stats.get('std', 0),
                quality_stats.get('min', 0),
                quality_stats.get('max', 0)
            ]
            
            bars = ax3.bar(metrics, values, color='#3498DB', alpha=0.8)
            ax3.set_title('Quality Score Statistics', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Quality Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Good Matches by Quality Level
        if correlation_data:
            categories = ['High Quality', 'Medium Quality', 'Low Quality']
            good_matches = [
                correlation_data.get('high_quality_matches', {}).get('good_matches', 0),
                correlation_data.get('medium_quality_matches', {}).get('good_matches', 0), 
                correlation_data.get('low_quality_matches', {}).get('good_matches', 0)
            ]
            total_matches = [
                correlation_data.get('high_quality_matches', {}).get('count', 1),
                correlation_data.get('medium_quality_matches', {}).get('count', 1),
                correlation_data.get('low_quality_matches', {}).get('count', 1)
            ]
            
            percentages = [g/t*100 if t > 0 else 0 for g, t in zip(good_matches, total_matches)]
            
            bars = ax4.bar(categories, percentages, color=['#2ECC71', '#F39C12', '#E74C3C'], alpha=0.8)
            ax4.set_title('Good Matches Rate by Quality Level', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Percentage of Good Matches (≥0.6)')
            ax4.set_ylim(0, 100)
            
            # Add value labels
            for bar, pct, good, total in zip(bars, percentages, good_matches, total_matches):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{pct:.1f}%\n({good}/{total})', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        charts['quality_analysis'] = self._fig_to_base64(fig)
        plt.close()
        
        return charts
    
    def _create_coverage_analysis_charts(self, coverage_data: Dict) -> Dict[str, str]:
        """Create coverage analysis charts."""
        charts = {}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Score Distribution 
        score_dist = coverage_data.get('score_distribution', {})
        if score_dist:
            categories = ['High (≥1.0)', 'Medium (0.6-0.99)', 'Low (<0.6)']
            counts = [
                score_dist.get('high', {}).get('count', 0),
                score_dist.get('medium', {}).get('count', 0),
                score_dist.get('low', {}).get('count', 0)
            ]
            colors = ['#2ECC71', '#F39C12', '#E74C3C']
            
            bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
            ax1.set_title('Match Score Distribution', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Number of Matches')
            
            # Add value labels with percentages
            total = sum(counts)
            for bar, count in zip(bars, counts):
                pct = count/total*100 if total > 0 else 0
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                        f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 2. Requirements Coverage
        req_coverage = coverage_data.get('requirements_coverage', {})
        if req_coverage:
            categories = ['High Confidence', 'Medium Confidence', 'Low Confidence']
            counts = [
                req_coverage.get('high_confidence', {}).get('count', 0),
                req_coverage.get('medium_confidence', {}).get('count', 0),
                req_coverage.get('low_confidence', {}).get('count', 0)
            ]
            
            bars = ax2.bar(categories, counts, color=['#2ECC71', '#F39C12', '#E74C3C'], alpha=0.8)
            ax2.set_title('Requirements by Confidence Level', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Number of Requirements')
            
            # Add value labels
            total_reqs = req_coverage.get('total_requirements', 1)
            for bar, count in zip(bars, counts):
                pct = count/total_reqs*100 if total_reqs > 0 else 0
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 3. Effort Estimation
        effort_data = coverage_data.get('effort_estimation', {})
        if effort_data:
            categories = ['Auto\nApprove', 'Quick\nReview', 'Detailed\nReview', 'Manual\nAnalysis']
            hours = [
                effort_data.get('auto_approve', {}).get('hours', 0),
                effort_data.get('quick_review', {}).get('hours', 0),
                effort_data.get('detailed_review', {}).get('hours', 0),
                effort_data.get('manual_analysis', {}).get('hours', 0)
            ]
            
            bars = ax3.bar(categories, hours, color=['#2ECC71', '#3498DB', '#F39C12', '#E74C3C'], alpha=0.8)
            ax3.set_title('Estimated Review Effort', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Hours')
            
            # Add value labels
            for bar, hour in zip(bars, hours):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hours)*0.01,
                        f'{hour:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        # 4. Coverage Statistics Summary
        coverage_stats = coverage_data.get('coverage_stats', {})
        if coverage_stats:
            metrics = ['Avg Score', 'Median Score', 'Std Dev', 'Min Score', 'Max Score']
            values = [
                coverage_stats.get('avg_score', 0),
                coverage_stats.get('median_score', 0),
                coverage_stats.get('std_score', 0),
                coverage_stats.get('min_score', 0),
                coverage_stats.get('max_score', 0)
            ]
            
            bars = ax4.bar(metrics, values, color='#3498DB', alpha=0.8)
            ax4.set_title('Coverage Score Statistics', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Score Value')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        charts['coverage_analysis'] = self._fig_to_base64(fig)
        plt.close()
        
        return charts