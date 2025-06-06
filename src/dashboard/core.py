"""
Dashboard Core - Main orchestrator for requirements traceability evaluation dashboard
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .charts import ChartGenerator
from .tables import TableGenerator  
from .templates import HTMLTemplateGenerator
from .data import DataProcessor
from .exports import DataExporter
from src.utils.repository_setup import RepositoryStructureManager  

class UnifiedEvaluationDashboard:
    """Unified dashboard that adapts based on available data."""
    
    def __init__(self, predictions_df: pd.DataFrame,
                ground_truth: Optional[Dict[str, List[Dict]]] = None,
                requirements_df: Optional[pd.DataFrame] = None,
                quality_results: Optional[Dict] = None,
                evaluation_results: Optional[Dict] = None,
                output_dir: Optional[str] = None,  # Changed to Optional
                repo_manager=None):
        
        self.predictions_df = predictions_df
        self.ground_truth = ground_truth
        self.requirements_df = requirements_df
        self.quality_results = quality_results
        self.evaluation_results = evaluation_results or {}
        
        # Setup repository manager FIRST
        if repo_manager is None:
            raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager
        
        # THEN determine output directory
        if output_dir is None:
            self.output_dir = self.repo_manager.structure['evaluation_dashboards']
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine dashboard capabilities
        self.capabilities = {
            'validation_mode': bool(ground_truth or evaluation_results.get('aggregate_metrics')),
            'exploration_mode': True,
            'quality_analysis': bool(quality_results or self._has_quality_columns()),
            'discovery_analysis': bool(evaluation_results.get('discovery_analysis')),
            'requirements_context': bool(requirements_df is not None)
        }
        
        # Initialize components AFTER repo_manager is set up
        self.data_processor = DataProcessor(ground_truth)
        self.chart_generator = ChartGenerator()
        self.table_generator = TableGenerator()
        self.template_generator = HTMLTemplateGenerator()
        self.data_exporter = DataExporter(self.output_dir, self.repo_manager) 
    def _has_quality_columns(self) -> bool:
        """Check if predictions_df contains quality analysis columns."""
        quality_columns = ['Quality_Grade', 'Quality_Score', 'Clarity_Score', 
                        'Completeness_Score', 'Verifiability_Score']
        return any(col in self.predictions_df.columns for col in quality_columns)
    
    def create_dashboard(self, dashboard_name: str = "unified_evaluation_dashboard") -> str:
        """Create unified HTML dashboard and return file path."""
        
        # Use existing evaluation results - no re-evaluation!
        if self.evaluation_results:
            print(f"✓ Using existing evaluation results with {len(self.evaluation_results.get('aggregate_metrics', {}))} metrics")
        else:
            print(f"✓ Using exploration mode (no evaluation results)")
        
        # Process data with existing evaluation results
        processed_data = self.data_processor.process_evaluation_data(
            self.evaluation_results,  # ← Use passed evaluation results
            self.predictions_df, 
            self.requirements_df
        )
        
        # Add capability info to processed data
        processed_data['capabilities'] = self.capabilities
        
        # Generate components
        charts = self.chart_generator.create_all_charts(processed_data)
        tables = self.table_generator.create_all_tables(processed_data)
        
        # Build HTML with capabilities
        html_content = self.template_generator.build_dashboard_html(
            processed_data, charts, tables, self.capabilities
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / f"{dashboard_name}.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Export data files
        self.data_exporter.export_all_data(processed_data)
        
        return str(dashboard_path)