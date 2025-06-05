"""
Data Export - Handle all data export operations and file management
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any

class DataExporter:
    """Handles all data export operations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def export_all_data(self, processed_data: Dict[str, Any]):
        """Export all processed data to various formats."""
        
        # Export predictions data
        if processed_data.get('predictions_data'):
            self._export_predictions(processed_data['predictions_data'])
        
        # Export discovery data
        if processed_data.get('discovery_data'):
            self._export_discovery_data(processed_data['discovery_data'])
        
        # Export summary statistics
        if processed_data.get('summary_stats'):
            self._export_summary_stats(processed_data['summary_stats'])
        
        # Export metadata
        if processed_data.get('metadata'):
            self._export_metadata(processed_data['metadata'])
    
    def _export_predictions(self, predictions_data: List[Dict]):
        """Export predictions data to CSV."""
        df = pd.DataFrame(predictions_data)
        output_path = self.output_dir / "all_predictions_with_context.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Exported predictions data: {output_path}")
    
    def _export_discovery_data(self, discovery_data: Dict[str, Any]):
        """Export discovery analysis data."""
        
        # High-scoring misses
        if discovery_data.get('high_scoring_misses'):
            df = pd.DataFrame(discovery_data['high_scoring_misses'])
            output_path = self.output_dir / "discovery_high_scoring_misses.csv"
            df.to_csv(output_path, index=False)
            print(f"✓ Exported discovery data: {output_path}")
        
        # Score gaps
        if discovery_data.get('score_gaps'):
            df = pd.DataFrame(discovery_data['score_gaps'])
            output_path = self.output_dir / "discovery_score_gaps.csv"
            df.to_csv(output_path, index=False)
            print(f"✓ Exported score gaps: {output_path}")
        
        # Discovery summary
        if discovery_data.get('summary'):
            output_path = self.output_dir / "discovery_summary.json"
            with open(output_path, 'w') as f:
                json.dump(discovery_data['summary'], f, indent=2)
            print(f"✓ Exported discovery summary: {output_path}")
    
    def _export_summary_stats(self, summary_stats: Dict[str, Any]):
        """Export summary statistics."""
        output_path = self.output_dir / "summary_statistics.json"
        with open(output_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"✓ Exported summary stats: {output_path}")
    
    def _export_metadata(self, metadata: Dict[str, Any]):
        """Export metadata information."""
        output_path = self.output_dir / "evaluation_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Exported metadata: {output_path}")