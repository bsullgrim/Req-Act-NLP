"""
Matching Workbook Generator - Create comprehensive Excel workbook for engineering review
Organizes matches by confidence level with discovery analysis and action items
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json

# Import project utilities
from src.utils.file_utils import SafeFileHandler
from src.utils.path_resolver import SmartPathResolver

logger = logging.getLogger(__name__)


class MatchingWorkbookGenerator:
    """Generate comprehensive Excel workbook for engineering teams to review matches."""
    
    def __init__(self, repo_manager=None):
        self.workbook_path = None
        if repo_manager is None:
         raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager

        # Initialize utilities
        self.file_handler = SafeFileHandler(self.repo_manager)
        self.path_resolver = SmartPathResolver(self.repo_manager)
        
    def create_workbook(self, 
                    enhanced_df: pd.DataFrame,
                    evaluation_results: Optional[Dict] = None,
                    output_path: Optional[str] = None,
                    repo_manager=None) -> str:
        """UNCHANGED SIGNATURE - only enhanced implementation"""
        
        # Setup repository manager (unchanged)
        if repo_manager is None:
            from src.utils.repository_setup import RepositoryStructureManager
            repo_manager = RepositoryStructureManager("outputs")
            repo_manager.setup_repository_structure()
        
        # Use proper output path (unchanged)
        if output_path is None:
            output_path = repo_manager.structure['engineering_review'] / "matching_workbook.xlsx"
        else:
            output_path = Path(output_path)
        
        # ENHANCEMENT: Add data validation (new)
        if enhanced_df is None or enhanced_df.empty:
            logger.error("❌ No data provided to workbook generator")
            raise ValueError("Enhanced DataFrame is empty or None")
        
        logger.info(f"📊 Creating workbook with {len(enhanced_df)} matches")
        
        # Create Excel writer (enhanced error handling)
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Tab 1: Executive Summary (enhanced)
                try:
                    exec_summary = self._create_executive_summary(enhanced_df, evaluation_results)
                    exec_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
                except Exception as e:
                    logger.warning(f"Executive summary failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # ENHANCEMENT: Fallback summary to ensure at least one sheet
                    fallback_summary = pd.DataFrame([
                        {"Metric": "Total Matches", "Value": len(enhanced_df)},
                        {"Metric": "Status", "Value": "Generated with fallback"}
                    ])
                    fallback_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
                
                # Tab 2-4: Confidence-based tabs (enhanced)
                try:
                    self._create_confidence_tabs(enhanced_df, writer)
                except Exception as e:
                    logger.error(f"Confidence tabs failed: {e}")
                    # ENHANCEMENT: Emergency fallback tab
                    enhanced_df.to_excel(writer, sheet_name='All Matches', index=False)
                
                # Tab 5: Discovery Analysis (unchanged logic, enhanced error handling)
                if evaluation_results and 'discovery_analysis' in evaluation_results:
                    try:
                        self._create_discovery_tab(evaluation_results['discovery_analysis'], writer)
                    except Exception as e:
                        logger.warning(f"Discovery tab failed: {e}")
                
                # Tab 6: Action Items (enhanced)
                try:
                    action_items = self._create_action_items(enhanced_df)
                    action_items.to_excel(writer, sheet_name='Action Items', index=False)
                except Exception as e:
                    logger.warning(f"Action items failed: {e}")
                
                # Format all worksheets (enhanced)
                try:
                    self._format_workbook(writer)
                except Exception as e:
                    logger.warning(f"Formatting failed: {e}")
        
        except Exception as e:
            logger.error(f"Excel creation failed: {e}")
            # ENHANCEMENT: CSV fallback
            fallback_path = output_path.with_suffix('.csv')
            enhanced_df.to_csv(fallback_path, index=False)
            return str(fallback_path)
        
        logger.info(f"✅ Matching workbook created: {output_path}")
        self.workbook_path = str(output_path)
        return str(output_path)
    
    def _create_executive_summary(self, enhanced_df: pd.DataFrame, 
                                evaluation_results: Optional[Dict]) -> pd.DataFrame:
        """Create executive summary with key metrics."""
        
        total_matches = len(enhanced_df)
        unique_reqs = enhanced_df['ID'].nunique()
        
        # Confidence distribution
        high_conf = len(enhanced_df[enhanced_df['Combined Score'] >= 0.8])
        med_conf = len(enhanced_df[(enhanced_df['Combined Score'] >= 0.5) & 
                                 (enhanced_df['Combined Score'] < 0.8)])
        low_conf = len(enhanced_df[enhanced_df['Combined Score'] < 0.5])
        
        summary_data = [
            # Overview metrics
            {"Category": "Overview", "Metric": "Total Matches", "Value": total_matches, 
             "Percentage": "100.0%", "Notes": "All algorithm suggestions"},
            {"Category": "Overview", "Metric": "Requirements Covered", "Value": unique_reqs,
             "Percentage": f"{unique_reqs/total_matches*100:.1f}%", "Notes": "Unique requirements with matches"},
            
            # Confidence distribution
            {"Category": "Confidence", "Metric": "High Confidence (≥0.8)", "Value": high_conf,
             "Percentage": f"{high_conf/total_matches*100:.1f}%", "Notes": "Ready for approval"},
            {"Category": "Confidence", "Metric": "Medium Confidence (0.5-0.8)", "Value": med_conf,
             "Percentage": f"{med_conf/total_matches*100:.1f}%", "Notes": "Needs review"},
            {"Category": "Confidence", "Metric": "Low Confidence (<0.5)", "Value": low_conf,
             "Percentage": f"{low_conf/total_matches*100:.1f}%", "Notes": "Manual investigation"},
            
            # Quality distribution (if available)
        ]
        
        if 'Quality_Grade' in enhanced_df.columns:
            quality_counts = enhanced_df['Quality_Grade'].value_counts()
            for grade, count in quality_counts.items():
                summary_data.append({
                    "Category": "Quality", "Metric": f"Requirements: {grade}", 
                    "Value": count, "Percentage": f"{count/total_matches*100:.1f}%",
                    "Notes": "Quality of underlying requirements"
                })
        # Add explanation coverage
        explanations_data = self._load_explanations()
        if explanations_data:
            summary_data.append({
                "Category": "Explainability", 
                "Metric": "Matches with Explanations", 
                "Value": len(explanations_data),
                "Percentage": f"{len(explanations_data)/total_matches*100:.1f}%",
                "Notes": "Detailed reasoning available"
            })
        # Evaluation metrics (if available)
        if evaluation_results and 'aggregate_metrics' in evaluation_results:
            agg_metrics = evaluation_results['aggregate_metrics']
            if 'f1_at_5' in agg_metrics:
                f1_score = agg_metrics['f1_at_5']['mean']
                summary_data.append({
                    "Category": "Performance", "Metric": "F1@5 Score", 
                    "Value": f"{f1_score:.3f}", "Percentage": f"{f1_score*100:.1f}%",
                    "Notes": "Algorithm validation against manual traces"
                })
        
        # Discovery metrics (if available)
        if evaluation_results and 'discovery_analysis' in evaluation_results:
            discovery = evaluation_results['discovery_analysis']['summary']
            summary_data.append({
                "Category": "Discovery", "Metric": "Novel Connections Found",
                "Value": discovery.get('total_high_scoring_misses', 0),
                "Percentage": f"{discovery.get('discovery_rate', 0)*100:.1f}%",
                "Notes": "High-scoring matches not in manual traces"
            })
        
        return pd.DataFrame(summary_data)
    
    def _create_confidence_tabs(self, enhanced_df: pd.DataFrame, writer):
        """UNCHANGED SIGNATURE - only enhanced implementation"""
        
        # ENHANCEMENT: Flexible score column detection
        score_col = None
        for col in ['Combined Score', 'Combined_Score', 'combined_score', 'score']:
            if col in enhanced_df.columns:
                score_col = col
                break
        
        if score_col is None:
            logger.warning("No score column found")
            # ENHANCEMENT: Create single tab with all data
            enhanced_df.to_excel(writer, sheet_name='All Matches', index=False)
            return
        
        # Load explanations if available (unchanged)
        explanations_data = self._load_explanations()
        
        # UNCHANGED: Same confidence level logic
        confidence_levels = [
            ("High Confidence (≥0.8)", enhanced_df[enhanced_df[score_col] >= 0.8]),
            ("Medium Confidence (0.5-0.8)", enhanced_df[(enhanced_df[score_col] >= 0.5) & 
                                                        (enhanced_df[score_col] < 0.8)]),
            ("Low Confidence (<0.5)", enhanced_df[enhanced_df[score_col] < 0.5])
        ]
        
        tabs_created = 0  # ENHANCEMENT: Track if any tabs created
        
        for tab_name, df_subset in confidence_levels:
            if len(df_subset) > 0:
                try:
                    # Select and order columns for engineering review (enhanced)
                    review_columns = self._get_review_columns(df_subset)
                    review_df = df_subset[review_columns].copy()
                    
                    # Sort by score (enhanced to use detected column)
                    review_df = review_df.sort_values(score_col, ascending=False)
                    
                    # ENHANCEMENT: Safe explanation addition
                    try:
                        review_df['Match Explanation'] = review_df.apply(
                            lambda row: self._get_explanation_text(row, explanations_data), axis=1
                        )
                        review_df['Key Evidence'] = review_df.apply(
                            lambda row: self._get_key_evidence(row, explanations_data), axis=1
                        )
                        review_df['Score Breakdown'] = review_df.apply(
                            lambda row: self._get_score_breakdown(row, explanations_data), axis=1
                        )
                    except Exception as e:
                        logger.warning(f"Could not add explanations: {e}")
                    
                    # Add review columns (unchanged)
                    review_df['Review Status'] = 'PENDING'
                    review_df['Assigned Engineer'] = ''
                    review_df['Review Notes'] = ''
                    review_df['Approval Decision'] = ''
                    
                    # Write to Excel (unchanged)
                    review_df.to_excel(writer, sheet_name=tab_name, index=False)
                    tabs_created += 1
                    
                    logger.info(f"✓ Created {tab_name} tab: {len(review_df)} matches")
                    
                except Exception as e:
                    logger.error(f"Failed to create {tab_name}: {e}")
        
        # ENHANCEMENT: Ensure at least one data tab exists
        if tabs_created == 0:
            enhanced_df.to_excel(writer, sheet_name='All Matches', index=False)
    
    def _get_review_columns(self, df: pd.DataFrame) -> List[str]:
        """UNCHANGED SIGNATURE - only enhanced implementation for column compatibility"""
        
        # ENHANCEMENT: Flexible column detection instead of hardcoded names
        base_columns = []
        
        # Find ID column
        for id_col in ['ID', 'Requirement_ID', 'requirement_id']:
            if id_col in df.columns:
                base_columns.append(id_col)
                break
        
        # Find name column
        for name_col in ['Requirement Name', 'Requirement_Name', 'requirement_name']:
            if name_col in df.columns:
                base_columns.append(name_col)
                break
        
        # Find text column
        for text_col in ['Requirement Text', 'Requirement_Text', 'requirement_text']:
            if text_col in df.columns:
                base_columns.append(text_col)
                break
        
        # Find activity column
        for act_col in ['Activity Name', 'Activity_Name', 'activity_name']:
            if act_col in df.columns:
                base_columns.append(act_col)
                break
        
        # Find score column
        for score_col in ['Combined Score', 'Combined_Score', 'combined_score']:
            if score_col in df.columns:
                base_columns.append(score_col)
                break
        
        # UNCHANGED: Add score components if available (but with flexible detection)
        score_columns = ['Dense Semantic', 'Semantic_Score', 'BM25 Score', 'BM25_Score', 
                        'Syntactic Score', 'Syntactic_Score', 'Domain Weighted', 'Domain_Score',
                        'Query Expansion', 'Query_Expansion_Score']
        available_score_cols = [col for col in score_columns if col in df.columns]
        base_columns.extend(available_score_cols)
        
        # UNCHANGED: Add quality information if available
        quality_columns = ['Quality_Grade', 'Quality_Score']
        available_quality_cols = [col for col in quality_columns if col in df.columns]
        base_columns.extend(available_quality_cols)
        
        # ENHANCEMENT: Ensure we have at least some columns
        if len(base_columns) < 2:
            base_columns = list(df.columns)  # Use all columns as fallback
        
        # Only return columns that exist in the DataFrame (unchanged)
        return [col for col in base_columns if col in df.columns]    
    
    def _create_discovery_tab(self, discovery_analysis: Dict, writer):
        """Create discovery analysis tab if evaluation results available."""
        
        high_scoring_misses = discovery_analysis.get('high_scoring_misses', [])
        
        if high_scoring_misses:
            discovery_df = pd.DataFrame(high_scoring_misses)
            
            # Add investigation columns
            discovery_df['Investigation Status'] = 'PENDING'
            discovery_df['Assigned Investigator'] = ''
            discovery_df['Investigation Notes'] = ''
            discovery_df['Outcome'] = ''
            
            # Sort by score
            discovery_df = discovery_df.sort_values('score', ascending=False)
            
            discovery_df.to_excel(writer, sheet_name='Discovery Analysis', index=False)
            logger.info(f"   ✓ Created Discovery Analysis tab: {len(discovery_df)} discoveries")
    
    def _create_action_items(self, enhanced_df: pd.DataFrame) -> pd.DataFrame:
        """Create prioritized action items for project management."""
        
        action_items = []
        
        # Group by confidence level for different action types
        high_conf = enhanced_df[enhanced_df['Combined Score'] >= 0.8]
        med_conf = enhanced_df[(enhanced_df['Combined Score'] >= 0.5) & 
                              (enhanced_df['Combined Score'] < 0.8)]
        low_conf = enhanced_df[enhanced_df['Combined Score'] < 0.5]
        
        # High confidence - approval workflow
        for _, row in high_conf.head(20).iterrows():  # Top 20 for manageable list
            action_items.append({
                'Priority': 'HIGH',
                'Action Type': 'APPROVE',
                'Requirement ID': row['ID'],
                'Requirement Name': row.get('Requirement Name', 'N/A'),
                'Activity Name': row['Activity Name'],
                'Score': row['Combined Score'],
                'Assigned To': '',
                'Due Date': '',
                'Status': 'PENDING',
                'Notes': f'High confidence match ({row["Combined Score"]:.3f}) - fast-track approval'
            })
        
        # Medium confidence - detailed review
        for _, row in med_conf.head(15).iterrows():  # Top 15
            action_items.append({
                'Priority': 'MEDIUM',
                'Action Type': 'REVIEW',
                'Requirement ID': row['ID'],
                'Requirement Name': row.get('Requirement Name', 'N/A'),
                'Activity Name': row['Activity Name'],
                'Score': row['Combined Score'],
                'Assigned To': '',
                'Due Date': '',
                'Status': 'PENDING',
                'Notes': f'Medium confidence ({row["Combined Score"]:.3f}) - needs detailed review'
            })
        
        # Low confidence - investigation
        for _, row in low_conf.head(10).iterrows():  # Top 10
            action_items.append({
                'Priority': 'LOW',
                'Action Type': 'INVESTIGATE',
                'Requirement ID': row['ID'],
                'Requirement Name': row.get('Requirement Name', 'N/A'),
                'Activity Name': row['Activity Name'],
                'Score': row['Combined Score'],
                'Assigned To': '',
                'Due Date': '',
                'Status': 'PENDING',
                'Notes': f'Low confidence ({row["Combined Score"]:.3f}) - manual investigation'
            })
        
        return pd.DataFrame(action_items)

    def _load_explanations(self) -> Dict:
        """Load match explanations from JSON file."""
        # Try multiple possible locations
        possible_paths = [
            self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json",
            Path("outputs/matching_results/aerospace_matches_explanations.json"),
            Path("aerospace_matches_explanations.json")
        ]
        
        for explanations_file in possible_paths:
            if explanations_file.exists():
                try:
                    with open(explanations_file, 'r', encoding='utf-8') as f:
                        explanations = json.load(f)
                        # Create lookup dictionary
                        explanations_dict = {}
                        for exp in explanations:
                            key = (exp['requirement_id'], exp['activity_name'])
                            explanations_dict[key] = exp
                        logger.info(f"✅ Loaded {len(explanations_dict)} match explanations from {explanations_file}")
                        return explanations_dict
                except Exception as e:
                    logger.warning(f"Could not load explanations from {explanations_file}: {e}")
        
        logger.warning("No explanations file found in expected locations")
        return {}

    def _get_explanation_text(self, row, explanations_data: Dict) -> str:
        """UNCHANGED SIGNATURE - enhanced implementation for column flexibility"""
        
        # ENHANCEMENT: Try multiple key formats for compatibility
        possible_keys = []
        
        # Try different ID column names
        for id_col in ['ID', 'Requirement_ID', 'requirement_id']:
            if id_col in row:
                # Try different activity column names
                for act_col in ['Activity Name', 'Activity_Name', 'activity_name']:
                    if act_col in row:
                        possible_keys.append((row[id_col], row[act_col]))
        
        # UNCHANGED: Original explanation logic, but with flexible keys
        for key in possible_keys:
            if key in explanations_data:
                exp = explanations_data[key]
                sem_level = exp.get('semantic_similarity_level', 'Unknown')
                parts = [f"{sem_level} semantic match"]
                scores = exp.get('scores', {})
                if scores.get('bm25', 0) > 0.5:
                    parts.append("strong term overlap")
                if scores.get('domain', 0) > 0.3:
                    parts.append("aerospace terms matched")
                return "; ".join(parts)
        
        # ENHANCEMENT: Flexible fallback based on available score columns
        score = 0
        for score_col in ['Combined Score', 'Combined_Score', 'combined_score', 'score']:
            if score_col in row and pd.notna(row[score_col]):
                score = row[score_col]
                break
        
        # UNCHANGED: Same fallback logic
        if score >= 0.8:
            return "Very strong match across multiple dimensions"
        elif score >= 0.6:
            return "Good match with solid evidence"
        elif score >= 0.4:
            return "Moderate match, review recommended"
        else:
            return "Weak match, manual verification needed"        
    
    def _get_key_evidence(self, row, explanations_data: Dict) -> str:
        """UNCHANGED SIGNATURE - enhanced implementation for column flexibility"""
        
        # ENHANCEMENT: Same flexible key detection as above
        possible_keys = []
        for id_col in ['ID', 'Requirement_ID', 'requirement_id']:
            if id_col in row:
                for act_col in ['Activity Name', 'Activity_Name', 'activity_name']:
                    if act_col in row:
                        possible_keys.append((row[id_col], row[act_col]))
        
        # UNCHANGED: Original evidence logic
        for key in possible_keys:
            if key in explanations_data:
                exp = explanations_data[key]
                shared_terms = exp.get('shared_terms', [])
                if shared_terms:
                    return f"Shared: {', '.join(shared_terms[:5])}"
        
        return "Review component scores"  # UNCHANGED fallback

    def _get_score_breakdown(self, row, explanations_data: Dict) -> str:
        """UNCHANGED SIGNATURE - enhanced implementation for column flexibility"""
        
        # ENHANCEMENT: Flexible score column detection
        breakdown = []
        score_mappings = [
            (['Semantic_Score', 'Dense Semantic'], 'Sem'),
            (['BM25_Score', 'BM25 Score'], 'BM25'), 
            (['Domain_Score', 'Domain Weighted'], 'Dom'),
            (['Query_Expansion_Score', 'Query Expansion'], 'QE'),
            (['Syntactic_Score', 'Syntactic Score'], 'Syn')
        ]
        
        # UNCHANGED: Same breakdown logic, but flexible column detection
        for possible_cols, label in score_mappings:
            for col in possible_cols:
                if col in row and pd.notna(row[col]):
                    breakdown.append(f"{label}:{row[col]:.2f}")
                    break
        
        return " | ".join(breakdown) if breakdown else "Scores unavailable"  # UNCHANGED
    
    def _format_workbook(self, writer):
        """UNCHANGED SIGNATURE - enhanced error handling only"""
        
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            
            # UNCHANGED: Same formatting logic, enhanced error handling
            workbook = writer.book
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            
            for sheet_name in workbook.sheetnames:
                try:  # ENHANCEMENT: Per-sheet error handling
                    worksheet = workbook[sheet_name]
                    
                    # Format headers (unchanged)
                    if worksheet.max_row > 0:
                        for cell in worksheet[1]:
                            cell.fill = header_fill
                            cell.font = header_font
                            cell.alignment = Alignment(horizontal="center")
                    
                    # Auto-adjust column widths (unchanged)
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                except Exception as e:  # ENHANCEMENT: Per-sheet error handling
                    logger.warning(f"Could not format sheet {sheet_name}: {e}")
                        
        except ImportError:
            logger.warning("openpyxl styling not available")  # UNCHANGED
        except Exception as e:  # ENHANCEMENT: Overall error handling
            logger.warning(f"Workbook formatting failed: {e}")

def create_matching_workbook(enhanced_df: pd.DataFrame, 
                           evaluation_results: Optional[Dict] = None,
                           output_path: str = "outputs/engineering_review/matching_workbook.xlsx",
                           repo_manager=None) -> str:  # Add repo_manager parameter
    """
    Convenience function to create matching workbook.
    
    Args:
        enhanced_df: Enhanced predictions DataFrame
        evaluation_results: Optional evaluation results with discovery analysis
        output_path: Where to save the workbook
        repo_manager: Repository manager instance
        
    Returns:
        Path to created workbook
    """
    generator = MatchingWorkbookGenerator(repo_manager)
    return generator.create_workbook(enhanced_df, evaluation_results, output_path, repo_manager)