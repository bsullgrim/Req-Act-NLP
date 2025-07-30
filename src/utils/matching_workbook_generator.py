import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.file_utils import SafeFileHandler
from src.utils.path_resolver import SmartPathResolver

logger = logging.getLogger(__name__)


class MatchingWorkbookGenerator:
    def __init__(self, repo_manager=None):
        if repo_manager is None:
            raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager
        self.file_handler = SafeFileHandler(repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager)
        self.workbook_path = None

    def create_workbook(self, enhanced_df: pd.DataFrame,
                        evaluation_results: Optional[Dict] = None,
                        output_path: Optional[str] = None,
                        repo_manager=None) -> str:

        if repo_manager is None:
            from src.utils.repository_setup import RepositoryStructureManager
            repo_manager = RepositoryStructureManager("outputs")
            repo_manager.setup_repository_structure()

        output_path = Path(output_path) if output_path else repo_manager.structure['engineering_review'] / "matching_workbook.xlsx"

        if enhanced_df is None or enhanced_df.empty:
            logger.error("❌ No data provided to workbook generator")
            raise ValueError("Enhanced DataFrame is empty or None")

        logger.info(f"📊 Creating workbook with {len(enhanced_df)} matches")

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                try:
                    summary = self._create_executive_summary(enhanced_df, evaluation_results)
                    summary.to_excel(writer, sheet_name='Executive Summary', index=False)
                except Exception as e:
                    logger.warning(f"Executive summary failed: {e}")
                    pd.DataFrame([{"Metric": "Total Matches", "Value": len(enhanced_df)}]).to_excel(writer, sheet_name='Executive Summary', index=False)

                try:
                    self._create_confidence_tabs(enhanced_df, writer)
                except Exception as e:
                    logger.error(f"Confidence tabs failed: {e}")
                    enhanced_df.to_excel(writer, sheet_name='All Matches', index=False)

                try:
                    self._format_workbook(writer)
                except Exception as e:
                    logger.warning(f"Formatting failed: {e}")

        except Exception as e:
            fallback_path = output_path.with_suffix('.csv')
            enhanced_df.to_csv(fallback_path, index=False)
            return str(fallback_path)

        self.workbook_path = str(output_path)
        logger.info(f"✅ Matching workbook created: {output_path}")
        return str(output_path)

    def _create_executive_summary(self, enhanced_df: pd.DataFrame, evaluation_results: Optional[Dict]) -> pd.DataFrame:
        total_matches = len(enhanced_df)
        unique_reqs = enhanced_df['Requirement_ID'].nunique()

        high = len(enhanced_df[enhanced_df['Combined_Score'] >= 0.8])
        med = len(enhanced_df[(enhanced_df['Combined_Score'] >= 0.5) & (enhanced_df['Combined_Score'] < 0.8)])
        low = len(enhanced_df[enhanced_df['Combined_Score'] < 0.5])

        data = []

        # Methodology: Matching Process Overview
        data.append({
            "Category": "Methodology",
            "Metric": "Matching Process",
            "Value": "Multi-method scoring (Semantic, BM25, Domain, Query Expansion)",
            "Percentage": "",
            "Notes": (
                "Each requirement is matched to activities using 4 methods. "
                "Each method generates a score, and these are combined using a weighted average "
                "to produce a final `Combined_Score`."
            )
        })

        # Methodology: Score Definitions
        data.extend([
            {
                "Category": "Scoring Definitions",
                "Metric": "Semantic Score",
                "Value": "0.0–1.0",
                "Percentage": "",
                "Notes": (
                    "Measures deep semantic similarity between requirement and activity using sentence embeddings. "
                    "Values closer to 1.0 indicate stronger contextual similarity."
                )
            },
            {
                "Category": "Scoring Definitions",
                "Metric": "BM25 Score",
                "Value": "0.0–1.0 (normalized)",
                "Percentage": "",
                "Notes": (
                    "Term overlap score using classic BM25. Highlights exact token matches. "
                    "Normalized to 0–1 scale across all matches."
                )
            },
            {
                "Category": "Scoring Definitions",
                "Metric": "Domain Score",
                "Value": "0.0–1.0",
                "Percentage": "",
                "Notes": (
                    "Measures presence of domain-specific keywords from aerospace glossary. "
                    "Higher scores reflect better terminology alignment."
                )
            },
            {
                "Category": "Scoring Definitions",
                "Metric": "Query Expansion Score",
                "Value": "0.0–1.0",
                "Percentage": "",
                "Notes": (
                    "Expands requirement text using synonyms and evaluates overlap. "
                    "Captures conceptual similarity when original terms differ."
                )
            },
            {
                "Category": "Scoring Definitions",
                "Metric": "Combined Score",
                "Value": "0.0–1.0",
                "Percentage": "",
                "Notes": (
                    "Weighted average of the four scores. Default weights: "
                    "semantic=0.4, bm25=0.2, domain=0.2, query_expansion=0.2. "
                    "Used to determine confidence thresholds."
                )
            }
        ])

        # Confidence thresholds
        data.extend([
            {
                "Category": "Confidence",
                "Metric": "High Confidence (≥0.8)",
                "Value": high,
                "Percentage": f"{high / total_matches * 100:.1f}%" if total_matches else "",
                "Notes": "Likely accurate — high score across multiple signals"
            },
            {
                "Category": "Confidence",
                "Metric": "Medium Confidence (0.5–0.8)",
                "Value": med,
                "Percentage": f"{med / total_matches * 100:.1f}%" if total_matches else "",
                "Notes": "Some alignment — review recommended"
            },
            {
                "Category": "Confidence",
                "Metric": "Low Confidence (<0.5)",
                "Value": low,
                "Percentage": f"{low / total_matches * 100:.1f}%" if total_matches else "",
                "Notes": "Likely incorrect — weak or conflicting signals"
            }
        ])

        # General overview stats
        data.insert(2, {
            "Category": "Overview",
            "Metric": "Total Matches",
            "Value": total_matches,
            "Percentage": "100.0%",
            "Notes": "All generated requirement-to-activity mappings"
        })
        data.insert(3, {
            "Category": "Overview",
            "Metric": "Requirements Covered",
            "Value": unique_reqs,
            "Percentage": f"{unique_reqs / total_matches * 100:.1f}%" if total_matches else "",
            "Notes": "Unique requirements matched to ≥1 activity"
        })

        # Optional: quality grading distribution
        if 'Quality_Grade' in enhanced_df.columns:
            for grade, count in enhanced_df['Quality_Grade'].value_counts().items():
                data.append({
                    "Category": "Quality",
                    "Metric": f"Requirements: {grade}",
                    "Value": count,
                    "Percentage": f"{count / total_matches * 100:.1f}%" if total_matches else "",
                    "Notes": "Distribution by manual review grade (if provided)"
                })

        # Explanation coverage
        explanations = self._load_explanations()
        if explanations:
            data.append({
                "Category": "Explainability",
                "Metric": "Matches with Explanations",
                "Value": len(explanations),
                "Percentage": f"{len(explanations) / total_matches * 100:.1f}%" if total_matches else "",
                "Notes": "Justifications available for match reasoning"
            })

        return pd.DataFrame(data)
    def _create_confidence_tabs(self, enhanced_df: pd.DataFrame, writer):
        score_col = next((col for col in ['Combined_Score', 'combined_score'] if col in enhanced_df.columns), None)
        explanations = self._load_explanations()

        levels = [
            ("High Confidence (≥0.8)", enhanced_df[enhanced_df[score_col] >= 0.8]),
            ("Medium Confidence (0.5-0.8)", enhanced_df[(enhanced_df[score_col] >= 0.5) & (enhanced_df[score_col] < 0.8)]),
            ("Low Confidence (<0.5)", enhanced_df[enhanced_df[score_col] < 0.5])
        ]

        for name, df in levels:
            if df.empty:
                continue
            df = df.copy().sort_values(score_col, ascending=False)
            df['Match Explanation'] = df.apply(lambda r: self._get_explanation_text(r, explanations), axis=1)
            df['Key Evidence'] = df.apply(lambda r: self._get_key_evidence(r, explanations), axis=1)
            df['Score Breakdown'] = df.apply(lambda r: self._get_score_breakdown(r, explanations), axis=1)
            df.to_excel(writer, sheet_name=name, index=False)

    def _get_explanation_text(self, row, explanations: Dict) -> str:
        key = self._make_key(row)
        exp = explanations.get(key)
        if exp:
            parts = [f"{exp.get('semantic_similarity_level', 'Unknown')} semantic match"]
            if exp.get('scores', {}).get('bm25', 0) > 0.5:
                parts.append("strong term overlap")
            if exp.get('scores', {}).get('domain', 0) > 0.3:
                parts.append("aerospace terms matched")
            return "; ".join(parts)
        return self._fallback_explanation(row.get('Combined_Score', 0))

    def _get_key_evidence(self, row, explanations: Dict) -> str:
        key = self._make_key(row)
        terms = explanations.get(key, {}).get('shared_terms')
        return f"Shared: {', '.join(terms[:5])}" if terms else "Review component scores"

    def _get_score_breakdown(self, row, explanations: Dict) -> str:
        key = self._make_key(row)
        scores = explanations.get(key, {}).get('scores', {})
        return " | ".join(f"{k}:{v:.2f}" for k, v in scores.items()) if scores else "Scores unavailable"

    def _get_field_from_explanations(self, row, explanations: Dict, field: str) -> str:
        key = self._make_key(row)
        match = explanations.get(key, {})
        explanation_section = match.get('explanations', {})

        field_map = {
            'semantic_explanation': 'semantic',
            'bm25_explanation': 'bm25',
            'domain_explanation': 'domain',
            'query_expansion_explanation': 'query_expansion'
        }

        actual_field = field_map.get(field)
        if actual_field and actual_field in explanation_section:
            return explanation_section[actual_field]

        return ''  # Fallback: empty string if not found

    def _make_key(self, row):
        for id_col in ['ID', 'Requirement_ID', 'requirement_id']:
            for act_col in ['Activity Name', 'Activity_Name', 'activity_name']:
                if id_col in row and act_col in row:
                    return (row[id_col], row[act_col])
        return (None, None)

    def _fallback_explanation(self, score: float) -> str:
        if score >= 0.8:
            return "Very strong match across multiple dimensions"
        elif score >= 0.6:
            return "Good match with solid evidence"
        elif score >= 0.4:
            return "Moderate match, review recommended"
        else:
            return "Weak match, manual verification needed"

    def _load_explanations(self) -> Dict:
        paths = [
            self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json",
            Path("outputs/matching_results/aerospace_matches_explanations.json"),
            Path("aerospace_matches_explanations.json")
        ]
        for path in paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return {(e['requirement_id'], e['activity_name']): e for e in json.load(f)}
        return {}

    def _format_workbook(self, writer):
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            wb = writer.book
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
                for col in ws.columns:
                    max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)
        except Exception as e:
            logger.warning(f"Formatting failed: {e}")
