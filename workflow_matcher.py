"""
Enhanced Workflow Matcher with Final Clean Matcher Integration
Extends the FinalCleanMatcher to support internal engineering workflow 
for dependency evaluation and team handoff with full explainability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import re

# Import the enhanced matcher
from final_clean_matcher import FinalCleanMatcher, MatchExplanation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowMatchResult:
    """Enhanced match result for workflow processing with explainability."""
    requirement_id: str
    requirement_name: str
    requirement_text: str
    activity_name: str
    combined_score: float
    semantic_score: float
    bm25_score: float
    syntactic_score: float
    domain_score: float
    query_expansion_score: float
    confidence_level: str
    match_type: str
    review_priority: str
    explanation: MatchExplanation
    engineering_notes: str = ""
    review_status: str = "PENDING"

class EnhancedWorkflowMatcher(FinalCleanMatcher):
    """Enhanced workflow matcher with explainability for engineering teams."""
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        super().__init__(model_name)
        self.confidence_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
        self.match_types = {
            'exact': 0.9,
            'strong': 0.7,
            'moderate': 0.4,
            'weak': 0.2
        }
    
    def categorize_match(self, scores: Dict[str, float]) -> Tuple[str, str, str]:
        """Categorize match quality and determine review priority."""
        combined = scores.get('combined', 0)
        semantic = scores.get('dense_semantic', 0)
        
        # Determine confidence level
        if combined >= self.confidence_thresholds['high']:
            confidence = 'HIGH'
        elif combined >= self.confidence_thresholds['medium']:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Determine match type
        if combined >= self.match_types['exact']:
            match_type = 'EXACT'
        elif combined >= self.match_types['strong']:
            match_type = 'STRONG'
        elif combined >= self.match_types['moderate']:
            match_type = 'MODERATE'
        else:
            match_type = 'WEAK'
        
        # Determine review priority based on confidence and semantic strength
        if confidence == 'HIGH' and semantic > 0.6:
            priority = 'AUTO-APPROVE'
        elif confidence == 'HIGH' or (confidence == 'MEDIUM' and semantic > 0.5):
            priority = 'QUICK-REVIEW'
        elif confidence == 'MEDIUM':
            priority = 'DETAILED-REVIEW'
        else:
            priority = 'MANUAL-ANALYSIS'
        
        return confidence, match_type, priority
    
    def generate_workflow_matches(self, matches_df: pd.DataFrame, 
                                explanations: List[MatchExplanation]) -> List[WorkflowMatchResult]:
        """Convert DataFrame and explanations to workflow match results."""
        workflow_matches = []
        
        # Create lookup for explanations by requirement+activity
        explanation_lookup = {}
        for exp in explanations:
            key = f"{exp.requirement_id}||{exp.activity_name}"
            explanation_lookup[key] = exp
        
        for _, row in matches_df.iterrows():
            # Get explanation for this match
            lookup_key = f"{row['ID']}||{row['Activity Name']}"
            explanation = explanation_lookup.get(lookup_key)
            
            if not explanation:
                # Create a basic explanation if not found
                explanation = MatchExplanation(
                    requirement_id=str(row['ID']),
                    requirement_text=str(row['Requirement Text']),
                    activity_name=str(row['Activity Name']),
                    combined_score=row['Combined Score'],
                    semantic_score=row.get('Dense Semantic', 0),
                    bm25_score=row.get('BM25 Score', 0),
                    syntactic_score=row.get('Syntactic Score', 0),
                    domain_score=row.get('Domain Weighted', 0),
                    query_expansion_score=row.get('Query Expansion', 0),
                    semantic_explanation="Generated from match data",
                    bm25_explanation="Generated from match data",
                    syntactic_explanation="Generated from match data",
                    domain_explanation="Generated from match data",
                    query_expansion_explanation="Generated from match data",
                    shared_terms=[],
                    semantic_similarity_level="Unknown",
                    match_quality="Unknown"
                )
            
            scores = {
                'combined': row['Combined Score'],
                'dense_semantic': row.get('Dense Semantic', 0),
                'bm25': row.get('BM25 Score', 0),
                'syntactic': row.get('Syntactic Score', 0),
                'domain_weighted': row.get('Domain Weighted', 0),
                'query_expansion': row.get('Query Expansion', 0)
            }
            
            confidence, match_type, priority = self.categorize_match(scores)
            
            workflow_match = WorkflowMatchResult(
                requirement_id=str(row['ID']),
                requirement_name=str(row.get('Requirement Name', '')),
                requirement_text=str(row['Requirement Text']),
                activity_name=str(row['Activity Name']),
                combined_score=row['Combined Score'],
                semantic_score=scores['dense_semantic'],
                bm25_score=scores['bm25'],
                syntactic_score=scores['syntactic'],
                domain_score=scores['domain_weighted'],
                query_expansion_score=scores['query_expansion'],
                confidence_level=confidence,
                match_type=match_type,
                review_priority=priority,
                explanation=explanation
            )
            workflow_matches.append(workflow_match)
        
        return workflow_matches

    def create_engineering_review_package(self, matches_df: pd.DataFrame, 
                                        explanations: List[MatchExplanation],
                                        output_dir: str = "engineering_review") -> Dict[str, str]:
        """Create complete package for internal engineering review with explanations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        workflow_matches = self.generate_workflow_matches(matches_df, explanations)
        files_created = {}
        
        # 1. Enhanced Excel workbook with explanations
        excel_file = output_path / "dependency_review_workbook_explained.xlsx"
        self._create_enhanced_engineering_excel(workflow_matches, excel_file)
        files_created['review_workbook'] = str(excel_file)
        
        # 2. Structured JSON with full explanations
        json_file = output_path / "matches_with_explanations.json"
        self._create_structured_json_with_explanations(workflow_matches, json_file)
        files_created['structured_data'] = str(json_file)
        
        # 3. Enhanced action items with explanation summaries
        action_file = output_path / "action_items_explained.csv"
        self._create_enhanced_action_items(workflow_matches, action_file)
        files_created['action_items'] = str(action_file)
        
        # 4. Explanation summary report
        explanation_report = output_path / "explanation_summary.html"
        self._create_explanation_summary_html(workflow_matches, explanation_report)
        files_created['explanation_report'] = str(explanation_report)
        
        # 5. Summary dashboard with explanation metrics
        summary_file = output_path / "match_summary_enhanced.json"
        self._create_enhanced_summary_dashboard(workflow_matches, summary_file)
        files_created['summary_dashboard'] = str(summary_file)
        
        # 6. Updated workflow guide
        guide_file = output_path / "workflow_guide_explained.md"
        self._create_enhanced_workflow_guide(guide_file)
        files_created['workflow_guide'] = str(guide_file)
        
        logger.info(f"Created enhanced engineering review package in {output_dir}")
        return files_created
    
    def _create_enhanced_engineering_excel(self, matches: List[WorkflowMatchResult], filepath: Path):
        """Create comprehensive Excel workbook with explanations for engineering review."""
        wb = openpyxl.Workbook()
        
        # Remove default sheet
        wb.remove(wb.active)
        
        # Define styles
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        high_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        medium_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
        low_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Sheet 1: Auto-Approve Candidates
        auto_approve = [m for m in matches if m.review_priority == 'AUTO-APPROVE']
        if auto_approve:
            ws1 = wb.create_sheet("Auto-Approve")
            self._populate_enhanced_excel_sheet(ws1, auto_approve, header_fill, header_font, high_fill)
        
        # Sheet 2: Quick Review Required
        quick_review = [m for m in matches if m.review_priority == 'QUICK-REVIEW']
        if quick_review:
            ws2 = wb.create_sheet("Quick Review")
            self._populate_enhanced_excel_sheet(ws2, quick_review, header_fill, header_font, medium_fill)
        
        # Sheet 3: Detailed Review Required
        detailed_review = [m for m in matches if m.review_priority == 'DETAILED-REVIEW']
        if detailed_review:
            ws3 = wb.create_sheet("Detailed Review")
            self._populate_enhanced_excel_sheet(ws3, detailed_review, header_fill, header_font, medium_fill)
        
        # Sheet 4: Manual Analysis Needed
        manual_analysis = [m for m in matches if m.review_priority == 'MANUAL-ANALYSIS']
        if manual_analysis:
            ws4 = wb.create_sheet("Manual Analysis")
            self._populate_enhanced_excel_sheet(ws4, manual_analysis, header_fill, header_font, low_fill)
        
        # Sheet 5: Enhanced Summary with Explanations
        ws_summary = wb.create_sheet("Summary")
        self._create_enhanced_summary_sheet(ws_summary, matches, header_fill, header_font)
        
        # Sheet 6: Explanation Guide
        ws_guide = wb.create_sheet("Explanation Guide")
        self._create_explanation_guide_sheet(ws_guide, header_fill, header_font)
        
        wb.save(filepath)
    
    def _populate_enhanced_excel_sheet(self, ws, matches: List[WorkflowMatchResult], 
                                     header_fill, header_font, row_fill):
        """Populate Excel sheet with enhanced match data and explanations."""
        headers = [
            "Req ID", "Requirement Name", "Activity Name", "Combined Score",
            "Confidence", "Match Type", "Priority", "Semantic Score", "BM25 Score",
            "Syntactic Score", "Domain Score", "Query Expansion", "Match Quality",
            "Semantic Explanation", "Key Evidence", "Shared Terms", 
            "Engineer Review", "Status", "Notes"
        ]
        
        # Add headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        
        # Add data
        for row, match in enumerate(matches, 2):
            ws.cell(row=row, column=1, value=match.requirement_id)
            ws.cell(row=row, column=2, value=self._truncate_text(match.requirement_name, 40))
            ws.cell(row=row, column=3, value=self._truncate_text(match.activity_name, 40))
            ws.cell(row=row, column=4, value=round(match.combined_score, 3))
            ws.cell(row=row, column=5, value=match.confidence_level)
            ws.cell(row=row, column=6, value=match.match_type)
            ws.cell(row=row, column=7, value=match.review_priority)
            ws.cell(row=row, column=8, value=round(match.semantic_score, 3))
            ws.cell(row=row, column=9, value=round(match.bm25_score, 3))
            ws.cell(row=row, column=10, value=round(match.syntactic_score, 3))
            ws.cell(row=row, column=11, value=round(match.domain_score, 3))
            ws.cell(row=row, column=12, value=round(match.query_expansion_score, 3))
            ws.cell(row=row, column=13, value=match.explanation.match_quality)
            ws.cell(row=row, column=14, value=self._truncate_text(match.explanation.semantic_explanation, 60))
            ws.cell(row=row, column=15, value=self._create_key_evidence_summary(match.explanation))
            ws.cell(row=row, column=16, value=", ".join(match.explanation.shared_terms[:3]))
            ws.cell(row=row, column=17, value="")  # Engineer Review (to be filled)
            ws.cell(row=row, column=18, value="PENDING")  # Status
            ws.cell(row=row, column=19, value="")  # Notes
            
            # Apply row fill
            for col in range(1, len(headers) + 1):
                ws.cell(row=row, column=col).fill = row_fill
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 60)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_key_evidence_summary(self, explanation: MatchExplanation) -> str:
        """Create a concise summary of key evidence for Excel display."""
        evidence = []
        
        if explanation.shared_terms:
            evidence.append(f"Shared: {', '.join(explanation.shared_terms[:2])}")
        
        if "High" in explanation.semantic_explanation:
            evidence.append("High semantic similarity")
        
        if "domain terms" in explanation.domain_explanation and explanation.domain_score > 0.1:
            evidence.append("Technical term match")
            
        if len(evidence) == 0:
            evidence.append("See detailed explanations")
            
        return "; ".join(evidence)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for Excel display."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _create_enhanced_summary_sheet(self, ws, matches: List[WorkflowMatchResult], 
                                     header_fill, header_font):
        """Create enhanced summary statistics sheet with explanation metrics."""
        ws.title = "Summary"
        
        # Summary statistics
        total_matches = len(matches)
        by_confidence = {}
        by_priority = {}
        by_match_quality = {}
        
        for match in matches:
            by_confidence[match.confidence_level] = by_confidence.get(match.confidence_level, 0) + 1
            by_priority[match.review_priority] = by_priority.get(match.review_priority, 0) + 1
            by_match_quality[match.explanation.match_quality] = by_match_quality.get(match.explanation.match_quality, 0) + 1
        
        # Write summary data
        row = 1
        ws.cell(row=row, column=1, value="ENHANCED MATCH ANALYSIS SUMMARY").font = Font(size=16, bold=True)
        row += 2
        
        ws.cell(row=row, column=1, value="Total Matches:").font = Font(bold=True)
        ws.cell(row=row, column=2, value=total_matches)
        row += 2
        
        ws.cell(row=row, column=1, value="By Confidence Level:").font = Font(bold=True)
        row += 1
        for conf, count in by_confidence.items():
            ws.cell(row=row, column=2, value=f"{conf}: {count} ({count/total_matches*100:.1f}%)")
            row += 1
        
        row += 1
        ws.cell(row=row, column=1, value="By Review Priority:").font = Font(bold=True)
        row += 1
        for priority, count in by_priority.items():
            estimated_hours = count * self._estimate_review_hours(priority)
            ws.cell(row=row, column=2, value=f"{priority}: {count} ({estimated_hours:.1f} hours)")
            row += 1
        
        row += 1
        ws.cell(row=row, column=1, value="By Match Quality:").font = Font(bold=True)
        row += 1
        for quality, count in by_match_quality.items():
            ws.cell(row=row, column=2, value=f"{quality}: {count} ({count/total_matches*100:.1f}%)")
            row += 1
    
    def _create_explanation_guide_sheet(self, ws, header_fill, header_font):
        """Create explanation guide sheet."""
        ws.title = "Explanation Guide"
        
        guide_content = [
            ["SCORING COMPONENT EXPLANATIONS", ""],
            ["", ""],
            ["Semantic Score", "Neural network-based meaning similarity (0-1)"],
            ["", "Very High (>0.7): Strong conceptual match"],
            ["", "High (0.5-0.7): Good conceptual alignment"],
            ["", "Medium (0.3-0.5): Moderate conceptual similarity"],
            ["", "Low (<0.3): Weak conceptual connection"],
            ["", ""],
            ["BM25 Score", "Statistical relevance ranking based on term frequency"],
            ["", "Shows exact term matches and their importance"],
            ["", "Higher scores indicate more shared technical terms"],
            ["", ""],
            ["Syntactic Score", "Structural similarity (grammar, dependencies)"],
            ["", "Compares sentence structure and linguistic patterns"],
            ["", "Useful for identifying functionally similar activities"],
            ["", ""],
            ["Domain Score", "Technical term overlap weighted by importance"],
            ["", "Emphasizes domain-specific terminology"],
            ["", "Higher scores indicate shared technical vocabulary"],
            ["", ""],
            ["Query Expansion", "Synonym and related term matching"],
            ["", "Uses domain dictionary to find vocabulary variants"],
            ["", "Helps overcome different terminology usage"],
            ["", ""],
            ["CONFIDENCE LEVELS", ""],
            ["", ""],
            ["HIGH (>0.7)", "Strong match, likely correct dependency"],
            ["MEDIUM (0.4-0.7)", "Moderate match, requires review"],
            ["LOW (<0.4)", "Weak match, may be false positive"],
            ["", ""],
            ["REVIEW PRIORITIES", ""],
            ["", ""],
            ["AUTO-APPROVE", "High confidence + strong semantic match"],
            ["QUICK-REVIEW", "High confidence or good semantic match"],
            ["DETAILED-REVIEW", "Medium confidence, needs thorough analysis"],
            ["MANUAL-ANALYSIS", "Low confidence, may be false positive"]
        ]
        
        for row_idx, (col1, col2) in enumerate(guide_content, 1):
            if col1.isupper() and col1.endswith("S"):  # Headers
                cell = ws.cell(row=row_idx, column=1, value=col1)
                cell.font = Font(bold=True, size=12)
            else:
                ws.cell(row=row_idx, column=1, value=col1)
                ws.cell(row=row_idx, column=2, value=col2)
        
        # Auto-adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 60
    
    def _create_structured_json_with_explanations(self, matches: List[WorkflowMatchResult], filepath: Path):
        """Create structured JSON with full explanations."""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_matches": len(matches),
                "tool_version": "2.0_with_explanations",
                "confidence_thresholds": self.confidence_thresholds
            },
            "matches": []
        }
        
        for match in matches:
            match_data = {
                "requirement": {
                    "id": match.requirement_id,
                    "name": match.requirement_name,
                    "text": match.requirement_text
                },
                "activity": {
                    "name": match.activity_name
                },
                "scores": {
                    "combined": round(match.combined_score, 3),
                    "semantic": round(match.semantic_score, 3),
                    "bm25": round(match.bm25_score, 3),
                    "syntactic": round(match.syntactic_score, 3),
                    "domain": round(match.domain_score, 3),
                    "query_expansion": round(match.query_expansion_score, 3)
                },
                "classification": {
                    "confidence_level": match.confidence_level,
                    "match_type": match.match_type,
                    "review_priority": match.review_priority,
                    "match_quality": match.explanation.match_quality
                },
                "explanations": {
                    "semantic": match.explanation.semantic_explanation,
                    "bm25": match.explanation.bm25_explanation,
                    "syntactic": match.explanation.syntactic_explanation,
                    "domain": match.explanation.domain_explanation,
                    "query_expansion": match.explanation.query_expansion_explanation
                },
                "evidence": {
                    "shared_terms": match.explanation.shared_terms,
                    "semantic_similarity_level": match.explanation.semantic_similarity_level
                },
                "workflow": {
                    "engineering_notes": match.engineering_notes,
                    "review_status": match.review_status,
                    "estimated_hours": self._estimate_review_hours(match.review_priority)
                }
            }
            data["matches"].append(match_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _create_enhanced_action_items(self, matches: List[WorkflowMatchResult], filepath: Path):
        """Create enhanced action items CSV with explanation summaries."""
        action_data = []
        
        for match in matches:
            # Create concise explanation summary for action tracking
            explanation_summary = f"Semantic: {match.explanation.semantic_similarity_level}"
            if match.explanation.shared_terms:
                explanation_summary += f" | Shared: {', '.join(match.explanation.shared_terms[:2])}"
            if match.explanation.match_quality != "Unknown":
                explanation_summary += f" | Quality: {match.explanation.match_quality}"
            
            action_data.append({
                "Requirement_ID": match.requirement_id,
                "Activity_Name": match.activity_name,
                "Priority": match.review_priority,
                "Confidence": match.confidence_level,
                "Combined_Score": round(match.combined_score, 3),
                "Match_Quality": match.explanation.match_quality,
                "Explanation_Summary": explanation_summary,
                "Shared_Terms": ", ".join(match.explanation.shared_terms[:3]),
                "Assigned_Engineer": "",  # To be filled
                "Due_Date": "",  # To be filled
                "Status": match.review_status,
                "Estimated_Hours": self._estimate_review_hours(match.review_priority),
                "Engineering_Notes": match.engineering_notes
            })
        
        df = pd.DataFrame(action_data)
        df.to_csv(filepath, index=False)
    
    def _create_explanation_summary_html(self, matches: List[WorkflowMatchResult], filepath: Path):
        """Create HTML summary report with explanations."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dependency Analysis Explanation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .match {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .match-header {{ font-weight: bold; color: #2c3e50; }}
        .score-breakdown {{ margin: 10px 0; }}
        .explanation {{ background-color: #f1f3f4; padding: 10px; border-left: 4px solid #4CAF50; margin: 5px 0; }}
        .high-confidence {{ border-left-color: #4CAF50; }}
        .medium-confidence {{ border-left-color: #FF9800; }}
        .low-confidence {{ border-left-color: #f44336; }}
        .evidence {{ font-style: italic; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dependency Analysis Explanation Summary</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary-stats">
        <h2>Summary Statistics</h2>
        <p><strong>Total Matches:</strong> {len(matches)}</p>
        <p><strong>Requirements Covered:</strong> {len(set(m.requirement_id for m in matches))}</p>
        <p><strong>Average Combined Score:</strong> {np.mean([m.combined_score for m in matches]):.3f}</p>
    </div>
    
    <h2>Top Matches with Explanations</h2>
"""
        
        # Show top 10 matches
        top_matches = sorted(matches, key=lambda x: x.combined_score, reverse=True)[:10]
        
        for i, match in enumerate(top_matches, 1):
            confidence_class = match.confidence_level.lower() + "-confidence"
            
            html_content += f"""
    <div class="match {confidence_class}">
        <div class="match-header">
            Match {i}: {match.requirement_id} → {match.activity_name}
        </div>
        <div class="score-breakdown">
            <strong>Combined Score:</strong> {match.combined_score:.3f} 
            | <strong>Confidence:</strong> {match.confidence_level}
            | <strong>Quality:</strong> {match.explanation.match_quality}
        </div>
        
        <div class="explanation">
            <strong>Semantic:</strong> {match.explanation.semantic_explanation}
        </div>
        
        <div class="explanation">
            <strong>Term Matching:</strong> {match.explanation.bm25_explanation}
        </div>
        
        <div class="explanation">
            <strong>Domain Analysis:</strong> {match.explanation.domain_explanation}
        </div>
        
        {f'<div class="evidence">Shared Terms: {", ".join(match.explanation.shared_terms)}</div>' if match.explanation.shared_terms else ''}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_enhanced_summary_dashboard(self, matches: List[WorkflowMatchResult], filepath: Path):
        """Create enhanced summary data for dashboard visualization."""
        # Calculate explanation-based metrics
        explanation_metrics = {
            "semantic_distribution": {},
            "match_quality_distribution": {},
            "evidence_strength": {}
        }
        
        for match in matches:
            # Semantic similarity distribution
            level = match.explanation.semantic_similarity_level
            explanation_metrics["semantic_distribution"][level] = explanation_metrics["semantic_distribution"].get(level, 0) + 1
            
            # Match quality distribution
            quality = match.explanation.match_quality
            explanation_metrics["match_quality_distribution"][quality] = explanation_metrics["match_quality_distribution"].get(quality, 0) + 1
            
            # Evidence strength (based on shared terms)
            if len(match.explanation.shared_terms) >= 3:
                evidence_level = "Strong"
            elif len(match.explanation.shared_terms) >= 1:
                evidence_level = "Moderate"
            else:
                evidence_level = "Weak"
            explanation_metrics["evidence_strength"][evidence_level] = explanation_metrics["evidence_strength"].get(evidence_level, 0) + 1
        
        summary = {
            "overview": {
                "total_matches": len(matches),
                "high_confidence": len([m for m in matches if m.confidence_level == 'HIGH']),
                "medium_confidence": len([m for m in matches if m.confidence_level == 'MEDIUM']),
                "low_confidence": len([m for m in matches if m.confidence_level == 'LOW']),
                "avg_combined_score": np.mean([m.combined_score for m in matches]),
                "avg_semantic_score": np.mean([m.semantic_score for m in matches])
            },
            "workflow_distribution": {
                "auto_approve": len([m for m in matches if m.review_priority == 'AUTO-APPROVE']),
                "quick_review": len([m for m in matches if m.review_priority == 'QUICK-REVIEW']),
                "detailed_review": len([m for m in matches if m.review_priority == 'DETAILED-REVIEW']),
                "manual_analysis": len([m for m in matches if m.review_priority == 'MANUAL-ANALYSIS'])
            },
            "estimated_effort": {
                "total_hours": sum(self._estimate_review_hours(m.review_priority) for m in matches),
                "by_priority": {
                    priority: sum(self._estimate_review_hours(m.review_priority) 
                                for m in matches if m.review_priority == priority)
                    for priority in set(m.review_priority for m in matches)
                }
            },
            "explanation_metrics": explanation_metrics,
            "score_distributions": {
                "semantic_scores": [m.semantic_score for m in matches],
                "bm25_scores": [m.bm25_score for m in matches],
                "domain_scores": [m.domain_score for m in matches],
                "combined_scores": [m.combined_score for m in matches]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _create_enhanced_workflow_guide(self, filepath: Path):
        """Create enhanced workflow guide with explanation information."""
        guide_content = """# Enhanced Dependency Analysis Workflow Guide

## Overview
This package contains automated dependency analysis results with detailed explanations for each match decision.

## File Structure
- `dependency_review_workbook_explained.xlsx`: Main review workbook with explanations
- `matches_with_explanations.json`: Machine-readable data with full explanations
- `action_items_explained.csv`: Project management tracking with explanation summaries
- `explanation_summary.html`: Visual summary report with top matches explained
- `match_summary_enhanced.json`: Dashboard data with explanation metrics

## Enhanced Review Process

### 1. Auto-Approve Sheet
- **High confidence matches (>70%) with strong semantic similarity**
- **Key indicators**: "Very High" or "High" semantic similarity, multiple shared terms
- **Explanation focus**: Verify the semantic explanation makes sense
- **Estimated time**: 15 minutes per match

### 2. Quick Review Sheet
- **High confidence OR medium confidence with good semantic match**
- **Key indicators**: Moderate to high semantic similarity, some shared technical terms
- **Explanation focus**: Check BM25 and domain explanations for technical alignment
- **Estimated time**: 30 minutes per match

### 3. Detailed Review Sheet
- **Medium confidence matches requiring thorough analysis**
- **Key indicators**: Mixed signals across different scoring components
- **Explanation focus**: Analyze all explanation components, especially syntactic patterns
- **Estimated time**: 2 hours per match

### 4. Manual Analysis Sheet
- **Low confidence matches that may be false positives**
- **Key indicators**: Low semantic similarity, few shared terms, weak explanations
- **Explanation focus**: Determine if match has any validity despite low scores
- **Estimated time**: 4 hours per match

## Explanation Components Guide

### Semantic Explanation
- **Very High (>0.7)**: Strong conceptual match, likely correct dependency
- **High (0.5-0.7)**: Good conceptual alignment, review context
- **Medium (0.3-0.5)**: Moderate similarity, may need domain expert review
- **Low (<0.3)**: Weak conceptual connection, likely false positive

### BM25 Explanation
- Shows exact term matches and their statistical importance
- Look for meaningful technical terms, not just common words
- Higher scores with technical terms indicate stronger matches

### Domain Explanation
- Highlights shared technical terminology
- Focus on domain-specific terms with high weights
- Empty explanations may indicate vocabulary mismatch

### Syntactic Explanation
- Compares sentence structure and linguistic patterns
- Useful for identifying functionally similar activities
- Higher scores suggest similar action patterns

### Query Expansion Explanation
- Shows synonym and related term matching
- Helps identify matches despite different terminology
- Review expanded terms for domain appropriateness

## Using the Explanation Data

### In Excel Workbook
1. **Key Evidence column**: Quick summary of main match indicators
2. **Explanation columns**: Detailed breakdowns for each scoring component
3. **Shared Terms column**: Direct vocabulary overlap
4. **Match Quality**: Overall assessment (EXCELLENT/GOOD/MODERATE/WEAK)

### In JSON Data
- Full programmatic access to all explanations
- Suitable for integration with other tools
- Contains structured evidence and reasoning

### In HTML Report
- Visual overview of top matches
- Color-coded by confidence level
- Easy sharing with stakeholders

## Review Workflow

1. **Start with Excel workbook** - gives best overview and input capability
2. **Use explanation columns** to understand why each match was suggested
3. **Cross-reference with HTML report** for visual confirmation of top matches
4. **Update status and notes** directly in Excel
5. **Export decisions** using action items CSV for project tracking

## Quality Indicators

### Strong Match Indicators
- Multiple shared technical terms
- High semantic similarity with clear explanation
- Domain-specific term overlap
- Consistent scores across components

### Weak Match Indicators
- No shared meaningful terms
- Low semantic similarity
- Generic or common-word overlap only
- Inconsistent scores across components

## Decision Guidelines

### Approve if:
- Semantic explanation makes logical sense
- Multiple technical terms are shared
- Domain explanation shows relevant technical overlap
- Match quality is EXCELLENT or GOOD

### Review Further if:
- Mixed signals across explanation components
- Semantic similarity is moderate but term overlap is low
- Domain explanation is weak but other indicators are strong
- Match quality is MODERATE

### Reject if:
- No meaningful shared technical terms
- Semantic explanation shows conceptual mismatch
- All explanation components are weak
- Match quality is WEAK with no redeeming factors

## Integration with Engineering Process

1. **Requirements Traceability**: Use approved matches for formal traceability matrix
2. **Architecture Review**: Reference explanations in design review meetings  
3. **Test Planning**: Use matched activities for test case derivation
4. **Documentation**: Include match reasoning in requirement specifications
5. **Change Impact**: Re-run analysis when requirements change

## Troubleshooting

### If explanations seem unclear:
- Check the original requirement and activity texts for context
- Review the explanation guide sheet in Excel workbook
- Consult domain experts for technical term validation

### If scores seem inconsistent:
- Different components measure different aspects of similarity
- Low semantic with high BM25 may indicate technical term match without conceptual alignment
- High semantic with low BM25 may indicate conceptual match with different terminology

### If no good matches found:
- Requirement may be too abstract or implementation-specific
- Activity descriptions may need more detail
- Consider if requirement should be decomposed

## Support

For questions about the matching algorithm or explanations, refer to:
- Explanation Guide sheet in Excel workbook
- Technical documentation in algorithm source code
- Domain expert consultation for technical term validation
"""
        
        with open(filepath, 'w') as f:
            f.write(guide_content)
    
    def _estimate_review_hours(self, priority: str) -> float:
        """Estimate review hours based on priority."""
        hours_map = {
            'AUTO-APPROVE': 0.25,
            'QUICK-REVIEW': 0.5,
            'DETAILED-REVIEW': 2.0,
            'MANUAL-ANALYSIS': 4.0
        }
        return hours_map.get(priority, 2.0)
    
    def run_enhanced_workflow_matching(self, requirements_file: str = "requirements.csv",
                                     activities_file: str = "activities.csv",
                                     gold_pairs_file: Optional[str] = None,
                                     min_sim: float = 0.35,
                                     top_n: int = 5,
                                     normalize_scores: bool = True,
                                     out_file: str = "enhanced_workflow_matches") -> pd.DataFrame:
        """Run enhanced matching with workflow support and explanations."""
        
        print("Starting Enhanced Workflow Matching with Explanations...")
        print("="*60)
        
        # Step 1: Run the final clean matching to get matches and explanations
        print("1. Running enhanced matching with explainability...")
        matches_df = self.run_final_matching(
            requirements_file=requirements_file,
            activities_file=activities_file,
            gold_pairs_file=gold_pairs_file,
            min_sim=min_sim,
            top_n=top_n,
            out_file=out_file
        )
        
        if matches_df.empty:
            print("No matches found. Exiting.")
            return matches_df
        
        # Step 2: Load explanations from the JSON file created by run_final_matching
        explanations_file = f"results/{out_file}_explanations.json"
        explanations = []
        
        try:
            with open(explanations_file, 'r', encoding='utf-8') as f:
                explanations_data = json.load(f)
                
            for exp_data in explanations_data:
                explanation = MatchExplanation(
                    requirement_id=exp_data['requirement_id'],
                    requirement_text=exp_data['requirement_text'],
                    activity_name=exp_data['activity_name'],
                    combined_score=exp_data['combined_score'],
                    semantic_score=exp_data['scores']['semantic'],
                    bm25_score=exp_data['scores']['bm25'],
                    syntactic_score=exp_data['scores'].get('syntactic', 0),
                    domain_score=exp_data['scores'].get('domain', 0),
                    query_expansion_score=exp_data['scores'].get('query_expansion', 0),
                    semantic_explanation=exp_data['explanations']['semantic'],
                    bm25_explanation=exp_data['explanations']['bm25'],
                    syntactic_explanation=exp_data['explanations'].get('syntactic', ''),
                    domain_explanation=exp_data['explanations'].get('domain', ''),
                    query_expansion_explanation=exp_data['explanations'].get('query_expansion', ''),
                    shared_terms=exp_data['shared_terms'],
                    semantic_similarity_level=exp_data['semantic_similarity_level'],
                    match_quality=exp_data['match_quality']
                )
                explanations.append(explanation)
                
        except FileNotFoundError:
            print(f"Warning: Could not load explanations from {explanations_file}")
            explanations = []
        
        print(f"   Found {len(matches_df)} matches with {len(explanations)} explanations")
        
        # Step 3: Create enhanced engineering review package
        print("2. Creating enhanced engineering review package...")
        files_created = self.create_engineering_review_package(
            matches_df, 
            explanations,
            output_dir="enhanced_engineering_review"
        )
        
        # Step 4: Enhanced summary for team
        print("\n" + "="*70)
        print("ENHANCED ENGINEERING REVIEW PACKAGE READY")
        print("="*70)
        
        # Analyze the distribution with explanations
        workflow_matches = self.generate_workflow_matches(matches_df, explanations)
        
        priority_counts = {}
        confidence_counts = {}
        quality_counts = {}
        
        for match in workflow_matches:
            priority_counts[match.review_priority] = priority_counts.get(match.review_priority, 0) + 1
            confidence_counts[match.confidence_level] = confidence_counts.get(match.confidence_level, 0) + 1
            quality_counts[match.explanation.match_quality] = quality_counts.get(match.explanation.match_quality, 0) + 1
        
        print(f"\nMatch Distribution:")
        print(f"  Total Matches: {len(workflow_matches)}")
        print(f"  Requirements Covered: {len(set(m.requirement_id for m in workflow_matches))}")
        
        print(f"\nBy Review Priority:")
        for priority, count in sorted(priority_counts.items()):
            estimated_hours = count * self._estimate_review_hours(priority)
            print(f"  {priority:15}: {count:3d} matches ({estimated_hours:5.1f} hours)")
        
        print(f"\nBy Confidence Level:")
        for confidence, count in sorted(confidence_counts.items()):
            print(f"  {confidence:8}: {count:3d} matches ({count/len(workflow_matches)*100:5.1f}%)")
        
        print(f"\nBy Match Quality:")
        for quality, count in sorted(quality_counts.items()):
            print(f"  {quality:10}: {count:3d} matches ({count/len(workflow_matches)*100:5.1f}%)")
        
        print(f"\nFiles Created:")
        for purpose, filepath in files_created.items():
            print(f"  {purpose:20}: {filepath}")
        
        print(f"\nImmediate Actions for Team Lead:")
        print(f"1. Open '{files_created['review_workbook']}' in Excel for detailed review")
        print(f"2. Review '{files_created['explanation_report']}' for visual summary")
        print(f"3. Assign engineers using '{files_created['action_items']}'")
        print(f"4. Reference '{files_created['workflow_guide']}' for explanation guidance")
        
        # Estimate total effort
        total_hours = sum(self._estimate_review_hours(m.review_priority) for m in workflow_matches)
        print(f"\nEstimated Total Review Effort: {total_hours:.1f} hours")
        
        return matches_df


def main():
    """Run enhanced workflow matching with full explainability."""
    
    print("="*70)
    print("ENHANCED WORKFLOW MATCHER - With Full Explainability")
    print("="*70)
    
    # Create enhanced workflow matcher
    workflow_matcher = EnhancedWorkflowMatcher()
    
    # Run complete workflow with explanations
    results = workflow_matcher.run_enhanced_workflow_matching(
        requirements_file="requirements.csv",
        activities_file="activities.csv",
        gold_pairs_file="manual_matches.csv",
        min_sim=0.35,
        top_n=5,
        out_file="enhanced_workflow_matches"
    )
    
    print(f"\nEnhanced Workflow Matching completed!")
    print(f"Check 'enhanced_engineering_review' folder for all deliverables")


if __name__ == "__main__":
    main()