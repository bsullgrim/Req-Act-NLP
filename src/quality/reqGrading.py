"""
Enhanced Requirements Quality Analyzer with Excel Output
Enhanced version of reqGrading.py that can create comprehensive Excel reports
"""

import pandas as pd
import chardet
import logging
import spacy
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass
from collections import Counter
import json

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for requirement quality metrics"""
    clarity_score: float
    completeness_score: float
    verifiability_score: float
    atomicity_score: float
    consistency_score: float
    total_issues: int
    severity_breakdown: Dict[str, int]

class RequirementAnalyzer:
    """Enhanced requirements quality analyzer with CSV and Excel output capabilities"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Please install it with: python -m spacy download {spacy_model}")
            raise
        
        # Enhanced term definitions with severity levels
        self.ambiguous_terms = {
            "clarity": {
                "high": {"appropriate", "sufficient", "adequate", "efficient", "reasonable", "acceptable"},
                "medium": {"good", "bad", "proper", "suitable", "normal", "standard"},
                "low": {"nice", "clean", "simple"}
            },
            "ambiguity": {
                "high": {"as needed", "if necessary", "where applicable", "to the extent possible", "as appropriate"},
                "medium": {"typically", "generally", "usually", "often", "sometimes"},
                "low": {"etc", "and so on", "among others"}
            }
        }
        
        self.modal_verbs = {
            "mandatory": {"shall", "must", "will"},
            "recommended": {"should", "ought"},
            "optional": {"may", "can", "might", "could"}
        }
        
        # Requirement keywords that indicate good structure
        self.requirement_indicators = {
            "system", "user", "application", "software", "interface", "database",
            "function", "feature", "capability", "performance", "security"
        }
        
        # Passive voice indicators
        self.passive_indicators = [
            r'\b(is|are|was|were|being|been)\s+\w+ed\b',
            r'\b(is|are|was|were)\s+\w+en\b'
        ]
        
    def detect_encoding(self, file_path: str) -> str:
        """Enhanced encoding detection with fallback options"""
        try:
            with open(file_path, "rb") as f:
                raw = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw)
                confidence = result.get("confidence", 0)
                encoding = result.get("encoding", "utf-8")
                
                if confidence < 0.7:
                    logger.warning(f"Low confidence ({confidence:.2f}) in detected encoding: {encoding}")
                
                return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8"
    
    def safe_read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Enhanced CSV reader with better error handling and validation"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
        
        # Try detected encoding first
        detected_encoding = self.detect_encoding(str(file_path))
        if detected_encoding not in encodings_to_try:
            encodings_to_try.insert(0, detected_encoding)
        
        last_error = None
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Successfully read {file_path.name} with encoding: {encoding} ({len(df)} rows)")
                
                if df.empty:
                    logger.warning(f"File {file_path.name} is empty")
                
                return df
                
            except UnicodeDecodeError as e:
                last_error = e
                logger.debug(f"Failed to read with {encoding}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                raise
        
        raise RuntimeError(f"Failed to read {file_path} with any encoding. Last error: {last_error}")
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score based on sentence and word complexity"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len([token for token in doc if not token.is_punct]) / len(sentences)
        
        # Complex word ratio (words with 3+ syllables - simplified heuristic)
        complex_words = sum(1 for token in doc if len(token.text) > 6 and token.is_alpha)
        total_words = sum(1 for token in doc if token.is_alpha)
        complex_ratio = complex_words / max(total_words, 1)
        
        # Simple readability score (lower is better, normalized 0-100)
        score = max(0, 100 - (avg_sentence_length * 2 + complex_ratio * 50))
        return min(score, 100)
    
    def check_passive_voice(self, text: str) -> List[str]:
        """Detect passive voice constructions"""
        issues = []
        for pattern in self.passive_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(f"Passive voice detected: {', '.join(set(matches))}")
        return issues
    
    def analyze_requirement(self, text: str, req_id: Optional[str] = None) -> Tuple[List[str], QualityMetrics]:
        """Enhanced requirement analysis with detailed metrics"""
        if pd.isna(text) or not str(text).strip():
            return ["Empty requirement"], QualityMetrics(0, 0, 0, 0, 0, 1, {"critical": 1})
        
        text = str(text).strip()
        doc = self.nlp(text)
        issues = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # 1. Enhanced Clarity Analysis
        clarity_issues = 0
        for token in doc:
            if token.pos_ in {"ADJ", "ADV"}:
                token_lower = token.text.lower()
                for severity, terms in self.ambiguous_terms["clarity"].items():
                    if token_lower in terms:
                        issues.append(f"Clarity ({severity}): ambiguous term '{token.text}'")
                        severity_counts[severity] += 1
                        clarity_issues += 1
        
        # 2. Enhanced Atomicity Check
        atomicity_issues = 0
        conjunction_count = sum(1 for token in doc if token.dep_ == "conj")
        if conjunction_count > 2:
            issues.append(f"Atomicity (high): multiple conjunctions ({conjunction_count}) suggest compound requirements")
            severity_counts["high"] += 1
            atomicity_issues += 1
        elif conjunction_count > 0:
            issues.append(f"Atomicity (medium): contains {conjunction_count} conjunction(s)")
            severity_counts["medium"] += 1
            atomicity_issues += 1
        
        # 3. Enhanced Verifiability
        verifiability_issues = 0
        measurable_entities = [ent for ent in doc.ents if ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"}]
        numbers_pattern = r'\b\d+(\.\d+)?\s*(seconds?|minutes?|hours?|days?|%|percent|times?|instances?)\b'
        has_numbers = bool(re.search(numbers_pattern, text.lower()))
        
        if not measurable_entities and not has_numbers:
            issues.append("Verifiability (high): no measurable criteria found")
            severity_counts["high"] += 1
            verifiability_issues += 1
        
        # 4. Enhanced Completeness
        completeness_issues = 0
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_object = any(token.dep_ in {"dobj", "pobj", "attr"} for token in doc)
        
        missing_components = []
        if not has_subject:
            missing_components.append("subject")
        if not has_verb:
            missing_components.append("verb")
        if not has_object:
            missing_components.append("object")
        
        if missing_components:
            issues.append(f"Completeness (high): missing {', '.join(missing_components)}")
            severity_counts["high"] += 1
            completeness_issues += len(missing_components)
        
        # 5. Enhanced Modal Verb Analysis
        modal_found = None
        modal_strength = None
        for strength, verbs in self.modal_verbs.items():
            for token in doc:
                if token.text.lower() in verbs:
                    modal_found = token.text.lower()
                    modal_strength = strength
                    break
            if modal_found:
                break
        
        if not modal_found:
            issues.append("Completeness (medium): no modal verb indicating requirement strength")
            severity_counts["medium"] += 1
            completeness_issues += 1
        
        # 6. Enhanced Ambiguity Detection
        ambiguity_issues = 0
        for severity, phrases in self.ambiguous_terms["ambiguity"].items():
            for phrase in phrases:
                if phrase in text.lower():
                    issues.append(f"Ambiguity ({severity}): vague phrase '{phrase}'")
                    severity_counts[severity] += 1
                    ambiguity_issues += 1
        
        # 7. Passive Voice Check
        passive_issues = self.check_passive_voice(text)
        for passive_issue in passive_issues:
            issues.append(f"Clarity (medium): {passive_issue}")
            severity_counts["medium"] += 1
        
        # 8. Length Analysis
        word_count = len([token for token in doc if token.is_alpha])
        if word_count < 5:
            issues.append("Completeness (medium): requirement may be too short")
            severity_counts["medium"] += 1
        elif word_count > 50:
            issues.append("Atomicity (low): requirement may be too long (consider splitting)")
            severity_counts["low"] += 1
        
        # 9. Readability
        readability = self.calculate_readability_score(text)
        if readability < 30:
            issues.append(f"Clarity (medium): low readability score ({readability:.1f}/100)")
            severity_counts["medium"] += 1
        
        # Calculate quality scores (0-100, higher is better)
        clarity_score = max(0, 100 - (clarity_issues * 20 + len(passive_issues) * 10))
        completeness_score = max(0, 100 - (completeness_issues * 25))
        verifiability_score = max(0, 100 - (verifiability_issues * 30))
        atomicity_score = max(0, 100 - (atomicity_issues * 30))
        consistency_score = 100 if modal_found else 80  # Simple consistency check
        
        metrics = QualityMetrics(
            clarity_score, completeness_score, verifiability_score,
            atomicity_score, consistency_score, len(issues), severity_counts
        )
        
        return issues, metrics
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary report"""
        total_reqs = len(df)
        reqs_with_issues = len(df[df["Total_Issues"] > 0])
        
        # Severity breakdown
        severity_summary = {
            "critical": df["Critical_Issues"].sum(),
            "high": df["High_Issues"].sum(),
            "medium": df["Medium_Issues"].sum(),
            "low": df["Low_Issues"].sum()
        }
        
        # Quality score statistics
        score_stats = {}
        score_columns = ["Clarity_Score", "Completeness_Score", "Verifiability_Score", 
                        "Atomicity_Score", "Consistency_Score"]
        
        for col in score_columns:
            if col in df.columns:
                score_stats[col.replace("_", " ")] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
        
        # Top issues
        all_issues = []
        for issues_list in df["Issues"]:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        issue_counter = Counter(all_issues)
        top_issues = issue_counter.most_common(10)
        
        return {
            "summary": {
                "total_requirements": total_reqs,
                "requirements_with_issues": reqs_with_issues,
                "issue_rate": (reqs_with_issues / total_reqs * 100) if total_reqs > 0 else 0
            },
            "severity_breakdown": severity_summary,
            "quality_scores": score_stats,
            "top_issues": top_issues
        }
    
    def create_excel_report(self, df: pd.DataFrame, output_file: str = None) -> str:
        """
        Create comprehensive Excel quality report from analysis results.
        """
        if output_file is None:
            output_file = "outputs/quality_analysis/requirements_quality_report.xlsx"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating Excel quality report: {output_path}")
        
        # Create Excel writer
        print("🧪 DEBUG REPORT CONTENTS")
        print(f"df empty: {df.empty}")
        print(f"dashboard preview:\n{self._create_dashboard_tab(df).head()}")
        print(f"critical preview:\n{self._create_critical_tab(df).head()}")
        print(f"recommendations preview:\n{self._create_recommendations_tab(df).head()}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            dashboard = self._create_dashboard_tab(df)
            critical = self._create_critical_tab(df)
            recommendations = self._create_recommendations_tab(df)

            wrote_sheet = False  # <-- TRACK IF ANYTHING GOT WRITTEN

            if not dashboard.empty:
                dashboard.to_excel(writer, sheet_name='Quality Dashboard', index=False)
                wrote_sheet = True

            if not critical.empty:
                critical.to_excel(writer, sheet_name='Critical Issues', index=False)
                wrote_sheet = True

            if not recommendations.empty:
                recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
                wrote_sheet = True

            if not df.empty:
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                wrote_sheet = True

            if not wrote_sheet:
                raise ValueError("❌ No sheets were written. All DataFrames were empty.")

            self._format_excel_workbook(writer)
                
        logger.info(f"Excel quality report saved: {output_path}")
        return str(output_path)

    def _create_dashboard_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quality dashboard summary."""
        
        total_reqs = len(df)
        dashboard_data = []
        
        # Overall metrics
        avg_quality = df['Quality_Score'].mean()
        needs_attention = len(df[df['Quality_Score'] < 50])
        
        dashboard_data.extend([
            {"Metric": "Total Requirements", "Value": total_reqs, "Status": "✓"},
            {"Metric": "Average Quality Score", "Value": f"{avg_quality:.1f}/100", 
             "Status": "✓" if avg_quality >= 70 else "⚠️"},
            {"Metric": "Need Attention", "Value": needs_attention,
             "Status": "✓" if needs_attention < total_reqs * 0.1 else "❌"}
        ])
        
        # Quality grade distribution
        grade_counts = df['Quality_Grade'].value_counts()
        for grade in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']:
            count = grade_counts.get(grade, 0)
            pct = count / total_reqs * 100
            dashboard_data.append({
                "Metric": f"{grade} Requirements", 
                "Value": f"{count} ({pct:.1f}%)",
                "Status": "✓" if grade in ['EXCELLENT', 'GOOD'] else "⚠️" if grade == 'FAIR' else "❌"
            })
        
        # Component averages
        components = ['Clarity_Score', 'Completeness_Score', 'Verifiability_Score', 
                      'Atomicity_Score', 'Consistency_Score']
        
        for comp in components:
            if comp in df.columns:
                avg_score = df[comp].mean()
                comp_name = comp.replace('_Score', '')
                dashboard_data.append({
                    "Metric": f"{comp_name} Average",
                    "Value": f"{avg_score:.1f}/100",
                    "Status": "✓" if avg_score >= 70 else "⚠️" if avg_score >= 50 else "❌"
                })
        
        return pd.DataFrame(dashboard_data)
    
    def _create_critical_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create critical issues tab."""
        
        # Filter critical/high issues
        critical_reqs = df[(df['Critical_Issues'] > 0) | (df['High_Issues'] > 0) | 
                          (df['Quality_Score'] < 35)].copy()
        
        if len(critical_reqs) == 0:
            return pd.DataFrame([{"Message": "No critical issues found!"}])
        
        # Select key columns
        critical_cols = ['ID', 'Quality_Score', 'Quality_Grade', 
                        'Critical_Issues', 'High_Issues', 'Issues']
        
        # Get available columns
        available_cols = [col for col in critical_cols if col in critical_reqs.columns]
        critical_issues = critical_reqs[available_cols].copy()
        
        # Sort by severity (worst first)
        critical_issues = critical_issues.sort_values('Quality_Score')
        
        # Add action columns
        critical_issues['Priority'] = critical_issues.apply(
            lambda row: 'CRITICAL' if row.get('Critical_Issues', 0) > 0 else 'HIGH', axis=1
        )
        critical_issues['Action_Status'] = 'PENDING'
        critical_issues['Assigned_To'] = ''
        critical_issues['Notes'] = ''
        
        return critical_issues
    
    def _create_recommendations_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create improvement recommendations."""
        
        recommendations = []
        
        # Analyze common issues
        all_issues = []
        for issues_list in df['Issues']:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        issue_counts = Counter(all_issues)
        
        # Generate recommendations for common issues
        for issue, count in issue_counts.most_common(10):
            recommendations.append({
                "Issue": issue,
                "Frequency": count,
                "Affected_Requirements": count,
                "Priority": "HIGH" if count > len(df) * 0.2 else "MEDIUM",
                "Recommendation": self._get_recommendation(issue),
                "Effort": "LOW" if "ambiguous" in issue.lower() else "MEDIUM"
            })
        
        return pd.DataFrame(recommendations)
    
    def _get_recommendation(self, issue: str) -> str:
        """Get specific recommendation for an issue."""
        issue_lower = issue.lower()
        
        if "ambiguous" in issue_lower:
            return "Replace vague terms with specific, measurable criteria"
        elif "passive voice" in issue_lower:
            return "Rewrite in active voice to clarify responsibility"
        elif "modal verb" in issue_lower:
            return "Add shall/should/may to indicate requirement priority"
        elif "conjunction" in issue_lower:
            return "Split compound requirements into separate statements"
        else:
            return "Review for clarity and completeness"
    
    def _format_excel_workbook(self, writer):
        """Apply basic formatting to Excel workbook."""
        
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            
            workbook = writer.book
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                
                # Format headers
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
                
                # Auto-adjust column widths
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
                    
        except ImportError:
            logger.warning("openpyxl not available for Excel formatting")
    
    def analyze_file(self, input_file: str = "requirements.csv", 
                    output_file: str = None,
                    requirement_column: str = "Requirement Text",
                    excel_report: bool = False) -> pd.DataFrame:
        """
        Enhanced analyze_file method that can create both CSV and Excel reports.
        """
        logger.info(f"Starting analysis of {input_file}")
        
        # Read and validate input
        df = self.safe_read_csv(input_file)
        
        if requirement_column not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Column '{requirement_column}' not found. Available columns: {available_cols}")
            raise ValueError(f"Column '{requirement_column}' not found in CSV")
        
        # Fill NaN values
        df = df.fillna({requirement_column: ""})
        
        logger.info(f"Analyzing {len(df)} requirements...")
        
        # Perform analysis
        analysis_results = []
        metrics_list = []
        
        for idx, requirement in enumerate(df[requirement_column]):
            req_id = df.get("ID", pd.Series([f"REQ_{idx:04d}"] * len(df))).iloc[idx]
            issues, metrics = self.analyze_requirement(requirement, str(req_id))
            analysis_results.append(issues)
            metrics_list.append(metrics)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} requirements")
        
        # Add results to DataFrame
        df["Issues"] = analysis_results
        df["Total_Issues"] = [len(issues) for issues in analysis_results]
        
        # Add individual metrics
        df["Clarity_Score"] = [m.clarity_score for m in metrics_list]
        df["Completeness_Score"] = [m.completeness_score for m in metrics_list]
        df["Verifiability_Score"] = [m.verifiability_score for m in metrics_list]
        df["Atomicity_Score"] = [m.atomicity_score for m in metrics_list]
        df["Consistency_Score"] = [m.consistency_score for m in metrics_list]
        
        # Add severity breakdowns
        df["Critical_Issues"] = [m.severity_breakdown.get("critical", 0) for m in metrics_list]
        df["High_Issues"] = [m.severity_breakdown.get("high", 0) for m in metrics_list]
        df["Medium_Issues"] = [m.severity_breakdown.get("medium", 0) for m in metrics_list]
        df["Low_Issues"] = [m.severity_breakdown.get("low", 0) for m in metrics_list]
        
        # Calculate overall quality score
        df["Quality_Score"] = (
            df["Clarity_Score"] * 0.25 +
            df["Completeness_Score"] * 0.25 +
            df["Verifiability_Score"] * 0.2 +
            df["Atomicity_Score"] * 0.2 +
            df["Consistency_Score"] * 0.1
        )
        
        # Calculate quality grade
        def assign_grade(score):
            if score >= 80:
                return "EXCELLENT"
            elif score >= 65:
                return "GOOD"
            elif score >= 50:
                return "FAIR"
            elif score >= 35:
                return "POOR"
            else:
                return "CRITICAL"
        
        df["Quality_Grade"] = df["Quality_Score"].apply(assign_grade)
        
        # Generate summary report
        summary_report = self.generate_summary_report(df)
        
        # Save CSV results (original functionality)
        if not output_file:
            input_path = Path(input_file)
            output_file = input_path.stem + "_quality_report.csv"
        
        df.to_csv(output_file, index=False)
        logger.info(f"CSV analysis complete. Results saved to '{output_file}'")
        
        # Save summary report
        summary_file = str(Path(output_file).with_suffix("")) + "_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_report, f, indent=2, default=str)
        logger.info(f"Summary report saved to '{summary_file}'")
        
        # Create Excel report if requested
        if excel_report:
            excel_path = self.create_excel_report(df)
            logger.info(f"Excel quality report created: {excel_path}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("REQUIREMENTS QUALITY ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Requirements: {summary_report['summary']['total_requirements']}")
        print(f"Requirements with Issues: {summary_report['summary']['requirements_with_issues']}")
        print(f"Issue Rate: {summary_report['summary']['issue_rate']:.1f}%")
        print(f"\nSeverity Breakdown:")
        for severity, count in summary_report['severity_breakdown'].items():
            print(f"  {severity.capitalize()}: {count}")
        
        if excel_report:
            print(f"\n📊 Excel report: outputs/quality_analysis/requirements_quality_report.xlsx")
        
        return df


def main():
    """Enhanced main function with Excel report option"""
    parser = argparse.ArgumentParser(description="Analyze requirements quality")
    parser.add_argument("input_file", nargs="?", default="requirements.csv",
                       help="Input CSV file containing requirements (default: requirements.csv)")
    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated)")
    parser.add_argument("-c", "--column", default="Requirement Text", 
                       help="Column name containing requirements (default: 'Requirement Text')")
    parser.add_argument("-m", "--model", default="en_core_web_sm",
                       help="spaCy model to use (default: en_core_web_sm)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--excel", action="store_true",
                       help="Create Excel report (default: CSV only)")
    
    args = parser.parse_args()
    


    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        analyzer = RequirementAnalyzer(args.model)
        analyzer.analyze_file(args.input_file, args.output, args.column, excel_report=args.excel)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()