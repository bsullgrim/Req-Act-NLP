"""
Enhanced Requirements Quality Analyzer with Excel Output
Clean version using existing project utils
"""

import pandas as pd
import logging
import spacy
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass
from collections import Counter
import json
import sys
import os

# Add project root to path for proper imports
# Since we're in src/quality/, we need to go up two levels to reach project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Go up from src/quality/ to project root
sys.path.insert(0, str(project_root))

# Import your existing utils (now from correct path)
try:
    from src.utils.file_utils import SafeFileHandler
    from src.utils.path_resolver import SmartPathResolver
    from src.utils.repository_setup import RepositoryStructureManager
    UTILS_AVAILABLE = True
    print("✅ Successfully imported project utils")
except ImportError as e:
    print(f"⚠️ Could not import utils: {e}")
    print(f"⚠️ Current location: {current_file}")
    print(f"⚠️ Project root: {project_root}")
    print("⚠️ Make sure the src/utils/ directory exists with the required files")
    UTILS_AVAILABLE = False
    sys.exit(1)  # Exit if utils not available since they're essential

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
    """Enhanced requirements quality analyzer using project utils"""
    
    def __init__(self, spacy_model: str = "en_core_web_trf", repo_manager=None):
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Trying fallback models...")
            # Try fallback models in order of preference
            fallback_models = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
            model_loaded = False
            
            for fallback in fallback_models:
                try:
                    self.nlp = spacy.load(fallback)
                    logger.info(f"Loaded fallback spaCy model: {fallback}")
                    model_loaded = True
                    break
                except OSError:
                    continue
            
            if not model_loaded:
                logger.error("No spaCy models available. Please install with:")
                logger.error("python -m spacy download en_core_web_trf")
                logger.error("or: python -m spacy download en_core_web_sm")
                raise OSError("No suitable spaCy model found")
        
        # Setup utils
        if repo_manager is None:
            self.repo_manager = RepositoryStructureManager("outputs")
            self.repo_manager.setup_repository_structure()
        else:
            self.repo_manager = repo_manager
        
        self.file_handler = SafeFileHandler(repo_manager=self.repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager=self.repo_manager)
        logger.info("✅ Initialized with project utils")
        
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
        
        # 3. Enhanced Verifiability (significantly improved)
        verifiability_issues = 0
        
        # Existing spaCy entity detection
        measurable_entities = [ent for ent in doc.ents if ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"}]
        
        # Enhanced patterns for measurable criteria
        verifiability_patterns = [
            # Numbers with units (your excellent insight!)
            r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|in|ft|yd|mil|thou)\b',  # Length
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|lb|oz|ton|tonnes?)\b',      # Mass/Weight
            r'\b\d+(?:\.\d+)?\s*(?:ms|sec|min|hr|day|week|month|year)s?\b',  # Time
            r'\b\d+(?:\.\d+)?\s*(?:Hz|kHz|MHz|GHz|rpm|bpm)\b',        # Frequency
            r'\b\d+(?:\.\d+)?\s*(?:V|mV|kV|A|mA|W|kW|MW|Ω|ohm)\b',     # Electrical
            r'\b\d+(?:\.\d+)?\s*(?:Pa|kPa|MPa|psi|bar|atm|torr)\b',   # Pressure
            r'\b\d+(?:\.\d+)?\s*(?:°C|°F|K|celsius|fahrenheit|kelvin)\b',  # Temperature
            r'\b\d+(?:\.\d+)?\s*(?:m/s|km/h|mph|fps|knots?)\b',       # Velocity
            r'\b\d+(?:\.\d+)?\s*(?:m/s²|g|G)\b',                      # Acceleration
            r'\b\d+(?:\.\d+)?\s*(?:J|kJ|cal|kcal|BTU|Wh|kWh)\b',      # Energy
            r'\b\d+(?:\.\d+)?\s*(?:N|kN|lbf|dyne)\b',                 # Force
            r'\b\d+(?:\.\d+)?\s*(?:bit|byte|kB|MB|GB|TB)\b',          # Data
            r'\b\d+(?:\.\d+)?\s*(?:dB|dBm|dBi)\b',                    # Decibels
            r'\b\d+(?:\.\d+)?\s*(?:lux|lumen|cd)\b',                  # Light
            
            # Percentages and ratios
            r'\b\d+(?:\.\d+)?%\b',
            r'\b\d+(?:\.\d+)?:\d+(?:\.\d+)?\b',  # Ratios like 3:1
            
            # Scientific notation with units
            r'\b\d+(?:\.\d+)?[eE][+-]?\d+\s*[a-zA-Z]+\b',
            
            # Numbers with explicit measurement contexts
            r'\b(?:less|more|greater|equal)\s+than\s+\d+(?:\.\d+)?',
            r'\b(?:within|±|plus|minus)\s*\d+(?:\.\d+)?',
            r'\b(?:maximum|minimum|nominal|typical)\s+(?:of\s+)?\d+(?:\.\d+)?',
            r'\b\d+(?:\.\d+)?\s*(?:maximum|minimum|nominal|typical)',
            
            # Performance criteria patterns
            r'\b(?:accuracy|precision|tolerance|error)\s+(?:of\s+)?[±]?\d+(?:\.\d+)?',
            r'\b(?:range|span)\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?',
            r'\b(?:between|from)\s+\d+(?:\.\d+)?\s*(?:to|and|-)\s*\d+(?:\.\d+)?',
            
            # Compliance and standards (verifiable through testing)
            r'\b(?:comply|conform|meet|satisfy)\s+(?:with\s+)?(?:standard|specification|requirement)',
            r'\b(?:ISO|IEEE|ANSI|ASTM|MIL-STD|DO-\d+|IEC)\s*[-]?\s*\d+',
            
            # Time-based criteria
            r'\b(?:within|before|after|during)\s+\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?)',
            r'\b(?:every|each)\s+\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?)',
            
            # Boolean/binary verifiable states
            r'\b(?:shall|must|will)\s+(?:be\s+)?(?:enabled|disabled|active|inactive|on|off)\b',
            r'\b(?:presence|absence)\s+of\b',
            
            # Count-based criteria
            r'\b(?:at\s+least|at\s+most|exactly|up\s+to)\s+\d+',
            r'\b\d+\s+(?:times|instances|occurrences|iterations|cycles)',
        ]
        
        # Check for any verifiable patterns
        has_verifiable_criteria = False
        
        # Check spaCy entities
        if measurable_entities:
            has_verifiable_criteria = True
        
        # Check enhanced patterns
        if not has_verifiable_criteria:
            text_lower = text.lower()
            for pattern in verifiability_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    has_verifiable_criteria = True
                    break
        
        # Additional context checks for edge cases
        if not has_verifiable_criteria:
            # Look for comparative terms that imply measurement
            comparative_terms = [
                'faster', 'slower', 'higher', 'lower', 'larger', 'smaller',
                'better', 'worse', 'more', 'less', 'greater', 'fewer'
            ]
            if any(term in text_lower for term in comparative_terms):
                # Check if there's a measurable baseline
                if re.search(r'\bthan\s+\d', text_lower):
                    has_verifiable_criteria = True  # Good: "faster than 100 rpm"
                # If comparative term but no measurable baseline, still not verifiable
                # This will fall through and get flagged below
        
        if not has_verifiable_criteria:
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
        
        # Add check for solution-specific language (domain-agnostic approach)
        # Focus on linguistic patterns that indicate HOW rather than WHAT
        implementation_indicators = [
            # Direct implementation verbs
            r'\b(?:implement|deploy|install|configure|setup|utilize|employ)\b',
            
            # "Using X" patterns (but exclude measurement contexts)
            r'\busing\s+(?:a|an|the)?\s*[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?!\s+(?:of|to|for|shall|must|will))',
            
            # "Via/through" with specific methods (exclude general phrases and verification)
            r'\b(?:via|through)\s+(?:a|an|the)?\s*[a-zA-Z]+(?:\s+[a-zA-Z]+){1,3}(?:\s+(?:method|process|technique|approach|system))?',
            
            # "By means of" always indicates implementation
            r'\bby\s+means\s+of\b',
            
            # "With" + specific technology/component names (capitalized or technical terms)
            r'\bwith\s+(?:a|an|the)?\s*[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:unit|module|component|device|sensor|actuator))?',
            
            # Technology stack indicators
            r'\b(?:based\s+on|built\s+on|powered\s+by)\s+',
            
            # Specific brand/product names (typically capitalized)
            r'\b(?:using|with|via)\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:brand|model|version|\d+))',
        ]
        
        # Exclude patterns that are clearly requirements, not implementation
        requirement_patterns = [
            r'\bwith(?:in)?\s+(?:less\s+than\s+)?[\d\.\-\+e]+',  # "with less than 10^-6", "within 5 seconds"
            r'\bwith(?:in)?\s+(?:a\s+)?(?:tolerance|accuracy|precision|error)\s+of',  # "with a tolerance of"
            r'\bwith(?:in)?\s+(?:maximum|minimum|nominal)\s+',  # "with maximum power"
            r'\busing\s+(?:no\s+more\s+than|less\s+than|up\s+to)\s+[\d\.\-\+e]+',  # "using no more than 5W"
            r'\bwith\s+(?:no|zero|minimal|low|high|maximum|minimum)\s+',  # "with minimal noise"
        ]
        
        # Exclude verification/testing language (NOT implementation)
        verification_patterns = [
            r'\bas\s+(?:verified|validated|demonstrated|tested|measured|confirmed|proven)\s+(?:through|by|via|using)',
            r'\bas\s+(?:shown|evidenced|established)\s+(?:through|by|via)',
            r'\b(?:verified|validated|demonstrated|tested|measured|confirmed|proven)\s+(?:through|by|via|using)\s+',
            r'\bsubject\s+to\s+(?:verification|validation|testing|analysis)',
            r'\bper\s+(?:test|verification|validation|analysis)',
            r'\b(?:acceptance|verification|validation|test)\s+(?:criteria|method|procedure)',
        ]
        
        # Check for implementation indicators
        text_lower = text.lower()
        has_implementation = False
        
        for pattern in implementation_indicators:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Double-check it's not actually a requirement pattern
                is_requirement = any(re.search(req_pattern, text_lower) for req_pattern in requirement_patterns)
                # Also check if it's verification language
                is_verification = any(re.search(ver_pattern, text_lower) for ver_pattern in verification_patterns)
                
                if not is_requirement and not is_verification:
                    has_implementation = True
                    break
        
        if has_implementation:
            issues.append("Design (high): Contains implementation details")
            severity_counts["high"] += 1

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
        """Create comprehensive Excel quality report from analysis results."""
        
        # Use structured path for output
        if output_file is None:
            output_file = self.file_handler.get_structured_path(
                'quality_analysis', 
                "requirements_quality_report.xlsx"
            )
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating Excel quality report: {output_path}")
        
        # Create Excel writer with explicit encoding to prevent character issues
        with pd.ExcelWriter(output_path, engine='openpyxl', options={'encoding': 'utf-8'}) as writer:
            # Create analysis tabs
            dashboard = self._create_dashboard_tab(df)
            critical = self._create_critical_tab(df)
            recommendations = self._create_recommendations_tab(df)

            wrote_sheet = False
            
            # Write sheets with error handling
            try:
                if not dashboard.empty:
                    dashboard.to_excel(writer, sheet_name='Quality Dashboard', index=False)
                    wrote_sheet = True
                    logger.debug("✓ Dashboard tab written")
            except Exception as e:
                logger.warning(f"Could not write dashboard tab: {e}")

            try:
                if not critical.empty:
                    critical.to_excel(writer, sheet_name='Critical Issues', index=False)
                    wrote_sheet = True
                    logger.debug("✓ Critical issues tab written")
            except Exception as e:
                logger.warning(f"Could not write critical issues tab: {e}")

            try:
                if not recommendations.empty:
                    recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
                    wrote_sheet = True
                    logger.debug("✓ Recommendations tab written")
            except Exception as e:
                logger.warning(f"Could not write recommendations tab: {e}")

            try:
                if not df.empty:
                    # Clean the dataframe for Excel output to prevent encoding issues
                    df_clean = df.copy()
                    # Convert any problematic columns to strings
                    for col in df_clean.columns:
                        if df_clean[col].dtype == 'object':
                            df_clean[col] = df_clean[col].astype(str)
                    
                    df_clean.to_excel(writer, sheet_name='Detailed Results', index=False)
                    wrote_sheet = True
                    logger.debug("✓ Detailed results tab written")
            except Exception as e:
                logger.warning(f"Could not write detailed results tab: {e}")

            if not wrote_sheet:
                raise ValueError("❌ No sheets were written. All DataFrames were empty or had errors.")

            # Apply formatting
            try:
                self._format_excel_workbook(writer)
                logger.debug("✓ Excel formatting applied")
            except Exception as e:
                logger.warning(f"Could not apply Excel formatting: {e}")
                
        logger.info(f"Excel quality report saved: {output_path}")
        return str(output_path)

    def _create_dashboard_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quality dashboard summary."""
        
        total_reqs = len(df)
        if total_reqs == 0:
            return pd.DataFrame([{"Metric": "No data", "Value": "0", "Status": "❌"}])
        
        dashboard_data = []
        
        # Overall metrics
        avg_quality = df['Quality_Score'].mean() if 'Quality_Score' in df.columns else 0
        needs_attention = len(df[df['Quality_Score'] < 50]) if 'Quality_Score' in df.columns else 0
        
        dashboard_data.extend([
            {"Metric": "Total Requirements", "Value": total_reqs, "Status": "✓"},
            {"Metric": "Average Quality Score", "Value": f"{avg_quality:.1f}/100", 
             "Status": "✓" if avg_quality >= 70 else "⚠️"},
            {"Metric": "Need Attention", "Value": needs_attention,
             "Status": "✓" if needs_attention < total_reqs * 0.1 else "❌"}
        ])
        
        # Quality grade distribution
        if 'Quality_Grade' in df.columns:
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
        critical_filter = (
            (df.get('Critical_Issues', 0) > 0) | 
            (df.get('High_Issues', 0) > 0) | 
            (df.get('Quality_Score', 100) < 35)
        )
        critical_reqs = df[critical_filter].copy()
        
        if len(critical_reqs) == 0:
            return pd.DataFrame([{"Message": "No critical issues found!"}])
        
        # Select key columns that exist
        potential_cols = ['ID', 'Quality_Score', 'Quality_Grade', 
                         'Critical_Issues', 'High_Issues', 'Issues']
        available_cols = [col for col in potential_cols if col in critical_reqs.columns]
        
        if not available_cols:
            return pd.DataFrame([{"Message": "No suitable columns found for critical analysis"}])
        
        critical_issues = critical_reqs[available_cols].copy()
        
        # Sort by severity (worst first)
        if 'Quality_Score' in critical_issues.columns:
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
        if 'Issues' in df.columns:
            for issues_list in df['Issues']:
                if isinstance(issues_list, list):
                    all_issues.extend(issues_list)
        
        if not all_issues:
            return pd.DataFrame([{"Message": "No issues found to analyze"}])
        
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
                if worksheet.max_row > 0:
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
        except Exception as e:
            logger.warning(f"Excel formatting failed: {e}")
    
    def save_with_proper_encoding(self, df: pd.DataFrame, output_file: str) -> str:
        """Save CSV with proper encoding to prevent character issues"""
        try:
            # Use UTF-8 with BOM to prevent encoding issues in Excel
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"CSV saved with UTF-8-sig encoding: {output_file}")
            return output_file
        except Exception as e:
            logger.warning(f"UTF-8-sig failed, trying latin1: {e}")
            try:
                df.to_csv(output_file, index=False, encoding='latin1')
                logger.info(f"CSV saved with latin1 encoding: {output_file}")
                return output_file
            except Exception as e2:
                logger.error(f"All encoding attempts failed: {e2}")
                raise
    
    def analyze_file(self, input_file: str = "requirements.csv", 
                    output_file: str = None,
                    requirement_column: str = "Requirement Text",
                    excel_report: bool = False) -> pd.DataFrame:
        """
        Enhanced analyze_file method using project utils for file operations.
        """
        logger.info(f"Starting analysis of {input_file}")
        
        # Use path resolver to find the requirements file
        print(f"🔍 Resolving file path for: {input_file}")
        resolved_paths = self.path_resolver.resolve_input_files({
            'requirements': input_file
        })
        input_file_path = resolved_paths['requirements']
        
        if not Path(input_file_path).exists():
            raise FileNotFoundError(f"Could not find requirements file: {input_file}")
        
        print(f"✅ Found requirements file: {input_file_path}")
        
        # Use SafeFileHandler to read with proper encoding detection
        df = self.file_handler.safe_read_csv(input_file_path)
        
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
        
        # Calculate overall quality score with stricter weighting
        df["Quality_Score"] = (
            df["Clarity_Score"] * 0.2 +           # Reduced from 0.25
            df["Completeness_Score"] * 0.2 +      # Reduced from 0.25  
            df["Verifiability_Score"] * 0.35 +    # Increased from 0.2 (most critical!)
            df["Atomicity_Score"] * 0.15 +        # Reduced from 0.2
            df["Consistency_Score"] * 0.1         # Same
        )
        
        # Apply issue penalties (stricter penalties for multiple issues)
        def apply_issue_penalty(row):
            base_score = row["Quality_Score"]
            total_issues = row["Total_Issues"]
            critical_issues = row["Critical_Issues"]
            high_issues = row["High_Issues"]
            
            # Heavy penalties for critical and high issues
            penalty = (critical_issues * 25) + (high_issues * 15) + (row["Medium_Issues"] * 5) + (row["Low_Issues"] * 2)
            
            # Additional penalty for multiple issues (compounds the problems)
            if total_issues > 3:
                penalty += (total_issues - 3) * 10  # Extra penalty for many issues
            
            return max(0, base_score - penalty)
        
        df["Quality_Score"] = df.apply(apply_issue_penalty, axis=1)
        
        # Much stricter quality grade thresholds
        def assign_grade(score):
            if score >= 95:
                return "EXCELLENT"
            elif score >= 85:
                return "GOOD"  
            elif score >= 70:
                return "FAIR"
            elif score >= 50:
                return "POOR"
            else:
                return "CRITICAL"
        
        df["Quality_Grade"] = df["Quality_Score"].apply(assign_grade)
        
        # Generate summary report
        summary_report = self.generate_summary_report(df)
        
        # Determine output file path using structured path
        if not output_file:
            input_stem = Path(input_file_path).stem
            output_file = self.file_handler.get_structured_path(
                'quality_analysis', 
                f"{input_stem}_quality_report.csv"
            )
        
        # Save CSV results with proper encoding
        try:
            self.save_with_proper_encoding(df, output_file)
            logger.info(f"CSV analysis complete. Results saved to '{output_file}'")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise
        
        # Save summary report
        try:
            summary_file = str(Path(output_file).with_suffix("")) + "_summary.json"
            with open(summary_file, "w", encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Summary report saved to '{summary_file}'")
        except Exception as e:
            logger.warning(f"Could not save summary report: {e}")
        
        # Create Excel report if requested
        if excel_report:
            try:
                excel_path = self.create_excel_report(df)
                logger.info(f"Excel quality report created: {excel_path}")
            except Exception as e:
                logger.error(f"Failed to create Excel report: {e}")
        
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
        
        # Print file locations
        print(f"\n📁 Output Files:")
        print(f"   📊 CSV Report: {output_file}")
        if excel_report:
            excel_location = self.file_handler.get_structured_path('quality_analysis', 'requirements_quality_report.xlsx')
            print(f"   📈 Excel Report: {excel_location}")
        
        return df


def main():
    """Enhanced main function with proper utils integration"""
    parser = argparse.ArgumentParser(description="Analyze requirements quality using project utils")
    parser.add_argument("input_file", nargs="?", default="requirements.csv",
                       help="Input CSV file containing requirements (default: requirements.csv)")
    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated)")
    parser.add_argument("-c", "--column", default="Requirement Text", 
                       help="Column name containing requirements (default: 'Requirement Text')")
    parser.add_argument("-m", "--model", default="en_core_web_trf",
                       help="spaCy model to use (default: en_core_web_trf)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--excel", action="store_true",
                       help="Create Excel report (default: CSV only)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("🔍 REQUIREMENTS QUALITY ANALYZER")
        print("=" * 50)
        print("✅ Using project utils for file operations and path resolution")
        
        # Show model being used
        print(f"🤖 NLP Model: {args.model}")
        if args.model == "en_core_web_trf":
            print("   📈 Using transformer model for enhanced accuracy")
        
        # Initialize analyzer with proper utils
        analyzer = RequirementAnalyzer(args.model)
        
        # Run analysis
        results_df = analyzer.analyze_file(
            args.input_file, 
            args.output, 
            args.column, 
            excel_report=args.excel
        )
        
        print(f"\n🎯 Analysis Complete!")
        print(f"   Analyzed {len(results_df)} requirements")
        print(f"   Generated quality scores and recommendations")
        print(f"   Files saved to outputs/quality_analysis/ directory")
        
        if args.excel:
            print(f"   Created comprehensive Excel report with multiple tabs")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"💡 The path resolver will search in these locations:")
        print(f"   • Current directory (.)")
        print(f"   • data/raw/")
        print(f"   • data/")
        print(f"   • examples/")
        print(f"\n💡 Make sure your requirements file exists in one of these locations")
        
    except ValueError as e:
        print(f"\n❌ Data error: {e}")
        print(f"💡 Common solutions:")
        print(f"   • Check that '{args.column}' column exists in your CSV")
        print(f"   • Verify CSV format and encoding")
        print(f"   • The SafeFileHandler will try multiple encodings automatically")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print("💡 Run with -v flag for detailed error information")


if __name__ == "__main__":
    main()