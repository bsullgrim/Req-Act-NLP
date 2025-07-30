"""
Enhanced Requirements Quality Analyzer with 4-Tab Excel Output
Provides comprehensive analysis with Summary, Quality, INCOSE, and Semantic tabs.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
from pathlib import Path
import spacy
import json
import re
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import argparse
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.file_utils import SafeFileHandler
from src.utils.path_resolver import SmartPathResolver
from src.utils.repository_setup import RepositoryStructureManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for requirement quality metrics."""
    clarity_score: float
    completeness_score: float
    verifiability_score: float
    atomicity_score: float
    consistency_score: float
    incose_compliance_score: float
    semantic_quality_score: float
    total_issues: int
    severity_breakdown: Dict[str, int]

@dataclass
class INCOSEAnalysis:
    """INCOSE pattern analysis results."""
    best_pattern: str
    compliance_score: float
    components_found: Dict[str, Optional[str]]
    missing_required: List[str]
    missing_optional: List[str]
    suggestions: List[str]
    template_recommendation: str

@dataclass
class SemanticAnalysis:
    """Semantic analysis results."""
    similarity_issues: List[Dict]
    contextual_ambiguities: List[str]
    entity_completeness: Dict[str, List[str]]
    tone_issues: List[str]
    improvement_suggestions: List[str]

class INCOSEPatternAnalyzer:
    """INCOSE requirements pattern analyzer."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.patterns = {
            'functional_performance': {
                'name': 'Functional/Performance',
                'required': ['AGENT', 'FUNCTION', 'PERFORMANCE'],
                'optional': ['INTERFACE_OUTPUT', 'TIMING', 'EVENT_TRIGGER', 'INTERFACE_INPUT', 'CONDITION'],
                'template': "The {AGENT} shall {FUNCTION} in accordance with {INTERFACE_OUTPUT} with {PERFORMANCE} [and {TIMING} upon {EVENT_TRIGGER}] while in {CONDITION}",
                'description': "Specifies what the system shall do and how well"
            },
            'suitability': {
                'name': 'Suitability',
                'required': ['AGENT', 'CHARACTERISTIC', 'PERFORMANCE'],
                'optional': ['CONDITION', 'CONDITION_DURATION'],
                'template': "The {AGENT} shall exhibit {CHARACTERISTIC} with {PERFORMANCE} while {CONDITION} [for {CONDITION_DURATION}]",
                'description': "Specifies quality characteristics the system must exhibit"
            },
            'environments': {
                'name': 'Environmental',
                'required': ['AGENT', 'CHARACTERISTIC', 'ENVIRONMENT'],
                'optional': ['EXPOSURE_DURATION'],
                'template': "The {AGENT} shall exhibit {CHARACTERISTIC} during/after exposure to {ENVIRONMENT} [for {EXPOSURE_DURATION}]",
                'description': "Specifies behavior under environmental conditions"
            },
            'design': {
                'name': 'Design Constraint',
                'required': ['AGENT', 'DESIGN_CONSTRAINTS'],
                'optional': ['PERFORMANCE', 'CONDITION'],
                'template': "The {AGENT} shall exhibit {DESIGN_CONSTRAINTS} [in accordance with {PERFORMANCE} while in {CONDITION}]",
                'description': "Specifies design limitations or constraints"
            }
        }
        
        # Component extraction patterns
        self.component_patterns = {
            'PERFORMANCE': [
                r'within\s+\d+(?:\.\d+)?\s*\w+',
                r'at\s+least\s+\d+(?:\.\d+)?',
                r'with\s+\d+(?:\.\d+)?%?\s*\w*',
                r'±\s*\d+(?:\.\d+)?',
                r'accuracy\s+of\s+\d+(?:\.\d+)?',
                r'throughput\s+of\s+\d+(?:\.\d+)?'
            ],
            'TIMING': [
                r'within\s+\d+\s*(?:ms|milliseconds?|sec|seconds?|min|minutes?)',
                r'upon\s+\w+',
                r'during\s+\w+',
                r'after\s+\d+\s*\w+'
            ],
            'CONDITION': [
                r'while\s+[\w\s]+',
                r'during\s+[\w\s]+',
                r'when\s+[\w\s]+',
                r'if\s+[\w\s]+',
                r'under\s+[\w\s]+'
            ],
            'ENVIRONMENT': [
                r'temperature\s+[\w\s]*',
                r'humidity\s+[\w\s]*',
                r'pressure\s+[\w\s]*',
                r'vibration\s+[\w\s]*',
                r'radiation\s+[\w\s]*'
            ]
        }
    
    def extract_incose_components(self, doc) -> Dict[str, Optional[str]]:
        """Extract INCOSE requirement components."""
        components = {comp: None for comp in [
            'AGENT', 'FUNCTION', 'CHARACTERISTIC', 'PERFORMANCE', 'CONDITION',
            'ENVIRONMENT', 'TIMING', 'INTERFACE_OUTPUT', 'INTERFACE_INPUT',
            'EVENT_TRIGGER', 'DESIGN_CONSTRAINTS'
        ]}
        
        text = doc.text
        text_lower = text.lower()
        
        # Extract AGENT (subject)
        for token in doc:
            if token.dep_ == "nsubj" and not token.text.lower() in ["it", "this", "that"]:
                components['AGENT'] = token.text
                break
        
        # Extract FUNCTION (main verb + direct objects)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                function_parts = [token.lemma_]
                for child in token.children:
                    if child.dep_ in ["dobj", "prep", "prt"]:
                        function_parts.append(child.text)
                components['FUNCTION'] = " ".join(function_parts)
                break
        
        # Extract components using patterns
        for comp_type, patterns in self.component_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match and not components[comp_type]:
                    components[comp_type] = match.group().strip()
                    break
        
        # Extract CHARACTERISTIC (quality attributes)
        quality_terms = ["accuracy", "reliability", "availability", "security", "performance", 
                        "efficiency", "maintainability", "usability", "compatibility"]
        for token in doc:
            if token.text.lower() in quality_terms:
                components['CHARACTERISTIC'] = token.text
                break

        # Extract DESIGN_CONSTRAINTS
        constraint_indicators = ["shall not", "must not", "cannot", "limited to", "constrained by"]
        for indicator in constraint_indicators:
            if indicator in text_lower:
                components['DESIGN_CONSTRAINTS'] = indicator
                break
        
        return components
    
    def analyze_incose_compliance(self, text: str) -> INCOSEAnalysis:
        """Analyze requirement against INCOSE patterns."""
        doc = self.nlp(text)
        components = self.extract_incose_components(doc)
        
        # Score each pattern
        pattern_scores = []
        for pattern_name, pattern_def in self.patterns.items():
            score = self.score_pattern_match(components, pattern_def)
            missing_req = [comp for comp in pattern_def['required'] if not components.get(comp)]
            missing_opt = [comp for comp in pattern_def['optional'] if not components.get(comp)]
            
            pattern_scores.append({
                'name': pattern_name,
                'score': score,
                'missing_required': missing_req,
                'missing_optional': missing_opt,
                'definition': pattern_def
            })
        
        # Find best matching pattern
        best_match = max(pattern_scores, key=lambda x: x['score'])
        
        # Generate suggestions
        suggestions = self.generate_pattern_suggestions(components, best_match['definition'])
        
        # Create template recommendation
        template_rec = self.create_template_recommendation(components, best_match['definition'])
        
        return INCOSEAnalysis(
            best_pattern=best_match['name'],
            compliance_score=best_match['score'],
            components_found=components,
            missing_required=best_match['missing_required'],
            missing_optional=best_match['missing_optional'],
            suggestions=suggestions,
            template_recommendation=template_rec
        )
    
    def score_pattern_match(self, components: Dict, pattern_def: Dict) -> float:
        """Score how well components match INCOSE pattern."""
        required_found = sum(1 for comp in pattern_def['required'] if components.get(comp))
        optional_found = sum(1 for comp in pattern_def['optional'] if components.get(comp))
        
        required_score = (required_found / len(pattern_def['required'])) * 80
        optional_bonus = (optional_found / len(pattern_def['optional'])) * 20 if pattern_def['optional'] else 0
        
        return min(100, required_score + optional_bonus)
    
    def generate_pattern_suggestions(self, components: Dict, pattern_def: Dict) -> List[str]:
        """Generate improvement suggestions based on missing components."""
        suggestions = []
        
        component_guidance = {
            'AGENT': "Specify the system, subsystem, or component responsible (e.g., 'The navigation system', 'The user interface')",
            'FUNCTION': "Define the specific action or capability (e.g., 'shall calculate', 'shall display', 'shall process')",
            'PERFORMANCE': "Add measurable criteria (e.g., 'within 2 seconds', '±0.1% tolerance')",
            'CONDITION': "Specify operational state (e.g., 'while in normal operation', 'during startup', 'when receiving input')",
            'TIMING': "Add temporal constraints (e.g., 'within 5 seconds', 'upon system startup', 'every 30 minutes')",
            'CHARACTERISTIC': "Define quality attribute (e.g., 'reliability', 'accuracy', 'availability', 'security')",
            'ENVIRONMENT': "Specify environmental conditions (e.g., 'temperature range -40°C to 85°C', 'humidity 0-95%')"
        }
        for comp in pattern_def['required']:
            if not components.get(comp):
                guidance = component_guidance.get(comp, f"Add {comp}")
                suggestions.append(f"Missing {comp}: {guidance}")
        
        return suggestions
    
    def create_template_recommendation(self, components: Dict, pattern_def: Dict) -> str:
        """Create a filled template recommendation."""
        template = pattern_def['template']
        
        # Fill in found components
        for comp, value in components.items():
            if value:
                template = template.replace(f"{{{comp}}}", value)
        
        # Highlight missing components
        for comp in pattern_def['required']:
            if not components.get(comp):
                template = template.replace(f"{{{comp}}}", f"<MISSING {comp}>")
        
        # Remove optional components
        template = re.sub(r'\[.*?\]', '', template)
        
        return template

class SemanticAnalyzer:
    """Semantic quality analyzer for requirements."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Ambiguous terms
        self.ambiguous_terms = {
            'vague_quantifiers': ['some', 'many', 'few', 'several', 'various'],
            'vague_qualities': ['appropriate', 'adequate', 'suitable', 'proper'],
            'vague_actions': ['handle', 'manage', 'deal', 'support', 'address']
        }
        
        # Subjective terms
        self.subjective_terms = {
            'emotional': ['amazing', 'terrible', 'excellent', 'awful'],
            'subjective': ['good', 'bad', 'best', 'worst', 'better'],
            'uncertainty': ['maybe', 'perhaps', 'possibly', 'might']
        }
    
    def analyze_semantic_quality(self, text: str) -> SemanticAnalysis:
        """Perform semantic analysis on requirement text."""
        doc = self.nlp(text)
        
        # Extract entities
        entity_completeness = self.extract_entities(doc)
        
        # Find ambiguities
        contextual_ambiguities = self.find_contextual_ambiguities(doc)
        
        # Analyze tone
        tone_issues = self.analyze_tone_and_subjectivity(doc)
        
        # Generate suggestions
        suggestions = self.generate_semantic_suggestions(doc)
        
        return SemanticAnalysis(
            similarity_issues=[],  # Will be filled by similarity analysis
            contextual_ambiguities=contextual_ambiguities,
            entity_completeness=entity_completeness,
            tone_issues=tone_issues,
            improvement_suggestions=suggestions
        )
    
    def extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract semantic entities from requirement."""
        entities = {
            'actors': [],
            'actions': [],
            'objects': [],
            'conditions': [],
            'standards': []
        }
        
        # Extract actors (subjects)
        for token in doc:
            if token.dep_ == "nsubj":
                entities['actors'].append(token.text)
        
        # Extract actions (main verbs)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                entities['actions'].append(token.lemma_)
        
        # Extract objects
        for token in doc:
            if token.dep_ in ["dobj", "pobj"]:
                entities['objects'].append(token.text)
        
        # Extract conditions
        for token in doc:
            if token.dep_ == "mark" and token.text.lower() in ["if", "when", "while", "during"]:
                # Get the clause
                clause_tokens = list(token.head.subtree)
                condition = " ".join([t.text for t in clause_tokens])
                entities['conditions'].append(condition)
        
        # Extract standards and compliance references
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(std in ent.text.upper() for std in ["ISO", "IEEE", "ANSI", "FIPS"]):
                entities['standards'].append(ent.text)
        return entities
    
    def find_contextual_ambiguities(self, doc) -> List[str]:
        """Find contextual ambiguities in text."""
        ambiguities = []
        
        for category, terms in self.ambiguous_terms.items():
            for token in doc:
                if token.text.lower() in terms:
                    context = self._get_token_context(token, doc)
                    ambiguities.append(f"{token.text} ({category}): {context}")
        
        return ambiguities
    
    def analyze_tone_and_subjectivity(self, doc) -> List[str]:
        """Detect inappropriate tone and subjective language"""
        issues = []
        
        for category, terms in self.subjective_terms.items():
            found_terms = [token.text for token in doc if token.text.lower() in terms]
            if found_terms:
                if category == "emotional":
                    issues.append(f"Emotional language detected: {found_terms} - use neutral, technical terms")
                elif category == "uncertainty":
                    issues.append(f"Uncertainty markers: {found_terms} - requirements should be definitive")
                elif category == "subjective":
                    issues.append(f"Subjective language: {found_terms} - use objective, measurable terms")
        
        return issues
    
    def generate_semantic_suggestions(self, doc) -> List[str]:
        """Generate improvement suggestions based on semantic analysis."""
        suggestions = []
        
        # Check for missing quantification
        has_numbers = any(token.like_num for token in doc)
        if not has_numbers:
            suggestions.append("Add quantitative criteria for verifiability")
        
        # Check for passive voice
        passive_pattern = r'\b(is|are|was|were|been|being)\s+\w+ed\b'
        if re.search(passive_pattern, doc.text, re.IGNORECASE):
            suggestions.append("Use active voice for clarity")

        # Check for vague action verbs
        vague_verbs = ["handle", "manage", "deal with", "work with", "support"]
        for token in doc:
            if token.lemma_ in vague_verbs:
                suggestions.append(f"Replace vague verb '{token.text}' with specific action (e.g., 'process', 'validate', 'calculate')")
        
        # Check for missing error handling
        if any(verb.lemma_ in ["process", "calculate", "validate"] for verb in doc if verb.pos_ == "VERB"):
            if "error" not in doc.text.lower() and "fail" not in doc.text.lower():
                suggestions.append("Consider adding error handling or failure mode specification")
          
        return suggestions
    
    def _get_token_context(self, token, doc, window=3) -> str:
        """Get context around a token."""
        start = max(0, token.i - window)
        end = min(len(doc), token.i + window + 1)
        context_tokens = doc[start:end]
        return " ".join([t.text for t in context_tokens])
    
    def find_similar_requirements(self, requirements_list: List[str], threshold: float = 0.95) -> List[Dict]:
        """Find potentially duplicate requirements using semantic similarity."""
        if not self.nlp.meta.get('vectors', 0):
            return []
        
        docs = [self.nlp(req) for req in requirements_list if req.strip()]
        similarities = []
        
        for i, doc1 in enumerate(docs):
            for j, doc2 in enumerate(docs[i+1:], i+1):
                try:
                    if doc1.vector_norm > 0 and doc2.vector_norm > 0:
                        similarity = doc1.similarity(doc2)
                        if similarity > threshold:
                            similarities.append({
                                'req1_index': i,
                                'req2_index': j,
                                'similarity': float(similarity),
                                'req1_text': requirements_list[i][:100] + "..." if len(requirements_list[i]) > 100 else requirements_list[i],
                                'req2_text': requirements_list[j][:100] + "..." if len(requirements_list[j]) > 100 else requirements_list[j],
                                'issue': f'Potential duplicate (similarity: {similarity:.2f})'
                            })
                except:
                    continue
        
        return similarities

class EnhancedRequirementAnalyzer:
    """Enhanced requirements quality analyzer with INCOSE patterns and advanced NLP."""
    
    def __init__(self, spacy_model: str = "en_core_web_lg", repo_manager=None):
        """Initialize with spaCy model and analyzers."""
        # Initialize utilities
        self.repo_manager = repo_manager or RepositoryStructureManager()
        self.file_handler = SafeFileHandler()
        self.path_resolver = SmartPathResolver()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"✅ Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"Model {spacy_model} not found, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize specialized analyzers
        self.incose_analyzer = INCOSEPatternAnalyzer(self.nlp)
        self.semantic_analyzer = SemanticAnalyzer(self.nlp)
        
        # Quality criteria
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
            'mandatory': ['shall', 'must', 'will'],
            'optional': ['should', 'may'],
            'forbidden': ['shall not', 'must not', 'will not']
        }
        
        self.passive_indicators = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were|been|being)\s+\w+en\b'
        ]
    
    def analyze_requirement(self, text: str, req_id: str = None) -> Tuple[List[str], QualityMetrics, INCOSEAnalysis, SemanticAnalysis]:
        """Analyze a single requirement with all analysis methods."""
        if pd.isna(text) or not str(text).strip():
            empty_incose = INCOSEAnalysis("", 0, {}, [], [], [], "")
            empty_semantic = SemanticAnalysis([], [], {}, [], [])
            return ["Empty requirement"], QualityMetrics(0, 0, 0, 0, 0, 0, 0, 1, {"critical": 1}), empty_incose, empty_semantic
        
        text = str(text).strip()
        doc = self.nlp(text)
        issues = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # Original quality analysis
        clarity_issues, clarity_score = self._analyze_clarity(doc, text, issues, severity_counts)
        atomicity_issues, atomicity_score = self._analyze_atomicity(doc, issues, severity_counts)
        verifiability_issues, verifiability_score = self._analyze_verifiability(doc, text, issues, severity_counts)
        completeness_issues, completeness_score = self._analyze_completeness(doc, text, issues, severity_counts)
        consistency_score = self._analyze_consistency(doc, issues, severity_counts)
        
        # INCOSE pattern analysis
        incose_analysis = self.incose_analyzer.analyze_incose_compliance(text)
        
        # Semantic analysis
        semantic_analysis = self.semantic_analyzer.analyze_semantic_quality(text)
        
        # Check for implementation details
        self._check_implementation_details(text, issues, severity_counts)

        # Calculate semantic score
        semantic_score = self._calculate_semantic_score(semantic_analysis)
        
        # Create metrics
        metrics = QualityMetrics(
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            verifiability_score=verifiability_score,
            atomicity_score=atomicity_score,
            consistency_score=consistency_score,
            incose_compliance_score=incose_analysis.compliance_score,
            semantic_quality_score=semantic_score,
            total_issues=len(issues),
            severity_breakdown=severity_counts
        )
        
        return issues, metrics, incose_analysis, semantic_analysis
    
    def _analyze_clarity(self, doc, text: str, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Analyze clarity."""
        clarity_issues = 0
        
        # Ambiguous terms
        for token in doc:
            if token.pos_ in {"ADJ", "ADV"}:
                token_lower = token.text.lower()
                for severity, terms in self.ambiguous_terms["clarity"].items():
                    if token_lower in terms:
                        issues.append(f"Clarity ({severity}): ambiguous term '{token.text}'")
                        severity_counts[severity] += 1
                        clarity_issues += 1
        
        # Passive voice
        passive_issues = []
        for pattern in self.passive_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Clarity (medium): Passive voice detected: {', '.join(set(passive_issues))}")
                severity_counts["medium"] += 1
                clarity_issues += 1
                break
        
        # # Sentence complexity
        # sentences = list(doc.sents)
        # for sent in sentences:
        #     if len([t for t in sent if not t.is_punct]) > 25:
        #         issues.append("Clarity (medium): Complex sentence")
        #         severity_counts["medium"] += 1
        #         clarity_issues += 1
        
        # Readability
        readability = self._calculate_readability_score(doc)
        if readability < 30:
            issues.append(f"Clarity (medium): low readability score ({readability:.1f}/100)")
            severity_counts["medium"] += 1
        
        clarity_score = max(0, 100 - (clarity_issues * 20 + len(passive_issues) * 10))
        clarity_score = max(0, 100 - (clarity_issues * 10))
        return clarity_issues, clarity_score
    
    def _analyze_completeness(self, doc, text: str, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Analyze completeness."""
        completeness_issues = 0
        
        # Check for modal verbs
        text_lower = text.lower()
        has_modal = any(modal in text_lower for modals in self.modal_verbs.values() for modal in modals)
        
        if not has_modal:
            issues.append("Completeness (high): Missing modal verb (shall/must/should)")
            severity_counts["high"] += 1
            completeness_issues += 1
        
        # Check for subject
        has_subject = any(token.dep_ == "nsubj" for token in doc)
        if not has_subject:
            issues.append("Completeness (high): Missing subject/actor")
            severity_counts["high"] += 1
            completeness_issues += 1
        
        # Check for action
        has_verb = any(token.pos_ == "VERB" for token in doc)
        if not has_verb:
            issues.append("Completeness (high): Missing action/verb")
            severity_counts["high"] += 1
            completeness_issues += 1
        
        # Length check
        word_count = len([token for token in doc if token.is_alpha])
        if word_count < 5:
            issues.append("Completeness (medium): Too brief")
            severity_counts["medium"] += 1
            completeness_issues += 1
        
        completeness_score = max(0, 100 - (completeness_issues * 15))
        return completeness_issues, completeness_score
    
    def _analyze_verifiability(self, doc, text: str, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Analyze verifiability."""
        verifiability_issues = 0
        
        # Check for measurable entities
        measurable_entities = [ent for ent in doc.ents if ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"}]
        
        # Enhanced patterns for measurable criteria
        verifiability_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|in|ft|°C|°F|K|Hz|rpm|V|A|W|Pa|psi|g|kg|lb)\b',
            r'\b\d+(?:\.\d+)?%\b',
            r'\b\d+(?:\.\d+)?[eE][+-]?\d+\s*[a-zA-Z]+\b',
            r'\b(?:less|more|greater|equal)\s+than\s+\d+(?:\.\d+)?',
            r'\b(?:within|±|plus|minus)\s*\d+(?:\.\d+)?',
            r'\b(?:at\s+least|at\s+most|exactly|up\s+to)\s+\d+',
            r'\b(?:ISO|IEEE|ANSI|ASTM|MIL-STD|DO-\d+|IEC|FIPS)\s*[-]?\s*\d+',
            r'\b(?:accuracy|precision|tolerance|error)\s+(?:of\s+)?[±]?\d+(?:\.\d+)?'
        ]
        
        has_verifiable_criteria = bool(measurable_entities)
        
        if not has_verifiable_criteria:
            text_lower = text.lower()
            for pattern in verifiability_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    has_verifiable_criteria = True
                    break
        
        if not has_verifiable_criteria:
            issues.append("Verifiability (high): no measurable criteria found")
            severity_counts["high"] += 1
            verifiability_issues += 1
        
        verifiability_score = max(0, 100 - (verifiability_issues * 20))
        return verifiability_issues, verifiability_score
    
    def _analyze_atomicity(self, doc, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Analyze atomicity"""
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
        
        # Length analysis
        word_count = len([token for token in doc if token.is_alpha])
        if word_count > 50:
            issues.append("Atomicity (low): requirement may be too long (consider splitting)")
            severity_counts["low"] += 1
        
        atomicity_score = max(0, 100 - (atomicity_issues * 30))
        return atomicity_issues, atomicity_score
        
    def _analyze_consistency(self, doc, issues: List[str], severity_counts: Dict[str, int]) -> float:
        """Analyze consistency."""
        # For now, just check if modal verb is present
        modal_found = False
        for strength, verbs in self.modal_verbs.items():
            for token in doc:
                if token.text.lower() in verbs:
                    modal_found = True
                    break
            if modal_found:
                break
        
        return 100 if modal_found else 80
    
    def _check_implementation_details(self, text: str, issues: List[str], severity_counts: Dict[str, int]):
        """Check for implementation details"""
        implementation_indicators = [
            r'\b(?:implement|deploy|install|configure|setup|utilize|employ)\b',
            r'\busing\s+(?:a|an|the)?\s*[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*',
            r'\b(?:via|through)\s+(?:a|an|the)?\s*[a-zA-Z]+',
            r'\bby\s+means\s+of\b',
            r'\bwith\s+(?:a|an|the)?\s*[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:database|server|protocol))?',
        ]
        
        # Exclude verification language
        verification_patterns = [
            r'\bas\s+(?:verified|validated|demonstrated|tested)\s+(?:through|by|via)',
            r'\bsubject\s+to\s+(?:verification|validation|testing)',
        ]
        
        text_lower = text.lower()
        has_implementation = False
        
        for pattern in implementation_indicators:
            if re.search(pattern, text_lower):
                is_verification = any(re.search(ver_pattern, text_lower) for ver_pattern in verification_patterns)
                if not is_verification:
                    has_implementation = True
                    break
        
        if has_implementation:
            issues.append("Design (high): Contains implementation details")
            severity_counts["high"] += 1

    def _calculate_readability_score(self, doc) -> float:
        """Calculate readability score"""
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
        
        avg_sentence_length = len([token for token in doc if not token.is_punct]) / len(sentences)
        complex_words = sum(1 for token in doc if len(token.text) > 6 and token.is_alpha)
        total_words = sum(1 for token in doc if token.is_alpha)
        complex_ratio = complex_words / max(total_words, 1)
        
        score = max(0, 100 - (avg_sentence_length * 2 + complex_ratio * 50))
        return min(score, 100)
   
    def _calculate_semantic_score(self, semantic_analysis: SemanticAnalysis) -> float:
        """Calculate semantic quality score."""
        base_score = 100
        
        # Penalize issues
        penalty = len(semantic_analysis.contextual_ambiguities) * 10
        penalty += len(semantic_analysis.tone_issues) * 5
        
        # Bonus for good entity completeness
        entity_bonus = 0
        if semantic_analysis.entity_completeness.get('actors'):
            entity_bonus += 5
        if semantic_analysis.entity_completeness.get('actions'):
            entity_bonus += 5
        if semantic_analysis.entity_completeness.get('objects'):
            entity_bonus += 5
        
        return max(0, min(100, base_score - penalty + entity_bonus))
    
    def analyze_file(self, input_file: str = "requirements.csv", 
                    output_file: str = None,
                    requirement_column: str = "Requirement Text",
                    excel_report: bool = True) -> pd.DataFrame:
        """Analyze requirements file with enhanced output."""
        logger.info(f"Starting enhanced analysis of {input_file}")
        
        # Resolve file path
        resolved_paths = self.path_resolver.resolve_input_files({'requirements': input_file})
        input_file_path = resolved_paths['requirements']
        
        if not Path(input_file_path).exists():
            raise FileNotFoundError(f"Could not find requirements file: {input_file}")
        
        # Read file
        df = self.file_handler.safe_read_csv(input_file_path)
        
        if requirement_column not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Column '{requirement_column}' not found. Available: {available_cols}")
            raise ValueError(f"Column '{requirement_column}' not found in CSV")
        
        # Ensure we have required columns per conventions.md
        if 'ID' not in df.columns:
            df['ID'] = [f"REQ_{i:04d}" for i in range(len(df))]
        
        # Get Requirement Name if available
        req_name_col = 'Requirement Name' if 'Requirement Name' in df.columns else None
        
        df = df.fillna({requirement_column: ""})
        logger.info(f"Analyzing {len(df)} requirements...")
        
        # Analyze each requirement
        all_analysis_results = []
        
        for idx, row in df.iterrows():
            req_id = row['ID']
            req_text = row[requirement_column]
            req_name = row[req_name_col] if req_name_col else ""
            
            issues, metrics, incose_analysis, semantic_analysis = self.analyze_requirement(req_text, req_id)
            
            # Build comprehensive result
            result = {
                # Core columns
                'ID': req_id,
                'Requirement Name': req_name,
                'Requirement Text': req_text,
                
                # Quality scores
                'Quality_Score': metrics.clarity_score * 0.2 + metrics.completeness_score * 0.2 + 
                                metrics.verifiability_score * 0.2 + metrics.atomicity_score * 0.2 + 
                                metrics.consistency_score * 0.2,
                'Quality_Grade': self._get_grade(metrics.clarity_score * 0.2 + metrics.completeness_score * 0.2 + 
                                                metrics.verifiability_score * 0.2 + metrics.atomicity_score * 0.2 + 
                                                metrics.consistency_score * 0.2),
                
                # Quality breakdown
                'Clarity_Score': metrics.clarity_score,
                'Completeness_Score': metrics.completeness_score,
                'Verifiability_Score': metrics.verifiability_score,
                'Atomicity_Score': metrics.atomicity_score,
                'Consistency_Score': metrics.consistency_score,
                
                # Issues
                'Total_Issues': metrics.total_issues,
                'Critical_Issues': metrics.severity_breakdown['critical'],
                'High_Issues': metrics.severity_breakdown['high'],
                'Medium_Issues': metrics.severity_breakdown['medium'],
                'Low_Issues': metrics.severity_breakdown['low'],
                'Issue_Details': '; '.join(issues),
                
                # INCOSE analysis
                'INCOSE_Compliance_Score': incose_analysis.compliance_score,
                'INCOSE_Best_Pattern': incose_analysis.best_pattern,
                'INCOSE_Missing_Required': ', '.join(incose_analysis.missing_required) if incose_analysis.missing_required else 'None',
                'INCOSE_Missing_Optional': ', '.join(incose_analysis.missing_optional) if incose_analysis.missing_optional else 'None',
                'INCOSE_Template': incose_analysis.template_recommendation,
                'INCOSE_Suggestions': '; '.join(incose_analysis.suggestions),
                
                # INCOSE Components found
                'Has_Agent': 'Yes' if incose_analysis.components_found.get('AGENT') else 'No',
                'Has_Function': 'Yes' if incose_analysis.components_found.get('FUNCTION') else 'No',
                'Has_Performance': 'Yes' if incose_analysis.components_found.get('PERFORMANCE') else 'No',
                'Has_Condition': 'Yes' if incose_analysis.components_found.get('CONDITION') else 'No',
                
                # Semantic analysis
                'Semantic_Quality_Score': metrics.semantic_quality_score,
                'Actors_Found': ', '.join(semantic_analysis.entity_completeness.get('actors', [])) if semantic_analysis.entity_completeness.get('actors') else 'None',
                'Actions_Found': ', '.join(semantic_analysis.entity_completeness.get('actions', [])) if semantic_analysis.entity_completeness.get('actions') else 'None',
                'Objects_Found': ', '.join(semantic_analysis.entity_completeness.get('objects', [])) if semantic_analysis.entity_completeness.get('objects') else 'None',
                'Conditions_Found': ', '.join(semantic_analysis.entity_completeness.get('conditions', [])) if semantic_analysis.entity_completeness.get('conditions') else 'None',
                
                'Ambiguous_Terms': '; '.join(semantic_analysis.contextual_ambiguities) if semantic_analysis.contextual_ambiguities else 'None',
                'Tone_Issues': '; '.join(semantic_analysis.tone_issues) if semantic_analysis.tone_issues else 'None',
                'Semantic_Suggestions': '; '.join(semantic_analysis.improvement_suggestions) if semantic_analysis.improvement_suggestions else 'None',
                
                # Will be filled after similarity analysis
                'Similar_Requirements': 0,
                'Most_Similar_ID': 'None',
                'Max_Similarity': 0.0,
                'Duplicate_Group': 'UNIQUE'
            }
            
            all_analysis_results.append(result)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} requirements")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_analysis_results)
        
        # Find similar requirements
        logger.info("Finding similar requirements...")
        req_texts = df[requirement_column].tolist()
        similarity_results = self.semantic_analyzer.find_similar_requirements(req_texts)
        
        # Update similarity information
        similarity_map = {}
        for sim_result in similarity_results:
            idx1, idx2 = sim_result['req1_index'], sim_result['req2_index']
            
            # Map to IDs
            id1 = results_df.iloc[idx1]['ID']
            id2 = results_df.iloc[idx2]['ID']
            
            if id1 not in similarity_map:
                similarity_map[id1] = []
            if id2 not in similarity_map:
                similarity_map[id2] = []
            
            similarity_map[id1].append((id2, sim_result['similarity']))
            similarity_map[id2].append((id1, sim_result['similarity']))
        
        # Update similarity columns
        for idx, row in results_df.iterrows():
            req_id = row['ID']
            if req_id in similarity_map:
                similar_reqs = similarity_map[req_id]
                results_df.at[idx, 'Similar_Requirements'] = len(similar_reqs)
                if similar_reqs:
                    most_similar = max(similar_reqs, key=lambda x: x[1])
                    results_df.at[idx, 'Most_Similar_ID'] = most_similar[0]
                    results_df.at[idx, 'Max_Similarity'] = most_similar[1]
        
        # Add duplicate groups
        results_df = self._add_duplicate_groups(results_df, similarity_map)
        
        # Save CSV results
        if output_file is None:
            output_file = Path(input_file_path).stem + "_quality_analysis"
        
        csv_path = Path(output_file).with_suffix('.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to {csv_path}")
        
        # Create Excel report if requested
        if excel_report:
            excel_path = self._create_excel_report(results_df, output_file)
            logger.info(f"Created Excel report: {excel_path}")
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _save_enhanced_results(self, df: pd.DataFrame, input_file_path: str, 
                              output_file: str, excel_report: bool) -> str:
        """Save enhanced analysis results"""
        
        # Determine output file path
        if not output_file:
            input_stem = Path(input_file_path).stem
            output_file = self.file_handler.get_structured_path(
                'quality_analysis', 
                f"{input_stem}_enhanced_quality_report.csv"
            )
        
        # Save CSV
        try:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"Enhanced CSV analysis saved to '{output_file}'")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise
        
        # Save enhanced summary
        summary_report = self._generate_enhanced_summary(df)
        summary_file = str(Path(output_file).with_suffix("")) + "_enhanced_summary.json"
        
        try:
            with open(summary_file, "w", encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Enhanced summary saved to '{summary_file}'")
        except Exception as e:
            logger.warning(f"Could not save enhanced summary: {e}")
        
        # Create enhanced Excel report
        if excel_report:
            try:
                excel_path = self._create_enhanced_excel_report(df)
                logger.info(f"Enhanced Excel report created: {excel_path}")
            except Exception as e:
                logger.error(f"Failed to create enhanced Excel report: {e}")
        
        return output_file
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade."""
        if score >= 90:
            return 'EXCELLENT'
        elif score >= 75:
            return 'GOOD'
        elif score >= 60:
            return 'FAIR'
        elif score >= 40:
            return 'POOR'
        else:
            return 'CRITICAL'
    
    def _add_duplicate_groups(self, df: pd.DataFrame, similarity_map: Dict[str, List[Tuple[str, float]]]) -> pd.DataFrame:
        """Add duplicate group IDs."""
        groups = {}
        group_id = 1
        processed = set()
        
        for req_id, similar_reqs in similarity_map.items():
            if req_id in processed:
                continue
            
            # Find all requirements with >95% similarity
            group_members = {req_id}
            for sim_id, sim_score in similar_reqs:
                if sim_score >= 0.95:
                    group_members.add(sim_id)
            
            if len(group_members) > 1:
                for member in group_members:
                    groups[member] = f"DUP_GROUP_{group_id:03d}"
                    processed.add(member)
                group_id += 1
        
        # Map to DataFrame
        df['Duplicate_Group'] = df['ID'].map(groups).fillna('UNIQUE')
        
        return df
    
    def _create_excel_report(self, df: pd.DataFrame, base_filename: str) -> Path:
        """Create comprehensive 4-tab Excel report."""
        output_file = self.file_handler.get_structured_path(
            'quality_analysis', 
            "requirements_quality_report.xlsx"
        )
        
        excel_path = Path(output_file)
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Tab 1: Summary
            summary_df = self._create_summary_tab(df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Tab 2: Quality Grading
            quality_cols = [
                'ID', 'Requirement Name', 'Requirement Text',
                'Quality_Score', 'Quality_Grade',
                'Clarity_Score', 'Completeness_Score', 'Verifiability_Score', 
                'Atomicity_Score', 'Consistency_Score',
                'Total_Issues', 'Critical_Issues', 'High_Issues', 'Medium_Issues', 'Low_Issues',
                'Issue_Details'
            ]
            quality_df = df[quality_cols]
            quality_df.to_excel(writer, sheet_name='Quality Grading', index=False)
            
            # Tab 3: INCOSE Analysis
            incose_cols = [
                'ID', 'Requirement Name', 'Requirement Text',
                'INCOSE_Compliance_Score', 'INCOSE_Best_Pattern',
                'Has_Agent', 'Has_Function', 'Has_Performance', 'Has_Condition',
                'INCOSE_Missing_Required', 'INCOSE_Missing_Optional', 'INCOSE_Suggestions'
            ]
            incose_df = df[incose_cols]
            incose_df.to_excel(writer, sheet_name='INCOSE Analysis', index=False)
            
            # Tab 4: Semantic Analysis
            semantic_cols = [
                'ID', 'Requirement Name', 'Requirement Text',
                'Semantic_Quality_Score',
                'Actors_Found', 'Actions_Found', 'Objects_Found', 'Conditions_Found',
                'Ambiguous_Terms', 'Tone_Issues',
                'Similar_Requirements', 'Most_Similar_ID', 'Max_Similarity', 'Duplicate_Group',
                'Semantic_Suggestions'
            ]
            semantic_df = df[semantic_cols]
            semantic_df.to_excel(writer, sheet_name='Semantic Analysis', index=False)
            
            # Format worksheets
            workbook = writer.book
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                
                # Format headers
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                    cell.font = Font(color="FFFFFF", bold=True)
                
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
        
        return excel_path
    
    def _create_summary_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive summary tab with methodology."""
        summary_data = []
        
        # Overall Statistics
        summary_data.extend([
            {'Section': 'OVERALL STATISTICS', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Total Requirements', 'Value': len(df), 'Details': ''},
            {'Section': '', 'Metric': 'Average Quality Score', 'Value': f"{df['Quality_Score'].mean():.1f}/100", 'Details': ''},
            {'Section': '', 'Metric': 'Average INCOSE Compliance', 'Value': f"{df['INCOSE_Compliance_Score'].mean():.1f}%", 'Details': ''},
            {'Section': '', 'Metric': 'Average Semantic Score', 'Value': f"{df['Semantic_Quality_Score'].mean():.1f}/100", 'Details': ''},
            {'Section': '', 'Metric': '', 'Value': '', 'Details': ''},
        ])
        
        # Grade Distribution
        summary_data.append({'Section': 'GRADE DISTRIBUTION', 'Metric': '', 'Value': '', 'Details': ''})
        grades = df['Quality_Grade'].value_counts()
        for grade in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']:
            count = grades.get(grade, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            summary_data.append({
                'Section': '',
                'Metric': f'{grade} Requirements',
                'Value': f'{count} ({pct:.1f}%)',
                'Details': self._get_grade_description(grade)
            })
        summary_data.append({'Section': '', 'Metric': '', 'Value': '', 'Details': ''})
        
        # Top Issues
        summary_data.append({'Section': 'TOP ISSUES', 'Metric': '', 'Value': '', 'Details': ''})
        all_issues = []
        for issues_text in df['Issue_Details']:
            if issues_text and str(issues_text) != 'nan':
                issues = issues_text.split(';')
                for issue in issues:
                    if ':' in issue:
                        issue_type = issue.split(':')[0].strip()
                        all_issues.append(issue_type)
        
        if all_issues:
            issue_counts = Counter(all_issues)
            for issue_type, count in issue_counts.most_common(5):
                summary_data.append({
                    'Section': '',
                    'Metric': issue_type,
                    'Value': count,
                    'Details': ''
                })
        summary_data.append({'Section': '', 'Metric': '', 'Value': '', 'Details': ''})
        
        # Duplicates
        dup_count = len(df[df['Duplicate_Group'] != 'UNIQUE'])
        summary_data.extend([
            {'Section': 'DUPLICATE ANALYSIS', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Potential Duplicates', 'Value': dup_count, 'Details': 'Requirements with >95% similarity'},
            {'Section': '', 'Metric': 'Duplicate Groups', 'Value': df[df['Duplicate_Group'] != 'UNIQUE']['Duplicate_Group'].nunique() if dup_count > 0 else 0, 'Details': ''},
            {'Section': '', 'Metric': '', 'Value': '', 'Details': ''},
        ])
        
        # Methodology
        summary_data.extend([
            {'Section': 'SCORING METHODOLOGY', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Quality Score Calculation', 'Value': '', 'Details': 'Average of 5 dimensions (Clarity, Completeness, Verifiability, Atomicity, Consistency)'},
            {'Section': '', 'Metric': 'Issue Severity Weights', 'Value': '', 'Details': 'Critical: -10, High: -5, Medium: -2, Low: -1 points'},
            {'Section': '', 'Metric': '', 'Value': '', 'Details': ''},
        ])
        
        # Quality Dimensions
        summary_data.extend([
            {'Section': 'QUALITY DIMENSIONS', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Clarity', 'Value': '', 'Details': 'Unambiguous language, active voice, reasonable length'},
            {'Section': '', 'Metric': 'Completeness', 'Value': '', 'Details': 'Has actor, action, modal verb (shall/must/should)'},
            {'Section': '', 'Metric': 'Verifiability', 'Value': '', 'Details': 'Measurable, testable, has acceptance criteria'},
            {'Section': '', 'Metric': 'Atomicity', 'Value': '', 'Details': 'Single requirement, not multiple bundled'},
            {'Section': '', 'Metric': 'Consistency', 'Value': '', 'Details': 'Uses consistent terminology, appropriate modal verbs'},
            {'Section': '', 'Metric': '', 'Value': '', 'Details': ''},
        ])
        
        # INCOSE Patterns
        summary_data.extend([
            {'Section': 'INCOSE PATTERNS', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Functional/Performance', 'Value': '', 'Details': 'The {AGENT} shall {FUNCTION} in accordance with {INTERFACE_OUTPUT} with {PERFORMANCE} [and {TIMING} upon {EVENT_TRIGGER}] while in {CONDITION}'},
            {'Section': '', 'Metric': 'Suitability', 'Value': '', 'Details': 'The {AGENT} shall exhibit {CHARACTERISTIC} with {PERFORMANCE} while {CONDITION} [for {CONDITION_DURATION}]'},
            {'Section': '', 'Metric': 'Environmental', 'Value': '', 'Details': 'The {AGENT} shall exhibit {CHARACTERISTIC} during/after exposure to {ENVIRONMENT} [for {EXPOSURE_DURATION}]'},
            {'Section': '', 'Metric': 'Design Constraint', 'Value': '', 'Details': 'The {AGENT} shall exhibit {DESIGN_CONSTRAINTS} [in accordance with {PERFORMANCE} while in {CONDITION}]'},
            {'Section': '', 'Metric': '', 'Value': '', 'Details': ''},
        ])
        
        # Semantic Checks
        summary_data.extend([
            {'Section': 'SEMANTIC ANALYSIS', 'Metric': '', 'Value': '', 'Details': ''},
            {'Section': '', 'Metric': 'Ambiguous Terms', 'Value': '', 'Details': 'Words like: appropriate, various, several, many, some'},
            {'Section': '', 'Metric': 'Entity Extraction', 'Value': '', 'Details': 'Identifies actors (who), actions (what), objects (on what)'},
            {'Section': '', 'Metric': 'Implementation Details', 'Value': '', 'Details': 'Detects "how" vs "what": using, via, through + technology'},
            {'Section': '', 'Metric': 'Similarity Threshold', 'Value': '95%', 'Details': 'For duplicate detection'},
        ])
        
        return pd.DataFrame(summary_data)
    
    def _get_grade_description(self, grade: str) -> str:
        """Get description for grade."""
        descriptions = {
            'EXCELLENT': 'Professional quality, ready for implementation',
            'GOOD': 'Minor issues only, acceptable for most uses',
            'FAIR': 'Several issues, needs improvement',
            'POOR': 'Significant issues, requires major revision',
            'CRITICAL': 'Severe issues, needs complete rewrite'
        }
        return descriptions.get(grade, '')
    
    def _print_summary(self, df: pd.DataFrame):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("REQUIREMENTS QUALITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Requirements: {len(df)}")
        print(f"  Average Quality Score: {df['Quality_Score'].mean():.1f}/100")
        print(f"  Average INCOSE Compliance: {df['INCOSE_Compliance_Score'].mean():.1f}%")
        print(f"  Average Semantic Score: {df['Semantic_Quality_Score'].mean():.1f}/100")
        
        print(f"\n🎯 Grade Distribution:")
        grades = df['Quality_Grade'].value_counts()
        for grade in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']:
            count = grades.get(grade, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"  {grade}: {count} ({pct:.1f}%)")
        
        print(f"\n📋 Key Findings:")
        print(f"  Requirements with issues: {len(df[df['Total_Issues'] > 0])}")
        print(f"  Missing actors: {len(df[df['Actors_Found'] == 'None'])}")
        print(f"  Missing modal verbs: {len(df[df['Has_Agent'] == 'No'])}")
        print(f"  Potential duplicates: {len(df[df['Duplicate_Group'] != 'UNIQUE'])}")
        
        print("="*70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Requirements Quality Analyzer with 4-Tab Excel Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reqGrading.py                    # Analyze requirements.csv
  python reqGrading.py -i my_reqs.csv    # Analyze custom file
  python reqGrading.py -c "Requirement"   # Use different column name

Output:
  • CSV file with all analysis results
  • Excel workbook with 4 tabs:
    - Summary: Overview and methodology
    - Quality Grading: Traditional quality metrics
    - INCOSE Analysis: Pattern compliance
    - Semantic Analysis: Entities and similarity
        """
    )
    
    parser.add_argument("-i", "--input", dest="input_file", default="requirements.csv",
                        help="Input CSV file with requirements")
    parser.add_argument("-o", "--output", dest="output_file", default=None,
                        help="Output filename (without extension)")
    parser.add_argument("-c", "--column", dest="requirement_column", default="Requirement Text",
                        help="Column name containing requirement text")
    parser.add_argument("-m", "--model", dest="spacy_model", default="en_core_web_lg",
                        help="spaCy model to use")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("="*70)
    print("🚀 ENHANCED REQUIREMENTS QUALITY ANALYZER v2.0")
    print("="*70)
    
    # Create analyzer
    analyzer = EnhancedRequirementAnalyzer(spacy_model=args.spacy_model)
    
    # Run analysis
    try:
        results_df = analyzer.analyze_file(
            input_file=args.input_file,
            output_file=args.output_file,
            requirement_column=args.requirement_column,
            excel_report=True
        )
        
        print(f"\n✅ Analysis complete!")
        output_name = args.output_file or Path(args.input_file).stem + "_quality_analysis"
        print(f"📁 CSV results: {output_name}.csv")
        print(f"📊 Excel report: {output_name}.xlsx")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()