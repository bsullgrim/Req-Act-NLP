"""
Enhanced Requirements Quality Analyzer v2.0
Advanced NLP with INCOSE pattern analysis and semantic features
"""

import pandas as pd
import logging
import spacy
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json
import sys
import os

# Add project root to path for proper imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import your existing utils
try:
    from src.utils.file_utils import SafeFileHandler
    from src.utils.path_resolver import SmartPathResolver
    from src.utils.repository_setup import RepositoryStructureManager
    UTILS_AVAILABLE = True
    print("✅ Successfully imported project utils")
except ImportError as e:
    print(f"⚠️ Could not import utils: {e}")
    UTILS_AVAILABLE = False
    sys.exit(1)

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Enhanced container for requirement quality metrics"""
    clarity_score: float
    completeness_score: float
    verifiability_score: float
    atomicity_score: float
    consistency_score: float
    incose_compliance_score: float  # New: INCOSE pattern compliance
    semantic_quality_score: float   # New: Semantic analysis score
    total_issues: int
    severity_breakdown: Dict[str, int]

@dataclass
class INCOSEAnalysis:
    """INCOSE pattern analysis results"""
    best_pattern: str
    compliance_score: float
    components_found: Dict[str, Optional[str]]
    missing_required: List[str]
    missing_optional: List[str]
    suggestions: List[str]
    template_recommendation: str

@dataclass
class SemanticAnalysis:
    """Semantic analysis results"""
    similarity_issues: List[Dict]
    contextual_ambiguities: List[str]
    entity_completeness: Dict[str, List[str]]
    tone_issues: List[str]
    improvement_suggestions: List[str]

class INCOSEPatternAnalyzer:
    """INCOSE requirements pattern analyzer"""
    
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
        """Extract INCOSE requirement components using advanced NLP"""
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
        """Analyze requirement against INCOSE patterns"""
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
        """Score how well components match INCOSE pattern"""
        required_found = sum(1 for comp in pattern_def['required'] if components.get(comp))
        optional_found = sum(1 for comp in pattern_def['optional'] if components.get(comp))
        
        required_score = (required_found / len(pattern_def['required'])) * 80
        optional_bonus = (optional_found / len(pattern_def['optional'])) * 20
        
        return min(100, required_score + optional_bonus)
    
    def generate_pattern_suggestions(self, components: Dict, pattern_def: Dict) -> List[str]:
        """Generate improvement suggestions based on missing components"""
        suggestions = []
        
        component_guidance = {
            'AGENT': "Specify the system, subsystem, or component responsible (e.g., 'The navigation system', 'The user interface')",
            'FUNCTION': "Define the specific action or capability (e.g., 'shall calculate', 'shall display', 'shall process')",
            'PERFORMANCE': "Add measurable criteria (e.g., 'within 2 seconds', 'with 99% accuracy', '±0.1% tolerance')",
            'CONDITION': "Specify operational state (e.g., 'while in normal operation', 'during startup', 'when receiving input')",
            'TIMING': "Add temporal constraints (e.g., 'within 5 seconds', 'upon system startup', 'every 30 minutes')",
            'CHARACTERISTIC': "Define quality attribute (e.g., 'reliability', 'accuracy', 'availability', 'security')",
            'ENVIRONMENT': "Specify environmental conditions (e.g., 'temperature range -40°C to 85°C', 'humidity 0-95%')"
        }
        
        for missing_comp in pattern_def['required']:
            if not components.get(missing_comp):
                suggestions.append(f"Add {missing_comp}: {component_guidance.get(missing_comp, f'Define {missing_comp} component')}")
        
        return suggestions
    
    def create_template_recommendation(self, components: Dict, pattern_def: Dict) -> str:
        """Create INCOSE-compliant template recommendation"""
        template = pattern_def['template']
        
        # Fill in known components
        filled_template = template
        for comp_name, comp_value in components.items():
            if comp_value:
                placeholder = f"{{{comp_name}}}"
                filled_template = filled_template.replace(placeholder, comp_value)
        
        return filled_template

class SemanticAnalyzer:
    """Advanced semantic analysis using spaCy vectors"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.similarity_threshold = 0.85
        self.subjective_terms = {
            "emotional": ["love", "hate", "amazing", "terrible", "beautiful", "ugly", "awesome", "horrible"],
            "uncertainty": ["maybe", "perhaps", "possibly", "probably", "likely", "might be"],
            "subjective": ["obviously", "clearly", "naturally", "of course", "simply"]
        }
    
    def analyze_semantic_quality(self, text: str) -> SemanticAnalysis:
        """Comprehensive semantic analysis"""
        doc = self.nlp(text)
        
        return SemanticAnalysis(
            similarity_issues=[],  # Will be filled by batch analysis
            contextual_ambiguities=self.analyze_contextual_ambiguity(doc),
            entity_completeness=self.analyze_entity_completeness(doc),
            tone_issues=self.analyze_tone_and_subjectivity(doc),
            improvement_suggestions=self.generate_semantic_suggestions(doc)
        )
    
    def analyze_contextual_ambiguity(self, doc) -> List[str]:
        """Find ambiguous terms with context analysis"""
        ambiguities = []
        ambiguous_base_terms = ["appropriate", "reasonable", "adequate", "sufficient", "good", "bad", "fast", "slow"]
        
        for token in doc:
            if token.text.lower() in ambiguous_base_terms:
                context_info = self.get_token_context(token)
                
                if context_info['modifying']:
                    ambiguities.append(
                        f"'{token.text}' modifying '{context_info['modifying']}' - "
                        f"specify {self.suggest_specific_criteria(token.text.lower(), context_info['modifying'])}"
                    )
                
                # Check for missing quantification
                if not context_info['has_numbers']:
                    ambiguities.append(
                        f"'{token.text}' lacks quantitative criteria - add measurable thresholds"
                    )
        
        return ambiguities
    
    def get_token_context(self, token) -> Dict:
        """Analyze the context around a token"""
        context = {
            'modifying': None,
            'has_numbers': False,
            'nearby_units': [],
            'sentence_numbers': []
        }
        
        # What is this token modifying?
        if token.head and token.head.pos_ == "NOUN":
            context['modifying'] = token.head.text
        
        # Are there numbers in the sentence?
        for sent_token in token.sent:
            if sent_token.like_num or sent_token.is_digit:
                context['has_numbers'] = True
                context['sentence_numbers'].append(sent_token.text)
        
        return context
    
    def suggest_specific_criteria(self, ambiguous_term: str, modified_noun: str) -> str:
        """Suggest specific criteria based on context"""
        suggestions = {
            ("appropriate", "time"): "specific time limit (e.g., '< 2 seconds')",
            ("appropriate", "response"): "response time threshold (e.g., 'within 500ms')",
            ("good", "performance"): "performance metrics (e.g., 'throughput ≥ 1000 ops/sec')",
            ("reasonable", "accuracy"): "accuracy percentage (e.g., '≥ 95% accuracy')",
            ("adequate", "security"): "security standards (e.g., 'AES-256 encryption')",
            ("sufficient", "memory"): "memory requirements (e.g., '≥ 4GB RAM')",
            ("fast", "processing"): "processing speed (e.g., '< 100ms processing time')",
        }
        
        key = (ambiguous_term, modified_noun.lower() if modified_noun else "")
        return suggestions.get(key, f"measurable criteria for {modified_noun or 'this attribute'}")
    
    def analyze_entity_completeness(self, doc) -> Dict[str, List[str]]:
        """Extract and categorize requirement entities"""
        entities = {
            'actors': [],
            'actions': [],
            'objects': [],
            'conditions': [],
            'constraints': [],
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
        
        # Extract standards and compliance references
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(std in ent.text.upper() for std in ["ISO", "IEEE", "ANSI", "FIPS"]):
                entities['standards'].append(ent.text)
        
        return entities
    
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
        """Generate improvement suggestions based on semantic analysis"""
        suggestions = []
        
        # Check for missing quantification
        has_numbers = any(token.like_num for token in doc)
        if not has_numbers:
            suggestions.append("Add quantitative criteria to make requirement verifiable")
        
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
    
    def find_similar_requirements(self, requirements_list: List[str], threshold: float = 0.85) -> List[Dict]:
        """Find potentially duplicate requirements using semantic similarity"""
        if not self.nlp.meta.get('vectors', 0):
            return []  # No vectors available
        
        docs = [self.nlp(req) for req in requirements_list]
        similarities = []
        
        for i, doc1 in enumerate(docs):
            for j, doc2 in enumerate(docs[i+1:], i+1):
                try:
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
                    continue  # Skip if similarity calculation fails
        
        return similarities

class EnhancedRequirementAnalyzer:
    """Enhanced requirements quality analyzer with INCOSE patterns and advanced NLP"""
    
    def __init__(self, spacy_model: str = "en_core_web_trf", repo_manager=None):
        # Initialize spaCy with fallback
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Trying fallback models...")
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
                raise OSError("No suitable spaCy model found")
        
        # Setup utils
        if repo_manager is None:
            self.repo_manager = RepositoryStructureManager("outputs")
            self.repo_manager.setup_repository_structure()
        else:
            self.repo_manager = repo_manager
        
        self.file_handler = SafeFileHandler(repo_manager=self.repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager=self.repo_manager)
        
        # Initialize analyzers
        self.incose_analyzer = INCOSEPatternAnalyzer(self.nlp)
        self.semantic_analyzer = SemanticAnalyzer(self.nlp)
        
        # Enhanced term definitions
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
        
        self.passive_indicators = [
            r'\b(is|are|was|were|being|been)\s+\w+ed\b',
            r'\b(is|are|was|were)\s+\w+en\b'
        ]
        
        logger.info("✅ Enhanced analyzer initialized with INCOSE patterns and semantic analysis")
    
    def analyze_requirement(self, text: str, req_id: Optional[str] = None) -> Tuple[List[str], QualityMetrics, INCOSEAnalysis, SemanticAnalysis]:
        """Enhanced requirement analysis with INCOSE patterns and semantic analysis"""
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
        
        # Add INCOSE-specific issues
        if incose_analysis.compliance_score < 70:
            issues.append(f"INCOSE Structure (medium): Best pattern '{incose_analysis.best_pattern}' only {incose_analysis.compliance_score:.0f}% complete")
            severity_counts["medium"] += 1
        
        for missing in incose_analysis.missing_required:
            issues.append(f"INCOSE Completeness (high): Missing required {missing} component")
            severity_counts["high"] += 1
        
        # Semantic analysis
        semantic_analysis = self.semantic_analyzer.analyze_semantic_quality(text)
        
        # Add semantic issues
        for ambiguity in semantic_analysis.contextual_ambiguities:
            issues.append(f"Semantic Clarity (high): {ambiguity}")
            severity_counts["high"] += 1
        
        for tone_issue in semantic_analysis.tone_issues:
            issues.append(f"Tone (medium): {tone_issue}")
            severity_counts["medium"] += 1
        
        # Calculate semantic quality score
        semantic_score = self._calculate_semantic_score(semantic_analysis)
        
        # Check for implementation details
        self._check_implementation_details(text, issues, severity_counts)
        
        # Enhanced metrics with new dimensions
        metrics = QualityMetrics(
            clarity_score=clarity_score,
            completeness_score=max(completeness_score, incose_analysis.compliance_score * 0.7),  # Boost with INCOSE
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
        """Analyze clarity with enhanced detection"""
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
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                passive_issues.extend(matches)
        
        if passive_issues:
            issues.append(f"Clarity (medium): Passive voice detected: {', '.join(set(passive_issues))}")
            severity_counts["medium"] += 1
        
        # Readability
        readability = self._calculate_readability_score(doc)
        if readability < 30:
            issues.append(f"Clarity (medium): low readability score ({readability:.1f}/100)")
            severity_counts["medium"] += 1
        
        clarity_score = max(0, 100 - (clarity_issues * 20 + len(passive_issues) * 10))
        return clarity_issues, clarity_score
    
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
    
    def _analyze_verifiability(self, doc, text: str, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Enhanced verifiability analysis"""
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
        
        verifiability_score = max(0, 100 - (verifiability_issues * 30))
        return verifiability_issues, verifiability_score
    
    def _analyze_completeness(self, doc, text: str, issues: List[str], severity_counts: Dict[str, int]) -> Tuple[int, float]:
        """Analyze completeness"""
        completeness_issues = 0
        
        # Check for structural components
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
        
        # Modal verb analysis
        modal_found = None
        for strength, verbs in self.modal_verbs.items():
            for token in doc:
                if token.text.lower() in verbs:
                    modal_found = token.text.lower()
                    break
            if modal_found:
                break
        
        if not modal_found:
            issues.append("Completeness (medium): no modal verb indicating requirement strength")
            severity_counts["medium"] += 1
            completeness_issues += 1
        
        # Length check
        word_count = len([token for token in doc if token.is_alpha])
        if word_count < 5:
            issues.append("Completeness (medium): requirement may be too short")
            severity_counts["medium"] += 1
        
        completeness_score = max(0, 100 - (completeness_issues * 25))
        return completeness_issues, completeness_score
    
    def _analyze_consistency(self, doc, issues: List[str], severity_counts: Dict[str, int]) -> float:
        """Analyze consistency"""
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
        """Calculate semantic quality score"""
        base_score = 100
        
        # Penalize issues
        penalty = len(semantic_analysis.contextual_ambiguities) * 15
        penalty += len(semantic_analysis.tone_issues) * 10
        
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
                    excel_report: bool = False) -> pd.DataFrame:
        """Enhanced file analysis with INCOSE and semantic features"""
        logger.info(f"Starting enhanced analysis of {input_file}")
        
        # Resolve file path
        print(f"🔍 Resolving file path for: {input_file}")
        resolved_paths = self.path_resolver.resolve_input_files({'requirements': input_file})
        input_file_path = resolved_paths['requirements']
        
        if not Path(input_file_path).exists():
            raise FileNotFoundError(f"Could not find requirements file: {input_file}")
        
        print(f"✅ Found requirements file: {input_file_path}")
        
        # Read file
        df = self.file_handler.safe_read_csv(input_file_path)
        
        if requirement_column not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Column '{requirement_column}' not found. Available columns: {available_cols}")
            raise ValueError(f"Column '{requirement_column}' not found in CSV")
        
        df = df.fillna({requirement_column: ""})
        logger.info(f"Analyzing {len(df)} requirements with enhanced NLP...")
        
        # Analyze each requirement
        analysis_results = []
        metrics_list = []
        incose_results = []
        semantic_results = []
        
        requirements_text = df[requirement_column].tolist()
        
        for idx, requirement in enumerate(requirements_text):
            req_id = df.get("ID", pd.Series([f"REQ_{idx:04d}"] * len(df))).iloc[idx]
            issues, metrics, incose_analysis, semantic_analysis = self.analyze_requirement(requirement, str(req_id))
            
            analysis_results.append(issues)
            metrics_list.append(metrics)
            incose_results.append(incose_analysis)
            semantic_results.append(semantic_analysis)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} requirements")
        
        # Find similar requirements (batch analysis)
        print("🔍 Analyzing requirement similarities...")
        similarity_results = self.semantic_analyzer.find_similar_requirements(requirements_text)
        
        # Add similarity issues to semantic results
        for sim_result in similarity_results:
            idx1, idx2 = sim_result['req1_index'], sim_result['req2_index']
            semantic_results[idx1].similarity_issues.append(sim_result)
            semantic_results[idx2].similarity_issues.append(sim_result)
        
        # Build enhanced DataFrame
        df = self._build_enhanced_dataframe(df, analysis_results, metrics_list, incose_results, semantic_results)
        
        # Generate reports
        output_file = self._save_enhanced_results(df, input_file_path, output_file, excel_report)
        
        # Print enhanced summary
        self._print_enhanced_summary(df, similarity_results)
        
        return df
    
    def _build_enhanced_dataframe(self, df: pd.DataFrame, analysis_results: List, 
                                 metrics_list: List, incose_results: List, 
                                 semantic_results: List) -> pd.DataFrame:
        """Build enhanced DataFrame with all analysis results"""
        
        # Add basic analysis results
        df["Issues"] = analysis_results
        df["Total_Issues"] = [len(issues) for issues in analysis_results]
        
        # Add quality scores
        df["Clarity_Score"] = [m.clarity_score for m in metrics_list]
        df["Completeness_Score"] = [m.completeness_score for m in metrics_list]
        df["Verifiability_Score"] = [m.verifiability_score for m in metrics_list]
        df["Atomicity_Score"] = [m.atomicity_score for m in metrics_list]
        df["Consistency_Score"] = [m.consistency_score for m in metrics_list]
        df["INCOSE_Compliance_Score"] = [m.incose_compliance_score for m in metrics_list]
        df["Semantic_Quality_Score"] = [m.semantic_quality_score for m in metrics_list]
        
        # Add severity breakdowns
        df["Critical_Issues"] = [m.severity_breakdown.get("critical", 0) for m in metrics_list]
        df["High_Issues"] = [m.severity_breakdown.get("high", 0) for m in metrics_list]
        df["Medium_Issues"] = [m.severity_breakdown.get("medium", 0) for m in metrics_list]
        df["Low_Issues"] = [m.severity_breakdown.get("low", 0) for m in metrics_list]
        
        # Add INCOSE analysis
        df["INCOSE_Best_Pattern"] = [incose.best_pattern for incose in incose_results]
        df["INCOSE_Missing_Required"] = [", ".join(incose.missing_required) for incose in incose_results]
        df["INCOSE_Suggestions"] = ["; ".join(incose.suggestions) for incose in incose_results]
        
        # Add semantic analysis
        df["Similarity_Issues"] = [len(semantic.similarity_issues) for semantic in semantic_results]
        df["Contextual_Ambiguities"] = [len(semantic.contextual_ambiguities) for semantic in semantic_results]
        df["Tone_Issues"] = [len(semantic.tone_issues) for semantic in semantic_results]
        
        # Calculate enhanced quality score
        df["Quality_Score"] = (
            df["Clarity_Score"] * 0.15 +           # Reduced
            df["Completeness_Score"] * 0.15 +      # Reduced
            df["Verifiability_Score"] * 0.30 +     # Still most important
            df["Atomicity_Score"] * 0.10 +         # Reduced
            df["Consistency_Score"] * 0.05 +       # Reduced
            df["INCOSE_Compliance_Score"] * 0.15 + # New: INCOSE compliance
            df["Semantic_Quality_Score"] * 0.10    # New: Semantic quality
        )
        
        # Apply enhanced penalties
        def apply_enhanced_penalty(row):
            base_score = row["Quality_Score"]
            
            # Standard penalties
            penalty = (row["Critical_Issues"] * 25 + row["High_Issues"] * 15 + 
                      row["Medium_Issues"] * 5 + row["Low_Issues"] * 2)
            
            # Additional penalties
            if row["Total_Issues"] > 3:
                penalty += (row["Total_Issues"] - 3) * 10
            
            # INCOSE penalties
            if row["INCOSE_Compliance_Score"] < 50:
                penalty += 20
            
            # Similarity penalties
            if row["Similarity_Issues"] > 0:
                penalty += row["Similarity_Issues"] * 10
            
            return max(0, base_score - penalty)
        
        df["Quality_Score"] = df.apply(apply_enhanced_penalty, axis=1)
        
        # Enhanced grade assignment
        def assign_enhanced_grade(score):
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
        
        df["Quality_Grade"] = df["Quality_Score"].apply(assign_enhanced_grade)
        
        return df
    
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
    
    def _generate_enhanced_summary(self, df: pd.DataFrame) -> Dict:
        """Generate enhanced summary with INCOSE and semantic metrics"""
        
        total_reqs = len(df)
        reqs_with_issues = len(df[df["Total_Issues"] > 0])
        
        # Basic metrics
        summary = {
            "basic_metrics": {
                "total_requirements": total_reqs,
                "requirements_with_issues": reqs_with_issues,
                "issue_rate": (reqs_with_issues / total_reqs * 100) if total_reqs > 0 else 0,
                "average_quality_score": df["Quality_Score"].mean()
            },
            "severity_breakdown": {
                "critical": int(df["Critical_Issues"].sum()),
                "high": int(df["High_Issues"].sum()),
                "medium": int(df["Medium_Issues"].sum()),
                "low": int(df["Low_Issues"].sum())
            },
            "incose_analysis": {
                "average_compliance": df["INCOSE_Compliance_Score"].mean(),
                "pattern_distribution": df["INCOSE_Best_Pattern"].value_counts().to_dict(),
                "low_compliance_count": len(df[df["INCOSE_Compliance_Score"] < 70])
            },
            "semantic_analysis": {
                "average_semantic_score": df["Semantic_Quality_Score"].mean(),
                "similarity_issues_count": int(df["Similarity_Issues"].sum()),
                "contextual_ambiguities_count": int(df["Contextual_Ambiguities"].sum()),
                "tone_issues_count": int(df["Tone_Issues"].sum())
            },
            "quality_distribution": df["Quality_Grade"].value_counts().to_dict()
        }
        
        return summary
    
    def _create_enhanced_excel_report(self, df: pd.DataFrame) -> str:
        """Create enhanced Excel report with INCOSE and semantic tabs"""
        
        output_file = self.file_handler.get_structured_path(
            'quality_analysis', 
            "enhanced_requirements_quality_report.xlsx"
        )
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Enhanced Dashboard
            dashboard = self._create_enhanced_dashboard_tab(df)
            dashboard.to_excel(writer, sheet_name='Enhanced Dashboard', index=False)
            
            # INCOSE Analysis Tab
            incose_tab = self._create_incose_analysis_tab(df)
            incose_tab.to_excel(writer, sheet_name='INCOSE Analysis', index=False)
            
            # Semantic Analysis Tab
            semantic_tab = self._create_semantic_analysis_tab(df)
            semantic_tab.to_excel(writer, sheet_name='Semantic Analysis', index=False)
            
            # Critical Issues (Enhanced)
            critical_tab = self._create_enhanced_critical_tab(df)
            critical_tab.to_excel(writer, sheet_name='Critical Issues', index=False)
            
            # Detailed Results
            df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Apply formatting
            try:
                self._format_enhanced_excel(writer)
            except Exception as e:
                logger.warning(f"Excel formatting failed: {e}")
        
        return str(output_path)
    
    def _create_enhanced_dashboard_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced dashboard with INCOSE and semantic metrics"""
        
        total_reqs = len(df)
        dashboard_data = []
        
        # Basic metrics
        avg_quality = df['Quality_Score'].mean()
        dashboard_data.extend([
            {"Category": "Overview", "Metric": "Total Requirements", "Value": total_reqs, "Status": "✓"},
            {"Category": "Overview", "Metric": "Average Quality Score", "Value": f"{avg_quality:.1f}/100", 
             "Status": "✓" if avg_quality >= 80 else "⚠️" if avg_quality >= 70 else "❌"},
        ])
        
        # INCOSE metrics
        avg_incose = df['INCOSE_Compliance_Score'].mean()
        low_incose = len(df[df['INCOSE_Compliance_Score'] < 70])
        dashboard_data.extend([
            {"Category": "INCOSE", "Metric": "Average INCOSE Compliance", "Value": f"{avg_incose:.1f}/100",
             "Status": "✓" if avg_incose >= 80 else "⚠️" if avg_incose >= 60 else "❌"},
            {"Category": "INCOSE", "Metric": "Low Compliance Requirements", "Value": low_incose,
             "Status": "✓" if low_incose < total_reqs * 0.1 else "❌"},
        ])
        
        # Semantic metrics
        similarity_issues = df['Similarity_Issues'].sum()
        avg_semantic = df['Semantic_Quality_Score'].mean()
        dashboard_data.extend([
            {"Category": "Semantic", "Metric": "Average Semantic Score", "Value": f"{avg_semantic:.1f}/100",
             "Status": "✓" if avg_semantic >= 80 else "⚠️"},
            {"Category": "Semantic", "Metric": "Potential Duplicates", "Value": int(similarity_issues),
             "Status": "✓" if similarity_issues == 0 else "⚠️"},
        ])
        
        # Quality grades
        grades = df['Quality_Grade'].value_counts()
        for grade in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']:
            count = grades.get(grade, 0)
            pct = count / total_reqs * 100
            dashboard_data.append({
                "Category": "Grades",
                "Metric": f"{grade} Requirements", 
                "Value": f"{count} ({pct:.1f}%)",
                "Status": "✓" if grade in ['EXCELLENT', 'GOOD'] else "⚠️" if grade == 'FAIR' else "❌"
            })
        
        return pd.DataFrame(dashboard_data)
    
    def _create_incose_analysis_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create INCOSE-specific analysis tab"""
        
        incose_data = []
        
        # Pattern distribution
        pattern_counts = df['INCOSE_Best_Pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            avg_compliance = df[df['INCOSE_Best_Pattern'] == pattern]['INCOSE_Compliance_Score'].mean()
            incose_data.append({
                "Analysis_Type": "Pattern Distribution",
                "Pattern": pattern,
                "Count": count,
                "Percentage": f"{count/len(df)*100:.1f}%",
                "Avg_Compliance": f"{avg_compliance:.1f}",
                "Status": "✓" if avg_compliance >= 80 else "⚠️" if avg_compliance >= 60 else "❌"
            })
        
        # Compliance analysis
        compliance_ranges = [
            ("Excellent (90-100)", len(df[df['INCOSE_Compliance_Score'] >= 90])),
            ("Good (80-89)", len(df[(df['INCOSE_Compliance_Score'] >= 80) & (df['INCOSE_Compliance_Score'] < 90)])),
            ("Fair (70-79)", len(df[(df['INCOSE_Compliance_Score'] >= 70) & (df['INCOSE_Compliance_Score'] < 80)])),
            ("Poor (50-69)", len(df[(df['INCOSE_Compliance_Score'] >= 50) & (df['INCOSE_Compliance_Score'] < 70)])),
            ("Critical (<50)", len(df[df['INCOSE_Compliance_Score'] < 50]))
        ]
        
        for range_name, count in compliance_ranges:
            incose_data.append({
                "Analysis_Type": "Compliance Range",
                "Pattern": range_name,
                "Count": count,
                "Percentage": f"{count/len(df)*100:.1f}%",
                "Avg_Compliance": "",
                "Status": "✓" if "Excellent" in range_name or "Good" in range_name else "⚠️" if "Fair" in range_name else "❌"
            })
        
        return pd.DataFrame(incose_data)
    
    def _create_semantic_analysis_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create semantic analysis tab"""
        
        semantic_data = []
        
        # Similarity issues
        similarity_issues = df['Similarity_Issues'].sum()
        if similarity_issues > 0:
            semantic_data.append({
                "Issue_Type": "Potential Duplicates",
                "Count": int(similarity_issues),
                "Severity": "Medium",
                "Recommendation": "Review similar requirements for consolidation",
                "Action": "Manual review required"
            })
        
        # Contextual ambiguities
        ambiguity_count = df['Contextual_Ambiguities'].sum()
        if ambiguity_count > 0:
            semantic_data.append({
                "Issue_Type": "Contextual Ambiguities",
                "Count": int(ambiguity_count),
                "Severity": "High",
                "Recommendation": "Replace ambiguous terms with specific criteria",
                "Action": "Add measurable thresholds"
            })
        
        # Tone issues
        tone_count = df['Tone_Issues'].sum()
        if tone_count > 0:
            semantic_data.append({
                "Issue_Type": "Tone Issues",
                "Count": int(tone_count),
                "Severity": "Medium",
                "Recommendation": "Use neutral, technical language",
                "Action": "Remove subjective terms"
            })
        
        return pd.DataFrame(semantic_data)
    
    def _create_enhanced_critical_tab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced critical issues tab"""
        
        # Filter for critical requirements
        critical_filter = (
            (df['Critical_Issues'] > 0) | 
            (df['High_Issues'] > 0) | 
            (df['Quality_Score'] < 50) |
            (df['INCOSE_Compliance_Score'] < 50) |
            (df['Similarity_Issues'] > 0)
        )
        
        critical_reqs = df[critical_filter].copy()
        
        if len(critical_reqs) == 0:
            return pd.DataFrame([{"Message": "No critical issues found!"}])
        
        # Select key columns
        columns_to_include = [
            'ID', 'Quality_Score', 'Quality_Grade', 'INCOSE_Compliance_Score', 
            'INCOSE_Best_Pattern', 'Similarity_Issues', 'Issues'
        ]
        
        available_cols = [col for col in columns_to_include if col in critical_reqs.columns]
        critical_issues = critical_reqs[available_cols].copy()
        
        # Sort by worst quality first
        critical_issues = critical_issues.sort_values('Quality_Score')
        
        # Add enhanced action columns
        critical_issues['Priority'] = critical_issues.apply(
            lambda row: 'CRITICAL' if row.get('Quality_Score', 100) < 35 else 'HIGH', axis=1
        )
        critical_issues['Primary_Issue'] = critical_issues.apply(
            lambda row: self._identify_primary_issue(row), axis=1
        )
        critical_issues['Action_Status'] = 'PENDING'
        critical_issues['Assigned_To'] = ''
        
        return critical_issues
    
    def _identify_primary_issue(self, row) -> str:
        """Identify the primary issue for a requirement"""
        if row.get('INCOSE_Compliance_Score', 100) < 50:
            return "INCOSE Pattern Compliance"
        elif row.get('Similarity_Issues', 0) > 0:
            return "Potential Duplicate"
        elif row.get('Quality_Score', 100) < 35:
            return "Multiple Quality Issues"
        else:
            return "Needs Review"
    
    def _format_enhanced_excel(self, writer):
        """Apply enhanced formatting to Excel workbook"""
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
                
                # Auto-adjust columns
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 60)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                    
        except ImportError:
            logger.warning("openpyxl not available for enhanced formatting")
    
    def _print_enhanced_summary(self, df: pd.DataFrame, similarity_results: List):
        """Print enhanced analysis summary"""
        
        print("\n" + "="*60)
        print("ENHANCED REQUIREMENTS QUALITY ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic metrics
        total_reqs = len(df)
        avg_quality = df["Quality_Score"].mean()
        print(f"📊 Total Requirements: {total_reqs}")
        print(f"📈 Average Quality Score: {avg_quality:.1f}/100")
        
        # Grade distribution
        grades = df["Quality_Grade"].value_counts()
        print(f"\n🎯 Quality Distribution:")
        for grade in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']:
            count = grades.get(grade, 0)
            pct = count / total_reqs * 100
            print(f"   {grade}: {count} ({pct:.1f}%)")
        
        # INCOSE analysis
        avg_incose = df["INCOSE_Compliance_Score"].mean()
        low_incose = len(df[df["INCOSE_Compliance_Score"] < 70])
        print(f"\n🏗️ INCOSE Pattern Analysis:")
        print(f"   Low Compliance Requirements: {low_incose}")
        
        # Pattern distribution
        pattern_dist = df["INCOSE_Best_Pattern"].value_counts()
        print(f"   Most Common Pattern: {pattern_dist.index[0] if len(pattern_dist) > 0 else 'None'}")
        
        # Semantic analysis
        avg_semantic = df["Semantic_Quality_Score"].mean()
        similarity_count = len(similarity_results)
        print(f"\n🧠 Semantic Analysis:")
        print(f"   Average Semantic Score: {avg_semantic:.1f}/100")
        print(f"   Potential Duplicates Found: {similarity_count}")
        
        if similarity_count > 0:
            print(f"   Top Similarity Pair: {similarity_results[0]['similarity']:.2f}")
        
        # Issue severity breakdown
        critical_issues = df["Critical_Issues"].sum()
        high_issues = df["High_Issues"].sum()
        medium_issues = df["Medium_Issues"].sum()
        low_issues = df["Low_Issues"].sum()
        
        print(f"\n⚠️ Issue Severity Breakdown:")
        print(f"   Critical: {critical_issues}")
        print(f"   High: {high_issues}")
        print(f"   Medium: {medium_issues}")
        print(f"   Low: {low_issues}")
        
        # Top issues by category
        print(f"\n🔍 Top Issue Categories:")
        all_issues = []
        for issues_list in df["Issues"]:
            all_issues.extend(issues_list)
        
        if all_issues:
            # Extract issue categories
            issue_categories = []
            for issue in all_issues:
                if ":" in issue:
                    category = issue.split(":")[0].strip()
                    issue_categories.append(category)
            
            if issue_categories:
                category_counts = Counter(issue_categories)
                for category, count in category_counts.most_common(5):
                    print(f"   {category}: {count}")
        
        # Recommendations
        print(f"\n💡 Key Recommendations:")
        if avg_incose < 70:
            print(f"   • Improve INCOSE pattern compliance (current: {avg_incose:.1f}%)")
        if similarity_count > 0:
            print(f"   • Review {similarity_count} potential duplicate requirements")
        if critical_issues > 0:
            print(f"   • Address {critical_issues} critical issues immediately")
        if high_issues > total_reqs * 0.2:
            print(f"   • High issue rate ({high_issues}/{total_reqs}) needs attention")
        
        print("="*60)


def main():
    """Main function with enhanced CLI"""
    parser = argparse.ArgumentParser(
        description="Enhanced Requirements Quality Analyzer v2.0 with INCOSE patterns and semantic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reqGrading.py                                    # Analyze requirements.csv
  python reqGrading.py -i my_reqs.csv --excel           # Create Excel report
  python reqGrading.py -c "Requirement" -m en_core_web_lg # Use different column and model
  python reqGrading.py -v --excel                       # Verbose mode with Excel output

Features:
  • INCOSE pattern compliance analysis
  • Semantic similarity detection  
  • Contextual ambiguity detection
  • Advanced NLP with spaCy transformers
  • Enhanced Excel reports with multiple tabs
  • Industry-standard recommendations
        """
    )
    
    parser.add_argument("-i", "--input", dest="input_file", default="requirements.csv",
                       help="Input CSV file containing requirements (default: requirements.csv)")
    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated)")
    parser.add_argument("-c", "--column", default="Requirement Text", 
                       help="Column name containing requirements (default: 'Requirement Text')")
    parser.add_argument("-m", "--model", default="en_core_web_trf",
                       help="spaCy model to use (default: en_core_web_trf)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--excel", action="store_true",
                       help="Create enhanced Excel report with INCOSE and semantic tabs")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                       help="Similarity threshold for duplicate detection (default: 0.85)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Enhanced Requirements Quality Analyzer v2.0")
    print("=" * 50)
    print(f"📁 Input file: {args.input_file}")
    print(f"📝 Requirement column: {args.column}")
    print(f"🤖 spaCy model: {args.model}")
    print(f"📊 Excel report: {'Yes' if args.excel else 'No'}")
    print(f"🔍 Similarity threshold: {args.similarity_threshold}")
    print("=" * 50)
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedRequirementAnalyzer(args.model)
        
        # Set similarity threshold
        analyzer.semantic_analyzer.similarity_threshold = args.similarity_threshold
        
        # Run enhanced analysis
        result_df = analyzer.analyze_file(
            args.input_file, 
            args.output, 
            args.column, 
            excel_report=args.excel
        )
        
        print(f"\n✅ Enhanced analysis complete!")
        print(f"📊 Analyzed {len(result_df)} requirements")
        print(f"📈 Average quality score: {result_df['Quality_Score'].mean():.1f}/100")
        
        # Show top issues
        critical_reqs = len(result_df[result_df['Quality_Grade'] == 'CRITICAL'])
        if critical_reqs > 0:
            print(f"⚠️ {critical_reqs} requirements need immediate attention")
        
        # Show INCOSE compliance
        avg_incose = result_df['INCOSE_Compliance_Score'].mean()
        print(f"🏗️ INCOSE compliance: {avg_incose:.1f}%")
        
        # Show duplicates
        duplicates = result_df['Similarity_Issues'].sum()
        if duplicates > 0:
            print(f"🔍 Found {int(duplicates)} potential duplicate pairs")
        
        print("\nReview the generated reports for detailed analysis and recommendations.")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()