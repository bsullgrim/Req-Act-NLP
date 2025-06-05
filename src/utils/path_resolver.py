"""
Path Resolver - Smart file path resolution and auto-detection
Handles finding files in multiple locations and suggesting alternatives
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SmartPathResolver:
    """Intelligently resolve file paths with auto-detection."""
    
    def __init__(self):
        # Common search locations in order of preference
        self.search_locations = [
            Path("."),              # Current directory
            Path("data/raw"),       # Repository structure
            Path("data"),           # Data directory
            Path("examples"),       # Examples directory
        ]
        
        # Common file patterns for auto-detection
        self.file_patterns = {
            'requirements': ['*requirement*', '*req*', '*specs*'],
            'activities': ['*activit*', '*task*', '*work*', '*function*'],
            'manual_matches': ['*manual*', '*match*', '*trace*', '*gold*', '*ground*', '*truth*']
        }

    def resolve_input_files(self, file_mapping: Dict[str, str]) -> Dict[str, str]:
        """
        Resolve file paths, auto-detecting locations if needed.
        
        Args:
            file_mapping: {'file_type': 'filename'} e.g., {'requirements': 'requirements.csv'}
        
        Returns:
            {'file_type': 'resolved_path'} or original path if not found
        """
        resolved_paths = {}
        
        for file_type, filename in file_mapping.items():
            resolved_path = self._find_file(filename, file_type)
            resolved_paths[file_type] = resolved_path
            
            if Path(resolved_path).exists():
                logger.info(f"✓ Resolved {file_type}: {resolved_path}")
            else:
                logger.warning(f"✗ Could not resolve {file_type}: {filename}")
        
        return resolved_paths

    def _find_file(self, filename: str, file_type: str) -> str:
        """Find a file in various locations, with auto-detection fallback."""
        
        # 1. Try exact path first
        if Path(filename).exists():
            return filename
        
        # 2. Try in search locations
        for location in self.search_locations:
            candidate = location / filename
            if candidate.exists():
                logger.info(f"Auto-detected {file_type} at: {candidate}")
                return str(candidate)
        
        # 3. Try pattern matching in search locations
        if file_type in self.file_patterns:
            patterns = self.file_patterns[file_type]
            
            for location in self.search_locations:
                if location.exists():
                    for pattern in patterns:
                        matches = list(location.glob(pattern + ".csv"))
                        if matches:
                            best_match = matches[0]  # Take first match
                            logger.info(f"Pattern-matched {file_type}: {best_match}")
                            return str(best_match)
        
        # 4. Return original if nothing found
        return filename

    def find_file_alternatives(self, missing_files: List[str]) -> Dict[str, List[str]]:
        """Find alternative files for missing ones."""
        
        alternatives = {}
        
        for missing_file in missing_files:
            file_base = Path(missing_file).stem.lower()
            file_alternatives = []
            
            # Determine file type from name
            file_type = self._guess_file_type(file_base)
            
            if file_type and file_type in self.file_patterns:
                patterns = self.file_patterns[file_type]
                
                # Search for alternatives
                for location in self.search_locations:
                    if location.exists():
                        for pattern in patterns:
                            for candidate in location.glob(pattern):
                                if candidate.is_file() and candidate.suffix in ['.csv', '.xlsx']:
                                    file_alternatives.append(str(candidate))
            
            if file_alternatives:
                alternatives[missing_file] = file_alternatives[:5]  # Limit to top 5
        
        return alternatives

    def _guess_file_type(self, filename_base: str) -> Optional[str]:
        """Guess file type from filename."""
        filename_lower = filename_base.lower()
        
        if any(term in filename_lower for term in ['requirement', 'req', 'spec']):
            return 'requirements'
        elif any(term in filename_lower for term in ['activit', 'task', 'work', 'function']):
            return 'activities'
        elif any(term in filename_lower for term in ['manual', 'match', 'trace', 'gold', 'ground', 'truth']):
            return 'manual_matches'
        
        return None

    def suggest_file_placement(self, file_types: List[str]) -> Dict[str, List[str]]:
        """Suggest where files should be placed for each file type."""
        
        suggestions = {}
        
        for file_type in file_types:
            file_suggestions = []
            
            # Repository structure (preferred)
            file_suggestions.append(f"data/raw/{file_type}.csv")
            
            # Current directory (simple)
            file_suggestions.append(f"./{file_type}.csv")
            
            # Examples directory
            file_suggestions.append(f"examples/{file_type}.csv")
            
            suggestions[file_type] = file_suggestions
        
        return suggestions

    def create_file_guidance(self, missing_files: List[str], alternatives: Dict[str, List[str]]) -> str:
        """Create user-friendly guidance for missing files."""
        
        guidance = "\n📁 FILE LOCATION GUIDANCE\n" + "=" * 50 + "\n"
        
        if alternatives:
            guidance += "\n💡 Found potential alternative files:\n"
            for missing_file, alts in alternatives.items():
                guidance += f"\n   {missing_file}:\n"
                for alt in alts[:3]:  # Show top 3
                    guidance += f"     - {alt}\n"
        
        guidance += "\n📂 Recommended file placement:\n"
        guidance += "   Option 1 (Repository structure):\n"
        guidance += "     ./data/raw/requirements.csv\n"
        guidance += "     ./data/raw/activities.csv\n"
        guidance += "     ./data/raw/manual_matches.csv\n\n"
        
        guidance += "   Option 2 (Current directory):\n"
        guidance += "     ./requirements.csv\n"
        guidance += "     ./activities.csv\n"
        guidance += "     ./manual_matches.csv\n\n"
        
        guidance += "🔧 The system will auto-detect files in these locations.\n"
        
        return guidance

    def validate_resolved_paths(self, resolved_paths: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate that resolved paths actually exist and are accessible."""
        
        all_valid = True
        issues = []
        
        for file_type, file_path in resolved_paths.items():
            path = Path(file_path)
            
            if not path.exists():
                all_valid = False
                issues.append(f"❌ {file_type}: {file_path} does not exist")
            elif not path.is_file():
                all_valid = False
                issues.append(f"❌ {file_type}: {file_path} is not a file")
            elif path.stat().st_size == 0:
                all_valid = False
                issues.append(f"❌ {file_type}: {file_path} is empty")
            else:
                logger.debug(f"✓ {file_type}: {file_path} validated")
        
        return all_valid, issues