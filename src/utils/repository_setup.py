"""
Repository Structure Manager - Handle repository setup and file organization
Creates and maintains the standard repository directory structure
FIXED: Added missing get_results_path() method
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class RepositoryStructureManager:
    """Manage repository directory structure and file organization."""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        
        # Standard repository structure from documentation
        self.structure = {
            # Data directories
            'data_raw': Path("data/raw"),
            
            # Output directories
            'matching_results': self.base_dir / "matching_results",
            'evaluation_results': self.base_dir / "evaluation_results", 
            'evaluation_dashboards': self.base_dir / "evaluation_results" / "dashboards",
            'engineering_review': self.base_dir / "engineering_review",
            'quality_analysis': self.base_dir / "quality_analysis",
            
            # Source directories
            'src_matching': Path("src/matching"),
            'src_evaluation': Path("src/evaluation"), 
            'src_dashboard': Path("src/dashboard"),
            'src_quality': Path("src/quality"),
            'src_utils': Path("src/utils"),
        }

    def setup_repository_structure(self, create_gitignore: bool = True) -> Dict[str, str]:
        """Create complete repository directory structure."""
        
        created_dirs = []
        
        for name, path in self.structure.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(path))
                logger.debug(f"✓ Created/verified directory: {path}")
            except Exception as e:
                logger.warning(f"Could not create directory {path}: {e}")
        
        if create_gitignore:
            self._create_gitignore()
        
        logger.info(f"✓ Repository structure setup complete ({len(created_dirs)} directories)")
        return {name: str(path) for name, path in self.structure.items()}
    
    def _create_gitignore(self):
        """Create .gitignore for output directories."""
        
        gitignore_path = Path(".gitignore")
        
        # Patterns to ignore
        ignore_patterns = [
            "# Matching outputs",
            "outputs/",
            "*.csv",
            "*.json", 
            "*.xlsx",
            "*.html",
            "# Python",
            "__pycache__/",
            "*.pyc",
            ".env",
            "# IDE",
            ".vscode/",
            ".idea/",
            "# OS",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        try:
            # Read existing .gitignore if it exists
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text()
            
            # Add new patterns if not already present
            new_patterns = []
            for pattern in ignore_patterns:
                if pattern not in existing_content:
                    new_patterns.append(pattern)
            
            if new_patterns:
                with open(gitignore_path, 'a') as f:
                    if existing_content and not existing_content.endswith('\n'):
                        f.write('\n')
                    f.write('\n'.join(new_patterns) + '\n')
                logger.debug("✓ Updated .gitignore")
                
        except Exception as e:
            logger.warning(f"Could not create/update .gitignore: {e}")

    # FIXED: Add the missing methods that the matcher expects
    def get_results_path(self):
        """Get matching results directory path - compatibility method."""
        results_dir = self.structure.get('matching_results', self.base_dir / 'matching_results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        return Path(results_dir)
    
    def get_evaluation_path(self):
        """Get evaluation results directory path."""
        eval_dir = self.structure.get('evaluation_results', self.base_dir / 'evaluation_results')
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        return Path(eval_dir)
    
    def get_engineering_path(self):
        """Get engineering review directory path."""
        eng_dir = self.structure.get('engineering_review', self.base_dir / 'engineering_review')
        Path(eng_dir).mkdir(parents=True, exist_ok=True)
        return Path(eng_dir)

    def organize_files(self, file_mapping: Dict[str, str]) -> Dict[str, str]:
        """Organize files into appropriate directories."""
        
        organized_files = {}
        
        for file_key, file_path in file_mapping.items():
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Determine target directory
            target_dir = self._get_target_directory(file_key, file_path)
            
            if target_dir:
                organized_path = self._move_to_target(file_path, target_dir)
                if organized_path:
                    organized_files[file_key] = organized_path
                else:
                    organized_files[file_key] = file_path  # Keep original if move failed
            else:
                organized_files[file_key] = file_path  # Keep original if no target
        
        return organized_files

    def _get_target_directory(self, file_key: str, file_path: str) -> Optional[Path]:
        """Determine target directory for a file."""
        
        file_name = Path(file_path).name.lower()
        
        # Dashboard files
        if 'dashboard' in file_key or file_name.endswith('.html'):
            return self.structure['evaluation_dashboards']
        
        # Engineering review files
        elif any(keyword in file_key for keyword in ['executive', 'action', 'summary', 'workbook']):
            return self.structure['engineering_review']
        
        # Quality analysis files
        elif 'quality' in file_key or 'quality' in file_name:
            return self.structure['quality_analysis']
        
        # Raw matching results
        elif 'match' in file_key and 'enhanced' not in file_key:
            return self.structure['matching_results']
        
        # Enhanced/detailed evaluation results
        elif any(keyword in file_key for keyword in ['enhanced', 'detailed', 'evaluation']):
            return self.structure['evaluation_results']
        
        # Default to evaluation results
        else:
            return self.structure['evaluation_results']

    def _move_to_target(self, source_path: str, target_dir: Path) -> Optional[str]:
        """Move file to target directory, avoiding overwrites."""
        
        source = Path(source_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source.name
        
        # Handle name conflicts
        if target_path.exists() and target_path != source:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"
        
        try:
            if source != target_path:  # Only move if different locations
                shutil.move(str(source), str(target_path))
                logger.info(f"   ✓ Moved {source.name} to {target_path}")
            return str(target_path)
        except Exception as e:
            logger.error(f"   ✗ Failed to move {source_path} to {target_dir}: {e}")
            return None

    def cleanup_stray_files(self, patterns: List[str]) -> None:
        """Clean up stray files matching patterns."""
        
        logger.info("🧹 Cleaning up stray files...")
        
        cleaned_count = 0
        for pattern in patterns:
            for file_path in Path(".").glob(pattern):
                if file_path.is_file():
                    try:
                        # Determine appropriate target
                        target_dir = self._get_target_directory('cleanup', str(file_path))
                        if target_dir:
                            organized_path = self._move_to_target(str(file_path), target_dir)
                            if organized_path:
                                cleaned_count += 1
                    except Exception as e:
                        logger.debug(f"Could not clean up {file_path}: {e}")
        
        # Clean up empty directories
        self._cleanup_empty_directories()
        
        if cleaned_count > 0:
            logger.info(f"✓ Cleaned up {cleaned_count} stray files")

    def _cleanup_empty_directories(self):
        """Remove empty directories created during processing."""
        
        # Common directories that might be left empty
        cleanup_dirs = [
            Path("evaluation_results"),
            Path("dashboard_data"),
            Path("results")
        ]
        
        for dir_path in cleanup_dirs:
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Only remove if empty
                    if not any(dir_path.rglob("*")):
                        shutil.rmtree(str(dir_path))
                        logger.debug(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.debug(f"Could not remove directory {dir_path}: {e}")

    def get_output_paths(self) -> Dict[str, str]:
        """Get all output directory paths as strings."""
        return {name: str(path) for name, path in self.structure.items() if 'output' in name or self.base_dir.name in str(path)}

    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate that the repository structure is properly set up."""
        
        issues = []
        
        # Check critical directories exist
        critical_dirs = [
            'matching_results',
            'evaluation_results', 
            'evaluation_dashboards',
            'engineering_review'
        ]
        
        for dir_name in critical_dirs:
            if dir_name in self.structure:
                dir_path = self.structure[dir_name]
                if not dir_path.exists():
                    issues.append(f"Missing critical directory: {dir_path}")
        
        # Check write permissions
        try:
            test_file = self.structure['matching_results'] / "test_write.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"No write permission to output directories: {e}")
        
        return len(issues) == 0, issues

    def create_readme(self) -> str:
        """Create README.md explaining the repository structure."""
        
        readme_content = f"""# Requirements Traceability Repository

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

### Input Data
- `data/raw/` - Original input files (requirements.csv, activities.csv, manual_matches.csv)

### Outputs
- `{self.base_dir}/matching_results/` - Raw algorithm matching results
- `{self.base_dir}/evaluation_results/` - Evaluation metrics and analysis
  - `dashboards/` - Interactive HTML dashboards
- `{self.base_dir}/engineering_review/` - Business deliverables
- `{self.base_dir}/quality_analysis/` - Requirements quality analysis

### Source Code
- `src/matching/` - Matching algorithms
- `src/evaluation/` - Evaluation framework
- `src/dashboard/` - Dashboard components
- `src/utils/` - Shared utilities

## Usage

1. Place input files in project root or `data/raw/`
2. Run the matcher: `python -m src.matching.matcher`
3. Check results in `{self.base_dir}/`

## Key Files

- **All Matches**: `{self.base_dir}/matching_results/`
- **Dashboards**: `{self.base_dir}/evaluation_results/dashboards/`
- **Reports**: `{self.base_dir}/engineering_review/`
"""
        
        readme_path = Path("README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("✓ Created README.md")
        return str(readme_path)