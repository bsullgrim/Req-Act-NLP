"""
Repository Structure Manager - Handle repository setup and file organization
Creates and maintains the standard repository directory structure
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
            #'data_processed': Path("data/processed"), 
            #'data_examples': Path("data/examples"),
            
            # Output directories
            'matching_results': self.base_dir / "matching_results",
            'evaluation_results': self.base_dir / "evaluation_results", 
            'evaluation_dashboards': self.base_dir / "evaluation_results" / "dashboards",
            #'evaluation_detailed': self.base_dir / "evaluation_results" / "detailed_results",
            #'evaluation_discovery': self.base_dir / "evaluation_results" / "discovery_analysis",
            'engineering_review': self.base_dir / "engineering_review",
            'quality_analysis': self.base_dir / "quality_analysis",
            
            # Source directories
            'src_matching': Path("src/matching"),
            'src_evaluation': Path("src/evaluation"), 
            'src_dashboard': Path("src/dashboard"),
            'src_quality': Path("src/quality"),
            'src_utils': Path("src/utils"),
            
            # Configuration directories
            #'configs': Path("configs"),
            #'configs_matching': Path("configs/matching_configs"),
            #'configs_evaluation': Path("configs/evaluation_configs"),
            
            # Other directories
            #'scripts': Path("scripts"),
            #'docs': Path("docs"),
            #'tests': Path("tests"),
            #'notebooks': Path("notebooks")
        }

    def setup_repository_structure(self, create_gitignore: bool = True) -> Dict[str, str]:
        """Create complete repository directory structure."""
        
        logger.info("🏗️ Setting up repository structure...")
        
        created_dirs = {}
        created_count = 0
        
        for name, path in self.structure.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_dirs[name] = str(path)
                created_count += 1
                logger.debug(f"Created directory: {path}")
        
        logger.info(f"✓ Repository structure created ({created_count} new directories)")
        
        # Create .gitignore for outputs
        if create_gitignore:
            self._create_gitignore()
        
        return created_dirs

    def _create_gitignore(self):
        """Create .gitignore file for the repository."""
        
        gitignore_path = Path(".gitignore")
        gitignore_content = f"""
# Generated outputs directory
{self.base_dir}/

# Python
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
env/
venv/
*.egg-info/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp

# Data files (uncomment if you want to ignore data)
# data/raw/*.csv
# data/raw/*.xlsx

# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content.strip())
            logger.info("✓ Created .gitignore")
        else:
            logger.debug(".gitignore already exists")

    def organize_outputs(self, files: Dict[str, str], cleanup_patterns: Optional[List[str]] = None) -> Dict[str, str]:
        """Organize output files into proper repository locations."""
        
        logger.info("📦 Organizing outputs into repository structure...")
        
        organized_files = {}
        
        # Move files to appropriate locations
        for file_key, file_path in files.items():
            if not Path(file_path).exists():
                logger.warning(f"File not found for organization: {file_path}")
                continue
            
            target_dir = self._get_target_directory(file_key, file_path)
            if target_dir:
                organized_path = self._move_to_target(file_path, target_dir)
                if organized_path:
                    organized_files[file_key] = organized_path
        
        # Clean up stray files if patterns provided
        if cleanup_patterns:
            self.cleanup_stray_files(cleanup_patterns)
        
        logger.info(f"✓ Organized {len(organized_files)} files")
        return organized_files

    def _get_target_directory(self, file_key: str, file_path: str) -> Optional[Path]:
        """Determine target directory based on file key and path."""
        
        file_name = Path(file_path).name.lower()
        
        # Dashboard files
        if 'dashboard' in file_key or file_name.endswith('.html'):
            return self.structure['evaluation_dashboards']
        
        # Discovery analysis files
        elif any(keyword in file_name for keyword in ['discovery', 'miss', 'gap']):
            return self.structure['evaluation_discovery']
        
        # Engineering review files
        elif any(keyword in file_key for keyword in ['executive', 'action', 'summary']):
            return self.structure['engineering_review']
        
        # Quality analysis files
        elif 'quality' in file_key or 'quality' in file_name:
            return self.structure['quality_analysis']
        
        # Raw matching results
        elif 'match' in file_key and 'enhanced' not in file_key:
            return self.structure['matching_results']
        
        # Enhanced/detailed evaluation results
        elif any(keyword in file_key for keyword in ['enhanced', 'detailed', 'evaluation']):
            return self.structure['evaluation_detailed']
        
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
- `data/processed/` - Cleaned and processed data files
- `data/examples/` - Sample/example datasets

### Outputs
- `{self.base_dir}/matching_results/` - Raw algorithm matching results
- `{self.base_dir}/evaluation_results/` - Evaluation metrics and analysis
  - `dashboards/` - Interactive HTML dashboards
  - `detailed_results/` - Detailed per-requirement analysis
  - `discovery_analysis/` - Discovery insights and missed connections
- `{self.base_dir}/engineering_review/` - Business deliverables
- `{self.base_dir}/quality_analysis/` - Requirements quality analysis

### Source Code
- `src/matching/` - Matching algorithms
- `src/evaluation/` - Evaluation framework
- `src/dashboard/` - Dashboard components
- `src/utils/` - Shared utilities

### Configuration
- `configs/` - Configuration files for different runs

## Usage

1. Place input files in `data/raw/`
2. Run the workflow: `python workflow_orchestrator.py`
3. Check results in `{self.base_dir}/`

## Key Files

- **Dashboard**: `{self.base_dir}/evaluation_results/dashboards/evaluation_dashboard.html`
- **Executive Summary**: `{self.base_dir}/engineering_review/executive_summary.csv`
- **All Matches**: `{self.base_dir}/matching_results/final_clean_matches.csv`
"""
        
        readme_path = Path("README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("✓ Created README.md")
        return str(readme_path)