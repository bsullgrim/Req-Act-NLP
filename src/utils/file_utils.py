"""
File Utilities - Safe file operations with encoding detection
Extracted from orchestrator for reusability across the codebase
"""

import pandas as pd
import chardet
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SafeFileHandler:
    """Handle all file operations with encoding detection and error handling."""
    def __init__(self, repo_manager=None):
        if repo_manager is None:
            from src.utils.repository_setup import RepositoryStructureManager
            self.repo_manager = RepositoryStructureManager("outputs")
        else:
            self.repo_manager = repo_manager

    def get_structured_path(self, file_type: str, filename: str) -> str:
        """Get properly structured path for file type."""
        if file_type == 'raw_data':
            return str(self.repo_manager.structure['data_raw'] / filename)
        elif file_type == 'matching_results':
            return str(self.repo_manager.structure['matching_results'] / filename)
        elif file_type == 'quality_analysis':
            return str(self.repo_manager.structure['quality_analysis'] / filename)
        else:
            return filename  # Fallback to original
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect the encoding of a file with fallback."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                if confidence < 0.7:
                    logger.warning(f"Low confidence ({confidence:.2f}) in detected encoding: {encoding}")
                
                return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8"

    def safe_read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Safely read CSV with automatic encoding detection."""
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

    def verify_csv_structure(self, file_path: str, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Verify CSV has required columns and return missing ones."""
        try:
            df = self.safe_read_csv(file_path, nrows=1)
            missing_columns = [col for col in required_columns if col not in df.columns]
            return len(missing_columns) == 0, missing_columns
        except Exception as e:
            logger.error(f"Could not verify structure of {file_path}: {e}")
            return False, required_columns

    def verify_input_files(self, file_specs: List[Dict[str, any]]) -> Tuple[bool, List[str]]:
        """
        Verify multiple input files exist and have required structure.
        
        Args:
            file_specs: List of dicts with 'path', 'name', 'required_columns', 'optional'
        
        Returns:
            (all_valid, error_messages)
        """
        all_valid = True
        error_messages = []
        
        for spec in file_specs:
            file_path = spec['path']
            file_name = spec['name']
            required_columns = spec.get('required_columns', [])
            is_optional = spec.get('optional', False)
            
            if not Path(file_path).exists():
                if is_optional:
                    logger.info(f"ℹ️ Optional file not found: {file_name} ({file_path})")
                else:
                    all_valid = False
                    error_messages.append(f"❌ {file_name} file missing: {file_path}")
                continue
                
            # Verify structure if required columns specified
            if required_columns:
                is_valid, missing_cols = self.verify_csv_structure(file_path, required_columns)
                if not is_valid:
                    all_valid = False
                    error_messages.append(
                        f"❌ {file_name} missing columns: {missing_cols} in {file_path}"
                    )
                else:
                    logger.info(f"✓ {file_name} file verified: {file_path}")
            else:
                # Just try to read it
                try:
                    self.safe_read_csv(file_path, nrows=1)
                    logger.info(f"✓ {file_name} file verified: {file_path}")
                except Exception as e:
                    all_valid = False
                    error_messages.append(f"❌ Cannot read {file_name} file {file_path}: {e}")
        
        return all_valid, error_messages

    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get comprehensive file information."""
        path = Path(file_path)
        
        info = {
            'exists': path.exists(),
            'path': str(path),
            'name': path.name,
            'size_bytes': 0,
            'encoding': None,
            'row_count': 0,
            'columns': [],
            'readable': False
        }
        
        if path.exists():
            try:
                info['size_bytes'] = path.stat().st_size
                info['encoding'] = self.detect_encoding(file_path)
                
                # Try to read and get structure info
                df = self.safe_read_csv(file_path, nrows=100)  # Sample for speed
                info['row_count'] = len(df)
                info['columns'] = list(df.columns)
                info['readable'] = True
                
            except Exception as e:
                logger.debug(f"Could not get full info for {file_path}: {e}")
        
        return info