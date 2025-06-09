import os
import spacy
from typing import Optional
import logging

# SOLUTION 1: Unified TRF Model Manager
class UnifiedTRFManager:
    """
    Singleton manager for shared TRF model across all components.
    Eliminates conflicts by ensuring only ONE model instance.
    """
    
    _instance = None
    _nlp = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def setup_environment(cls):
        """Setup environment for stable TRF model usage."""
        
        # Critical OpenMP settings for TRF models
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OMP_NUM_THREADS'] = '2'  # TRF can use 2 threads efficiently
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Optional: Force CPU usage for consistency
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU if causing issues
        
        print("🔧 Environment configured for TRF model stability")
    
    def get_trf_model(self, model_name: str = "en_core_web_trf") -> spacy.Language:
        """Get or create the shared TRF model instance."""
        
        if not self._model_loaded:
            self.setup_environment()
            
            try:
                print(f"📦 Loading TRF model: {model_name}...")
                self._nlp = spacy.load(model_name)
                self._model_loaded = True
                print(f"✅ TRF model loaded successfully")
                
                # Verify transformer components
                if 'transformer' in self._nlp.pipe_names:
                    print(f"🧠 Transformer component found: {self._nlp.get_pipe('transformer')}")
                
            except OSError as e:
                print(f"❌ TRF model not found: {e}")
                print("📦 Installing en_core_web_trf...")
                
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
                
                self._nlp = spacy.load(model_name)
                self._model_loaded = True
                print(f"✅ TRF model installed and loaded")
                
            except Exception as e:
                print(f"❌ Failed to load TRF model: {e}")
                print("🔄 Falling back to en_core_web_sm...")
                self._nlp = spacy.load("en_core_web_sm")
                self._model_loaded = True
        
        return self._nlp
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self._nlp:
            return {
                'name': self._nlp.meta.get('name', 'unknown'),
                'version': self._nlp.meta.get('version', 'unknown'),
                'has_transformer': 'transformer' in self._nlp.pipe_names,
                'pipe_names': self._nlp.pipe_names,
                'vectors': self._nlp.vocab.vectors_length
            }
        return {}