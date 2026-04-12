"""
Configuration loader for Question Analyzer
Reads from config.yaml and provides centralized config access
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the application."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        
        if not config_path.exists():
            print(f"⚠️  Config file not found at {config_path}")
            print("Using default configuration...")
            self._config = self._default_config()
            return
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            print(f"✓ Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False
            },
            "models": {
                "embedding": {
                    "name": "all-MiniLM-L6-v2",
                    "device": "cuda"
                },
                "generation": {
                    "provider": "groq",
                    "name": "llama-3.1-8b-instant",
                    "max_tokens": 300,
                    "temperature": 0.0
                }
            },
            "database": {
                "index_path": "data/index.faiss",
                "metadata_path": "data/index_metadata.pkl",
                "question_index_path": "data/question_index.faiss",
                "question_meta_path": "data/questions_meta.jsonl"
            },
            "data": {
                "input_folder": "data",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "supported_formats": [".txt", ".pdf"]
            },
            "rag": {
                "top_k": 5,
                "provider": "groq"
            },
            "tokens": {
                "groq_api_key": ""
            },
            "logging": {
                "level": "INFO",
                "file": "logs/app.log"
            }
        }
    
    def get(self, key: str, default=None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.embedding.name')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def embedding_model(self) -> str:
        return self.get("models.embedding.name", "all-MiniLM-L6-v2")
    
    @property
    def embedding_device(self) -> str:
        return self.get("models.embedding.device", "cuda")
    
    @property
    def generation_model(self) -> str:
        return self.get("models.generation.name", "llama-3.1-8b-instant")
    
    @property
    def generation_provider(self) -> str:
        return self.get("models.generation.provider", "groq")
    
    @property
    def generation_max_tokens(self) -> int:
        return self.get("models.generation.max_tokens", 300)
    
    @property
    def generation_temperature(self) -> float:
        return self.get("models.generation.temperature", 0.0)
    
    @property
    def index_path(self) -> str:
        return self.get("database.index_path", "data/index.faiss")
    
    @property
    def metadata_path(self) -> str:
        return self.get("database.metadata_path", "data/index_metadata.pkl")
    
    @property
    def question_index_path(self) -> str:
        return self.get("database.question_index_path", "data/question_index.faiss")
    
    @property
    def question_meta_path(self) -> str:
        return self.get("database.question_meta_path", "data/questions_meta.jsonl")
    
    @property
    def data_folder(self) -> str:
        return self.get("data.input_folder", "data")
    
    @property
    def chunk_size(self) -> int:
        return self.get("data.chunk_size", 500)
    
    @property
    def chunk_overlap(self) -> int:
        return self.get("data.chunk_overlap", 100)
    
    @property
    def supported_formats(self) -> list:
        return self.get("data.supported_formats", [".txt", ".pdf"])
    
    @property
    def rag_top_k(self) -> int:
        return self.get("rag.top_k", 5)
    
    @property
    def use_hf_api(self) -> bool:
        return self.get("rag.use_hf_api", True)
    
    @property
    def fallback_to_local(self) -> bool:
        return self.get("rag.fallback_to_local", True)
    
    @property
    def rag_provider(self) -> str:
        return self.get("rag.provider", "groq")
    
    @property
    def server_host(self) -> str:
        return self.get("server.host", "0.0.0.0")
    
    @property
    def server_port(self) -> int:
        return self.get("server.port", 8000)
    
    @property
    def groq_api_key(self) -> str:
        """Get Groq API key from config, environment variable, or groq_key.txt file"""
        # First, try config file
        key = self.get("tokens.groq_api_key", "").strip()
        if key:
            return key
        
        # Try environment variable
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if key:
            return key
        
        # Try groq_key.txt in project root
        root = Path(__file__).resolve().parents[1]
        key_file = root / "groq_key.txt"
        if key_file.exists():
            try:
                key = key_file.read_text(encoding="utf-8").strip()
                if key:
                    return key
            except Exception:
                pass
        
        return ""


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
