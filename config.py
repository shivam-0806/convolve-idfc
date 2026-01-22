"""
Configuration management for Document AI system
Uses Pydantic for type-safe configuration with validation
"""
import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class OCRSettings(BaseSettings):
    """OCR engine configuration"""
    engine: str = Field(default="easyocr", description="OCR engine to use")
    languages: List[str] = Field(default=["en"], description="Languages for OCR")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @field_validator('engine')
    @classmethod
    def validate_engine(cls, v):
        allowed = ['easyocr', 'tesseract']
        if v not in allowed:
            raise ValueError(f"OCR engine must be one of {allowed}")
        return v


class LLMSettings(BaseSettings):
    """LLM configuration for Ollama"""
    backend: str = Field(default="ollama", description="LLM backend")
    model: str = Field(default="llama3.2", description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="Maximum tokens")
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['ollama']
        if v not in allowed:
            raise ValueError(f"LLM backend must be one of {allowed}")
        return v


class YOLOSettings(BaseSettings):
    """YOLO visual detection configuration"""
    enabled: bool = Field(default=True, description="Enable YOLO detection")
    model_path: Optional[str] = Field(default=None, description="Path to YOLO weights")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="IOU threshold for validation")
    use_opencv_fallback: bool = Field(default=True, description="Fallback to OpenCV if YOLO fails")


class ProcessingSettings(BaseSettings):
    """General processing configuration"""
    max_image_size: int = Field(default=2048, ge=512, le=4096)
    timeout_per_document: int = Field(default=30, ge=10, le=120)
    enable_logging: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()


class MasterDataSettings(BaseSettings):
    """Master data file paths"""
    dealers_file: Optional[str] = Field(default="master_data/dealers.txt")
    models_file: Optional[str] = Field(default="master_data/models.txt")


class Config(BaseSettings):
    """Main configuration object"""
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    yolo: YOLOSettings = Field(default_factory=YOLOSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    master_data: MasterDataSettings = Field(default_factory=MasterDataSettings)
    
    class Config:
        env_prefix = "DOC_AI_"
        env_nested_delimiter = "__"
        case_sensitive = False


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration (useful for testing)"""
    global _config
    _config = Config()
    return _config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully:")
    print(f"  OCR Engine: {config.ocr.engine}")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  YOLO Enabled: {config.yolo.enabled}")
    print(f"  Processing Timeout: {config.processing.timeout_per_document}s")
