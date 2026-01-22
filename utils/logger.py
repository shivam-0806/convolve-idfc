"""
Structured logging infrastructure for Document AI system
"""
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class DocumentAILogger:
    """Custom logger with structured logging support"""
    
    def __init__(self, name: str = "DocumentAI", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self.context: Dict[str, Any] = {}
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"doc_ai_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs):
        """Set context for logging (e.g., doc_id, component)"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
    
    def _format_message(self, msg: str) -> str:
        """Format message with context"""
        if self.context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{ctx_str}] {msg}"
        return msg
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(msg), **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(msg), **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(msg), **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(msg), **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(msg), **kwargs)
    
    def log_extraction(self, doc_id: str, field: str, value: Any, confidence: float, method: str):
        """Log field extraction with structured data"""
        self.info(
            f"Extracted {field}: {value} (confidence: {confidence:.2f}, method: {method})",
            extra={'doc_id': doc_id, 'field': field, 'confidence': confidence}
        )
    
    def log_performance(self, doc_id: str, component: str, duration_sec: float):
        """Log performance metrics"""
        self.debug(
            f"{component} completed in {duration_sec:.2f}s",
            extra={'doc_id': doc_id, 'component': component, 'duration': duration_sec}
        )


# Global logger instance
_logger: Optional[DocumentAILogger] = None


def get_logger(name: str = "DocumentAI", level: str = "INFO") -> DocumentAILogger:
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = DocumentAILogger(name, level)
    return _logger


if __name__ == "__main__":
    # Test logger
    logger = get_logger()
    logger.set_context(doc_id="test_123")
    logger.info("Testing logger")
    logger.log_extraction("test_123", "dealer_name", "Test Dealer", 0.95, "llm")
    logger.clear_context()
