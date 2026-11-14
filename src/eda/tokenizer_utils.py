"""
Tokenizer utility supporting HuggingFace transformers (defaulting to t5-base)
with automatic fallback to regex BPE whitespace tokenization.
"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PipelineTokenizer:
    """Wrapper around HuggingFace or fallback tokenizer."""

    def __init__(self, model_name: str = "t5-base", use_hf: bool = True):
        self.model_name = model_name
        self.hf_tokenizer = None
        
        if use_hf:
            try:
                from transformers import AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Loaded HuggingFace tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load HF tokenizer '{model_name}': {e}. Falling back to BPE/regex tokenizer.")
                self.hf_tokenizer = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of token strings."""
        if not text:
            return []
        if self.hf_tokenizer is not None:
            try:
                return self.hf_tokenizer.tokenize(text)
            except Exception as e:
                logger.warning(f"HF tokenization failed, using fallback: {e}")
        
        # Fallback BPE / whitespace regex tokenizer
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def token_count(self, text: str) -> int:
        """Returns the number of tokens in text."""
        return len(self.tokenize(text))
