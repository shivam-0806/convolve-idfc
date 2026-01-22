"""
Llama 3 Field Extractor via Ollama
Implements intelligent field extraction using LLM reasoning
"""
import json
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from fuzzywuzzy import fuzz, process

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[!] Ollama not available. Install with: pip install ollama")

from utils.logger import get_logger

logger = get_logger()


class LlamaFieldExtractor:
    """Extract fields using Llama 3 via Ollama"""
    
    def __init__(self, 
                 model: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 timeout: int = 30,
                 master_dealers: Optional[List[str]] = None,
                 master_models: Optional[List[str]] = None):
        """
        Initialize Llama extractor
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "llama3")
            base_url: Ollama server URL
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            master_dealers: List of known dealers for fuzzy matching
            master_models: List of known models for matching
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama package not installed. Install with: pip install ollama")
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.master_dealers = master_dealers or []
        self.master_models = master_models or []
        
        # Verify Ollama connection (OFFLINE - no internet required)
        logger.info(f"Connecting to Ollama (OFFLINE mode) at {base_url}...")
        try:
            response = ollama.list()
            # Handle both dict and object response formats
            if isinstance(response, dict):
                models_list = response.get('models', [])
            else:
                models_list = getattr(response, 'models', [])
            
            # Extract model names - handle both dict and object formats
            available_models = []
            for m in models_list:
                if isinstance(m, dict):
                    available_models.append(m.get('name', m.get('model', '')))
                else:
                    available_models.append(getattr(m, 'name', getattr(m, 'model', '')))
            
            logger.info(f"✓ Ollama connected (OFFLINE). Available models: {available_models}")
            
            if not any(self.model in m for m in available_models):
                logger.warning(f"Model {self.model} not found locally. Available: {available_models}")
                logger.warning(f"Run: ollama pull {self.model}")
                raise ValueError(f"Model {self.model} not available. Please run: ollama pull {self.model}")
        except (ConnectionError, ValueError):
            # Re-raise these specific errors
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}. Is Ollama running? (ollama serve)")

    
    def _build_prompt(self, ocr_text: str) -> str:
        """
        Build extraction prompt following task.txt specifications
        
        Args:
            ocr_text: Concatenated OCR text
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are a professional invoice auditor. Extract data from this messy OCR text.

Extract the following fields:

1. Dealer Name: Identify the tractor dealer company (NOT the manufacturer like Mahindra, Swaraj, etc.). 
   Look for companies with words like "Tractors", "Motors", "Traders", "Corporation", "Ltd", "Enterprises".
   
2. Model Name: Extract the exact tractor model (e.g., '575 DI', 'Swaraj 744 FE', 'Mahindra 475 DI').
   Include brand and model number. It is not supposed to be very long, should be within 6-7 words.

3. Horse Power: Extract the HP value as a number. Look for patterns like "HP: 48" or "48 HP".
   Typical tractor range is 15-200 HP.

4. Asset Cost: Locate the final 'Grand Total', 'Total Amount', or 'Net Amount'. 
   Ignore subtotals and partial amounts. Look for the largest amount in rupees.
   Format: numeric value without currency symbols.

5. Vernacular Support: Handle keywords in:
   - Hindi: मूल्य (price), कुल (total), एचपी (HP)
   - Gujarati: કિંમત (price), કુલ (total)

IMPORTANT RULES:
- Dealer name should be the DEALERSHIP, not the tractor manufacturer
- Return ONLY a valid JSON object, no additional text
- If a field cannot be found, use null for numbers or empty string for text
- Be precise with model names, include all parts (e.g., "744 FE" not just "744")

OCR TEXT:
---
{ocr_text}
---

Return ONLY this JSON format:
{{
  "dealer_name": "string or empty",
  "model_name": "string or empty", 
  "horse_power": number or null,
  "asset_cost": number or null
}}
"""
        return prompt.format(ocr_text=ocr_text)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract JSON
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed field dictionary
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # Try to extract from code blocks
                code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if code_block:
                    json_str = code_block.group(1)
                else:
                    logger.warning("No JSON found in LLM response")
                    return {}
            else:
                json_str = json_match.group(0)
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate and clean
            result = {
                'dealer_name': str(data.get('dealer_name', '')).strip(),
                'model_name': str(data.get('model_name', '')).strip(),
                'horse_power': data.get('horse_power'),
                'asset_cost': data.get('asset_cost')
            }
            
            # Convert to proper types
            if result['horse_power'] is not None:
                try:
                    result['horse_power'] = int(float(result['horse_power']))
                except (ValueError, TypeError):
                    result['horse_power'] = None
            
            if result['asset_cost'] is not None:
                try:
                    result['asset_cost'] = float(result['asset_cost'])
                except (ValueError, TypeError):
                    result['asset_cost'] = None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {}
    
    def _fuzzy_match_dealer(self, dealer_name: str) -> Tuple[str, float]:
        """
        Fuzzy match dealer name against master list
        
        Args:
            dealer_name: Extracted dealer name
            
        Returns:
            (matched_name, confidence_score)
        """
        if not dealer_name or not self.master_dealers:
            return dealer_name, 0.8  # Default confidence
        
        # Find best match
        result = process.extractOne(dealer_name, self.master_dealers, scorer=fuzz.token_sort_ratio)
        
        if result:
            matched_name, score = result[0], result[1] / 100.0
            
            # Use matched name if score is high enough
            if score >= 0.7:
                logger.info(f"Fuzzy matched '{dealer_name}' -> '{matched_name}' (score: {score:.2f})")
                return matched_name, score
        
        return dealer_name, 0.6  # Lower confidence for unmatched
    
    def _calculate_confidence(self, fields: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for extracted fields
        
        Args:
            fields: Extracted fields dictionary
            
        Returns:
            Dictionary of field confidences
        """
        confidences = {}
        
        # Dealer name
        if fields.get('dealer_name'):
            _, dealer_conf = self._fuzzy_match_dealer(fields['dealer_name'])
            confidences['dealer_name'] = dealer_conf
        else:
            confidences['dealer_name'] = 0.0
        
        # Model name (higher confidence if has numbers and letters)
        model = fields.get('model_name', '')
        if model and any(c.isdigit() for c in model) and any(c.isalpha() for c in model):
            confidences['model_name'] = 0.85
        elif model:
            confidences['model_name'] = 0.6
        else:
            confidences['model_name'] = 0.0
        
        # Horse power (high confidence if in valid range)
        hp = fields.get('horse_power')
        if hp and 15 <= hp <= 200:
            confidences['horse_power'] = 0.9
        elif hp:
            confidences['horse_power'] = 0.5
        else:
            confidences['horse_power'] = 0.0
        
        # Asset cost (high confidence if in valid range)
        cost = fields.get('asset_cost')
        if cost and 300000 <= cost <= 3000000:
            confidences['asset_cost'] = 0.9
        elif cost:
            confidences['asset_cost'] = 0.5
        else:
            confidences['asset_cost'] = 0.0
        
        return confidences
    
    def extract_fields_llm(self, ocr_data: List[Dict]) -> Dict[str, Any]:
        """
        Extract fields using Llama 3
        
        Args:
            ocr_data: List of OCR results with text and bbox
            
        Returns:
            Dictionary with extracted fields and metadata
        """
        start_time = time.time()
        
        # Concatenate OCR text
        ocr_text = " ".join(item['text'] for item in ocr_data)
        
        logger.debug(f"OCR text length: {len(ocr_text)} chars")
        
        # Build prompt
        prompt = self._build_prompt(ocr_text)
        
        try:
            # Call Ollama
            logger.info(f"Calling Llama model: {self.model}")
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                }
            )
            
            # Extract response text
            response_text = response.get('response', '')
            logger.debug(f"LLM response: {response_text[:200]}...")
            
            # Parse response
            fields = self._parse_llm_response(response_text)
            
            # Apply fuzzy matching for dealer
            if fields.get('dealer_name'):
                matched_dealer, _ = self._fuzzy_match_dealer(fields['dealer_name'])
                fields['dealer_name'] = matched_dealer
            
            # Calculate confidences
            field_confidences = self._calculate_confidence(fields)
            
            # Overall confidence (weighted average)
            weights = {'dealer_name': 0.3, 'model_name': 0.25, 'horse_power': 0.2, 'asset_cost': 0.25}
            overall_conf = sum(field_confidences[f] * weights[f] for f in weights)
            
            duration = time.time() - start_time
            logger.info(f"LLM extraction completed in {duration:.2f}s (confidence: {overall_conf:.2f})")
            
            return {
                'dealer_name': fields.get('dealer_name', ''),
                'model_name': fields.get('model_name', ''),
                'horse_power': fields.get('horse_power'),
                'asset_cost': fields.get('asset_cost'),
                'confidence': round(overall_conf, 2),
                'field_confidences': field_confidences,
                'extraction_method': 'llm',
                'llm_duration_sec': round(duration, 2)
            }
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {
                'dealer_name': '',
                'model_name': '',
                'horse_power': None,
                'asset_cost': None,
                'confidence': 0.0,
                'field_confidences': {},
                'extraction_method': 'llm-failed',
                'error': str(e)
            }


if __name__ == "__main__":
    # Test extractor
    print("Testing LlamaFieldExtractor...")
    
    # Sample OCR data
    ocr_data = [
        {'text': 'Krishna Tractors Ltd', 'bbox': [100, 100, 300, 120]},
        {'text': 'Authorized Dealer', 'bbox': [100, 130, 250, 150]},
        {'text': 'Swaraj 744 FE', 'bbox': [100, 200, 200, 220]},
        {'text': 'HP: 48', 'bbox': [100, 230, 150, 250]},
        {'text': 'Total Amount: Rs. 8,01,815/-', 'bbox': [100, 400, 300, 420]}
    ]
    
    try:
        extractor = LlamaFieldExtractor(model="llama3.2")
        result = extractor.extract_fields_llm(ocr_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And model is pulled: ollama pull llama3.2")
