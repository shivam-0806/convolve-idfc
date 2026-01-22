"""
Field extraction logic for invoice/quotation documents
Supports English, Hindi (Devanagari), Gujarati
"""
import re
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz, process


class FieldExtractor:
    """Extract specific fields from OCR text"""

    def __init__(self, master_dealers: Optional[List[str]] = None,
                 master_models: Optional[List[str]] = None):

        self.master_dealers = master_dealers or []
        self.master_models = master_models or []

        # ---------------------------
        # Language-aware patterns
        # ---------------------------

        self.hp_patterns = [
            r'(\d+)\s*(?:HP|H\.P\.|Horse\s*Power)',
            r'(?:HP|H\.P\.|Horse\s*Power)\s*[:\-]?\s*(\d+)',
            r'(\d+)\s*(?:एचपी)',               # Hindi
            r'(\d+)\s*(?:હોર્સ\s*પાવર)',       # Gujarati
        ]

        self.cost_patterns = [
            r'(?:Rs\.?|INR|₹)\s*(\d[\d,]*)',
            r'(?:Total|Amount|Cost|Price|Grand\s*Total)\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*(\d[\d,]*)',
            r'(?:कुल\s*राशि|कुल)\s*[:\-]?\s*(\d[\d,]*)',        # Hindi
            r'(?:કુલ\s*રકમ|કુલ)\s*[:\-]?\s*(\d[\d,]*)',        # Gujarati
            r'(\d[\d,]+)\s*(?:only|/-|rupees)',
        ]

        self.dealer_keywords = [
            'industries', 'corporation', 'ltd', 'limited', 'pvt', 'private',
            'agro', 'tractors', 'motors', 'traders', 'enterprises',
            'एग्रीकल्चर', 'ट्रैक्टर',           # Hindi
            'ટ્રેક્ટર', 'એગ્રી',                 # Gujarati
            # Add brand names that often appear in dealer names
            'kubota', 'mahindra', 'swaraj', 'sonalika', 'massey',
            'john deere', 'new holland', 'eicher', 'tafe', 'vst',
            'indo farm', 'captain', 'ace', 'preet', 'farmtrac'
        ]

        self.model_keywords = [
            'swaraj', 'mahindra', 'john deere', 'sonalika',
            'new holland', 'massey', 'eicher', 'farmtrac',
            'powertrac', 'tafe', 'kubota', 'vst', 'indo farm',
            'captain', 'ace', 'preet'
        ]

    # --------------------------------------------------
    # Utility: sort OCR lines in reading order
    # --------------------------------------------------
    # --------------------------------------------------
    # Utility: sort OCR lines in reading order
    # --------------------------------------------------
    def _sort_ocr(self, ocr_data: List[Dict]) -> List[Dict]:
        return sorted(
            ocr_data,
            key=lambda x: (x['bbox'][1], x['bbox'][0])
        )

    def _find_contributing_indices(self, ocr_data: List[Dict], target_text: str) -> List[int]:
        """Find indices of OCR items that likely contributed to the target text"""
        indices = []
        if not target_text: return indices
        
        target_words = set(re.findall(r'\w+', target_text.lower()))
        
        for i, item in enumerate(ocr_data):
            item_words = set(re.findall(r'\w+', item['text'].lower()))
            # If significant overlap in words
            if target_words & item_words:
                indices.append(i)
                continue
                
            # Or if target is a substring of item (or vice-versa)
            t_clean = target_text.lower().replace(' ', '')
            i_clean = item['text'].lower().replace(' ', '')
            if t_clean in i_clean or i_clean in t_clean:
                indices.append(i)
        
        return list(set(indices))

    # --------------------------------------------------
    # Dealer Name
    # --------------------------------------------------
    def extract_dealer_name(self, ocr_data: List[Dict]) -> Tuple[str, float, List[int]]:
        ocr_data = self._sort_ocr(ocr_data)
        candidates = []

        # Expanded exclusion patterns
        exclude_patterns = [
            r'@', r'www\.|\.com|\.in',
            r'GSTIN|PAN|TIN|CIN',
            r'EPBX|FAX|EMAIL|MOBILE|TEL|PHONE',
            r'INVOICE|TAX\s*INVOICE|QUOTATION|ESTIMATE',
            r'WELCOME|THANK\s*YOU',
            r'^\d+$',
            r'Ref\s*No|Date|Block|Dist|Dear\s*Sir',
            r'Village|Villege|District|Distt',
            r'Customer|Sir|Madam|Mr\.|Mrs\.',
            r'Shreel|Shree(?!\s*[A-Z])',
            r'^Branch$|^Address$|^Phone\s*No',
            r'^Name\s*[:;]?$|^GST\s*IN$',
            r'^TYRES?$|^Signature$|^Date$',
            r'^Place$|^City$|^State$',
            r'^To\s*[:;]?$|^From\s*[:;]?$|^Subject$',
            r'Financed\s*By',
            r'Details',
            r'^Sales$|^Service$|^Parts$', # Generic single words
            r'^Size$|^Qnty$|^Rate$|^Amount$',
            r'^Description$',
        ]

        # Explicit dealer keywords (high confidence)
        dealer_suffixes = [
            r'Tractors?', r'Motors?', r'Automobiles?', r'Agencies', 
            r'Enterprises', r'Traders', r'Sales', r'Service'
        ]
        
        # Look for "Authorized Dealer" label
        auth_dealer_idx = -1
        for i, item in enumerate(ocr_data[:20]):
            if re.search(r'Authori[sz]ed\s*Dealer', item['text'], re.IGNORECASE):
                auth_dealer_idx = i
                break
        
        # If found "Authorized Dealer", look at lines immediately above/below
        if auth_dealer_idx >= 0:
            # Check line above (often company name)
            if auth_dealer_idx > 0:
                text = ocr_data[auth_dealer_idx-1]['text'].strip()
                if len(text) > 5 and not any(re.search(p, text, re.IGNORECASE) for p in exclude_patterns):
                    candidates.append((text, 0.95, auth_dealer_idx-1))
            
            # Check line same (if long) or below
            if auth_dealer_idx + 1 < len(ocr_data):
                text = ocr_data[auth_dealer_idx+1]['text'].strip()
                if len(text) > 5 and not any(re.search(p, text, re.IGNORECASE) for p in exclude_patterns):
                    candidates.append((text, 0.90, auth_dealer_idx+1))

        # Standard search top 20 lines
        for i, item in enumerate(ocr_data[:20]):
            text = item['text'].strip()
            conf = item.get('confidence', 0.5)

            if len(text) < 4:
                continue

            if any(re.search(p, text, re.IGNORECASE) for p in exclude_patterns):
                continue
            
            # Exclude if starts with "Cost of" or "Price of"
            if re.match(r'^(Cost|Price|Value)\s+of', text, re.IGNORECASE):
                continue

            # Skip if looks like phone number
            if sum(c.isdigit() for c in text) > 6:
                continue
            
            # Skip if just a keyword
            if text.lower() in ['tractor', 'tractors', 'sales', 'service', 'amount', 'total', 'pulley']:
                continue

            score = conf
            text_lower = text.lower()
            
            # Boost if contains dealer suffix
            if any(re.search(s, text, re.IGNORECASE) for s in dealer_suffixes):
                score *= 1.4
            
            # Boost if contains simple dealer keywords
            if any(k in text_lower for k in self.dealer_keywords):
                score *= 1.2

            # Boost if "Corporation" or "Ltd" (but check if it's manufacturer)
            if 'corporation' in text_lower or 'ltd' in text_lower:
                if 'escort' in text_lower or 'mahindra' in text_lower or 'tafe' in text_lower:
                    score *= 0.8  # Likely manufacturer, reduce slightly but keep as fallback
                else:
                    score *= 1.3
            
            # Position boost
            if i < 5:
                score += 0.2
            elif i < 10:
                score += 0.1
            
            if score > 0.6:
                candidates.append((text, score, i))

        if candidates:
            # Sort by score primarily
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_text, best_conf, best_idx = candidates[0]
            
            # Cleanup common issues
            best_text = re.sub(r'^[.,\-_:]+\s*', '', best_text)
            
            # Return text, conf, and LIST of source indices (just one here)
            return best_text, min(best_conf, 0.95), [best_idx]

        return "", 0.0, []

    # --------------------------------------------------
    # Model Name
    # --------------------------------------------------
    def extract_model_name(self, ocr_data: List[Dict]) -> Tuple[str, float, List[int]]:
        ocr_data = self._sort_ocr(ocr_data)
        candidates = []

        # Enhanced model patterns specific to tractors
        model_patterns = [
            # Patterns with explicit prefixes
            r'(?:Model|Tractor)\s*[:\-\.]\s*([A-Z0-9\s\-\+]+)',
            # Brand + Model Number (e.g. Swaraj 744, Mahindra 475)
            r'(?:Swaraj|Mahindra|Sonalika|Farmtrac|Powertrac|Eicher|John\s*Deere|New\s*Holland|Kubota)\s+([A-Z0-9\s\-\+]+)',
            # Common Tractor Suffixes
            r'(\d{3,4}\s*(?:DI|FE|XM|XP|PLUS|TECH|HR|HST|HDM|4WD))',
            r'([A-Z]+\s+\d{3,4}\s*(?:DI|FE|XM))',
            # Specific format matches seen in data
            r'(?:PT|MF)\s*(\d{3,4}[A-Z0-9\s]*)',  # PT 434, MF 1035
            r'(?:TTG|Tar)\s*(\d{3,4}\s*[A-Z]*)',  # TTG 855
        ]

        full_text = " ".join(item['text'] for item in ocr_data)
        
        # 1. Search in full text (good for split lines)
        for pattern in model_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                model_text = match.group(1).strip()
                if len(model_text) > 3 and any(c.isdigit() for c in model_text):
                    # Clean up
                    model_text = re.sub(r'^(?:No\.?|Name|Type)\s*', '', model_text, flags=re.IGNORECASE)
                    
                    # Find provenance
                    sources = self._find_contributing_indices(ocr_data, model_text)
                    candidates.append((model_text, 0.90, sources))

        # 2. Search in individual items
        for i, item in enumerate(ocr_data):
            text = item['text'].strip()
            if len(text) < 3: continue
            
            # Check for model keywords
            if any(s in text for s in ['DI', 'FE', 'XM', 'XP', '4WD', 'HP']):
                if any(c.isdigit() for c in text): # Must have digit
                     # Avoid just "HP" or "4WD"
                    if len(text) > 4:
                         candidates.append((text, 0.85, [i]))

        if candidates:
            # Sort by length (longer usually better for model names) and confidence
            candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            
            best_text, best_conf, best_indices = candidates[0]
            
            # Clean up trailing junk
            best_text = re.sub(r'\s*\(?HP.*$', '', best_text, flags=re.IGNORECASE)
            best_text = best_text.strip()
            
            return best_text, best_conf, best_indices

        return "", 0.0, []

    # --------------------------------------------------
    # Horse Power
    # --------------------------------------------------
    def extract_horse_power(self, ocr_data: List[Dict]) -> Tuple[Optional[int], float, List[int]]:
        text = " ".join(item['text'] for item in ocr_data)
        candidates = []
        
        # Enhanced HP patterns
        hp_patterns = [
            r'HP\s*[:\-\.]?\s*(\d{2,3})',
            r'(\d{2,3})\s*HP',
            r'(\d{2,3})\s*H\.P\.',
            r'\(HP[-:\s]*(\d{2,3})\)',  # (HP- 39)
            r'Power\s*[:\-\.]?\s*(\d{2,3})',
        ]

        # 1. Regex search
        for pattern in hp_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    hp = int(match.group(1))
                    if 15 <= hp <= 100: # Tractor range
                        # Find source
                        src_txt = match.group(0)
                        sources = self._find_contributing_indices(ocr_data, src_txt)
                        candidates.append((hp, 0.90, sources))
                except: pass
        
        # 2. Standalone number search with HP keyword proximity
        for i, item in enumerate(ocr_data):
            txt = item['text'].upper()
            if 'HP' in txt or 'H.P' in txt:
                # Check digits in same item
                digits = re.findall(r'(\d{2,3})', txt)
                for d in digits:
                    if 15 <= int(d) <= 100: candidates.append((int(d), 0.85, [i]))
                
                # Check neighbors
                for j in range(max(0, i-2), min(len(ocr_data), i+3)):
                    nb_txt = ocr_data[j]['text']
                    if nb_txt.isdigit():
                        val = int(nb_txt)
                        if 15 <= val <= 100:
                            candidates.append((val, 0.80, [j]))

        if candidates:
            # Most common candidate
            from collections import Counter
            counts = Counter([c[0] for c in candidates])
            best_hp = counts.most_common(1)[0][0]
            
            # Find associated metadata for best HP
            for c_val, c_conf, c_src in candidates:
                if c_val == best_hp:
                    return best_hp, 0.90, c_src

        return None, 0.0, []

    # --------------------------------------------------
    # Asset Cost
    # --------------------------------------------------
    def extract_asset_cost(self, ocr_data: List[Dict]) -> Tuple[Optional[float], float, List[int]]:
        candidates = []
        
        # Common valid cost range for tractors (3L to 30L)
        MIN_COST = 300000
        MAX_COST = 3000000 
        
        sorted_ocr = sorted(ocr_data, key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))

        # Helper to clean and parse amount
        def parse_amount(s):
            s = s.replace(',', '').replace('Rs.', '').replace('Rs', '').replace('/-', '').replace('=', '').strip()
            try:
                return float(s)
            except: return 0.0

        # 1. Look for row with "Total" or "Amount" and a number on the same line or line below
        for i, item in enumerate(sorted_ocr):
            txt = item['text'].lower()
            if 'total' in txt or 'amount' in txt or 'cost' in txt or 'price' in txt:
                # Check neighbors
                for j in range(max(0, i-2), min(len(sorted_ocr), i+5)):
                    val = parse_amount(sorted_ocr[j]['text'])
                    if MIN_COST <= val <= MAX_COST:
                        candidates.append((val, 0.95, [j]))

        # 2. Look for explicit currency amounts anywhere
        for i, item in enumerate(sorted_ocr):
            matches = re.findall(r'(?:Rs\.?|₹)\s*([\d,]+)', item['text'], re.IGNORECASE)
            for m in matches:
                val = parse_amount(m)
                if MIN_COST <= val <= MAX_COST:
                    candidates.append((val, 0.90, [i]))

        # 3. Look for large numbers that look like money (e.g. 7,96,000)
        for i, item in enumerate(sorted_ocr):
            txt = item['text']
            # Match number like X,XX,XXX
            if re.search(r'\d{1,2},\d{2},\d{3}', txt):
                 val = parse_amount(txt)
                 if MIN_COST <= val <= MAX_COST:
                     candidates.append((val, 0.85, [i]))

        if candidates:
            # Prefer larger amounts (Total is usually largest amount) within range
            # But not too large (avoid phone numbers interpreted as int)
            # Filter out likely phone numbers (10 digits starting with 6-9) unless formatted usually
            valid_candidates = [c for c in candidates if c[0] < 10000000] 
            if valid_candidates:
                best = max(valid_candidates, key=lambda x: x[0])
                return best[0], best[1], best[2]

        return None, 0.0, []

    # --------------------------------------------------
    # Final Aggregation with Hybrid ML/LLM Support
    # --------------------------------------------------
    def extract_all_fields(self, ocr_data: List[Dict],
                           signature_result: Dict,
                           stamp_result: Dict,
                           image: 'np.ndarray' = None,
                           use_ml: bool = False,
                           ml_extractor = None,
                           ml_confidence_threshold: float = 0.7) -> Dict:
        """
        Extract all fields using hybrid LLM + rule-based approach
        
        Args:
            ocr_data: OCR results
            signature_result: Signature detection result
            stamp_result: Stamp detection result
            image: Original document image (optional for LLM)
            use_ml: Whether to use ML/LLM-based extraction
            ml_extractor: LlamaFieldExtractor instance (or LayoutLMv3 for backward compatibility)
            ml_confidence_threshold: Minimum confidence for ML/LLM results (0.0-1.0)
            
        Returns:
            Dictionary with extracted fields and metadata
        """
        # Initialize result tracking
        ml_result = None
        extraction_method = "rule-based"
        
        # Try ML/LLM extraction first if enabled
        if use_ml and ml_extractor is not None:
            try:
                # Check if it's LLM extractor (has extract_fields_llm method)
                if hasattr(ml_extractor, 'extract_fields_llm'):
                    # Llama-based extraction (OFFLINE via Ollama)
                    ml_result = ml_extractor.extract_fields_llm(ocr_data)
                    extraction_method = "llm"
                # Check if it's LayoutLMv3 extractor (has extract_fields method)
                elif hasattr(ml_extractor, 'extract_fields') and image is not None:
                    # LayoutLMv3-based extraction (requires image)
                    ml_result = ml_extractor.extract_fields(image, ocr_data)
                    extraction_method = "layoutlmv3"
                else:
                    print(f"  [!] Unknown ML extractor type, falling back to rules")
                    extraction_method = "hybrid-fallback"
                
                # ALWAYS use ML/LLM results if extraction succeeded (no confidence threshold check)
                if ml_result:
                    # Use ML/LLM results directly
                    ml_result['signature'] = signature_result
                    ml_result['stamp'] = stamp_result
                    
                    # Calculate overall confidence including signature/stamp
                    sig_conf = signature_result.get('confidence', 0.9 if signature_result['present'] else 0.0)
                    stamp_conf = stamp_result.get('confidence', 0.9 if stamp_result['present'] else 0.0)
                    
                    weights = {
                        'fields': 0.90,
                        'signature': 0.05,
                        'stamp': 0.05
                    }
                    
                    # Get field confidence from ML result
                    ml_field_conf = ml_result.get('confidence', 0.5)
                    
                    overall_conf = (
                        ml_field_conf * weights['fields'] +
                        sig_conf * weights['signature'] +
                        stamp_conf * weights['stamp']
                    )
                    
                    ml_result['confidence'] = round(overall_conf, 2)
                    ml_result['extraction_method'] = extraction_method
                    
                    return ml_result

                    
            except Exception as e:
                print(f"  [!] ML/LLM extraction failed, falling back to rules: {e}")
                extraction_method = "hybrid-fallback"
        
        # Rule-based extraction (original logic)
        dealer, d_conf, _ = self.extract_dealer_name(ocr_data)
        model, m_conf, _ = self.extract_model_name(ocr_data)
        hp, hp_conf, _ = self.extract_horse_power(ocr_data)
        cost, c_conf, _ = self.extract_asset_cost(ocr_data)

        sig_conf = signature_result.get('confidence', 0.9 if signature_result['present'] else 0.0)
        stamp_conf = stamp_result.get('confidence', 0.9 if stamp_result['present'] else 0.0)

        weights = {
            'dealer': 0.30,
            'model': 0.25,
            'hp': 0.15,
            'cost': 0.20,
            'signature': 0.05,
            'stamp': 0.05
        }

        score, total = 0.0, 0.0
        for val, w in zip(
            [d_conf, m_conf, hp_conf, c_conf, sig_conf, stamp_conf],
            weights.values()
        ):
            if val > 0:
                score += val * w
                total += w

        overall_conf = round(score / total, 2) if total else 0.0

        return {
            "dealer_name": dealer,
            "model_name": model,
            "horse_power": hp,
            "asset_cost": cost,
            "signature": signature_result,
            "stamp": stamp_result,
            "confidence": overall_conf,
            "extraction_method": extraction_method,
            "field_confidences": {
                "dealer_name": d_conf,
                "model_name": m_conf,
                "horse_power": hp_conf,
                "asset_cost": c_conf
            }
        }




# --------------------------------------------------
# Master Data Loading
# --------------------------------------------------
def load_master_data(dealers_file: Optional[str] = None,
                     models_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Load master data from files

    Args:
        dealers_file: Path to dealers master file
        models_file: Path to models master file

    Returns:
        Tuple of (dealers_list, models_list)
    """
    dealers = []
    models = []

    if dealers_file:
        try:
            with open(dealers_file, 'r', encoding='utf-8') as f:
                dealers = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: Dealers file not found: {dealers_file}")

    if models_file:
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                models = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: Models file not found: {models_file}")

    return dealers, models
