"""
PaddleOCR engine for multilingual text extraction
Supports English, Hindi (Devanagari), Gujarati, and other Indian languages
"""
import os
# Disable oneDNN/MKL-DNN to avoid compatibility issues
# These MUST be set before importing paddleocr/paddle
os.environ['PADDLE_USE_ONEDNN'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'  # Disable PIR (Program Intermediate Representation)
os.environ['ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ONEDNN_VERBOSE'] = '0'

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import warnings
import gc
import sys
warnings.filterwarnings('ignore')


class OCREngine:
    """Wrapper for PaddleOCR for multilingual text extraction with memory optimization"""
    
    # Maximum image dimensions to avoid memory overflow
    MAX_IMAGE_SIZE = 2000  # pixels (reduced to prevent OOM crashes)
    
    def __init__(self, languages: List[str] = ['en', 'hi', 'ta']):
        """
        Initialize OCR engine (EasyOCR primary, Tesseract fallback)
        
        Args:
            languages: List of language codes (en, hi, ta, gu, etc.)
        """
        self.initialized = False
        self.engine_type = None
        self.languages = languages
        
        # Map language codes to engine-specific codes
        lang_map_easy = {
            'en': 'en',
            'hi': 'hi',      # Hindi (Devanagari)
            'ta': 'ta',      # Tamil
            'gu': 'gu',      # Gujarati
            'te': 'te',      # Telugu
            'kn': 'kn',      # Kannada
            'mr': 'mr',      # Marathi
        }
        
        # Tesseract language codes (different format)
        lang_map_tesseract = {
            'en': 'eng',
            'hi': 'hin',
            'ta': 'tam',
            'gu': 'guj',
            'te': 'tel',
            'kn': 'kan',
            'mr': 'mar',
        }
        
        # Try EasyOCR first (best accuracy, supports multiple languages)
        try:
            import easyocr
            # Convert language list for EasyOCR
            # NOTE: Some languages like Tamil have restrictions (can only be used with English)
            # We use English + primary language to avoid conflicts
            easy_langs = ['en']  # Always start with English
            
            # Add the first non-English language if specified
            if languages and languages[0] != 'en':
                primary_lang = lang_map_easy.get(languages[0], languages[0])
                if primary_lang not in easy_langs:
                    easy_langs.append(primary_lang)
            
            print(f"  Loading EasyOCR with languages: {easy_langs}...")
            self.reader = easyocr.Reader(easy_langs, gpu=False, verbose=False)
            self.engine_type = 'easyocr'
            self.initialized = True
            print(f"  ✓ EasyOCR engine initialized successfully")
            return
        except ImportError:
            print(f"  ⓘ EasyOCR not available, trying Tesseract...")
        except Exception as e:
            print(f"  ⚠ EasyOCR initialization failed: {e}")
            print(f"  Trying Tesseract as fallback...")
        
        # Fallback to Tesseract (lightweight, widely available)
        try:
            import pytesseract
            from pytesseract import Output
            
            # Build Tesseract language string
            tesseract_langs = []
            for lang in languages:
                mapped = lang_map_tesseract.get(lang, 'eng')
                if mapped not in tesseract_langs:
                    tesseract_langs.append(mapped)
            
            self.tesseract_lang = '+'.join(tesseract_langs)
            
            print(f"  Loading Tesseract OCR with languages: {self.tesseract_lang}...")
            
            # Test if tesseract is installed
            try:
                version = pytesseract.get_tesseract_version()
                print(f"  Found Tesseract version: {version}")
            except:
                print(f"  ✗ Tesseract not installed on system")
                print(f"  Install with: sudo apt-get install tesseract-ocr")
                print(f"  For languages: sudo apt-get install tesseract-ocr-hin tesseract-ocr-tam")
                raise ImportError("Tesseract not found")
            
            self.pytesseract = pytesseract
            self.engine_type = 'tesseract'
            self.initialized = True
            print(f"  ✓ Tesseract engine initialized successfully")
            
        except ImportError as e:
            print(f"  ✗ Error: No OCR engine available")
            print(f"  Install EasyOCR with: pip install easyocr")
            print(f"  Or install Tesseract with: sudo apt-get install tesseract-ocr pytesseract")
            self.initialized = False
            self.engine_type = None
        except Exception as e:
            print(f"  ✗ Error initializing Tesseract: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            self.engine_type = None
    
    def _resize_image_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds maximum dimensions to save memory
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (resized_image, scale_factor)
        """
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.MAX_IMAGE_SIZE:
            scale_factor = self.MAX_IMAGE_SIZE / max_dim
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            print(f"  Resizing image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.2f})")
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized, scale_factor
        
        return image, 1.0
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from image using available OCR engine
        
        Args:
            image: Input image as numpy array (RGB or BGR)
            
        Returns:
            List of dictionaries containing:
            - text: extracted text
            - bbox: bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - confidence: confidence score (0-1)
        """
        if not self.initialized:
            print(f"  Warning: OCR engine not initialized")
            return []
        
        extracted_items = []
        try:
            # Resize image if too large to avoid memory issues
            original_image = image.copy()
            image, scale_factor = self._resize_image_if_needed(image)
            
            # Ensure image is in correct format
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
            
            # Run OCR based on engine type
            if self.engine_type == 'easyocr':
                # EasyOCR processing
                try:
                    result = self.reader.readtext(image)
                except MemoryError as e:
                    print(f"  ✗ Memory Error during OCR - image too large or insufficient memory")
                    print(f"  Try reducing MAX_IMAGE_SIZE or using a smaller image")
                    gc.collect()
                    return []
                except Exception as e:
                    print(f"  ✗ OCR extraction error: {e}")
                    gc.collect()
                    raise
                
                # Parse EasyOCR results
                # EasyOCR returns: List of (bbox, text, confidence)
                for item in result:
                    try:
                        bbox_coords = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text = item[1]
                        confidence = item[2]
                        
                        if not text or not isinstance(text, str):
                            continue
                        
                        # Scale bbox back if image was resized
                        if scale_factor != 1.0:
                            bbox_int = [[int(x / scale_factor), int(y / scale_factor)] 
                                       for x, y in bbox_coords]
                        else:
                            bbox_int = [[int(x), int(y)] for x, y in bbox_coords]
                        
                        extracted_items.append({
                            'text': text.strip(),
                            'bbox': bbox_int,
                            'confidence': float(confidence)
                        })
                    except Exception as e:
                        # Silently skip problematic items
                        continue
                
                print(f"  [OK] EasyOCR extracted {len(extracted_items)} text items")
            
            elif self.engine_type == 'tesseract':
                # Tesseract processing
                try:
                    from pytesseract import Output
                    # Get detailed data including bounding boxes
                    data = self.pytesseract.image_to_data(
                        image, 
                        lang=self.tesseract_lang,
                        output_type=Output.DICT
                    )
                except MemoryError as e:
                    print(f"  ✗ Memory Error during OCR - image too large or insufficient memory")
                    print(f"  Try reducing MAX_IMAGE_SIZE or using a smaller image")
                    gc.collect()
                    return []
                except Exception as e:
                    print(f"  ✗ OCR extraction error: {e}")
                    gc.collect()
                    raise
                
                # Parse Tesseract results
                # Tesseract returns: dict with keys: text, conf, left, top, width, height
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    try:
                        text = data['text'][i]
                        conf = float(data['conf'][i]) / 100.0  # Convert 0-100 to 0-1
                        
                        # Skip empty text or low confidence
                        if not text or not isinstance(text, str) or text.strip() == '':
                            continue
                        if conf < 0.3:  # Skip very low confidence
                            continue
                        
                        # Get bounding box
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        # Convert to 4-point bbox format and scale back if resized
                        if scale_factor != 1.0:
                            bbox_int = [
                                [int(x / scale_factor), int(y / scale_factor)],
                                [int((x + w) / scale_factor), int(y / scale_factor)],
                                [int((x + w) / scale_factor), int((y + h) / scale_factor)],
                                [int(x / scale_factor), int((y + h) / scale_factor)]
                            ]
                        else:
                            bbox_int = [
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ]
                        
                        extracted_items.append({
                            'text': text.strip(),
                            'bbox': bbox_int,
                            'confidence': conf
                        })
                    except Exception as e:
                        # Silently skip problematic items
                        continue
                
                print(f"  [OK] Tesseract extracted {len(extracted_items)} text items")
            else:
                print(f"  Warning: No active OCR engine found.")
                return []
            
            # Apply validation and filtering
            filtered_items = self._validate_and_filter(extracted_items)
            
            # Cleanup memory after extraction
            del image, original_image
            gc.collect()
            
            return filtered_items
            
        except Exception as e:
            print(f"  ✗ OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()  # Cleanup on error
            return []
    
    def get_full_text(self, image: np.ndarray) -> str:
        """
        Get all extracted text as a single string
        
        Args:
            image: Input image
            
        Returns:
            Concatenated text
        """
        extracted = self.extract_text(image)
        # Sort by vertical position (top to bottom)
        sorted_items = sorted(extracted, key=lambda x: x['bbox'][0][1])
        return ' '.join([item['text'] for item in sorted_items])
    
    def get_text_with_positions(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Get text with simplified bounding box positions
        
        Args:
            image: Input image
            
        Returns:
            List of (text, (x, y, w, h)) tuples
        """
        extracted = self.extract_text(image)
        results = []
        
        for item in extracted:
            bbox = item['bbox']
            # Convert polygon to rectangle
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            
            results.append((item['text'], (x, y, w, h)))
        
        return results
    
    def _validate_and_filter(self, results: List[Dict]) -> List[Dict]:
        """
        Validate and filter OCR results to remove low-quality extractions
        
        Args:
            results: Raw OCR results
            
        Returns:
            Filtered results
        """
        import re
        
        filtered = []
        for item in results:
            text = item['text'].strip()
            confidence = item.get('confidence', 0.0)
            
            # Filter out very low confidence results
            if confidence < 0.3:
                continue
            
            # Filter out very short text (likely noise)
            if len(text) < 2:
                continue
            
            # Filter out text that's mostly special characters
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text) if text else 0
            if alpha_ratio < 0.5:
                continue
            
            filtered.append(item)
        
        return filtered


class FallbackOCR:
    """Fallback text extraction using simple line detection"""
    
    def __init__(self):
        """Initialize fallback OCR"""
        self.initialized = True
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Simple fallback text extraction
        
        Args:
            image: Input image
            
        Returns:
            List of extracted text items (empty in this fallback)
        """
        # This fallback returns empty - can be extended with simple pattern matching
        return []
    
    def get_full_text(self, image: np.ndarray) -> str:
        """Get all text as string"""
        extracted = self.extract_text(image)
        return ' '.join([item['text'] for item in extracted])
