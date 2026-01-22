"""
Visual detection for signatures and stamps
Supports YOLOv8 (primary) and OpenCV (fallback)
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from utils.logger import get_logger

logger = get_logger()


class VisualDetector:
    """Detector for signatures and stamps in documents"""
    
    def __init__(self, 
                 yolo_model_path: Optional[str] = None,
                 use_yolo: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        """
        Initialize detector with YOLO and OpenCV support
        
        Args:
            yolo_model_path: Path to YOLO weights file (.pt)
            use_yolo: Whether to use YOLO (if False, use OpenCV only)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for validation
        """
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.yolo_model = None
        
        # Try to load YOLO model
        if self.use_yolo:
            if yolo_model_path and Path(yolo_model_path).exists():
                try:
                    self.yolo_model = YOLO(yolo_model_path)
                    logger.info(f"Loaded YOLO model from {yolo_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model: {e}")
                    logger.warning("Falling back to OpenCV detection")
                    self.use_yolo = False
            else:
                logger.info("YOLO model path not provided or invalid, using OpenCV")
                self.use_yolo = False
        else:
            if not YOLO_AVAILABLE:
                logger.info("YOLO not available, using OpenCV detection")
            else:
                logger.info("YOLO disabled, using OpenCV detection")
    
    def detect_signature(self, image: np.ndarray) -> Dict:
        """
        Detect signature in document using contour analysis
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with 'present' (bool), 'bbox' (list), and 'confidence' (float)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be signatures
            signature_candidates = []
            h, w = image.shape[:2]
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Signature characteristics:
                # - Medium size (not too small, not too large)
                # - Aspect ratio between 1:4 and 4:1
                # - Located in lower 70% of document
                # - Has reasonable area
                
                if area < 500 or area > (h * w * 0.15):  # Size filter
                    continue
                
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio < 0.25 or aspect_ratio > 6:  # More flexible aspect ratio
                    continue
                
                # Prefer signatures in lower 70% of document
                if y > h * 0.3:
                    # Calculate stroke density (signatures have moderate density)
                    roi = binary[y:y+ch, x:x+cw]
                    density = np.sum(roi > 0) / (cw * ch) if (cw * ch) > 0 else 0
                    
                    # Signatures typically have 5-30% density
                    if 0.05 < density < 0.35:
                        score = 0.7
                        # Boost score for lower position
                        if y > h * 0.5:
                            score += 0.2
                        
                        signature_candidates.append({
                            'bbox': [x, y, x + cw, y + ch],
                            'area': area,
                            'y_position': y,
                            'density': density,
                            'score': score
                        })
            
            # Select best candidate (highest score, prefer lower position)
            if signature_candidates:
                best = max(signature_candidates, key=lambda x: (x['score'], x['y_position'] / h))
                
                return {
                    'present': True,
                    'bbox': best['bbox'],
                    'confidence': best['score']
                }
            
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.1}
            
        except Exception as e:
            print(f"Signature detection failed: {e}")
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.1}
    
    def detect_stamp(self, image: np.ndarray) -> Dict:
        """
        Detect stamp in document using multiple detection methods
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with 'present' (bool), 'bbox' (list), and 'confidence' (float)
        """
        try:
            h, w = image.shape[:2]
            
            # Method 1: Color-based detection (for colored stamps)
            color_candidates = self._detect_stamp_by_color(image)
            
            # Method 2: Grayscale contour detection (for black/dark stamps)
            gray_candidates = self._detect_stamp_by_grayscale(image)
            
            # Method 3: Edge-based circular detection
            circle_candidates = self._detect_stamp_by_circles(image)
            
            # Combine all candidates
            all_candidates = color_candidates + gray_candidates + circle_candidates
            
            if not all_candidates:
                return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.1}
            
            # Score and select best candidate
            best = max(all_candidates, key=lambda x: x['score'])
            
            # Confidence based on score and detection method
            confidence = min(0.95, best['score'])
            
            return {
                'present': True,
                'bbox': best['bbox'],
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Stamp detection failed: {e}")
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.1}
    
    def _detect_stamp_by_color(self, image: np.ndarray) -> List[Dict]:
        """Detect colored stamps using HSV color space"""
        candidates = []
        
        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for stamps
            color_ranges = [
                # Red (two ranges due to wrap-around)
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255])),
                # Blue
                (np.array([100, 50, 50]), np.array([130, 255, 255])),
                # Purple
                (np.array([130, 50, 50]), np.array([170, 255, 255])),
                # Dark blue (common for stamps)
                (np.array([90, 30, 30]), np.array([110, 255, 200])),
            ]
            
            # Combine all color masks
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = image.shape[:2]
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Stamp size filter (can be small)
                if area < 500 or area > (h * w * 0.25):
                    continue
                
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Score based on circularity and position
                score = circularity * 0.7
                if y > h * 0.3:  # Prefer lower portion
                    score += 0.2
                
                candidates.append({
                    'bbox': [x, y, x + cw, y + ch],
                    'score': score,
                    'method': 'color'
                })
        
        except Exception as e:
            print(f"  Color-based stamp detection failed: {e}")
        
        return candidates
    
    def _detect_stamp_by_grayscale(self, image: np.ndarray) -> List[Dict]:
        """Detect black/dark stamps using grayscale analysis"""
        candidates = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply binary threshold to detect dark regions
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to connect stamp components
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = image.shape[:2]
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Stamp size filter
                if area < 800 or area > (h * w * 0.2):
                    continue
                
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Stamps should be reasonably circular
                if circularity < 0.3:
                    continue
                
                # Score based on circularity and position
                score = circularity * 0.8
                if y > h * 0.3:
                    score += 0.15
                
                candidates.append({
                    'bbox': [x, y, x + cw, y + ch],
                    'score': score,
                    'method': 'grayscale'
                })
        
        except Exception as e:
            print(f"  Grayscale stamp detection failed: {e}")
        
        return candidates
    
    def _detect_stamp_by_circles(self, image: np.ndarray) -> List[Dict]:
        """Detect stamps using Hough Circle Transform"""
        candidates = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect circles
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=150
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                h, w = image.shape[:2]
                
                for circle in circles[0, :]:
                    cx, cy, r = circle
                    
                    # Create bounding box
                    x = max(0, cx - r)
                    y = max(0, cy - r)
                    x2 = min(w, cx + r)
                    y2 = min(h, cy + r)
                    
                    # Score based on size and position
                    score = 0.7
                    if y > h * 0.3:
                        score += 0.2
                    
                    candidates.append({
                        'bbox': [int(x), int(y), int(x2), int(y2)],
                        'score': score,
                        'method': 'circle'
                    })
        
        except Exception as e:
            print(f"  Circle-based stamp detection failed: {e}")
        
        return candidates
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IOU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def detect_with_yolo(self, image: np.ndarray, class_names: List[str] = ['signature', 'stamp']) -> Dict[str, Dict]:
        """
        Detect signature and stamp using YOLO
        
        Args:
            image: Input image
            class_names: List of class names to detect
            
        Returns:
            Dictionary with detection results
        """
        if not self.yolo_model:
            raise RuntimeError("YOLO model not loaded")
        
        results = {'signature': None, 'stamp': None}
        
        try:
            # Run inference
            predictions = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
            
            for pred in predictions:
                boxes = pred.boxes
                
                for box in boxes:
                    # Get bbox and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Map class to name
                    if cls < len(class_names):
                        class_name = class_names[cls]
                        
                        # Store best detection for each class
                        if class_name in results:
                            if results[class_name] is None or conf > results[class_name]['confidence']:
                                results[class_name] = {
                                    'present': True,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': conf,
                                    'method': 'yolo'
                                }
            
            # Fill in non-detected classes
            for class_name in ['signature', 'stamp']:
                if results[class_name] is None:
                    results[class_name] = {
                        'present': False,
                        'bbox': [0, 0, 0, 0],
                        'confidence': 0.0,
                        'method': 'yolo'
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            raise
    
    def detect_both(self, image: np.ndarray, use_yolo_primary: bool = True) -> Tuple[Dict, Dict]:
        """
        Detect both signature and stamp (hybrid YOLO + OpenCV)
        
        Args:
            image: Input image
            use_yolo_primary: Try YOLO first if available
            
        Returns:
            Tuple of (signature_result, stamp_result)
        """
        # Try YOLO first if available and enabled
        if self.use_yolo and use_yolo_primary and self.yolo_model:
            try:
                logger.debug("Using YOLO for visual detection")
                yolo_results = self.detect_with_yolo(image)
                
                signature = yolo_results['signature']
                stamp = yolo_results['stamp']
                
                # If YOLO found both with good confidence, use it
                if (signature['present'] and stamp['present'] and 
                    signature['confidence'] >= self.confidence_threshold and
                    stamp['confidence'] >= self.confidence_threshold):
                    return signature, stamp
                
                # If YOLO found one but not the other, supplement with OpenCV
                if not signature['present']:
                    logger.debug("YOLO didn't find signature, trying OpenCV")
                    signature = self.detect_signature(image)
                    signature['method'] = 'opencv-fallback'
                
                if not stamp['present']:
                    logger.debug("YOLO didn't find stamp, trying OpenCV")
                    stamp = self.detect_stamp(image)
                    stamp['method'] = 'opencv-fallback'
                
                return signature, stamp
                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}, falling back to OpenCV")
        
        # Fallback to OpenCV
        logger.debug("Using OpenCV for visual detection")
        signature = self.detect_signature(image)
        stamp = self.detect_stamp(image)
        signature['method'] = 'opencv'
        stamp['method'] = 'opencv'
        
        return signature, stamp
    
    def visualize_detections(self, image: np.ndarray, signature: Dict, stamp: Dict) -> np.ndarray:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image: Input image
            signature: Signature detection result
            stamp: Stamp detection result
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        # Draw signature box (green)
        if signature['present']:
            bbox = signature['bbox']
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(vis_image, 'Signature', (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw stamp box (blue)
        if stamp['present']:
            bbox = stamp['bbox']
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(vis_image, 'Stamp', (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return vis_image
