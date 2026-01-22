"""
Visual detection for signatures and stamps
Supports YOLOv5 (primary) and OpenCV (fallback)
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    torch = None

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
        Initialize detector with YOLOv5 and OpenCV support
        
        Args:
            yolo_model_path: Path to YOLOv5 weights file (.pt)
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
                    # Load YOLOv5 model using torch.hub
                    self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=False, verbose=False)
                    self.yolo_model.conf = self.confidence_threshold
                    logger.info(f"Loaded YOLOv5 model from {yolo_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load YOLOv5 model: {e}")
                    logger.warning("Falling back to OpenCV detection")
                    self.use_yolo = False
            else:
                logger.info("YOLO model path not provided or invalid, using OpenCV")
                self.use_yolo = False
        else:
            if not YOLO_AVAILABLE:
                logger.info("PyTorch not available, using OpenCV detection")
            else:
                logger.info("YOLO disabled, using OpenCV detection")
    
    # Copy all OpenCV detection methods from original file
    # (detect_signature, detect_stamp, helper methods, etc.)
    
    def detect_with_yolo(self, image: np.ndarray, class_names: List[str] = ['signature', 'stamp']) -> Dict[str, Dict]:
        """
        Detect signature and stamp using YOLOv5
        
        Args:
            image: Input image
            class_names: List of class names to detect
            
        Returns:
            Dictionary with detection results
        """
        if not self.yolo_model:
            raise RuntimeError("YOLO model not loaded")
        
        results_dict = {'signature': None, 'stamp': None}
        
        try:
            # Run inference (YOLOv5 returns results object)
            results = self.yolo_model(image)
            
            # Parse results (YOLOv5 format)
            predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
            
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred
                cls = int(cls)
                
                # Map class to name
                if cls < len(class_names):
                    class_name = class_names[cls]
                    
                    # Store best detection for each class
                    if class_name in results_dict:
                        if results_dict[class_name] is None or conf > results_dict[class_name]['confidence']:
                            results_dict[class_name] = {
                                'present': True,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'method': 'yolov5'
                            }
            
            # Fill in non-detected classes
            for class_name in ['signature', 'stamp']:
                if results_dict[class_name] is None:
                    results_dict[class_name] = {
                        'present': False,
                        'bbox': [0, 0, 0, 0],
                        'confidence': 0.0,
                        'method': 'yolov5'
                    }
            
            return results_dict
            
        except Exception as e:
            logger.error(f"YOLOv5 detection failed: {e}")
            raise
