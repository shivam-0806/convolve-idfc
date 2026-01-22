"""
Image preprocessing utilities for Document AI system
"""
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array of image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Try loading with PIL first (handles more formats)
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        # Fallback to OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better OCR results
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    # Keep the image in color for PaddleOCR
    # Apply gentle denoising
    if len(image.shape) == 3:
        # Color image - use colored denoising
        enhanced = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)
    else:
        # Grayscale image
        enhanced = cv2.fastNlMeansDenoising(image, h=10)
        # Convert to RGB for consistency
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # Optional: Increase contrast slightly
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced



def resize_image(image: np.ndarray, max_dimension: int = 2000) -> Tuple[np.ndarray, float]:
    """
    Resize image if it's too large, maintaining aspect ratio
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dimension:
        return image, 1.0
    
    # Calculate scale factor
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def preprocess_for_ocr(image: np.ndarray, enhance: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for OCR
    
    Args:
        image: Input image
        enhance: Whether to apply enhancement
        
    Returns:
        Preprocessed image
    """
    # Resize if too large
    processed, _ = resize_image(image)
    
    # Enhance if requested
    if enhance:
        processed = enhance_image(processed)
    
    return processed


def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for signature/stamp detection
    Keep original colors for better detection
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Resize if too large
    processed, _ = resize_image(image)
    
    # Apply slight denoising but keep colors
    if len(processed.shape) == 3:
        processed = cv2.fastNlMeansDenoisingColored(processed, h=10, hColor=10)
    
    # Sharpen the image to enhance edges (helps with stamp/signature detection)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    processed = cv2.filter2D(processed, -1, kernel)
    
    return processed


def convert_pdf_to_images(pdf_path: str) -> list:
    """
    Convert PDF to list of images
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of images as numpy arrays
    """
    try:
        from pdf2image import convert_from_path
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=200)
        
        # Convert PIL images to numpy arrays
        np_images = [np.array(img) for img in images]
        
        return np_images
    except ImportError:
        raise ImportError("pdf2image not installed. Install with: pip install pdf2image")
    except Exception as e:
        raise ValueError(f"Failed to convert PDF: {str(e)}")


def load_document(file_path: str) -> list:
    """
    Load document (PDF or image) and return list of images
    
    Args:
        file_path: Path to document file
        
    Returns:
        List of images (one per page)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return convert_pdf_to_images(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return [load_image(file_path)]
    else:
        raise ValueError(f"Unsupported file format: {ext}")
