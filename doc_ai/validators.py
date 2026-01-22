"""
Output schema validation using Pydantic
Ensures type safety and data integrity for extraction results
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates [x1, y1, x2, y2]"""
    coordinates: List[int] = Field(..., min_length=4, max_length=4)
    
    @field_validator('coordinates')
    @classmethod
    def validate_bbox(cls, v):
        """Validate bounding box format and values"""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        
        x1, y1, x2, y2 = v
        
        # Check non-negative
        if any(coord < 0 for coord in v):
            raise ValueError("Bounding box coordinates must be non-negative")
        
        # Check x2 > x1 and y2 > y1
        if x2 <= x1:
            raise ValueError(f"x2 ({x2}) must be greater than x1 ({x1})")
        if y2 <= y1:
            raise ValueError(f"y2 ({y2}) must be greater than y1 ({y1})")
        
        return v
    
    def to_list(self) -> List[int]:
        """Convert to list format"""
        return self.coordinates
    
    def area(self) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.coordinates
        return (x2 - x1) * (y2 - y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bbox"""
        x1, y1, x2, y2 = self.coordinates
        ox1, oy1, ox2, oy2 = other.coordinates
        
        # Calculate intersection
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0  # No intersection
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        union_area = self.area() + other.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


class VisualElement(BaseModel):
    """Visual element detection result (stamp or signature)"""
    present: bool = Field(..., description="Whether element is detected")
    bbox: List[int] = Field(default=[0, 0, 0, 0], description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")
    method: str = Field(default="unknown", description="Detection method used")
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v, info):
        """Validate bbox if element is present"""
        if info.data.get('present', False):
            # Only validate if present
            if v == [0, 0, 0, 0]:
                raise ValueError("Bounding box required when element is present")
            
            # Basic validation
            if len(v) != 4:
                raise ValueError("Bounding box must have 4 coordinates")
            
            x1, y1, x2, y2 = v
            if any(c < 0 for c in v):
                raise ValueError("Coordinates must be non-negative")
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid bounding box dimensions")
        
        return v


class ExtractionFields(BaseModel):
    """Extracted document fields"""
    dealer_name: str = Field(default="", description="Dealer/company name")
    model_name: str = Field(default="", description="Tractor model name")
    horse_power: Optional[int] = Field(default=None, ge=10, le=200, description="Horse power")
    asset_cost: Optional[float] = Field(default=None, ge=100000, le=5000000, description="Asset cost in rupees")
    
    @field_validator('horse_power')
    @classmethod
    def validate_hp(cls, v):
        """Validate horse power range for tractors"""
        if v is not None and not (15 <= v <= 200):
            raise ValueError(f"Horse power {v} outside typical tractor range (15-200)")
        return v
    
    @field_validator('asset_cost')
    @classmethod
    def validate_cost(cls, v):
        """Validate asset cost range"""
        if v is not None and not (300000 <= v <= 3000000):
            raise ValueError(f"Asset cost {v} outside typical range (3L-30L)")
        return v


class DocumentResult(BaseModel):
    """Complete document extraction result"""
    doc_id: str = Field(..., description="Document identifier")
    fields: Dict[str, Any] = Field(..., description="Extracted fields")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    field_confidences: Dict[str, float] = Field(default_factory=dict, description="Per-field confidence")
    extraction_method: str = Field(default="unknown", description="Extraction method used")
    processing_time_sec: float = Field(..., ge=0.0, description="Processing time in seconds")
    cost_estimate_usd: float = Field(default=0.001, ge=0.0, description="Estimated processing cost")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    @field_validator('field_confidences')
    @classmethod
    def validate_confidences(cls, v):
        """Validate confidence scores are in range"""
        for field, conf in v.items():
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence for {field} must be between 0.0 and 1.0")
        return v
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return self.model_dump(exclude_none=True)


def validate_extraction_result(result: Dict[str, Any]) -> DocumentResult:
    """
    Validate extraction result against schema
    
    Args:
        result: Raw extraction result dictionary
        
    Returns:
        Validated DocumentResult object
        
    Raises:
        ValidationError: If validation fails
    """
    return DocumentResult(**result)


def create_error_result(doc_id: str, error_msg: str) -> Dict[str, Any]:
    """Create standardized error result"""
    return {
        'doc_id': doc_id,
        'fields': {
            'dealer_name': '',
            'model_name': '',
            'horse_power': None,
            'asset_cost': None,
            'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        },
        'confidence': 0.0,
        'field_confidences': {},
        'extraction_method': 'error',
        'processing_time_sec': 0.0,
        'cost_estimate_usd': 0.0,
        'error': error_msg
    }


if __name__ == "__main__":
    # Test validators
    print("Testing BoundingBox...")
    bbox = BoundingBox(coordinates=[100, 100, 200, 200])
    print(f"  Area: {bbox.area()}")
    
    bbox2 = BoundingBox(coordinates=[150, 150, 250, 250])
    print(f"  IOU: {bbox.iou(bbox2):.2f}")
    
    print("\nTesting VisualElement...")
    sig = VisualElement(present=True, bbox=[100, 100, 200, 150], confidence=0.95, method="yolo")
    print(f"  Signature: {sig.present}, confidence: {sig.confidence}")
    
    print("\nTesting ExtractionFields...")
    fields = ExtractionFields(
        dealer_name="Test Dealer Ltd",
        model_name="Swaraj 744 FE",
        horse_power=48,
        asset_cost=800000.0
    )
    print(f"  {fields.model_dump()}")
    
    print("\nValidation complete!")
