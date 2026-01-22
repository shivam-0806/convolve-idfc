"""
Production-Grade Document AI Executable
Implements hybrid approach: Llama 3 (Offline via Ollama) + Rule-Based + YOLO/OpenCV

Usage:
    python executable.py <input_file> [--output <output_file>]
    
Example:
    python executable.py train/172561841_pg1.png --output result.json
"""

import os
# CRITICAL: Set these BEFORE any paddle imports
os.environ['PADDLE_USE_ONEDNN'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import time
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from doc_ai.preprocessing import load_document, preprocess_for_ocr, preprocess_for_detection
from doc_ai.ocr_engine import OCREngine
from doc_ai.visual_detector import VisualDetector
from doc_ai.field_extractor import FieldExtractor, load_master_data
from doc_ai.logger import get_logger
from doc_ai.validators import create_error_result

# Try to import LLM extractor (Ollama-based, fully offline)
try:
    from doc_ai.llm_extractor import LlamaFieldExtractor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class DocumentAI:
    """Production-Grade Document AI Pipeline with Offline LLM Support"""
    
    def __init__(self, dealers_file=None, models_file=None,
                 use_llm=True, llm_model="llama3.2",
                 llm_confidence_threshold=0.7,
                 yolo_model_path=None,
                 use_yolo=True):
        """
        Initialize Document AI system
        
        Args:
            dealers_file: Path to dealers master file
            models_file: Path to models master file
            use_llm: Whether to use Llama 3 for extraction (via Ollama - OFFLINE)
            llm_model: LLM model name (default: llama3.2)
            llm_confidence_threshold: Confidence threshold for LLM
            yolo_model_path: Path to YOLO weights
            use_yolo: Whether to use YOLO for visual detection
        """
        self.logger = get_logger()

        
        # Load master data
        dealers, models = load_master_data(dealers_file, models_file)
        
        # Set YOLO model path to default if not provided
        if yolo_model_path is None and use_yolo:
            default_yolo_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
            if os.path.exists(default_yolo_path):
                yolo_model_path = default_yolo_path

        
        # Initialize components
        self.ocr_engine = OCREngine()
        self.visual_detector = VisualDetector(
            yolo_model_path=yolo_model_path,
            use_yolo=use_yolo
        )
        self.field_extractor = FieldExtractor(
            master_dealers=dealers,
            master_models=models
        )
        
        # Initialize LLM extractor if enabled (OFFLINE via Ollama)
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_confidence_threshold = llm_confidence_threshold
        self.llm_extractor = None
        
        if use_llm:
            if not LLM_AVAILABLE:
                self.logger.warning("Llama 3 requested but not available. Using rule-based extraction only.")
                self.use_llm = False
            else:
                try:

                    self.llm_extractor = LlamaFieldExtractor(
                        model=llm_model,
                        master_dealers=dealers,
                        master_models=models
                    )

                except Exception as e:
                    self.logger.error(f"Failed to load Llama 3: {e}")
                    self.logger.warning("Falling back to rule-based extraction")
                    self.use_llm = False
        
        extraction_mode = "Hybrid (Llama 3 OFFLINE + Rules)" if self.use_llm else "Rule-based only"
        visual_mode = "YOLO + OpenCV" if use_yolo else "OpenCV only"

    
    def process_document(self, file_path: str) -> dict:
        """
        Process a single document and extract fields
        
        Args:
            file_path: Path to document (PDF or image)
            
        Returns:
            Dictionary with extracted fields and metadata
        """
        start_time = time.time()
        doc_id = os.path.basename(file_path)
        
        self.logger.set_context(doc_id=doc_id)
        self.logger.info(f"Processing: {file_path}")
        
        # Step 1: Load document
        try:
            images = load_document(file_path)
            if not images:
                raise ValueError("No images loaded from document")
            
            image = images[0]  # Process first page

            
        except Exception as e:
            self.logger.error(f"Failed to load document: {e}")
            return create_error_result(doc_id, str(e))
        
        # Step 2: Preprocess for OCR
        try:
            ocr_image = preprocess_for_ocr(image, enhance=False)

        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {e}, using original")
            ocr_image = image
        
        # Step 3: Extract text with OCR
        try:
            ocr_data = self.ocr_engine.extract_text(ocr_image)

        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            ocr_data = []
        
        # Step 4: Detect signature and stamp
        try:
            detection_image = preprocess_for_detection(image)
            signature_result, stamp_result = self.visual_detector.detect_both(detection_image)
            

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            signature_result = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
            stamp_result = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        # Step 5: Extract fields (Hybrid: LLM + Rules)
        try:
            fields = self.field_extractor.extract_all_fields(
                ocr_data, 
                signature_result, 
                stamp_result,
                image=image,
                use_ml=self.use_llm,
                ml_extractor=self.llm_extractor,
                ml_confidence_threshold=self.llm_confidence_threshold
            )
            extraction_method = fields.get('extraction_method', 'unknown')

        except Exception as e:
            self.logger.error(f"Field extraction failed: {e}")
            fields = self._create_empty_fields(signature_result, stamp_result)
        
        # Calculate processing time
        
        # Calculate CPU cost (assuming $0.05/hour for standard CPU)
        processing_time = time.time() - start_time
        cpu_cost_per_hour = 0.05  # USD per hour
        cpu_cost = (processing_time / 3600) * cpu_cost_per_hour
        

        
        # Create result
        result = {
            'doc_id': doc_id,
            'fields': {
                'dealer_name': fields.get('dealer_name', ''),
                'model_name': fields.get('model_name', ''),
                'horse_power': fields.get('horse_power'),
                'asset_cost': fields.get('asset_cost'),
                'signature': signature_result,
                'stamp': stamp_result
            },
            'confidence': fields.get('confidence', 0.0),
            'processing_time_sec': round(processing_time, 2),
            'cost_estimate_usd': round(cpu_cost, 6)  # CPU cost
        }
        
        self.logger.info(f"✓ Processing complete in {processing_time:.2f}s")
        self.logger.info(f"  Overall confidence: {result['confidence']:.2%}")
        self.logger.clear_context()
        
        return result
    
    def _create_empty_fields(self, signature_result, stamp_result) -> dict:
        """Create empty fields result"""
        return {
            'dealer_name': '',
            'model_name': '',
            'horse_power': None,
            'asset_cost': None,
            'signature': signature_result,
            'stamp': stamp_result,
            'confidence': 0.0,
            'field_confidences': {},
            'extraction_method': 'error'
        }


def process_batch(args):
    """Process multiple documents in batch mode"""
    from glob import glob
    from datetime import datetime
    
    # Find all image files in input folder
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.PNG', '*.JPG', '*.JPEG', '*.PDF']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.input_folder, ext)))
    
    if not image_files:
        print(f"No image files found in {args.input_folder}")
        return
    
    # Limit to batch_size if specified
    if args.batch_size and args.batch_size > 0:
        image_files = image_files[:args.batch_size]
    
    total_files = len(image_files)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"{'='*60}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Total files to process: {total_files}")
    print(f"{'='*60}\n")
    
    # Initialize system once for all processing
    try:
        print("Initializing Document AI system...")
        ai = DocumentAI(
            dealers_file=args.dealers,
            models_file=args.models,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            llm_confidence_threshold=args.llm_threshold,
            yolo_model_path=args.yolo_model,
            use_yolo=not args.no_yolo
        )
        print("✓ System initialized\n")
    except Exception as e:
        print(f"Error: Failed to initialize system: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process all files
    all_results = []
    start_time = time.time()
    
    for file_idx, file_path in enumerate(image_files):
        file_num = file_idx + 1
        print(f"\n[{file_num}/{total_files}] Processing: {os.path.basename(file_path)}")
        
        try:
            result = ai.process_document(file_path)
            all_results.append(result)

            # Save individual JSON file for this document
            doc_id = result.get('doc_id', f'doc_{file_num}')
            clean_id = doc_id.rsplit('.', 1)[0] if '.' in doc_id else doc_id
            individual_file = os.path.join(args.output_folder, f"{clean_id}.json")
            with open(individual_file, 'w') as f:
                json.dump(result, f, indent=2)

            
            # Print summary
            conf = result.get('confidence', 0)
            method = result.get('extraction_method', 'unknown')
            print(f"  ✓ Confidence: {conf:.0%} | Method: {method}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_result = {
                'doc_id': os.path.basename(file_path),
                'error': str(e),
                'status': 'failed'
            }
            all_results.append(error_result)
    
    total_time = time.time() - start_time
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f"results_{timestamp}.json"
    results_filepath = os.path.join(args.output_folder, results_filename)
    
    # Calculate statistics
    successful = sum(1 for r in all_results if 'error' not in r)
    failed = len(all_results) - successful
    avg_confidence = sum(r.get('confidence', 0) for r in all_results if 'error' not in r) / max(successful, 1)
    
    with open(results_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_files': total_files,
                'successful': successful,
                'failed': failed,
                'average_confidence': round(avg_confidence, 2),
                'processing_time_sec': round(total_time, 2),
                'timestamp': datetime.now().isoformat()
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average confidence: {avg_confidence:.2%}")
    print(f"Total time: {total_time:.2f}s")
    print(f"\nResults saved to: {results_filename}")
    print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Production-Grade Document AI system (Offline LLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python executable.py train/172561841_pg1.png
  python executable.py train/172561841_pg1.png --output result.json
  python executable.py document.pdf --dealers master_data/dealers.txt
  python executable.py document.pdf --no-llm --no-yolo

Note: LLM runs OFFLINE via Ollama (no internet required at runtime)
      Make sure Ollama is running: ollama serve
      And model is pulled: ollama pull llama3.2
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input document file (PDF or image) - not used in batch mode')
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--dealers', help='Dealers master file')
    parser.add_argument('--models', help='Models master file')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON output')
    
    # Batch processing options
    parser.add_argument('--input-folder', help='Input folder containing images for batch processing')
    parser.add_argument('--output-folder', default='results', help='Output folder for batch processing results (default: results)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Number of images to process (default: all images in folder)')
    
    # LLM options
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable Llama 3, use rule-based only')
    parser.add_argument('--llm-model', default='llama3.2',
                       help='LLM model name (default: llama3.2, runs OFFLINE via Ollama)')
    parser.add_argument('--llm-threshold', type=float, default=0.7,
                       help='LLM confidence threshold (0.0-1.0, default: 0.7)')
    
    # YOLO options
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO, use OpenCV only')
    parser.add_argument('--yolo-model', help='Path to YOLO weights file')
    
    args = parser.parse_args()
    
    # Check if batch mode or single file mode
    batch_mode = args.input_folder is not None
    
    if batch_mode:
        # Batch processing mode
        if not args.output_folder:
            sys.exit(1)
        
        if not os.path.exists(args.input_folder):
            print(f"Error: Input folder not found: {args.input_folder}", file=sys.stderr)
            sys.exit(1)
        
        # Create output folder if it doesn't exist
        os.makedirs(args.output_folder, exist_ok=True)
        
        # Process batch
        process_batch(args)
    else:
        # Single file mode
        if not args.input:
            print("Error: Either provide input file or use --input-folder for batch processing", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        
        # Initialize system
        try:
            ai = DocumentAI(
                dealers_file=args.dealers,
                models_file=args.models,
                use_llm=not args.no_llm,
                llm_model=args.llm_model,
                llm_confidence_threshold=args.llm_threshold,
                yolo_model_path=args.yolo_model,
                use_yolo=not args.no_yolo
            )
        except Exception as e:
            print(f"Error: Failed to initialize system: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Process document
        try:
            result = ai.process_document(args.input)
        except Exception as e:
            print(f"Error: Processing failed: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Output result
        json_indent = 2 if args.pretty else None
        json_output = json.dumps(result, indent=json_indent)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n✓ Result saved to: {args.output}")
        else:
            print("\n" + "="*60)
            print("RESULT:")
            print("="*60)
            print(json_output)


if __name__ == '__main__':
    main()