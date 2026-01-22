"""
Test script to verify the Document AI system on sample images
"""
import os
# CRITICAL: Set these BEFORE any paddle imports to avoid oneDNN/PIR errors

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from executable import DocumentAI


def test_single_image():
    """Test on a single image"""
    print("="*60)
    print("Testing Document AI System")
    print("="*60)
    
    # Initialize system
    print("\n1. Initializing system...")
    ai = DocumentAI(
        dealers_file='master_data/dealers.txt',
        models_file='master_data/models.txt'
    )
    
    # Find a test image
    train_dir = Path('train')
    if not train_dir.exists():
        print("Error: train directory not found")
        return
    
    # Get first PNG file
    images = list(train_dir.glob('*.png'))
    if not images:
        print("Error: No PNG files found in train directory")
        return
    
    test_image = images[0]
    print(f"\n2. Testing with: {test_image.name}")
    
    # Process document
    result = ai.process_document(str(test_image))
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    # Save result
    output_file = Path('sample_output') / f'test_{test_image.stem}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Result saved to: {output_file}")


def test_multiple_images(num_images=5):
    """Test on multiple images"""
    print("="*60)
    print(f"Testing on {num_images} images")
    print("="*60)
    
    # Initialize system
    ai = DocumentAI(
        dealers_file='master_data/dealers.txt',
        models_file='master_data/models.txt'
    )
    
    # Get test images
    train_dir = Path('train')
    images = list(train_dir.glob('*.png'))[:num_images]
    
    results = []
    
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{num_images}] Processing: {image_path.name}")
        result = ai.process_document(str(image_path))
        results.append(result)
        
        # Quick summary
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Time: {result['processing_time_sec']:.2f}s")
    
    # Calculate statistics
    avg_time = sum(r['processing_time_sec'] for r in results) / len(results)
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Total documents: {len(results)}")
    print(f"Average processing time: {avg_time:.2f}s")
    print(f"Average confidence: {avg_conf:.2%}")
    print(f"Total cost estimate: ${sum(r['cost_estimate_usd'] for r in results):.4f}")
    
    # Save all results
    output_file = Path('sample_output') / 'batch_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def compare_methods(num_samples=5):
    """Compare ML-based vs rule-based extraction"""
    print("="*60)
    print(f"Comparing Extraction Methods on {num_samples} samples")
    print("="*60)
    
    # Get test images
    train_dir = Path('train')
    images = list(train_dir.glob('*.png'))[:num_samples]
    
    if len(images) == 0:
        print("Error: No images found in train directory")
        return
    
    # Initialize both systems
    print("\nInitializing ML-based system...")
    ai_ml = DocumentAI(
        dealers_file='master_data/dealers.txt',
        models_file='master_data/models.txt',
        use_layoutlm=True
    )
    
    print("\nInitializing rule-based system...")
    ai_rule = DocumentAI(
        dealers_file='master_data/dealers.txt',
        models_file='master_data/models.txt',
        use_layoutlm=False
    )
    
    # Compare results
    comparison_results = []
    
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{num_samples}] Processing: {image_path.name}")
        
        # ML extraction
        print("  Running ML extraction...")
        result_ml = ai_ml.process_document(str(image_path))
        
        # Rule-based extraction
        print("  Running rule-based extraction...")
        result_rule = ai_rule.process_document(str(image_path))
        
        # Compare
        comparison = {
            'doc_id': image_path.name,
            'ml': {
                'dealer': result_ml['fields']['dealer_name'],
                'model': result_ml['fields']['model_name'],
                'hp': result_ml['fields']['horse_power'],
                'cost': result_ml['fields']['asset_cost'],
                'confidence': result_ml['confidence'],
                'method': result_ml.get('extraction_method', 'unknown')
            },
            'rule': {
                'dealer': result_rule['fields']['dealer_name'],
                'model': result_rule['fields']['model_name'],
                'hp': result_rule['fields']['horse_power'],
                'cost': result_rule['fields']['asset_cost'],
                'confidence': result_rule['confidence'],
                'method': result_rule.get('extraction_method', 'unknown')
            }
        }
        
        # Check agreement
        comparison['agreement'] = {
            'dealer': comparison['ml']['dealer'] == comparison['rule']['dealer'],
            'model': comparison['ml']['model'] == comparison['rule']['model'],
            'hp': comparison['ml']['hp'] == comparison['rule']['hp'],
            'cost': comparison['ml']['cost'] == comparison['rule']['cost']
        }
        
        comparison_results.append(comparison)
        
        # Print comparison
        print(f"\n  Results:")
        print(f"    {'Field':<12} {'ML Result':<30} {'Rule Result':<30} {'Match':<8}")
        print(f"    {'-'*12} {'-'*30} {'-'*30} {'-'*8}")
        print(f"    {'Dealer':<12} {str(comparison['ml']['dealer'])[:28]:<30} {str(comparison['rule']['dealer'])[:28]:<30} {'✓' if comparison['agreement']['dealer'] else '✗':<8}")
        print(f"    {'Model':<12} {str(comparison['ml']['model'])[:28]:<30} {str(comparison['rule']['model'])[:28]:<30} {'✓' if comparison['agreement']['model'] else '✗':<8}")
        print(f"    {'HP':<12} {str(comparison['ml']['hp'])[:28]:<30} {str(comparison['rule']['hp'])[:28]:<30} {'✓' if comparison['agreement']['hp'] else '✗':<8}")
        print(f"    {'Cost':<12} {str(comparison['ml']['cost'])[:28]:<30} {str(comparison['rule']['cost'])[:28]:<30} {'✓' if comparison['agreement']['cost'] else '✗':<8}")
        print(f"    {'Confidence':<12} {comparison['ml']['confidence']:.2%}{'':23} {comparison['rule']['confidence']:.2%}")
    
    # Calculate overall statistics
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    total_fields = num_samples * 4
    total_agreements = sum(
        sum(1 for agree in comp['agreement'].values() if agree)
        for comp in comparison_results
    )
    agreement_rate = total_agreements / total_fields if total_fields > 0 else 0
    
    avg_ml_conf = sum(r['ml']['confidence'] for r in comparison_results) / len(comparison_results)
    avg_rule_conf = sum(r['rule']['confidence'] for r in comparison_results) / len(comparison_results)
    
    print(f"Documents processed: {num_samples}")
    print(f"Total fields compared: {total_fields}")
    print(f"Fields in agreement: {total_agreements} ({agreement_rate:.1%})")
    print(f"Average ML confidence: {avg_ml_conf:.2%}")
    print(f"Average Rule confidence: {avg_rule_conf:.2%}")
    
    # Save comparison
    output_file = Path('sample_output') / 'method_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Document AI system')
    parser.add_argument('--batch', type=int, help='Test on N images', metavar='N')
    parser.add_argument('--compare', type=int, help='Compare ML vs rule-based on N images', metavar='N')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_methods(args.compare)
        elif args.batch:
            test_multiple_images(args.batch)
        else:
            test_single_image()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
