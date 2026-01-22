# Setup Guide - Document AI System

## Overview

This guide walks you through setting up and running the production-grade Document AI system for tractor loan invoice processing.

## Prerequisites

### System Requirements
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for YOLOv5)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 5GB for dependencies and models

### Required Software
- Python 3.10+
- Git
- CUDA Toolkit (for GPU acceleration)
- Ollama (for offline LLM)

## Installation

### Step 1: Clone the Repository

```bash
cd ~/Documents
git clone <repository-url> idfc
cd idfc
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies**:
- `easyocr` - OCR engine
- `torch`, `torchvision` - For YOLOv5
- `ollama` - Offline LLM client
- `opencv-python` - Image processing
- `fuzzywuzzy` - Fuzzy matching
- `pdf2image` - PDF support

### Step 4: Install Ollama

#### Download and Install
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit: https://ollama.ai/download
```

#### Start Ollama Service
```bash
# Start Ollama in background
ollama serve &
```

#### Pull Llama Model
```bash
# Download llama3.2 model (one-time, ~2GB)
ollama pull llama3.2
```

#### Verify Ollama
```bash
# Test that Ollama is running
curl http://localhost:11434/api/version
```

### Step 5: Setup YOLOv5 Weights

The YOLOv5 weights should be placed in the `weights/` directory:

```bash
# Ensure weights directory exists
mkdir -p weights

# Place your best.pt file
# weights/best.pt (provided separately)
```

### Step 6: Prepare Master Data (Optional)

Create master data files for better accuracy:

```bash
mkdir -p master_data

# Create dealers list
cat > master_data/dealers.txt << EOF
International Tractors Ltd
Escorts Kubota Limited
Mahindra & Mahindra
John Deere India
EOF

# Create models list
cat > master_data/models.txt << EOF
DI-745 III HDM+4WD
Swaraj 744 FE
Mahindra 475 DI
575 DI
EOF
```

## Configuration

### Directory Structure

After setup, your directory should look like:

```
idfc/
├── .venv/                    # Virtual environment
├── doc_ai/                     # Utility modules (renamed from utils)
│   ├── __init__.py
│   ├── logger.py
│   ├── ocr_engine.py
│   ├── visual_detector.py
│   ├── field_extractor.py
│   ├── llm_extractor.py
│   └── ...
├── weights/
│   └── best.pt              # YOLOv5 weights
├── master_data/             # Optional
│   ├── dealers.txt
│   └── models.txt
├── train/                   # Sample images
├── executable.py            # Main entry point
├── requirements.txt
└── README.md
```

## Running the System

### Basic Usage

#### Single File Processing

```bash
# Activate virtual environment
source .venv/bin/activate

# Process single document (with YOLOv5 + LLM)
python executable.py train/172561841_pg1.png --pretty

# Save to file
python executable.py train/document.png --output result.json --pretty
```

#### Batch Processing

```bash
# Process all images in folder
python executable.py --input-folder train/ --output-folder results/

# Process only 5 images
python executable.py --input-folder train/ --output-folder results/ --batch-size 5

# Batch with custom LLM threshold
python executable.py --input-folder train/ --output-folder results/ \
  --batch-size 10 --llm-threshold 0.6
```

### Advanced Options

```bash
# Disable LLM (rule-based only)
python executable.py document.png --no-llm

# Disable YOLO (OpenCV only)
python executable.py document.png --no-yolo

# Custom master data
python executable.py document.png \
  --dealers master_data/dealers.txt \
  --models master_data/models.txt

# All options
python executable.py document.png \
  --output result.json \
  --dealers master_data/dealers.txt \
  --models master_data/models.txt \
  --llm-threshold 0.6 \
  --pretty
```

### CLI Reference

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Input file (single mode) | Required |
| `--input-folder` | Input folder (batch mode) | None |
| `--output-folder` | Output folder (batch mode) | None |
| `--batch-size N` | Number of images to process | All files |
| `--output, -o` | Output JSON file | stdout |
| `--pretty` | Pretty print JSON | False |
| `--no-llm` | Disable LLM extraction | False |
| `--llm-threshold` | LLM confidence (0.0-1.0) | 0.7 |
| `--no-yolo` | Disable YOLO detection | False |
| `--yolo-model` | YOLO weights path | weights/best.pt |
| `--dealers` | Dealers master file | None |
| `--models` | Models master file | None |

## Testing

### Quick Test

```bash
source .venv/bin/activate

# Test single file (should complete in ~20-60s)
python executable.py train/172561841_pg1.png --pretty
```

**Expected Output**:
```json
{
  "doc_id": "172561841_pg1.png",
  "fields": {
    "dealer_name": "International Tractors Ltd",
    "model_name": "DI-745 III HDM+4WD 50 HP",
    "horse_power": 50,
    "asset_cost": 911769.0,
    ...
  },
  "confidence": 0.86,
  "extraction_method": "llm",
  ...
}
```
