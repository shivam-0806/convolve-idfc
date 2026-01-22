#!/bin/bash
# Production-Grade Document AI - Installation Script

set -e  # Exit on error

echo "================================================"
echo "Production-Grade Document AI - Installation"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✓ Python dependencies installed"
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama is running
    if pgrep -x "ollama" > /dev/null; then
        echo "✓ Ollama is running"
    else
        echo "⚠ Ollama is not running. Start with: ollama serve &"
    fi
    
    # Check if llama3.2 model is available
    if ollama list | grep -q "llama3.2"; then
        echo "✓ llama3.2 model is available"
    else
        echo "⚠ llama3.2 model not found"
        echo "  Pull it with: ollama pull llama3.2"
    fi
else
    echo "⚠ Ollama not installed"
    echo "  Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. ollama serve &  (if not running)"
echo "  3. ollama pull llama3.2  (if not downloaded)"
echo "  4. python3 main.py train/172561841_pg1.png --pretty"
echo ""
