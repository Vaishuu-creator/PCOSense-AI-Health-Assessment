#!/bin/bash

# PCOSense ML Application Startup Script

echo "============================================================"
echo "           PCOSense - AI PCOS Assessment System"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Check required packages
echo "Checking required packages..."

packages=("flask" "flask_cors" "sklearn" "pandas" "numpy")
missing_packages=()

for package in "${packages[@]}"; do
    if ! python3 -c "import ${package}" &> /dev/null; then
        missing_packages+=("${package}")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  Missing packages detected: ${missing_packages[*]}"
    echo ""
    echo "Installing missing packages..."
    pip install flask flask-cors scikit-learn pandas numpy --break-system-packages
    echo ""
fi

echo "✓ All required packages are installed"
echo ""

# Check if model files exist
if [ ! -f "pcos_model.pkl" ]; then
    echo "⚠️  Model file not found. Training model..."
    python3 train_pcos_model.py
    echo ""
fi

echo "✓ Model files ready"
echo ""

# Start the Flask API
echo "============================================================"
echo "Starting Flask API Server..."
echo "============================================================"
echo ""
echo "API will be available at: http://localhost:5000"
echo "Open pcosense_ml.html in your browser to use the application"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""


python3 flask_api.py