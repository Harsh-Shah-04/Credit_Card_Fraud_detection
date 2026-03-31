#!/bin/bash

# Credit Card Fraud Detection - Streamlit App Launcher
# This script automatically sets up and runs the Streamlit application

echo ""
echo "==============================================="
echo "Credit Card Fraud Detection System"
echo "Streamlit Application Launcher"
echo "==============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Python version:"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/4] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully!"
else
    echo ""
    echo "[2/4] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/4] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/4] Installing/updating dependencies..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "==============================================="
echo "Setup completed successfully!"
echo "==============================================="
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your default browser."
echo ""
echo "Press Ctrl+C to stop the server at any time."
echo ""

# Check if models exist
if [ ! -f "models/ann_model.h5" ]; then
    echo "WARNING: Model files not found in models/ directory!"
    echo "Please ensure the following files exist:"
    echo "  - models/ann_model.h5"
    echo "  - models/best_ml_model.pkl"
    echo "  - models/scaler.pkl"
    echo "  - models/ann_threshold.pkl"
    echo ""
fi

# Run Streamlit app
streamlit run streamlit_app.py
