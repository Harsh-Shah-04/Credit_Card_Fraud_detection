#!/bin/bash

# Setup script for Streamlit Cloud and local deployment

echo "Setting up Credit Card Fraud Detection Application..."

# Update pip
pip install --upgrade pip

# Install dependencies with retry logic
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

echo "Setup complete! Models should be in the 'models/' directory."
echo "You can now run: streamlit run streamlit_app.py"
