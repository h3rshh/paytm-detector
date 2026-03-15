#!/bin/bash
# Quick test setup script

echo "🚀 Setting up conda environment for testing..."

# Create conda environment (stored in conda's envs folder, not here)
conda create -n paytm-detector python=3.11 -y

# Activate and install dependencies
conda activate paytm-detector
pip install -r requirements.txt

echo "✅ Environment ready!"
echo "To test:"
echo "  conda activate paytm-detector"
echo "  uvicorn app.main:app --reload"
echo ""
echo "To cleanup later:"
echo "  conda deactivate"
echo "  conda remove -n paytm-detector --all -y"