#!/bin/bash

echo "=========================================="
echo "Kalshi Market Predictor - Setup Script"
echo "=========================================="
echo ""

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"

# Check Node
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi
echo "✓ Node.js found: $(node --version)"

echo ""
echo "=========================================="
echo "Setting up Backend..."
echo "=========================================="
cd backend

echo "Installing Python dependencies..."
pip install -r requirements_minimal.txt

echo ""
echo "Training ML model..."
python simple_test.py

echo ""
echo "=========================================="
echo "Setting up Frontend..."
echo "=========================================="
cd ../frontend

echo "Installing Node dependencies..."
npm install

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the app:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend"
echo "  uvicorn api_server:app --reload"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "Then visit: http://localhost:3000"
echo ""
