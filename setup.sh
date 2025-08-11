#!/bin/bash

# Data Analyst Agent Setup Script

echo "🚀 Setting up Data Analyst Agent..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install pandas==2.1.3
pip install numpy==1.25.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install requests==2.31.0
pip install beautifulsoup4==4.12.2
pip install duckdb==0.9.2
pip install anthropic==0.34.0
pip install scipy==1.11.4
pip install python-multipart==0.0.6
pip install lxml==4.9.3
pip install html5lib==1.1
pip install openpyxl==3.1.2
pip install httpx==0.25.2

echo "✅ All dependencies installed!"

# Check for API key
if [ -z "$CLAUDE_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: CLAUDE_API_KEY environment variable is not set!"
    echo "Please set your Claude API key:"
    echo "export CLAUDE_API_KEY='your_api_key_here'"
    echo ""
    echo "Or create a .env file with:"
    echo "CLAUDE_API_KEY=your_api_key_here"
else
    echo "✅ Claude API key found in environment"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the server:"
echo "1. source venv/bin/activate"
echo "2. export CLAUDE_API_KEY='your_api_key_here'  # if not already set"
echo "3. uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To test the server:"
echo "python test_local.py"
echo ""
echo "🚀 Happy analyzing!"
