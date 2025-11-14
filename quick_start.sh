#!/bin/bash

echo "🧠 Psychology Agent + RLHF - Quick Start"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d "env" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✓ Dependencies installed"

# Check for API keys
echo ""
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: No API keys found in environment"
    echo ""
    echo "Please set at least one API key:"
    echo "  export OPENAI_API_KEY='your-key'"
    echo "  export ANTHROPIC_API_KEY='your-key'"
    echo ""
    echo "Or create a .env file (see .env.example)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create data directories
echo ""
echo "📁 Setting up data directories..."
mkdir -p data/user_profiles
mkdir -p data/rlhf
mkdir -p data/validated_dialogues
mkdir -p data/clinical_guidelines

echo "✓ Data directories created"

# Run the application
echo ""
echo "🚀 Starting Psychology Agent..."
echo "========================================"
echo ""

python3 main.py

# Cleanup
echo ""
echo "👋 Thanks for using Psychology Agent!"
