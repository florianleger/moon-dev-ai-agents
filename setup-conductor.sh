#!/bin/bash
# Conductor Setup Script for Moon Dev AI Agents
# Run this script when initializing a new Conductor workspace

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ENV="$HOME/dev/moon-dev-ai-agents/.env"
TARGET_ENV="$SCRIPT_DIR/.env"

echo "=== Moon Dev AI Agents - Conductor Setup ==="

# 1. Copy .env file from source project
if [ -f "$SOURCE_ENV" ]; then
    echo "Copying .env from $SOURCE_ENV..."
    cp "$SOURCE_ENV" "$TARGET_ENV"
    echo "✓ .env copied successfully"
else
    echo "⚠ Warning: Source .env not found at $SOURCE_ENV"
    echo "  Please manually create .env based on .env_example"
fi

# 2. Activate conda environment and install dependencies
echo ""
echo "Setting up Python environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'tflow'..."

    # Source conda for script context
    eval "$(conda shell.bash hook)"

    if conda activate tflow 2>/dev/null; then
        echo "✓ Conda environment 'tflow' activated"

        # 3. Install/update dependencies
        echo ""
        echo "Installing Python dependencies..."
        if pip install -r "$SCRIPT_DIR/requirements.txt" --quiet 2>/dev/null; then
            echo "✓ Dependencies installed"
        else
            echo "⚠ Warning: Some dependencies failed to install"
            echo "  Try running: pip cache purge && pip install -r requirements.txt"
        fi
    else
        echo "⚠ Warning: Conda environment 'tflow' not found"
        echo "  Create it with: conda create -n tflow python=3.11"
        echo "  Then run this script again"
    fi
else
    echo "⚠ Warning: Conda not found in PATH"
    echo "  Please install Miniconda/Anaconda and try again"
fi

# 4. Verify setup
echo ""
echo "=== Setup Summary ==="
if [ -f "$TARGET_ENV" ]; then
    echo "✓ .env file: Present"
else
    echo "✗ .env file: Missing"
fi

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ Python: $PYTHON_VERSION"
else
    echo "✗ Python: Not found"
fi

echo ""
echo "=== Setup Complete ==="
echo "You can now run agents with: python src/agents/<agent_name>.py"
echo "Or run the main orchestrator with: python src/main.py"
