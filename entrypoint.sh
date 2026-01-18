#!/bin/bash
# Moon Dev AI Trading Bot - Entrypoint Script
# Validates environment and starts the bot

set -e

echo "=========================================="
echo "  Moon Dev AI Trading Bot Starting..."
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check required env var
check_env() {
    local var_name=$1
    local is_required=$2

    if [ -z "${!var_name}" ]; then
        if [ "$is_required" = "required" ]; then
            echo -e "${RED}ERROR: Required environment variable $var_name is not set${NC}"
            return 1
        else
            echo -e "${YELLOW}WARNING: Optional environment variable $var_name is not set${NC}"
        fi
    else
        echo -e "${GREEN}OK: $var_name is configured${NC}"
    fi
    return 0
}

echo ""
echo "Checking environment variables..."
echo ""

# Track if all required vars are set
ALL_REQUIRED_SET=true

# Required for HyperLiquid trading
if ! check_env "HYPER_LIQUID_KEY" "required"; then
    ALL_REQUIRED_SET=false
fi

# At least one LLM API key required
if [ -z "$ANTHROPIC_KEY" ] && [ -z "$OPENAI_KEY" ] && [ -z "$DEEPSEEK_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}ERROR: At least one LLM API key required (ANTHROPIC_KEY, OPENAI_KEY, DEEPSEEK_KEY, or GROQ_API_KEY)${NC}"
    ALL_REQUIRED_SET=false
else
    echo -e "${GREEN}OK: At least one LLM API key is configured${NC}"
fi

# Optional but recommended
echo ""
echo "Checking optional environment variables..."
# Note: MOONDEV_API_KEY no longer needed - using Binance WebSocket for liquidations
check_env "COINGECKO_API_KEY" "optional"
check_env "BIRDEYE_API_KEY" "optional"

echo ""

# Check if paper trading mode is enabled
if [ "$PAPER_TRADING" = "true" ] || [ "$PAPER_TRADING" = "True" ] || [ "$PAPER_TRADING" = "1" ]; then
    echo -e "${YELLOW}=========================================="
    echo "  PAPER TRADING MODE ENABLED"
    echo "  No real trades will be executed"
    echo "==========================================${NC}"
fi

# Check if testnet is enabled
if [ "$USE_TESTNET" = "true" ] || [ "$USE_TESTNET" = "True" ] || [ "$USE_TESTNET" = "1" ]; then
    echo -e "${YELLOW}=========================================="
    echo "  TESTNET MODE ENABLED"
    echo "  Using HyperLiquid testnet"
    echo "==========================================${NC}"
fi

# Create data directories if they don't exist
echo ""
echo "Creating data directories..."
mkdir -p /app/src/data/ramf
mkdir -p /app/src/data/execution_results
mkdir -p /app/logs
echo -e "${GREEN}Data directories ready${NC}"

# Final check
echo ""
if [ "$ALL_REQUIRED_SET" = false ]; then
    echo -e "${RED}=========================================="
    echo "  STARTUP ABORTED"
    echo "  Missing required environment variables"
    echo "==========================================${NC}"
    exit 1
fi

echo -e "${GREEN}=========================================="
echo "  Environment validated successfully"
echo "  Starting trading bot..."
echo "==========================================${NC}"
echo ""

# Execute the main command
exec "$@"
