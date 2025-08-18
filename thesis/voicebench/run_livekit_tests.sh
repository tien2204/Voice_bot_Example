#!/bin/bash

# Script to run LiveKit model tests for voicebench

# Source .env file if it exists in the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  echo "Sourcing environment variables from $ENV_FILE"
  set -a; . "$ENV_FILE"; set +a
fi

# IMPORTANT: Set these environment variables before running the script.
# These are expected by the LiveKitAssistant (potentially after modification)
# or by a .env file that it might load.
#
# export LIVEKIT_API_KEY="your_api_key"
# export LIVEKIT_API_SECRET="your_api_secret"
# export LIVEKIT_URL="your_livekit_ws_LIVEKIT_URL" # e.g., ws://localhost:7880 or wss://your-project.livekit.cloud

# Check if required environment variables are set
if [ -z "$LIVEKIT_API_KEY" ] || [ -z "$LIVEKIT_API_SECRET" ] || [ -z "$LIVEKIT_URL" ]; then
  echo "Error: Required environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, or LIVEKIT_URL are not set."
  echo "Please set them and try again. Example:"
  echo "  export LIVEKIT_API_KEY=\"YOUR_KEY\""
  echo "  export LIVEKIT_API_SECRET=\"YOUR_SECRET\""
  echo "  export LIVEKIT_URL=\"ws://localhost:7880\""
  exit 1
fi

MODEL_NAME="livekit"
DATASET="alpacaeval" # Default dataset, can be parameterized if needed
SPLIT="test"       # Default split, can be parameterized if needed

MODALITIES=("audio")

# Ensure python and main.py are accessible.
PYTHON_CMD="python"
MAIN_SCRIPT_PATH="$SCRIPT_DIR/main.py"

# Check if main.py exists
if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
    echo "Error: main.py not found at $MAIN_SCRIPT_PATH. Make sure the path is correct and you are in the voicebench directory."
    exit 1
fi

echo "Starting LiveKit model tests..."

for MODALITY in "${MODALITIES[@]}"; do
  echo "-----------------------------------------------------"
  echo "Running test for Model: $MODEL_NAME, Dataset: $DATASET, Split: $SPLIT, Modality: $MODALITY"
  echo "-----------------------------------------------------"

  COMMAND="$PYTHON_CMD $MAIN_SCRIPT_PATH --model $MODEL_NAME --data $DATASET --split $SPLIT --modality $MODALITY"
  echo "Executing: $COMMAND"

  $COMMAND

  if [ $? -ne 0 ]; then
    echo "Error running test for modality $MODALITY. See output above."
  else
    echo "Successfully completed test for modality $MODALITY. Output file: ${MODEL_NAME}-${DATASET}-${SPLIT}-${MODALITY}.jsonl"
  fi
  echo ""
done

echo "All LiveKit model tests completed."