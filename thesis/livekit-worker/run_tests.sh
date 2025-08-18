#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run tests using Python's unittest discovery
echo "Running tests in ${SCRIPT_DIR}/tests/ ..."
python -m unittest discover -s "${SCRIPT_DIR}/tests" -p "test_*.py"