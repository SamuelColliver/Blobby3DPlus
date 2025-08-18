#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory> <num_cores>"
    echo "Example: $0 /path/to/simulation 8"
    exit 1
fi

DIRECTORY="$1"
NUM_CORES="$2"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist"
    exit 1
fi

# Check if num_cores is a valid number
if ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]]; then
    echo "Error: num_cores must be a positive integer"
    exit 1
fi

echo "Changing to directory: $DIRECTORY"
echo "Running Blobby3D with $NUM_CORES cores..."

# Change to the directory and run the command
cd "$DIRECTORY" || {
    echo "Error: Failed to change to directory '$DIRECTORY'"
    exit 1
}

# Run the blobby command
# Using caffeinate to prevent sleep and nohup to run in background
nohup caffeinate -i "/Users/scol0322/Library/Mobile Documents/com~apple~CloudDocs/PhD-Code/Blobby3DPlus/Blobby3D" -t "$NUM_CORES" -f MODEL_OPTIONS &

# Get the process ID
PID=$!

echo "Blobby3D started with PID: $PID"
echo "Output will be logged to: $(pwd)/nohup.out"
echo "To check if it's still running: ps -p $PID"
echo "To kill the process: kill $PID"