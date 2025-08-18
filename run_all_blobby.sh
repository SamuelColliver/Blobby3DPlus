#!/bin/bash

# Configuration - modify these values as needed
NUM_CORES=4                    # Number of cores to use for each run
SCRIPT_PATH="./run_blobby.sh"  # Path to the run_blobby.sh script

# Check if parent directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    echo "Example: $0 /path/to/simulations"
    echo ""
    echo "This script will run Blobby3D with $NUM_CORES cores in all subdirectories of the given parent directory."
    exit 1
fi

PARENT_DIR="$1"

# Check if parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Parent directory '$PARENT_DIR' does not exist"
    exit 1
fi

# Check if the run_blobby.sh script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found"
    echo "Make sure run_blobby.sh is in the same directory as this script"
    exit 1
fi

# Check if the script is executable
if [ ! -x "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' is not executable"
    echo "Run: chmod +x $SCRIPT_PATH"
    exit 1
fi

echo "==========================================="
echo "Batch Blobby Runner"
echo "Parent directory: $PARENT_DIR"
echo "Cores per run: $NUM_CORES"
echo "Script path: $SCRIPT_PATH"
echo "==========================================="
echo ""

# Counter for tracking runs
count=0
failed=0

# Loop through all directories in the parent directory
for dir in "$PARENT_DIR"/*; do
    # Check if it's actually a directory (not a file)
    if [ -d "$dir" ]; then
        echo "Starting run in: $(basename "$dir")"
        echo "Full path: $dir"
        
        # Call the run_blobby.sh script
        if "$SCRIPT_PATH" "$dir" "$NUM_CORES"; then
            echo "✓ Successfully started in $(basename "$dir")"
            ((count++))
        else
            echo "✗ Failed to start in $(basename "$dir")"
            ((failed++))
        fi
        
        echo ""
        
        # Small delay between launches to avoid overwhelming the system
        sleep 2
    fi
done

echo "==========================================="
echo "Batch run completed!"
echo "Successfully started: $count directories"
echo "Failed: $failed directories"
echo "Total cores in use: $((count * NUM_CORES))"
echo ""
echo "To monitor all processes:"
echo "ps aux | grep Blobby3D"
echo ""
echo "Output files will be in each directory as 'nohup.out'"
echo "==========================================="