#!/bin/bash

# Install script for pyblobby3d package

echo "Installing pyblobby3d package..."

# Uninstall any existing version of pyblobby3d
pip uninstall pyblobby3d -y

# Navigate to the pyblobby3d directory
cd ../pyblobby3d

# Install in development mode
pip install -e .

echo "Installation complete!"
echo "You can now import pyblobby3d in Python"

# Test the installation
python -c "
try:
    from pyblobby3d import PostBlobby3D, SpectralModel
    print('✓ pyblobby3d imported successfully')
except ImportError as e:
    print(f'✗ Import failed: {e}')
"