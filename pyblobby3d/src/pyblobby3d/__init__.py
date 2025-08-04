#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup python Blobby3D postprocess codes.

@author: mathewvaridel
"""

from .post_blobby3d import PostBlobby3D
from .moments import SpectralModel
from .meta import Metadata, WavelengthWindow

# Version information
__version__ = "0.1.0"
__author__ = "Mathew Varidel & Samuel Colliver"

# Export main classes
__all__ = [
    'PostBlobby3D',
    'SpectralModel', 
    'Metadata',
    'WavelengthWindow'
]