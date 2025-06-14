#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: __init__.py.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/9/6
Last Modified: 2024/9/6
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.
    
Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/9/6] wux024: Initial file creation
"""

from .inferencer import InferenceEngine
from .constant import PLATFORM, ENGINEtoDEVICE, ENGINEtoBackend, OpenCV_TARGETS

__all__ = ['InferenceEngine','PLATFORM', 'ENGINEtoDEVICE', 
           'ENGINEtoBackend', 'OpenCV_TARGETS']