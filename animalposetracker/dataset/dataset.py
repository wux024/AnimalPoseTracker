#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: dataset.py.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/9/8
Last Modified: 2024/9/8
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.
    
Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/9/8] wux024: Initial file creation
"""

class AnimalPoseAnnotation(object):
    def __init__(self, keypoints = 17, classes = 1):
        self.keypoints = keypoints
        self.classes = classes
    
    def load_annotation(self, annotation_file):
        pass

    def save_annotation(self, annotation_file, format = 'COCO'):
        pass

    def get_keypoints(self):
        pass

    def set_keypoints(self, keypoints):
        pass

    def get_classes(self):
        pass


