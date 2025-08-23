#!/usr/bin/env python3
"""
Enhanced Image Quality Assessment Modules
========================================

This package provides comprehensive, step-by-step image quality analysis with
detailed reporting of preprocessing steps, algorithms used, and technical details.

Modules:
- config: Configuration management
- preprocessing_tracker: Step tracking and documentation  
- raw_image_analyzer: Raw image property analysis
- brain_extraction: Brain extraction with multiple methods
- noise_estimation: Noise estimation with multiple methods
- quality_metrics: SNR, CNR, and uniformity calculations
- main_analyzer: Main integration and coordination
"""

from .config import QualityConfig
from .preprocessing_tracker import PreprocessingStepTracker
from .raw_image_analyzer import RawImageAnalyzer
from .brain_extraction import BrainExtractionModule
from .noise_estimation import NoiseEstimationModule
from .quality_metrics import QualityMetricsCalculator
from .main_analyzer import EnhancedQualityAssessment, create_test_config

__all__ = [
    'QualityConfig',
    'PreprocessingStepTracker',
    'RawImageAnalyzer', 
    'BrainExtractionModule',
    'NoiseEstimationModule',
    'QualityMetricsCalculator',
    'EnhancedQualityAssessment',
    'create_test_config'
]

__version__ = "1.0.0"
__author__ = "Enhanced YOPD Analysis"
