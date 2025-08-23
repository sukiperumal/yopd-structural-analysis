#!/usr/bin/env python3
"""
Main integration module for enhanced image quality assessment
Coordinates all analysis stages and provides comprehensive reporting
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import json
from typing import Dict, Optional

# Import our modules
from .config import QualityConfig
from .preprocessing_tracker import PreprocessingStepTracker
from .raw_image_analyzer import RawImageAnalyzer
from .brain_extraction import BrainExtractionModule
from .noise_estimation import NoiseEstimationModule
from .quality_metrics import QualityMetricsCalculator

class EnhancedQualityAssessment:
    """
    Main class for comprehensive image quality analysis
    Coordinates all analysis stages with detailed technical reporting
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config if config else QualityConfig()
        self.logger = self._setup_logging()
        self.main_tracker = PreprocessingStepTracker()
        
        # Initialize analysis modules
        self.raw_analyzer = RawImageAnalyzer()
        self.brain_extractor = BrainExtractionModule(self.config.brain_extraction_method)
        self.noise_estimator = NoiseEstimationModule(self.config.noise_estimation_method)
        self.metrics_calculator = QualityMetricsCalculator(self.config)
        
        print("Enhanced Quality Assessment initialized")
        print(f"Configuration: {self.config.get_config_summary()}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('enhanced_quality_assessment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """
        Comprehensive analysis of a single image
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE IMAGE QUALITY ANALYSIS")
        print(f"{'='*60}")
        print(f"File: {os.path.basename(image_path)}")
        print(f"Full path: {image_path}")
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image
            print(f"\nLoading image...")
            img = nib.load(image_path)
            data = img.get_fdata()
            print(f"✓ Image loaded successfully: {data.shape}")
            
            # Initialize results dictionary
            results = {
                'file_path': image_path,
                'subject_id': self._extract_subject_id(image_path),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'config_used': self.config.get_config_summary()
            }
            
            # STAGE 1: Raw image analysis (before any processing)
            print(f"\n{'-'*40}")
            print("STAGE 1: RAW IMAGE ANALYSIS")
            print(f"{'-'*40}")
            
            raw_metrics = self.raw_analyzer.analyze_raw_properties(data, img.header)
            acquisition_metrics = self.raw_analyzer.assess_acquisition_quality(data)
            
            results['stage_1_raw_analysis'] = raw_metrics
            results['stage_1_acquisition_assessment'] = acquisition_metrics
            results['stage_1_processing_report'] = self.raw_analyzer.get_analysis_report()
            
            # STAGE 2: Brain extraction
            print(f"\n{'-'*40}")
            print("STAGE 2: BRAIN EXTRACTION")
            print(f"{'-'*40}")
            
            brain_mask, brain_extraction_details = self.brain_extractor.extract_brain(data)
            brain_quality_metrics = self.brain_extractor.visualize_extraction_quality(data, brain_mask)
            
            # Calculate brain volume in ml
            voxel_volume_mm3 = np.prod(img.header.get_zooms()[:3])
            brain_volume_ml = np.sum(brain_mask) * voxel_volume_mm3 / 1000
            
            results['stage_2_brain_extraction'] = {
                'method': self.config.brain_extraction_method,
                'brain_volume_voxels': int(np.sum(brain_mask)),
                'brain_volume_ml': float(brain_volume_ml),
                'brain_coverage_percentage': float(np.sum(brain_mask) / np.sum(data > 0) * 100) if np.sum(data > 0) > 0 else 0,
                'extraction_details': brain_extraction_details,
                'quality_assessment': brain_quality_metrics
            }
            results['stage_2_processing_report'] = self.brain_extractor.get_extraction_report()
            
            # STAGE 3: Noise estimation
            print(f"\n{'-'*40}")
            print("STAGE 3: NOISE ESTIMATION")
            print(f"{'-'*40}")
            
            noise_estimate, noise_params = self.noise_estimator.estimate_noise(data, brain_mask)
            
            # Optional: Compare different noise methods
            if hasattr(self, '_compare_methods') and self._compare_methods:
                noise_comparison = self.noise_estimator.compare_noise_methods(data, brain_mask)
                results['stage_3_noise_method_comparison'] = noise_comparison
            
            results['stage_3_noise_analysis'] = {
                'method': self.config.noise_estimation_method,
                'noise_estimate': float(noise_estimate),
                'noise_parameters': noise_params
            }
            results['stage_3_processing_report'] = self.noise_estimator.get_noise_report()
            
            # STAGE 4: Quality metrics calculation
            print(f"\n{'-'*40}")
            print("STAGE 4: QUALITY METRICS CALCULATION")
            print(f"{'-'*40}")
            
            # SNR calculation
            snr_value, snr_details = self.metrics_calculator.calculate_snr_detailed(
                data, brain_mask, noise_estimate, noise_params
            )
            
            # CNR calculation
            cnr_value, cnr_details = self.metrics_calculator.calculate_cnr_detailed(
                data, brain_mask, noise_estimate, noise_params
            )
            
            # Uniformity assessment
            uniformity_value, uniformity_details = self.metrics_calculator.assess_uniformity_detailed(
                data, brain_mask
            )
            
            results['stage_4_snr_analysis'] = snr_details
            results['stage_4_cnr_analysis'] = cnr_details
            results['stage_4_uniformity_analysis'] = uniformity_details
            results['stage_4_processing_report'] = self.metrics_calculator.get_metrics_report()
            
            # STAGE 5: Quality assessment summary
            print(f"\n{'-'*40}")
            print("STAGE 5: QUALITY ASSESSMENT SUMMARY")
            print(f"{'-'*40}")
            
            quality_flags = self.metrics_calculator.assess_quality_flags(
                snr_value, uniformity_value, np.sum(brain_mask)
            )
            
            quality_score = self.metrics_calculator.calculate_quality_score(
                snr_value, uniformity_value, np.sum(brain_mask)
            )
            
            results['stage_5_quality_summary'] = {
                **quality_flags,
                'quality_score': float(quality_score),
                'thresholds_used': {
                    'min_snr': self.config.min_snr,
                    'max_uniformity': self.config.max_intensity_nonuniformity,
                    'min_volume_voxels': self.config.min_brain_volume
                }
            }
            
            # Legacy format for backward compatibility
            self._add_legacy_format(results, data, img.header, brain_mask, 
                                  snr_value, cnr_value, uniformity_value, noise_estimate)
            
            # Overall processing summary
            results['overall_processing_summary'] = self.main_tracker.get_report()
            
            # Final summary
            self._print_final_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed for {image_path}: {e}")
            return self._create_error_metrics(image_path, str(e))
    
    def _add_legacy_format(self, results: Dict, data: np.ndarray, header, brain_mask: np.ndarray,
                          snr: float, cnr: float, uniformity: float, noise: float):
        """Add legacy format fields for backward compatibility"""
        raw_metrics = results.get('stage_1_raw_analysis', {})
        
        results.update({
            'image_shape': data.shape,
            'voxel_size': header.get_zooms()[:3],
            'orig_mean_intensity': raw_metrics.get('raw_mean_intensity', np.nan),
            'orig_std_intensity': raw_metrics.get('raw_std_intensity', np.nan),
            'orig_min_intensity': raw_metrics.get('raw_min_intensity', np.nan),
            'orig_max_intensity': raw_metrics.get('raw_max_intensity', np.nan),
            'intensity_range': raw_metrics.get('raw_intensity_range', np.nan),
            'brain_volume_voxels': int(np.sum(brain_mask)),
            'brain_volume_ml': float(np.sum(brain_mask) * np.prod(header.get_zooms()[:3]) / 1000),
            'orig_snr': float(snr) if not np.isnan(snr) else np.nan,
            'orig_cnr': float(cnr) if not np.isnan(cnr) else np.nan,
            'orig_noise_level': float(noise) if not np.isnan(noise) else np.nan,
            'orig_uniformity': float(uniformity) if not np.isnan(uniformity) else np.nan,
            'snr_adequate': results['stage_5_quality_summary']['snr_adequate'],
            'uniformity_good': results['stage_5_quality_summary']['uniformity_good'],
            'volume_adequate': results['stage_5_quality_summary']['volume_adequate'],
            'overall_quality_good': results['stage_5_quality_summary']['overall_quality_good']
        })
    
    def _print_final_summary(self, results: Dict):
        """Print final analysis summary"""
        print(f"\n{'='*60}")
        print("FINAL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        summary = results.get('stage_5_quality_summary', {})
        
        print(f"Subject ID: {results.get('subject_id', 'Unknown')}")
        print(f"SNR: {results.get('orig_snr', 'N/A'):.2f} (Adequate: {summary.get('snr_adequate', False)})")
        print(f"CNR: {results.get('orig_cnr', 'N/A'):.2f}")
        print(f"Uniformity: {results.get('orig_uniformity', 'N/A'):.4f} (Good: {summary.get('uniformity_good', False)})")
        print(f"Volume: {results.get('brain_volume_ml', 'N/A'):.1f} ml (Adequate: {summary.get('volume_adequate', False)})")
        print(f"Overall Quality: {'GOOD' if summary.get('overall_quality_good', False) else 'POOR'}")
        print(f"Quality Score: {summary.get('quality_score', 0):.1f}/10.0")
        
        # Processing summary
        total_stages = 5
        successful_stages = sum([
            'stage_1_raw_analysis' in results and 'raw_analysis_error' not in results['stage_1_raw_analysis'],
            'stage_2_brain_extraction' in results,
            'stage_3_noise_analysis' in results,
            'stage_4_snr_analysis' in results and 'error' not in results['stage_4_snr_analysis'],
            'stage_5_quality_summary' in results
        ])
        
        print(f"Processing Success: {successful_stages}/{total_stages} stages completed")
        print(f"{'='*60}")
    
    def _extract_subject_id(self, file_path: str) -> str:
        """Extract subject ID from file path"""
        filename = os.path.basename(file_path)
        if filename.startswith('sub-'):
            return filename.split('_')[0]
        return filename.split('.')[0]
    
    def _create_error_metrics(self, image_path: str, error_msg: str) -> Dict:
        """Create error metrics for failed analyses"""
        return {
            'file_path': image_path,
            'subject_id': self._extract_subject_id(image_path),
            'error': error_msg,
            'analysis_failed': True,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'config_used': self.config.get_config_summary()
        }
    
    def save_results(self, results: Dict, output_dir: str, filename_prefix: str = "detailed_analysis"):
        """Save analysis results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        subject_id = results.get('subject_id', 'unknown')
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{subject_id}_{timestamp}.json"
        
        filepath = output_path / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {filepath}")
        return str(filepath)

def create_test_config():
    """Create a test configuration"""
    config = QualityConfig()
    config.output_root = r"c:\Users\Pesankar\OneDrive\Documents\GitHub\yopd-structural-analysis\image_analysis\outputs"
    config.brain_extraction_method = "threshold_based"
    config.noise_estimation_method = "edge_regions"
    config.min_snr = 10.0  # Lower threshold for testing
    return config
