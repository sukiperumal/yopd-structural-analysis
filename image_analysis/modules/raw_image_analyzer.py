#!/usr/bin/env python3
"""
Raw image analysis module for enhanced image quality assessment
Analyzes image properties BEFORE any preprocessing
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple
from .preprocessing_tracker import PreprocessingStepTracker

class RawImageAnalyzer:
    """Analyze raw image properties before any processing"""
    
    def __init__(self):
        self.tracker = PreprocessingStepTracker()
    
    def analyze_raw_properties(self, image_data: np.ndarray, img_header) -> Dict:
        """
        Analyze raw image properties before any processing
        
        Technical Details:
        - Direct analysis of NIfTI image data
        - No preprocessing applied
        - Basic intensity and spatial characteristics
        """
        try:
            print("Analyzing raw image properties...")
            
            # Basic intensity statistics
            raw_metrics = {
                'raw_mean_intensity': float(np.mean(image_data)),
                'raw_std_intensity': float(np.std(image_data)),
                'raw_min_intensity': float(np.min(image_data)),
                'raw_max_intensity': float(np.max(image_data)),
                'raw_intensity_range': float(np.max(image_data) - np.min(image_data)),
                'raw_nonzero_voxels': int(np.sum(image_data > 0)),
                'raw_zero_voxels': int(np.sum(image_data == 0)),
                'image_shape': image_data.shape,
                'voxel_size': img_header.get_zooms()[:3],
                'voxel_volume_mm3': float(np.prod(img_header.get_zooms()[:3])),
                'total_image_volume_mm3': float(np.prod(image_data.shape) * np.prod(img_header.get_zooms()[:3])),
            }
            
            print(f"  Image shape: {raw_metrics['image_shape']}")
            print(f"  Voxel size: {raw_metrics['voxel_size']}")
            print(f"  Intensity range: {raw_metrics['raw_min_intensity']:.1f} - {raw_metrics['raw_max_intensity']:.1f}")
            print(f"  Mean intensity: {raw_metrics['raw_mean_intensity']:.1f}")
            print(f"  Non-zero voxels: {raw_metrics['raw_nonzero_voxels']:,}")
            
            # Intensity distribution analysis (only on non-zero voxels)
            nonzero_data = image_data[image_data > 0]
            if len(nonzero_data) > 0:
                percentiles = [1, 5, 25, 50, 75, 95, 99]
                percentile_values = np.percentile(nonzero_data, percentiles)
                
                raw_metrics.update({
                    'raw_intensity_percentiles': {
                        f'{p}st' if p == 1 else f'{p}th': float(v) 
                        for p, v in zip(percentiles, percentile_values)
                    },
                    'raw_intensity_skewness': float(stats.skew(nonzero_data)),
                    'raw_intensity_kurtosis': float(stats.kurtosis(nonzero_data))
                })
                
                print(f"  Median intensity: {raw_metrics['raw_intensity_percentiles']['50th']:.1f}")
                print(f"  Intensity skewness: {raw_metrics['raw_intensity_skewness']:.3f}")
            
            # Log the analysis step
            self.tracker.log_step(
                step_name="raw_image_analysis",
                method="direct_numpy_analysis",
                parameters={'analysis_type': 'raw_properties'},
                input_shape=image_data.shape,
                output_shape=image_data.shape,
                success=True
            )
            
            print("✓ Raw image analysis completed successfully")
            return raw_metrics
            
        except Exception as e:
            print(f"✗ Raw image analysis failed: {e}")
            
            # Log the failed step
            self.tracker.log_step(
                step_name="raw_image_analysis",
                method="direct_numpy_analysis",
                parameters={'error': str(e)},
                input_shape=image_data.shape if 'image_data' in locals() else (0,),
                output_shape=(0,),
                success=False
            )
            
            return {'raw_analysis_error': str(e)}
    
    def assess_acquisition_quality(self, image_data: np.ndarray) -> Dict:
        """
        Assess potential acquisition quality issues
        """
        try:
            print("Assessing acquisition quality indicators...")
            
            # Check for potential motion artifacts (high variance in expected uniform regions)
            # Sample edge regions that should be relatively uniform
            edge_thickness = min(5, min(image_data.shape) // 20)
            edge_regions = [
                image_data[:edge_thickness, :, :],
                image_data[-edge_thickness:, :, :],
                image_data[:, :edge_thickness, :],
                image_data[:, -edge_thickness:, :]
            ]
            
            edge_variances = [np.var(region[region > 0]) for region in edge_regions if np.sum(region > 0) > 0]
            
            # Check for intensity distribution anomalies
            nonzero_data = image_data[image_data > 0]
            if len(nonzero_data) > 1000:  # Need sufficient data
                # Check for bimodal distribution (possible partial volume effects)
                hist, _ = np.histogram(nonzero_data, bins=50)
                hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')  # Simple smoothing
                peaks = []
                for i in range(1, len(hist_smooth)-1):
                    if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                        peaks.append(i)
                
                acquisition_metrics = {
                    'edge_variance_mean': float(np.mean(edge_variances)) if edge_variances else np.nan,
                    'edge_variance_std': float(np.std(edge_variances)) if len(edge_variances) > 1 else np.nan,
                    'intensity_histogram_peaks': len(peaks),
                    'intensity_distribution_uniformity': float(np.std(hist_smooth) / np.mean(hist_smooth)) if np.mean(hist_smooth) > 0 else np.nan,
                    'potential_motion_artifacts': len(edge_variances) > 0 and np.max(edge_variances) > 2 * np.mean(edge_variances),
                    'potential_distribution_anomalies': len(peaks) > 2
                }
                
                print(f"  Edge variance (motion indicator): {acquisition_metrics['edge_variance_mean']:.1f}")
                print(f"  Histogram peaks detected: {acquisition_metrics['intensity_histogram_peaks']}")
                print(f"  Potential motion artifacts: {acquisition_metrics['potential_motion_artifacts']}")
                
                return acquisition_metrics
            else:
                print("  Insufficient data for acquisition quality assessment")
                return {'insufficient_data': True}
                
        except Exception as e:
            print(f"✗ Acquisition quality assessment failed: {e}")
            return {'acquisition_assessment_error': str(e)}
    
    def get_analysis_report(self) -> Dict:
        """Get comprehensive analysis report"""
        return self.tracker.get_report()
