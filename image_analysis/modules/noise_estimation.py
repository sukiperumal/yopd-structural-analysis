#!/usr/bin/env python3
"""
Noise estimation module for enhanced image quality assessment
Provides multiple noise estimation methods with detailed technical reporting
"""

import numpy as np
from scipy.stats import median_abs_deviation
from typing import Tuple, Dict
from .preprocessing_tracker import PreprocessingStepTracker

class NoiseEstimationModule:
    """Dedicated module for noise estimation with multiple methods"""
    
    def __init__(self, method: str = "edge_regions"):
        self.method = method
        self.tracker = PreprocessingStepTracker()
    
    def estimate_noise_edge_regions(self, image_data: np.ndarray) -> Tuple[float, Dict]:
        """
        Noise estimation from edge regions
        
        Technical Details:
        - Algorithm: Standard deviation of edge region intensities
        - Assumption: Edge regions contain primarily noise
        - Implementation: 6-sided edge sampling + robust statistics
        """
        print("Estimating noise using edge regions method...")
        original_shape = image_data.shape
        
        try:
            # Adaptive edge thickness based on image size
            edge_thickness = min(10, min(image_data.shape) // 10)
            print(f"  Edge thickness: {edge_thickness} voxels")
            
            # Sample from 6 sides of the image volume
            edge_slices = [
                image_data[:edge_thickness, :, :],   # Front
                image_data[-edge_thickness:, :, :],  # Back
                image_data[:, :edge_thickness, :],   # Left
                image_data[:, -edge_thickness:, :],  # Right
                image_data[:, :, :edge_thickness],   # Bottom
                image_data[:, :, -edge_thickness:]   # Top
            ]
            
            # Combine all edge data
            edge_data = np.concatenate([edge.flatten() for edge in edge_slices])
            initial_edge_count = len(edge_data)
            
            print(f"  Initial edge voxels: {initial_edge_count:,}")
            
            # Remove outliers using IQR method for robust statistics
            q25, q75 = np.percentile(edge_data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Filter out outliers
            filtered_edge_data = edge_data[
                (edge_data >= lower_bound) & (edge_data <= upper_bound)
            ]
            
            filtered_edge_count = len(filtered_edge_data)
            outlier_percentage = (initial_edge_count - filtered_edge_count) / initial_edge_count * 100
            
            print(f"  Filtered edge voxels: {filtered_edge_count:,}")
            print(f"  Outliers removed: {outlier_percentage:.1f}%")
            
            # Calculate noise estimate
            if len(filtered_edge_data) > 0:
                noise_estimate = np.std(filtered_edge_data)
                edge_mean = np.mean(filtered_edge_data)
                edge_median = np.median(filtered_edge_data)
            else:
                # Fallback if filtering removes everything
                noise_estimate = np.std(edge_data)
                edge_mean = np.mean(edge_data)
                edge_median = np.median(edge_data)
                print("  Warning: All edge data filtered out, using unfiltered data")
            
            print(f"  Noise estimate (std): {noise_estimate:.3f}")
            print(f"  Edge region mean intensity: {edge_mean:.3f}")
            
            # FIXED: Handle zero noise estimates (binary masks, etc.)
            if noise_estimate < 1e-6:
                print(f"  ⚠️  Very low noise estimate ({noise_estimate:.6f}) - possibly binary/mask data")
                # Use minimum noise floor based on signal range
                if len(filtered_edge_data) > 0:
                    signal_range = np.ptp(filtered_edge_data)
                else:
                    signal_range = np.ptp(edge_data)
                min_noise_floor = max(0.001, signal_range * 0.01)  # 1% of signal range
                noise_estimate = max(noise_estimate, min_noise_floor)
                print(f"  Applied minimum noise floor: {noise_estimate:.6f}")
            
            parameters = {
                'edge_thickness': edge_thickness,
                'total_edge_voxels': initial_edge_count,
                'filtered_edge_voxels': filtered_edge_count,
                'outlier_removal_method': 'iqr_1.5',
                'outlier_percentage': float(outlier_percentage),
                'noise_estimate': float(noise_estimate),
                'iqr_bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                'edge_data_stats': {
                    'mean': float(edge_mean),
                    'std': float(np.std(filtered_edge_data) if len(filtered_edge_data) > 0 else np.std(edge_data)),
                    'median': float(edge_median),
                    'q25': float(q25),
                    'q75': float(q75)
                }
            }
            
            self.tracker.log_step(
                step_name="noise_estimation_edge",
                method="edge_region_std_robust",
                parameters=parameters,
                input_shape=original_shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ Edge regions noise estimation completed successfully")
            return noise_estimate, parameters
            
        except Exception as e:
            print(f"✗ Edge regions noise estimation failed: {e}")
            
            parameters = {'error': str(e), 'fallback_noise': 1.0}
            self.tracker.log_step(
                step_name="noise_estimation_edge",
                method="edge_region_std_robust",
                parameters=parameters,
                input_shape=original_shape,
                output_shape=(1,),
                success=False
            )
            return 1.0, parameters
    
    def estimate_noise_mad(self, image_data: np.ndarray, 
                          brain_mask: np.ndarray) -> Tuple[float, Dict]:
        """
        Noise estimation using Median Absolute Deviation
        
        Technical Details:
        - Algorithm: MAD (Median Absolute Deviation)
        - Library: scipy.stats.median_abs_deviation
        - Robust to outliers, suitable for Rician noise in MRI
        """
        print("Estimating noise using MAD (Median Absolute Deviation) method...")
        
        try:
            # Get background data (non-brain regions)
            background_data = image_data[~brain_mask]
            background_data = background_data[background_data > 0]  # Remove true zeros
            
            initial_background_count = len(background_data)
            print(f"  Background voxels: {initial_background_count:,}")
            
            if len(background_data) == 0:
                print("  No background data available, falling back to edge regions method")
                return self.estimate_noise_edge_regions(image_data)
            
            # MAD-based noise estimation
            # Scale='normal' converts MAD to standard deviation equivalent
            mad_noise = median_abs_deviation(background_data, scale='normal')
            
            # Additional statistics for validation
            background_mean = np.mean(background_data)
            background_median = np.median(background_data)
            background_std = np.std(background_data)
            background_mad_raw = median_abs_deviation(background_data)
            
            print(f"  Background median: {background_median:.3f}")
            print(f"  Background MAD (raw): {background_mad_raw:.3f}")
            print(f"  Noise estimate (MAD scaled): {mad_noise:.3f}")
            print(f"  Background std (for comparison): {background_std:.3f}")
            
            parameters = {
                'background_voxels': initial_background_count,
                'mad_scale': 'normal',
                'noise_estimate': float(mad_noise),
                'background_stats': {
                    'median': float(background_median),
                    'mad_raw': float(background_mad_raw),
                    'mad_scaled': float(mad_noise),
                    'mean': float(background_mean),
                    'std': float(background_std)
                }
            }
            
            self.tracker.log_step(
                step_name="noise_estimation_mad",
                method="median_absolute_deviation",
                parameters=parameters,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ MAD noise estimation completed successfully")
            return mad_noise, parameters
            
        except Exception as e:
            print(f"✗ MAD noise estimation failed: {e}")
            print("  Falling back to edge regions method...")
            return self.estimate_noise_edge_regions(image_data)
    
    def estimate_noise_background_roi(self, image_data: np.ndarray, 
                                    brain_mask: np.ndarray) -> Tuple[float, Dict]:
        """
        Noise estimation using background ROI approach
        
        Technical Details:
        - Algorithm: Standard deviation of background regions
        - More conservative than edge regions, uses entire background
        """
        print("Estimating noise using background ROI method...")
        
        try:
            # Get background data
            background_data = image_data[~brain_mask]
            background_data = background_data[background_data > 0]
            
            if len(background_data) == 0:
                print("  No background data available, falling back to edge regions method")
                return self.estimate_noise_edge_regions(image_data)
            
            # Apply robust filtering to background
            q10, q90 = np.percentile(background_data, [10, 90])
            filtered_background = background_data[
                (background_data >= q10) & (background_data <= q90)
            ]
            
            if len(filtered_background) == 0:
                filtered_background = background_data
            
            noise_estimate = np.std(filtered_background)
            
            print(f"  Background voxels: {len(background_data):,}")
            print(f"  Filtered background: {len(filtered_background):,}")
            print(f"  Noise estimate: {noise_estimate:.3f}")
            
            parameters = {
                'background_voxels': len(background_data),
                'filtered_background_voxels': len(filtered_background),
                'noise_estimate': float(noise_estimate),
                'filtering_percentiles': [10, 90],
                'background_stats': {
                    'mean': float(np.mean(filtered_background)),
                    'std': float(np.std(filtered_background)),
                    'median': float(np.median(filtered_background))
                }
            }
            
            self.tracker.log_step(
                step_name="noise_estimation_background_roi",
                method="background_std_robust",
                parameters=parameters,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ Background ROI noise estimation completed successfully")
            return noise_estimate, parameters
            
        except Exception as e:
            print(f"✗ Background ROI noise estimation failed: {e}")
            print("  Falling back to edge regions method...")
            return self.estimate_noise_edge_regions(image_data)
    
    def estimate_noise(self, image_data: np.ndarray, brain_mask: np.ndarray = None) -> Tuple[float, Dict]:
        """
        Estimate noise using the configured method
        """
        if self.method == "edge_regions":
            return self.estimate_noise_edge_regions(image_data)
        elif self.method == "mad" and brain_mask is not None:
            return self.estimate_noise_mad(image_data, brain_mask)
        elif self.method == "background_roi" and brain_mask is not None:
            return self.estimate_noise_background_roi(image_data, brain_mask)
        else:
            if brain_mask is None and self.method in ["mad", "background_roi"]:
                print(f"Warning: {self.method} requires brain_mask, falling back to edge_regions")
            else:
                print(f"Warning: Unknown method '{self.method}', using edge_regions")
            return self.estimate_noise_edge_regions(image_data)
    
    def compare_noise_methods(self, image_data: np.ndarray, brain_mask: np.ndarray) -> Dict:
        """
        Compare different noise estimation methods
        """
        print("Comparing different noise estimation methods...")
        
        comparison = {}
        
        # Edge regions method
        try:
            noise_edge, params_edge = self.estimate_noise_edge_regions(image_data)
            comparison['edge_regions'] = {
                'noise_estimate': noise_edge,
                'method_success': True,
                'parameters': params_edge
            }
        except Exception as e:
            comparison['edge_regions'] = {
                'noise_estimate': np.nan,
                'method_success': False,
                'error': str(e)
            }
        
        # MAD method
        try:
            noise_mad, params_mad = self.estimate_noise_mad(image_data, brain_mask)
            comparison['mad'] = {
                'noise_estimate': noise_mad,
                'method_success': True,
                'parameters': params_mad
            }
        except Exception as e:
            comparison['mad'] = {
                'noise_estimate': np.nan,
                'method_success': False,
                'error': str(e)
            }
        
        # Background ROI method
        try:
            noise_bg, params_bg = self.estimate_noise_background_roi(image_data, brain_mask)
            comparison['background_roi'] = {
                'noise_estimate': noise_bg,
                'method_success': True,
                'parameters': params_bg
            }
        except Exception as e:
            comparison['background_roi'] = {
                'noise_estimate': np.nan,
                'method_success': False,
                'error': str(e)
            }
        
        # Summary statistics
        successful_estimates = [
            comparison[method]['noise_estimate'] 
            for method in comparison 
            if comparison[method]['method_success'] and not np.isnan(comparison[method]['noise_estimate'])
        ]
        
        if successful_estimates:
            comparison['summary'] = {
                'mean_estimate': float(np.mean(successful_estimates)),
                'std_estimate': float(np.std(successful_estimates)),
                'min_estimate': float(np.min(successful_estimates)),
                'max_estimate': float(np.max(successful_estimates)),
                'successful_methods': len(successful_estimates),
                'coefficient_of_variation': float(np.std(successful_estimates) / np.mean(successful_estimates))
            }
            
            print(f"  Methods comparison summary:")
            print(f"    Successful methods: {len(successful_estimates)}/3")
            print(f"    Mean estimate: {comparison['summary']['mean_estimate']:.3f}")
            print(f"    Estimate range: {comparison['summary']['min_estimate']:.3f} - {comparison['summary']['max_estimate']:.3f}")
            print(f"    CV: {comparison['summary']['coefficient_of_variation']:.3f}")
        else:
            comparison['summary'] = {'error': 'No successful noise estimations'}
        
        return comparison
    
    def get_noise_report(self) -> Dict:
        """Get comprehensive noise estimation report"""
        return self.tracker.get_report()
