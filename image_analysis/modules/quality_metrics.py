#!/usr/bin/env python3
"""
Quality metrics calculator module for enhanced image quality assessment
Calculates SNR, CNR, and uniformity metrics with detailed technical reporting
"""

import numpy as np
from scipy.stats import median_abs_deviation
from typing import Tuple, Dict
from .preprocessing_tracker import PreprocessingStepTracker

class QualityMetricsCalculator:
    """Calculate quality metrics with detailed technical reporting"""
    
    def __init__(self, config):
        self.config = config
        self.tracker = PreprocessingStepTracker()
    
    def calculate_snr_detailed(self, image_data: np.ndarray, 
                              brain_mask: np.ndarray,
                              noise_estimate: float,
                              noise_params: Dict) -> Tuple[float, Dict]:
        """
        Calculate SNR with detailed technical reporting
        
        Technical Details:
        - Signal: Median intensity within brain mask (robust to outliers)
        - Noise: Pre-calculated noise estimate from noise estimation module
        - Formula: SNR = Signal / Noise
        """
        print("Calculating Signal-to-Noise Ratio (SNR)...")
        
        try:
            # Signal calculation
            brain_intensities = image_data[brain_mask]
            if len(brain_intensities) == 0:
                print("✗ Empty brain mask - cannot calculate SNR")
                return 0.0, {'error': 'Empty brain mask'}
            
            # Use median for robustness against outliers
            signal_median = np.median(brain_intensities)
            signal_mean = np.mean(brain_intensities)
            signal_std = np.std(brain_intensities)
            
            print(f"  Brain voxels: {len(brain_intensities):,}")
            print(f"  Signal (median): {signal_median:.3f}")
            print(f"  Signal (mean): {signal_mean:.3f}")
            print(f"  Noise estimate: {noise_estimate:.3f}")
            
            # SNR calculation
            snr = signal_median / noise_estimate if noise_estimate > 0 else 0.0
            
            print(f"  SNR: {snr:.2f}")
            
            # Additional signal statistics for comprehensive reporting
            signal_percentiles = np.percentile(brain_intensities, [5, 25, 50, 75, 95])
            
            snr_details = {
                'snr_value': float(snr),
                'signal_median': float(signal_median),
                'signal_mean': float(signal_mean),
                'signal_std': float(signal_std),
                'noise_estimate': float(noise_estimate),
                'noise_method': self.config.noise_estimation_method,
                'noise_parameters': noise_params,
                'brain_voxel_count': len(brain_intensities),
                'signal_percentiles': {
                    '5th': float(signal_percentiles[0]),
                    '25th': float(signal_percentiles[1]),
                    '50th': float(signal_percentiles[2]),
                    '75th': float(signal_percentiles[3]),
                    '95th': float(signal_percentiles[4])
                },
                'signal_range': {
                    'min': float(np.min(brain_intensities)),
                    'max': float(np.max(brain_intensities)),
                    'range': float(np.max(brain_intensities) - np.min(brain_intensities))
                }
            }
            
            self.tracker.log_step(
                step_name="snr_calculation",
                method=f"median_signal_{self.config.noise_estimation_method}_noise",
                parameters=snr_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ SNR calculation completed successfully")
            return snr, snr_details
            
        except Exception as e:
            print(f"✗ SNR calculation failed: {e}")
            
            error_details = {'error': str(e), 'snr_value': np.nan}
            self.tracker.log_step(
                step_name="snr_calculation",
                method=f"median_signal_{self.config.noise_estimation_method}_noise",
                parameters=error_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=False
            )
            return np.nan, error_details
    
    def calculate_cnr_detailed(self, image_data: np.ndarray, 
                              brain_mask: np.ndarray,
                              noise_estimate: float,
                              noise_params: Dict) -> Tuple[float, Dict]:
        """
        Calculate CNR with detailed technical reporting
        
        Technical Details:
        - Contrast: Difference between brain and background signal
        - Noise: Same estimation as SNR
        - Formula: CNR = (Signal_brain - Signal_background) / Noise
        """
        print("Calculating Contrast-to-Noise Ratio (CNR)...")
        
        try:
            brain_intensities = image_data[brain_mask]
            background_intensities = image_data[~brain_mask & (image_data > 0)]
            
            if len(brain_intensities) == 0 or len(background_intensities) == 0:
                print("✗ Insufficient data for CNR calculation")
                return 0.0, {'error': 'Insufficient data for CNR calculation'}
            
            # Signal calculations (using medians for robustness)
            brain_signal = np.median(brain_intensities)
            background_signal = np.median(background_intensities)
            contrast = brain_signal - background_signal
            
            print(f"  Brain voxels: {len(brain_intensities):,}")
            print(f"  Background voxels: {len(background_intensities):,}")
            print(f"  Brain signal (median): {brain_signal:.3f}")
            print(f"  Background signal (median): {background_signal:.3f}")
            print(f"  Contrast: {contrast:.3f}")
            print(f"  Noise estimate: {noise_estimate:.3f}")
            
            # CNR calculation
            cnr = contrast / noise_estimate if noise_estimate > 0 else 0.0
            
            print(f"  CNR: {cnr:.2f}")
            
            # Detailed statistics for both regions
            brain_percentiles = np.percentile(brain_intensities, [5, 25, 50, 75, 95])
            bg_percentiles = np.percentile(background_intensities, [5, 25, 50, 75, 95])
            
            cnr_details = {
                'cnr_value': float(cnr),
                'brain_signal_median': float(brain_signal),
                'background_signal_median': float(background_signal),
                'contrast': float(contrast),
                'noise_estimate': float(noise_estimate),
                'noise_method': self.config.noise_estimation_method,
                'noise_parameters': noise_params,
                'brain_voxel_count': len(brain_intensities),
                'background_voxel_count': len(background_intensities),
                'brain_signal_stats': {
                    'mean': float(np.mean(brain_intensities)),
                    'std': float(np.std(brain_intensities)),
                    'min': float(np.min(brain_intensities)),
                    'max': float(np.max(brain_intensities)),
                    'percentiles': {
                        '5th': float(brain_percentiles[0]),
                        '25th': float(brain_percentiles[1]),
                        '50th': float(brain_percentiles[2]),
                        '75th': float(brain_percentiles[3]),
                        '95th': float(brain_percentiles[4])
                    }
                },
                'background_signal_stats': {
                    'mean': float(np.mean(background_intensities)),
                    'std': float(np.std(background_intensities)),
                    'min': float(np.min(background_intensities)),
                    'max': float(np.max(background_intensities)),
                    'percentiles': {
                        '5th': float(bg_percentiles[0]),
                        '25th': float(bg_percentiles[1]),
                        '50th': float(bg_percentiles[2]),
                        '75th': float(bg_percentiles[3]),
                        '95th': float(bg_percentiles[4])
                    }
                }
            }
            
            self.tracker.log_step(
                step_name="cnr_calculation",
                method=f"brain_background_contrast_{self.config.noise_estimation_method}_noise",
                parameters=cnr_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ CNR calculation completed successfully")
            return cnr, cnr_details
            
        except Exception as e:
            print(f"✗ CNR calculation failed: {e}")
            
            error_details = {'error': str(e), 'cnr_value': np.nan}
            self.tracker.log_step(
                step_name="cnr_calculation",
                method=f"brain_background_contrast_{self.config.noise_estimation_method}_noise",
                parameters=error_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=False
            )
            return np.nan, error_details
    
    def assess_uniformity_detailed(self, image_data: np.ndarray, 
                                  brain_mask: np.ndarray) -> Tuple[float, Dict]:
        """
        Assess intensity uniformity with detailed technical reporting
        
        Technical Details:
        - Method: MAD-based uniformity assessment
        - Formula: Uniformity = MAD(brain_intensities) / Median(brain_intensities)
        - Lower values indicate better uniformity
        """
        print("Assessing intensity uniformity...")
        
        try:
            brain_intensities = image_data[brain_mask]
            
            if len(brain_intensities) == 0:
                print("✗ Empty brain mask - cannot assess uniformity")
                return 1.0, {'error': 'Empty brain mask'}
            
            print(f"  Brain voxels: {len(brain_intensities):,}")
            
            # Remove extreme outliers (1st and 99th percentile)
            q1, q99 = np.percentile(brain_intensities, [1, 99])
            filtered_intensities = brain_intensities[
                (brain_intensities >= q1) & (brain_intensities <= q99)
            ]
            
            if len(filtered_intensities) == 0:
                filtered_intensities = brain_intensities
                print("  Warning: Outlier filtering removed all data, using original data")
            
            outlier_percentage = (len(brain_intensities) - len(filtered_intensities)) / len(brain_intensities) * 100
            print(f"  Outliers removed: {outlier_percentage:.1f}%")
            print(f"  Filtered voxels: {len(filtered_intensities):,}")
            
            # Uniformity calculation using MAD
            median_intensity = np.median(filtered_intensities)
            mad_intensity = median_abs_deviation(filtered_intensities)
            
            # Primary uniformity metric (MAD-based)
            uniformity_mad = mad_intensity / median_intensity if median_intensity > 0 else 1.0
            
            # Alternative uniformity measures for comparison
            mean_intensity = np.mean(filtered_intensities)
            std_intensity = np.std(filtered_intensities)
            coefficient_of_variation = std_intensity / mean_intensity if mean_intensity > 0 else 1.0
            
            print(f"  Median intensity: {median_intensity:.3f}")
            print(f"  MAD: {mad_intensity:.3f}")
            print(f"  Uniformity (MAD/median): {uniformity_mad:.4f}")
            print(f"  CV (std/mean): {coefficient_of_variation:.4f}")
            
            # Additional uniformity metrics
            intensity_range = np.max(filtered_intensities) - np.min(filtered_intensities)
            relative_range = intensity_range / median_intensity if median_intensity > 0 else 0
            
            uniformity_details = {
                'uniformity_mad_based': float(uniformity_mad),
                'coefficient_of_variation': float(coefficient_of_variation),
                'median_intensity': float(median_intensity),
                'mad_intensity': float(mad_intensity),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'original_voxel_count': len(brain_intensities),
                'filtered_voxel_count': len(filtered_intensities),
                'outlier_percentage': float(outlier_percentage),
                'intensity_range': {
                    'min': float(np.min(filtered_intensities)),
                    'max': float(np.max(filtered_intensities)),
                    'range': float(intensity_range),
                    'relative_range': float(relative_range),
                    'q1_threshold': float(q1),
                    'q99_threshold': float(q99)
                },
                'method': 'mad_based_with_outlier_removal',
                'outlier_removal_percentiles': [1, 99]
            }
            
            self.tracker.log_step(
                step_name="uniformity_assessment",
                method="mad_based_robust",
                parameters=uniformity_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=True
            )
            
            print("✓ Uniformity assessment completed successfully")
            return uniformity_mad, uniformity_details
            
        except Exception as e:
            print(f"✗ Uniformity assessment failed: {e}")
            
            error_details = {'error': str(e), 'uniformity_value': np.nan}
            self.tracker.log_step(
                step_name="uniformity_assessment",
                method="mad_based_robust",
                parameters=error_details,
                input_shape=image_data.shape,
                output_shape=(1,),
                success=False
            )
            return np.nan, error_details
    
    def calculate_quality_score(self, snr: float, uniformity: float, volume_voxels: int) -> float:
        """
        Calculate composite quality score (0-10)
        
        Technical Details:
        - SNR component (0-4 points): Normalized to 30 as max "good" SNR
        - Uniformity component (0-3 points): Lower uniformity = better score
        - Volume component (0-3 points): Normalized to 1M voxels as reference
        """
        try:
            print("Calculating composite quality score...")
            
            # SNR component (0-4 points)
            snr_score = min(snr / 30 * 4, 4) if not np.isnan(snr) else 0
            
            # Uniformity component (0-3 points) - lower uniformity is better
            uniformity_score = max(0, 3 - uniformity * 10) if not np.isnan(uniformity) else 0
            
            # Volume component (0-3 points)
            volume_score = min(volume_voxels / 1000000 * 3, 3)
            
            total_score = snr_score + uniformity_score + volume_score
            final_score = min(total_score, 10.0)
            
            print(f"  SNR score: {snr_score:.2f}/4.0")
            print(f"  Uniformity score: {uniformity_score:.2f}/3.0")
            print(f"  Volume score: {volume_score:.2f}/3.0")
            print(f"  Total score: {final_score:.2f}/10.0")
            
            return final_score
            
        except Exception as e:
            print(f"✗ Quality score calculation failed: {e}")
            return 0.0
    
    def assess_quality_flags(self, snr: float, uniformity: float, volume_voxels: int) -> Dict:
        """
        Assess quality flags based on thresholds
        """
        print("Assessing quality flags...")
        
        flags = {
            'snr_adequate': snr >= self.config.min_snr if not np.isnan(snr) else False,
            'uniformity_good': uniformity <= self.config.max_intensity_nonuniformity if not np.isnan(uniformity) else False,
            'volume_adequate': volume_voxels >= self.config.min_brain_volume,
        }
        
        flags['overall_quality_good'] = all(flags.values())
        
        print(f"  SNR adequate (>= {self.config.min_snr}): {flags['snr_adequate']}")
        print(f"  Uniformity good (<= {self.config.max_intensity_nonuniformity}): {flags['uniformity_good']}")
        print(f"  Volume adequate (>= {self.config.min_brain_volume:,}): {flags['volume_adequate']}")
        print(f"  Overall quality good: {flags['overall_quality_good']}")
        
        return flags
    
    def get_metrics_report(self) -> Dict:
        """Get comprehensive metrics calculation report"""
        return self.tracker.get_report()
