#!/usr/bin/env python3
"""
Brain extraction module for enhanced image quality assessment
Provides multiple brain extraction methods with detailed technical reporting
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, filters
from typing import Tuple, Dict
from .preprocessing_tracker import PreprocessingStepTracker

class BrainExtractionModule:
    """Dedicated module for brain extraction with multiple methods"""
    
    def __init__(self, method: str = "threshold_based"):
        self.method = method
        self.tracker = PreprocessingStepTracker()
    
    def extract_brain_threshold(self, image_data: np.ndarray, 
                               threshold_factor: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Threshold-based brain extraction
        
        Technical Details:
        - Algorithm: Otsu's method + morphological operations
        - Library: scipy.ndimage, skimage.morphology
        - Parameters: threshold_factor for intensity threshold
        """
        print(f"Performing threshold-based brain extraction (factor={threshold_factor})...")
        original_shape = image_data.shape
        
        try:
            # Step 1: Initial intensity threshold
            max_intensity = np.max(image_data)
            threshold = threshold_factor * max_intensity
            binary_mask = image_data > threshold
            
            initial_brain_voxels = np.sum(binary_mask)
            print(f"  Initial threshold ({threshold:.1f}): {initial_brain_voxels:,} voxels")
            
            # Step 2: Morphological operations to clean up mask
            # Remove small objects (noise)
            binary_mask = morphology.remove_small_objects(
                binary_mask, min_size=1000, connectivity=3
            )
            
            after_cleanup_voxels = np.sum(binary_mask)
            print(f"  After small object removal: {after_cleanup_voxels:,} voxels")
            
            # Fill holes
            binary_mask = ndimage.binary_fill_holes(binary_mask)
            
            after_fill_voxels = np.sum(binary_mask)
            print(f"  After hole filling: {after_fill_voxels:,} voxels")
            
            # Morphological closing to smooth boundaries
            binary_mask = morphology.binary_closing(
                binary_mask, morphology.ball(radius=2)
            )
            
            # Step 3: Keep largest connected component (brain)
            labeled = measure.label(binary_mask)
            props = measure.regionprops(labeled)
            if props:
                largest_region = max(props, key=lambda x: x.area)
                brain_mask = labeled == largest_region.label
                print(f"  Largest connected component: {largest_region.area:,} voxels")
            else:
                brain_mask = binary_mask
                print("  Warning: No connected components found, using raw mask")
            
            final_brain_voxels = np.sum(brain_mask)
            print(f"  Final brain mask: {final_brain_voxels:,} voxels")
            
            # Calculate extraction success metrics
            brain_coverage = final_brain_voxels / np.sum(image_data > 0) if np.sum(image_data > 0) > 0 else 0
            print(f"  Brain coverage: {brain_coverage:.1%} of non-zero voxels")
            
            # Log the step
            parameters = {
                'threshold_factor': threshold_factor,
                'threshold_value': threshold,
                'max_intensity': max_intensity,
                'morphology_operations': ['remove_small_objects', 'binary_fill_holes', 'binary_closing'],
                'connectivity': 3,
                'min_object_size': 1000,
                'closing_radius': 2,
                'initial_voxels': int(initial_brain_voxels),
                'final_voxels': int(final_brain_voxels),
                'brain_coverage_percentage': float(brain_coverage * 100)
            }
            
            self.tracker.log_step(
                step_name="brain_extraction_threshold",
                method="threshold_morphology",
                parameters=parameters,
                input_shape=original_shape,
                output_shape=brain_mask.shape,
                success=True
            )
            
            print("✓ Threshold-based brain extraction completed successfully")
            return brain_mask.astype(bool), parameters
            
        except Exception as e:
            print(f"✗ Threshold-based brain extraction failed: {e}")
            
            self.tracker.log_step(
                step_name="brain_extraction_threshold",
                method="threshold_morphology",
                parameters={'error': str(e)},
                input_shape=original_shape,
                output_shape=original_shape,
                success=False
            )
            
            # Fallback to simple threshold
            fallback_mask = image_data > (0.1 * np.max(image_data))
            return fallback_mask, {'error': str(e), 'fallback_used': True}
    
    def extract_brain_bet_style(self, image_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        BET-style brain extraction (simplified implementation)
        
        Technical Details:
        - Algorithm: Edge-based deformable model
        - Inspired by: FSL BET (Brain Extraction Tool)
        - Implementation: Simplified gradient-based approach
        """
        print("Performing BET-style brain extraction...")
        original_shape = image_data.shape
        
        try:
            # Step 1: Intensity normalization
            normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
            print("  Intensity normalization completed")
            
            # Step 2: Gradient-based edge detection
            gradients = [np.gradient(normalized, axis=i) for i in range(3)]
            gradient_mag = np.sqrt(np.sum([g**2 for g in gradients], axis=0))
            
            # Step 3: Threshold based on gradient magnitude
            # Only consider gradients in regions with reasonable intensity
            mask_region = normalized > 0.1
            if np.sum(mask_region) == 0:
                raise ValueError("No suitable intensity region found for gradient analysis")
            
            edge_threshold = np.percentile(gradient_mag[mask_region], 75)
            edge_mask = gradient_mag > edge_threshold
            
            print(f"  Edge threshold: {edge_threshold:.6f}")
            print(f"  Edge voxels detected: {np.sum(edge_mask):,}")
            
            # Step 4: Morphological operations
            # Invert edge mask to get interior regions
            brain_mask = ~edge_mask
            brain_mask = ndimage.binary_fill_holes(brain_mask)  # Use scipy.ndimage instead
            brain_mask = morphology.remove_small_objects(brain_mask, min_size=5000)
            
            # Keep only regions with reasonable intensity
            brain_mask = brain_mask & mask_region
            
            final_brain_voxels = np.sum(brain_mask)
            brain_coverage = final_brain_voxels / np.sum(image_data > 0) if np.sum(image_data > 0) > 0 else 0
            
            print(f"  Final brain mask: {final_brain_voxels:,} voxels")
            print(f"  Brain coverage: {brain_coverage:.1%} of non-zero voxels")
            
            parameters = {
                'normalization': 'min_max',
                'gradient_method': 'numpy_gradient',
                'edge_threshold_percentile': 75,
                'edge_threshold_value': edge_threshold,
                'min_object_size': 5000,
                'final_voxels': int(final_brain_voxels),
                'brain_coverage_percentage': float(brain_coverage * 100)
            }
            
            self.tracker.log_step(
                step_name="brain_extraction_bet_style",
                method="gradient_deformable",
                parameters=parameters,
                input_shape=original_shape,
                output_shape=brain_mask.shape,
                success=True
            )
            
            print("✓ BET-style brain extraction completed successfully")
            return brain_mask.astype(bool), parameters
            
        except Exception as e:
            print(f"✗ BET-style brain extraction failed: {e}")
            print("  Falling back to threshold method...")
            # Fallback to threshold method
            return self.extract_brain_threshold(image_data)
    
    def extract_brain(self, image_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Extract brain using the configured method
        """
        if self.method == "threshold_based":
            return self.extract_brain_threshold(image_data, **kwargs)
        elif self.method == "bet_style":
            return self.extract_brain_bet_style(image_data, **kwargs)
        else:
            print(f"Warning: Unknown method '{self.method}', using threshold_based")
            return self.extract_brain_threshold(image_data, **kwargs)
    
    def visualize_extraction_quality(self, image_data: np.ndarray, brain_mask: np.ndarray) -> Dict:
        """
        Assess the quality of brain extraction
        """
        try:
            print("Assessing brain extraction quality...")
            
            # Calculate quality metrics
            total_nonzero = np.sum(image_data > 0)
            brain_voxels = np.sum(brain_mask)
            
            if total_nonzero == 0:
                return {'error': 'No non-zero voxels in image'}
            
            # Coverage metrics
            coverage = brain_voxels / total_nonzero
            
            # Intensity comparison
            brain_intensities = image_data[brain_mask]
            non_brain_intensities = image_data[~brain_mask & (image_data > 0)]
            
            quality_metrics = {
                'brain_voxel_count': int(brain_voxels),
                'total_nonzero_voxels': int(total_nonzero),
                'brain_coverage_percentage': float(coverage * 100),
                'mean_brain_intensity': float(np.mean(brain_intensities)) if len(brain_intensities) > 0 else 0,
                'mean_non_brain_intensity': float(np.mean(non_brain_intensities)) if len(non_brain_intensities) > 0 else 0,
                'brain_intensity_std': float(np.std(brain_intensities)) if len(brain_intensities) > 0 else 0
            }
            
            # Quality flags
            quality_flags = {
                'reasonable_coverage': 0.3 <= coverage <= 0.8,  # Brain should be 30-80% of non-zero voxels
                'good_contrast': quality_metrics['mean_brain_intensity'] > 1.5 * quality_metrics['mean_non_brain_intensity'] if quality_metrics['mean_non_brain_intensity'] > 0 else False,
                'sufficient_size': brain_voxels > 100000  # At least 100k voxels for adult brain
            }
            
            quality_metrics.update(quality_flags)
            
            print(f"  Brain coverage: {quality_metrics['brain_coverage_percentage']:.1f}%")
            print(f"  Reasonable coverage: {quality_flags['reasonable_coverage']}")
            print(f"  Good contrast: {quality_flags['good_contrast']}")
            print(f"  Sufficient size: {quality_flags['sufficient_size']}")
            
            return quality_metrics
            
        except Exception as e:
            print(f"✗ Brain extraction quality assessment failed: {e}")
            return {'quality_assessment_error': str(e)}
    
    def get_extraction_report(self) -> Dict:
        """Get comprehensive extraction report"""
        return self.tracker.get_report()
