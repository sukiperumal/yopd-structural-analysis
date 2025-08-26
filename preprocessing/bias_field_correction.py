#!/usr/bin/env python3
"""
Bias Field Correction Module
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from scipy.stats import pearsonr
from scipy.optimize import minimize
from skimage import filters, morphology, segmentation
from skimage.restoration import denoise_bilateral
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BiasFieldConfig:
    """Configuration for bias field correction"""
    output_root: str = "./bias_field_outputs"
    method: str = "polynomial"  # "polynomial", "histogram", "homomorphic"
    poly_order: int = 3
    max_iterations: int = 20
    convergence_threshold: float = 0.001
    smoothing_sigma: float = 2.0
    max_uniformity: float = 0.3
    
class QualityMetrics:
    """Calculate quality metrics"""
    
    @staticmethod
    def calculate_uniformity(image_data: np.ndarray, brain_mask: np.ndarray) -> float:
        """Calculate intensity uniformity using coefficient of variation"""
        try:
            if np.sum(brain_mask) == 0:
                return 1.0  # Worst case
            
            brain_intensities = image_data[brain_mask]
            if len(brain_intensities) == 0:
                return 1.0
            
            mean_intensity = np.mean(brain_intensities)
            if mean_intensity == 0:
                return 1.0
            
            std_intensity = np.std(brain_intensities)
            coefficient_of_variation = std_intensity / mean_intensity
            
            return float(coefficient_of_variation)
            
        except Exception:
            return 1.0
    
    @staticmethod
    def create_brain_mask(image_data: np.ndarray, threshold_percentile: float = 15) -> np.ndarray:
        """Create brain mask using Otsu thresholding and morphological operations"""
        try:
            # Initial threshold using percentile
            threshold = np.percentile(image_data[image_data > 0], threshold_percentile)
            
            # Create initial mask
            mask = image_data > threshold
            
            # Apply Otsu thresholding for refinement
            try:
                otsu_thresh = filters.threshold_otsu(image_data[image_data > 0])
                if otsu_thresh > threshold:
                    mask = image_data > otsu_thresh
            except:
                pass
            
            # Morphological operations to clean up the mask
            mask = ndimage.binary_fill_holes(mask)
            mask = morphology.binary_closing(mask, morphology.ball(3))
            mask = morphology.remove_small_objects(mask, min_size=1000)
            
            # Get largest connected component (brain)
            labeled = ndimage.label(mask)[0]
            if np.max(labeled) > 0:
                largest_component = np.argmax(np.bincount(labeled.flat)[1:]) + 1
                mask = labeled == largest_component
            
            return mask.astype(bool)
            
        except Exception:
            # Fallback: simple thresholding
            return image_data > (0.1 * np.max(image_data))

class PolynomialBiasCorrection:
    """Polynomial-based bias field correction"""
    
    def __init__(self, order: int = 3, max_iterations: int = 20):
        self.order = order
        self.max_iterations = max_iterations
    
    def _create_polynomial_basis(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Create polynomial basis functions"""
        z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
        # Normalize coordinates to [-1, 1]
        z = 2 * z / (shape[0] - 1) - 1
        y = 2 * y / (shape[1] - 1) - 1
        x = 2 * x / (shape[2] - 1) - 1
        
        basis_functions = []
        
        # Generate polynomial terms up to specified order
        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                for k in range(self.order + 1 - i - j):
                    if i + j + k <= self.order:
                        basis_functions.append((x**k) * (y**j) * (z**i))
        
        return np.stack(basis_functions, axis=-1)
    
    def correct(self, image_data: np.ndarray, brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform polynomial bias field correction"""
        shape = image_data.shape
        
        # Create polynomial basis
        basis = self._create_polynomial_basis(shape)
        basis_flat = basis.reshape(-1, basis.shape[-1])
        
        # Get brain voxels
        brain_indices = np.where(brain_mask.flatten())[0]
        brain_intensities = image_data.flatten()[brain_indices]
        brain_basis = basis_flat[brain_indices]
        
        # Iterative correction
        corrected_image = image_data.copy()
        
        for iteration in range(self.max_iterations):
            log_intensities = np.log(brain_intensities + 1e-10)
            
            # Fit polynomial to log intensities
            try:
                coeffs = np.linalg.lstsq(brain_basis, log_intensities, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                coeffs = np.linalg.pinv(brain_basis) @ log_intensities
            
            # Calculate bias field
            log_bias_field = (basis_flat @ coeffs).reshape(shape)
            bias_field = np.exp(log_bias_field - np.mean(log_bias_field))
            
            # Apply correction
            corrected_image = image_data / bias_field
            
            # Update brain intensities for next iteration
            brain_intensities = corrected_image.flatten()[brain_indices]
            
            # Check convergence (coefficient change)
            if iteration > 0:
                coeff_change = np.mean(np.abs(coeffs - prev_coeffs))
                if coeff_change < 0.001:
                    break
            
            prev_coeffs = coeffs.copy()
        
        return corrected_image, bias_field

class HistogramBiasCorrection:
    """Histogram-based bias field correction"""
    
    def __init__(self, num_classes: int = 3, max_iterations: int = 10):
        self.num_classes = num_classes
        self.max_iterations = max_iterations
    
    def correct(self, image_data: np.ndarray, brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform histogram-based bias field correction"""
        from sklearn.mixture import GaussianMixture
        
        # Get brain intensities
        brain_intensities = image_data[brain_mask]
        
        if len(brain_intensities) < 100:
            # Not enough data, return original
            return image_data, np.ones_like(image_data)
        
        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=self.num_classes, random_state=42)
        brain_intensities_reshaped = brain_intensities.reshape(-1, 1)
        
        try:
            gmm.fit(brain_intensities_reshaped)
            labels = gmm.predict(image_data[brain_mask].reshape(-1, 1))
        except:
            # Fallback: return original
            return image_data, np.ones_like(image_data)
        
        # Create tissue probability maps
        tissue_maps = np.zeros((*image_data.shape, self.num_classes))
        brain_coords = np.where(brain_mask)
        
        for i in range(self.num_classes):
            tissue_mask = labels == i
            tissue_maps[brain_coords[0][tissue_mask], 
                       brain_coords[1][tissue_mask], 
                       brain_coords[2][tissue_mask], i] = 1
        
        # Smooth tissue maps
        for i in range(self.num_classes):
            tissue_maps[..., i] = ndimage.gaussian_filter(tissue_maps[..., i], sigma=2.0)
        
        # Estimate bias field using expectation-maximization approach
        bias_field = np.ones_like(image_data)
        
        for iteration in range(self.max_iterations):
            # Update tissue means
            tissue_means = []
            for i in range(self.num_classes):
                weights = tissue_maps[..., i]
                if np.sum(weights) > 0:
                    mean_intensity = np.sum(image_data * weights) / np.sum(weights)
                    tissue_means.append(mean_intensity)
                else:
                    tissue_means.append(np.mean(image_data[brain_mask]))
            
            # Update bias field
            estimated_image = np.zeros_like(image_data)
            for i, mean_val in enumerate(tissue_means):
                estimated_image += tissue_maps[..., i] * mean_val
            
            # Smooth the bias field estimate
            ratio = image_data / (estimated_image + 1e-10)
            ratio[~brain_mask] = 1.0
            bias_field = ndimage.gaussian_filter(ratio, sigma=3.0)
            bias_field[bias_field <= 0] = 1.0
        
        corrected_image = image_data / bias_field
        return corrected_image, bias_field

class HomomorphicBiasCorrection:
    """Homomorphic filtering-based bias field correction"""
    
    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma
    
    def correct(self, image_data: np.ndarray, brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform homomorphic filtering bias correction"""
        # Log transform
        log_image = np.log(image_data + 1e-10)
        
        # Apply Gaussian filter to estimate low-frequency component (bias)
        log_bias = ndimage.gaussian_filter(log_image, sigma=self.sigma)
        
        # Subtract bias in log domain
        log_corrected = log_image - log_bias
        
        # Transform back
        corrected_image = np.exp(log_corrected)
        bias_field = np.exp(log_bias - np.mean(log_bias[brain_mask]))
        
        return corrected_image, bias_field

class BiasFieldCorrection:
    """Main Bias Field Correction class"""
    
    def __init__(self, config: BiasFieldConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('bias_field_correction')
    
    def _get_corrector(self):
        """Get the appropriate bias correction method"""
        if self.config.method == "polynomial":
            return PolynomialBiasCorrection(
                order=self.config.poly_order,
                max_iterations=self.config.max_iterations
            )
        elif self.config.method == "histogram":
            return HistogramBiasCorrection(
                max_iterations=self.config.max_iterations
            )
        elif self.config.method == "homomorphic":
            return HomomorphicBiasCorrection(
                sigma=self.config.smoothing_sigma
            )
        else:
            raise ValueError(f"Unknown correction method: {self.config.method}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict:
        """Correct bias field using selected method"""
        self.logger.info(f"Correcting bias field ({self.config.method}): {os.path.basename(input_path)}")
        
        try:
            # Load original image
            img_orig = nib.load(input_path)
            data_orig = img_orig.get_fdata().astype(np.float64)
            
            # Handle negative values and zeros
            data_orig[data_orig < 0] = 0
            if np.max(data_orig) == 0:
                raise ValueError("Image contains only zero values")
            
            # Create brain mask
            brain_mask = QualityMetrics.create_brain_mask(data_orig)
            
            if np.sum(brain_mask) < 1000:
                self.logger.warning("Brain mask too small, using liberal threshold")
                brain_mask = data_orig > (0.05 * np.max(data_orig))
            
            # Calculate original uniformity
            orig_uniformity = QualityMetrics.calculate_uniformity(data_orig, brain_mask)
            
            # Apply bias field correction
            corrector = self._get_corrector()
            data_corrected, bias_field = corrector.correct(data_orig, brain_mask)
            
            # Ensure corrected data is positive
            data_corrected = np.maximum(data_corrected, 0)
            
            # Preserve original intensity scale
            brain_mean_orig = np.mean(data_orig[brain_mask])
            brain_mean_corrected = np.mean(data_corrected[brain_mask])
            if brain_mean_corrected > 0:
                scale_factor = brain_mean_orig / brain_mean_corrected
                data_corrected *= scale_factor
                bias_field *= scale_factor
            
            # Save corrected image
            img_corrected = nib.Nifti1Image(
                data_corrected.astype(data_orig.dtype),
                img_orig.affine,
                img_orig.header
            )
            nib.save(img_corrected, output_path)
            
            # Save bias field
            bias_field_path = output_path.replace('.nii.gz', '_bias_field.nii.gz')
            bias_img = nib.Nifti1Image(
                bias_field.astype(np.float32),
                img_orig.affine,
                img_orig.header
            )
            nib.save(bias_img, bias_field_path)
            
            # Calculate corrected uniformity
            corrected_uniformity = QualityMetrics.calculate_uniformity(data_corrected, brain_mask)
            
            # Calculate improvement
            uniformity_improvement = (orig_uniformity - corrected_uniformity) / orig_uniformity if orig_uniformity > 0 else 0
            
            # Quality check
            uniformity_adequate = corrected_uniformity <= self.config.max_uniformity
            
            # Calculate additional metrics
            bias_field_stats = {
                'bias_field_mean': float(np.mean(bias_field)),
                'bias_field_std': float(np.std(bias_field)),
                'bias_field_range': float(np.max(bias_field) - np.min(bias_field)),
                'bias_field_min': float(np.min(bias_field)),
                'bias_field_max': float(np.max(bias_field))
            }
            
            correlation = pearsonr(data_orig.flatten(), data_corrected.flatten())[0]
            
            metrics = {
                'subject_id': self._extract_subject_id(input_path),
                'input_file': input_path,
                'output_file': output_path,
                'bias_field_file': bias_field_path,
                'processing_step': 'bias_field_correction',
                'method': self.config.method,
                'timestamp': datetime.now().isoformat(),
                'original_uniformity': float(orig_uniformity),
                'corrected_uniformity': float(corrected_uniformity),
                'uniformity_improvement': float(uniformity_improvement),
                'uniformity_improvement_percent': float(uniformity_improvement * 100),
                'uniformity_adequate': bool(uniformity_adequate),
                'brain_mask_voxels': int(np.sum(brain_mask)),
                'brain_mask_percent': float(100 * np.sum(brain_mask) / brain_mask.size),
                **bias_field_stats,
                'correlation_with_original': float(correlation),
                'success': True,
                'error_message': ''
            }
            
            self.results.append(metrics)
            
            if uniformity_adequate:
                self.logger.info(f"✓ Bias field correction completed with adequate uniformity ({corrected_uniformity:.4f})")
            else:
                self.logger.warning(f"⚠ Bias field correction completed but uniformity above threshold ({corrected_uniformity:.4f})")
            
            return metrics
            
        except Exception as e:
            error_metrics = {
                'subject_id': self._extract_subject_id(input_path),
                'input_file': input_path,
                'output_file': output_path,
                'processing_step': 'bias_field_correction',
                'method': self.config.method,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error_message': str(e)
            }
            self.results.append(error_metrics)
            self.logger.error(f"✗ Bias field correction failed: {e}")
            return error_metrics
    
    def process_batch(self, input_directory: str, output_directory: str) -> List[Dict]:
        """Process batch of images"""
        self.logger.info(f"Starting batch bias field correction: {input_directory}")
        
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in ['.nii', '.nii.gz']:
            image_files.extend(list(input_dir.rglob(f'*{ext}')))
        
        if not image_files:
            self.logger.error("No image files found")
            return []
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            # Create subject-specific output path
            subject_id = self._extract_subject_id(str(image_file))
            output_file = output_dir / f"{subject_id}_bias_corrected.nii.gz"
            
            try:
                self.process(str(image_file), str(output_file))
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
        
        # Save results
        self.save_metrics(str(output_dir))
        return self.results
    
    def save_metrics(self, output_dir: str):
        """Save metrics to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = Path(output_dir) / "bias_field_correction_metrics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved metrics to {csv_file}")
            return str(csv_file)
        return None
    
    def _extract_subject_id(self, file_path: str) -> str:
        return Path(file_path).stem.split('_')[0]

def main():
    """Main function for bias field correction"""
    parser = argparse.ArgumentParser(description='MRI Bias Field Correction using Pure Python')
    parser.add_argument('--input', '-i', required=True, help='Input directory or single image file')
    parser.add_argument('--output', '-o', default='./bias_field_outputs', help='Output directory')
    parser.add_argument('--single', action='store_true', help='Process single image instead of batch')
    parser.add_argument('--method', '-m', choices=['polynomial', 'histogram', 'homomorphic'], 
                       default='polynomial', help='Bias correction method')
    parser.add_argument('--poly-order', type=int, default=3, help='Polynomial order (for polynomial method)')
    parser.add_argument('--max-iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--smoothing-sigma', type=float, default=2.0, help='Smoothing sigma (for homomorphic method)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = BiasFieldConfig(
        output_root=args.output,
        method=args.method,
        poly_order=args.poly_order,
        max_iterations=args.max_iterations,
        smoothing_sigma=args.smoothing_sigma
    )
    
    # Initialize processor
    processor = BiasFieldCorrection(config)
    
    print("MRI Bias Field Correction - Pure Python")
    print("=" * 45)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    if args.method == "polynomial":
        print(f"Polynomial order: {args.poly_order}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Mode: {'Single image' if args.single else 'Batch processing'}")
    print()
    
    try:
        if args.single:
            # Process single image
            output_file = Path(args.output) / f"{processor._extract_subject_id(args.input)}_bias_corrected.nii.gz"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            result = processor.process(args.input, str(output_file))
            processor.save_metrics(args.output)
            
            if result['success']:
                print(f"✓ Successfully processed: {result['output_file']}")
                print(f"  Method: {result['method']}")
                print(f"  Uniformity improvement: {result['uniformity_improvement_percent']:.1f}%")
                print(f"  Final uniformity: {result['corrected_uniformity']:.4f}")
                print(f"  Brain mask coverage: {result['brain_mask_percent']:.1f}%")
            else:
                print(f"✗ Processing failed: {result['error_message']}")
        else:
            # Process batch
            results = processor.process_batch(args.input, args.output)
            successful = sum(1 for r in results if r.get('success', False))
            print(f"\nBatch processing completed:")
            print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")
            
            if successful > 0:
                avg_improvement = np.mean([r.get('uniformity_improvement_percent', 0) 
                                         for r in results if r.get('success', False)])
                print(f"Average uniformity improvement: {avg_improvement:.1f}%")
        
        print(f"\nResults saved to: {Path(args.output).absolute()}")
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())