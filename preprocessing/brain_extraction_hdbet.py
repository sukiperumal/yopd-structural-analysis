#!/usr/bin/env python3
"""
Brain Extraction Module
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Only import if absolutely necessary
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

@dataclass
class UltraConservativeConfig:
    """Ultra-conservative configuration"""
    output_root: str = "./brain_extraction_outputs"
    hdbet_path: str = "C:/Users/Pesankar/OneDrive/Documents/GitHub/yopd-structural-analysis/external_tools/HD-BET"
    brain_extraction_threshold: float = 0.5
    max_memory_mb: int = 1024  # Very conservative 1GB limit
    slice_processing: bool = True  # Process slice by slice
    emergency_cleanup: bool = True  # Aggressive cleanup
    skip_large_files: bool = True  # Skip files that are too large

class MemoryMonitor:
    """Monitor system memory usage"""
    
    @staticmethod
    def get_available_memory_mb() -> float:
        """Get available system memory in MB"""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except:
            return 1024  # Default assumption
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    @staticmethod
    def emergency_cleanup():
        """Emergency memory cleanup"""
        gc.collect()
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(50, 5, 5)  # More aggressive GC

class SliceProcessor:
    """Process images slice by slice to avoid memory issues"""
    
    @staticmethod
    def get_file_size_mb(filepath: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(filepath) / (1024 * 1024)
        except:
            return 0
    
    @staticmethod
    def estimate_memory_needed(img_shape: tuple, dtype=np.float32) -> float:
        """Estimate memory needed in MB"""
        bytes_per_voxel = np.dtype(dtype).itemsize
        # Multiply by 3 to account for intermediate arrays
        return (np.prod(img_shape) * bytes_per_voxel * 3) / (1024 * 1024)
    
    @staticmethod
    def can_process_safely(filepath: str, max_memory_mb: int) -> Tuple[bool, str]:
        """Check if file can be processed safely"""
        try:
            # Check file size first
            file_size = SliceProcessor.get_file_size_mb(filepath)
            if file_size > max_memory_mb:
                return False, f"File too large: {file_size:.1f}MB > {max_memory_mb}MB"
            
            # Check available system memory
            available_memory = MemoryMonitor.get_available_memory_mb()
            if available_memory < max_memory_mb:
                return False, f"Insufficient memory: {available_memory:.1f}MB available"
            
            # Try to peek at image dimensions without loading
            img = nib.load(filepath)
            estimated_memory = SliceProcessor.estimate_memory_needed(img.shape)
            
            if estimated_memory > max_memory_mb:
                return False, f"Estimated memory needed: {estimated_memory:.1f}MB > {max_memory_mb}MB"
            
            return True, "Safe to process"
            
        except Exception as e:
            return False, f"Error checking file: {str(e)}"

class UltraConservativeBrainExtraction:
    """Ultra-conservative brain extraction that won't crash your laptop"""
    
    def __init__(self, config: UltraConservativeConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []
        self.memory_monitor = MemoryMonitor()
        
    def _setup_logging(self):
        """Minimal logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger('brain_extraction')
    
    def _check_memory_before_processing(self) -> bool:
        """Check memory before each operation"""
        available = self.memory_monitor.get_available_memory_mb()
        current_usage = self.memory_monitor.get_memory_usage_mb()
        
        if available < self.config.max_memory_mb or current_usage > self.config.max_memory_mb:
            self.logger.warning(f"Memory check failed. Available: {available:.1f}MB, Usage: {current_usage:.1f}MB")
            if self.config.emergency_cleanup:
                self.memory_monitor.emergency_cleanup()
            return False
        return True
    
    def process(self, input_path: str, output_path: str, method: str = "threshold") -> Dict:
        """Ultra-safe brain extraction"""
        self.logger.info(f"Checking: {os.path.basename(input_path)}")
        
        # Pre-flight safety check
        can_process, reason = SliceProcessor.can_process_safely(input_path, self.config.max_memory_mb)
        if not can_process:
            self.logger.warning(f"Skipping {input_path}: {reason}")
            return self._create_error_metrics(input_path, output_path, method, f"Skipped: {reason}")
        
        try:
            start_time = datetime.now()
            
            # Memory check before processing
            if not self._check_memory_before_processing():
                raise MemoryError("Insufficient memory before processing")
            
            self.logger.info(f"Processing: {os.path.basename(input_path)}")
            
            if method == "hdbet":
                success = self._run_hdbet_minimal(input_path, output_path)
            else:
                success = self._run_slice_based_extraction(input_path, output_path)
            
            if not success:
                raise RuntimeError("Brain extraction failed")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            metrics = self._create_success_metrics(input_path, output_path, method, processing_time)
            
            self.results.append(metrics)
            self.logger.info(f"✓ Completed in {processing_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            error_metrics = self._create_error_metrics(input_path, output_path, method, str(e))
            self.results.append(error_metrics)
            self.logger.error(f"✗ Failed: {e}")
            return error_metrics
        
        finally:
            # Always cleanup after each image
            self.memory_monitor.emergency_cleanup()
    
    def _run_slice_based_extraction(self, input_path: str, output_path: str) -> bool:
        """Process image slice by slice to minimize memory usage"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load image header only (no data)
            self.logger.info("Loading image header...")
            img = nib.load(input_path)
            shape = img.shape
            affine = img.affine
            header = img.header
            
            # Create output array
            self.logger.info("Initializing output array...")
            output_data = np.zeros(shape, dtype=np.float32)
            
            # Process slice by slice along the largest dimension
            largest_dim = np.argmax(shape)
            num_slices = shape[largest_dim]
            
            self.logger.info(f"Processing {num_slices} slices along dimension {largest_dim}")
            
            # Calculate global threshold first (sample-based)
            self.logger.info("Calculating threshold...")
            global_max = self._estimate_global_max_safe(img)
            threshold = self.config.brain_extraction_threshold * global_max
            
            # Process each slice
            for i in range(num_slices):
                if i % 10 == 0:
                    self.logger.info(f"Processing slice {i+1}/{num_slices}")
                
                # Memory check every few slices
                if i % 5 == 0 and not self._check_memory_before_processing():
                    self.logger.warning("Memory issue during slice processing")
                    self.memory_monitor.emergency_cleanup()
                
                # Extract single slice
                slice_data = self._extract_slice_safe(img, i, largest_dim)
                
                # Apply threshold
                slice_mask = slice_data > threshold
                
                # Basic morphological operations (very conservative)
                if SCIPY_AVAILABLE and slice_mask.any():
                    try:
                        slice_mask = ndimage.binary_fill_holes(slice_mask)
                    except:
                        pass  # Skip if any issues
                
                # Apply mask
                processed_slice = slice_data * slice_mask
                
                # Store in output
                if largest_dim == 0:
                    output_data[i, :, :] = processed_slice
                elif largest_dim == 1:
                    output_data[:, i, :] = processed_slice
                else:
                    output_data[:, :, i] = processed_slice
                
                # Cleanup slice data
                del slice_data, slice_mask, processed_slice
                
                if i % 5 == 0:
                    gc.collect()
            
            # Save result
            self.logger.info("Saving result...")
            output_img = nib.Nifti1Image(output_data, affine, header)
            nib.save(output_img, output_path)
            
            # Cleanup
            del output_data, output_img
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Slice-based extraction error: {e}")
            return False
    
    def _estimate_global_max_safe(self, img) -> float:
        """Safely estimate global maximum by sampling"""
        try:
            # Sample a few slices to estimate maximum
            shape = img.shape
            largest_dim = np.argmax(shape)
            num_slices = shape[largest_dim]
            
            # Sample 5-10 slices evenly distributed
            sample_indices = np.linspace(0, num_slices-1, min(10, num_slices), dtype=int)
            max_values = []
            
            for idx in sample_indices:
                slice_data = self._extract_slice_safe(img, idx, largest_dim)
                max_values.append(np.max(slice_data))
                del slice_data
                gc.collect()
            
            return max(max_values) if max_values else 1.0
            
        except Exception:
            return 1.0  # Fallback value
    
    def _extract_slice_safe(self, img, slice_idx: int, dimension: int) -> np.ndarray:
        """Safely extract a single slice"""
        try:
            # Use memory mapping and extract only one slice
            if dimension == 0:
                slice_data = np.array(img.dataobj[slice_idx, :, :], dtype=np.float32)
            elif dimension == 1:
                slice_data = np.array(img.dataobj[:, slice_idx, :], dtype=np.float32)
            else:
                slice_data = np.array(img.dataobj[:, :, slice_idx], dtype=np.float32)
            
            return slice_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting slice {slice_idx}: {e}")
            # Return empty slice of correct shape
            shape = list(img.shape)
            shape.pop(dimension)
            return np.zeros(shape, dtype=np.float32)
    
    def _run_hdbet_minimal(self, input_path: str, output_path: str) -> bool:
        """Minimal HD-BET execution"""
        try:
            if not os.path.exists(self.config.hdbet_path):
                self.logger.warning("HD-BET not found")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "HD_BET.entry_point",
                "-i", input_path,
                "-o", output_path,
                "-device", "cpu",
                "--disable_tta"
            ]
            
            self.logger.info("Running HD-BET...")
            # Very short timeout to prevent hanging
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            return result.returncode == 0 and os.path.exists(output_path)
            
        except subprocess.TimeoutExpired:
            self.logger.error("HD-BET timed out")
            return False
        except Exception as e:
            self.logger.error(f"HD-BET error: {e}")
            return False
    
    def _create_success_metrics(self, input_path: str, output_path: str, method: str, processing_time: float) -> Dict:
        """Create success metrics"""
        return {
            'subject_id': self._extract_subject_id(input_path),
            'input_file': input_path,
            'output_file': output_path,
            'processing_step': 'brain_extraction',
            'extraction_method': method,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'success': True,
            'error_message': '',
            'ultra_conservative_mode': True
        }
    
    def _create_error_metrics(self, input_path: str, output_path: str, method: str, error: str) -> Dict:
        """Create error metrics"""
        return {
            'subject_id': self._extract_subject_id(input_path),
            'input_file': input_path,
            'output_file': output_path,
            'processing_step': 'brain_extraction',
            'extraction_method': method,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': 0,
            'success': False,
            'error_message': error,
            'ultra_conservative_mode': True
        }
    
    def process_batch_one_by_one(self, input_directory: str, output_directory: str, 
                                method: str = "threshold") -> List[Dict]:
        """Process batch one image at a time with frequent cleanup"""
        self.logger.info(f"Starting ultra-conservative batch processing: {input_directory}")
        
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_files = []
        for ext in ['.nii', '.nii.gz']:
            image_files.extend(list(input_dir.rglob(f'*{ext}')))
        
        if not image_files:
            self.logger.error("No image files found")
            return []
        
        self.logger.info(f"Found {len(image_files)} images")
        
        # Process extremely carefully
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"\n=== [{i}/{len(image_files)}] {image_file.name} ===")
            
            # Memory check before each image
            available = self.memory_monitor.get_available_memory_mb()
            self.logger.info(f"Available memory: {available:.1f}MB")
            
            if available < self.config.max_memory_mb:
                self.logger.warning("Low memory - performing emergency cleanup")
                self.memory_monitor.emergency_cleanup()
                
                # Check again
                available = self.memory_monitor.get_available_memory_mb()
                if available < self.config.max_memory_mb:
                    self.logger.error("Still insufficient memory - skipping remaining images")
                    break
            
            # Process single image
            subject_id = self._extract_subject_id(str(image_file))
            output_file = output_dir / f"{subject_id}_brain.nii.gz"
            
            try:
                self.process(str(image_file), str(output_file), method=method)
            except Exception as e:
                self.logger.error(f"Critical error processing {image_file}: {e}")
            
            # Aggressive cleanup after each image
            self.memory_monitor.emergency_cleanup()
            
            # Progress summary every 5 images
            if i % 5 == 0:
                successful = sum(1 for r in self.results if r.get('success', False))
                self.logger.info(f"Progress: {successful}/{i} successful ({successful/i*100:.1f}%)")
        
        # Save final results
        self.save_metrics(str(output_dir))
        return self.results
    
    def save_metrics(self, output_dir: str):
        """Save metrics"""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = Path(output_dir) / "brain_extraction_metrics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved metrics to {csv_file}")
            return str(csv_file)
        return None
    
    def _extract_subject_id(self, file_path: str) -> str:
        """Extract subject ID"""
        return Path(file_path).stem.replace('.nii', '').split('_')[0]

def main():
    """Ultra-conservative main function"""
    parser = argparse.ArgumentParser(description='Ultra-Conservative Brain Extraction (Won\'t Crash Your Laptop)')
    parser.add_argument('--input', '-i', required=True, help='Input directory or single image file')
    parser.add_argument('--output', '-o', default='./brain_extraction_outputs', help='Output directory')
    parser.add_argument('--single', action='store_true', help='Process single image')
    parser.add_argument('--method', choices=['hdbet', 'threshold'], default='threshold')
    parser.add_argument('--hdbet-path', help='HD-BET installation path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Brain extraction threshold')
    parser.add_argument('--max-memory', type=int, default=1024, help='Max memory in MB (default: 1024)')
    parser.add_argument('--test-only', action='store_true', help='Only test which files can be processed safely')
    
    args = parser.parse_args()
    
    # Ultra-conservative configuration
    config = UltraConservativeConfig(
        output_root=args.output,
        brain_extraction_threshold=args.threshold,
        max_memory_mb=args.max_memory
    )
    
    if args.hdbet_path:
        config.hdbet_path = args.hdbet_path
    
    processor = UltraConservativeBrainExtraction(config)
    
    print("Ultra-Conservative Brain Extraction")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print(f"Max memory: {args.max_memory}MB")
    print(f"Available memory: {MemoryMonitor.get_available_memory_mb():.1f}MB")
    print()
    
    if args.test_only:
        # Test mode - just check which files can be processed
        print("=== TESTING MODE - Checking files ===")
        input_dir = Path(args.input)
        image_files = []
        for ext in ['.nii', '.nii.gz']:
            image_files.extend(list(input_dir.rglob(f'*{ext}')))
        
        for image_file in image_files:
            can_process, reason = SliceProcessor.can_process_safely(str(image_file), args.max_memory)
            status = "✓ CAN PROCESS" if can_process else "✗ SKIP"
            print(f"{status}: {image_file.name} - {reason}")
        return 0
    
    try:
        start_time = datetime.now()
        
        if args.single:
            # Single image processing
            output_file = Path(args.output) / f"{processor._extract_subject_id(args.input)}_brain.nii.gz"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            
            result = processor.process(args.input, str(output_file), method=args.method)
            processor.save_metrics(args.output)
            
            if result['success']:
                print(f"✓ Success: {result['output_file']}")
                print(f"  Time: {result.get('processing_time_seconds', 0):.2f}s")
            else:
                print(f"✗ Failed: {result['error_message']}")
        
        else:
            # Batch processing
            results = processor.process_batch_one_by_one(args.input, args.output, method=args.method)
            successful = sum(1 for r in results if r.get('success', False))
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\nBatch processing completed in {total_time:.2f}s:")
            print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")
            if results:
                print(f"Success rate: {successful/len(results)*100:.1f}%")
        
        print(f"\nResults: {Path(args.output).absolute()}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())