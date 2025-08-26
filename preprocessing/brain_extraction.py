#!/usr/bin/env python3
"""
Brain Extraction Module
=========================================
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import warnings
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import only essential libraries to reduce memory footprint
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available - using basic operations")

@dataclass
class BrainExtractionConfig:
    """Ultra-lightweight configuration for brain extraction"""
    output_root: str = "./brain_extraction_outputs"
    hdbet_path: str = "C:/Users/Pesankar/OneDrive/Documents/GitHub/yopd-structural-analysis/external_tools/HD-BET"
    threshold: float = 0.25  # More aggressive threshold for better brain extraction
    min_brain_volume: int = 800000
    memory_limit_gb: float = 1.0  # Strict memory limit
    use_memory_mapping: bool = True  # Always use memory mapping
    batch_size: int = 1  # Process only one file at a time
    max_workers: int = 1  # Default to single process
    max_retries: int = 3  # Retry failed files
    cooldown_seconds: int = 30  # Cooldown between batches to free memory
    save_mask: bool = True  # Save brain mask for inspection

class UltraLightBrainExtractor:
    """Ultra-lightweight brain extraction using only nibabel and numpy"""
    
    def __init__(self, config: BrainExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []
        
        # Ensure output directory exists
        Path(config.output_root).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ultra-lightweight brain extractor with max {config.max_workers} workers")
        
    def _setup_logging(self):
        """Setup lightweight logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger('brain_extraction')
    
    def extract_brain_lightweight(self, input_path: str, output_path: str) -> bool:
        """Ultra-lightweight brain extraction using only nibabel and numpy"""
        try:
            self.logger.info(f"Processing: {os.path.basename(input_path)}")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load image with memory mapping
            img = nib.load(input_path, mmap=True)
            
            # Get data only when needed and as float32 to save memory
            data = np.asarray(img.dataobj, dtype=np.float32)
            
            # Calculate threshold based on image intensity
            # Using percentiles instead of max for robustness
            non_zero = data[data > 0]
            if len(non_zero) == 0:
                self.logger.error("Image contains no signal")
                return False
                
            p95 = np.percentile(non_zero, 95)
            threshold = p95 * self.config.threshold
            
            # Create binary mask (in-place to save memory)
            self.logger.info("Creating brain mask...")
            mask = data > threshold
            
            # Basic morphological operations if scipy is available
            if SCIPY_AVAILABLE:
                # Fill holes in each slice to save memory (3D is too intensive)
                self.logger.info("Filling holes in each slice...")
                for i in range(mask.shape[2]):
                    if np.any(mask[:,:,i]):
                        mask[:,:,i] = ndimage.binary_fill_holes(mask[:,:,i])
                
                # Find largest connected component in 3D
                self.logger.info("Finding largest component...")
                structure = ndimage.generate_binary_structure(3, 2)
                labeled, n_labels = ndimage.label(mask, structure)
                
                if n_labels > 0:
                    # Find largest component by count
                    sizes = np.bincount(labeled.ravel())
                    sizes[0] = 0  # Ignore background
                    largest_label = np.argmax(sizes)
                    
                    # Keep only largest component
                    mask = labeled == largest_label
                    
                del labeled
                gc.collect()
            
            # Apply mask to original data
            self.logger.info("Applying mask...")
            brain_data = data * mask
            
            # Save extracted brain
            self.logger.info("Saving extracted brain...")
            out_img = nib.Nifti1Image(brain_data, img.affine, img.header)
            nib.save(out_img, output_path)
            
            # Optionally save mask for quality control
            if self.config.save_mask:
                mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")
                mask_img = nib.Nifti1Image(mask.astype(np.int8), img.affine, img.header)
                nib.save(mask_img, mask_path)
                del mask_img
            
            # Clean up
            del data, mask, brain_data, out_img, img
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting brain: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str, batch_size: int = None):
        """Process files in small batches with cooldown periods"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        # Find all input files
        input_files = []
        for ext in ['.nii', '.nii.gz']:
            input_files.extend(list(Path(input_dir).rglob(f'*{ext}')))
        
        if not input_files:
            self.logger.error(f"No image files found in {input_dir}")
            return []
            
        total_files = len(input_files)
        self.logger.info(f"Found {total_files} images to process")
        
        # Process in small batches with cooldown periods
        results = []
        processed_count = 0
        
        for i in range(0, total_files, batch_size):
            batch = input_files[i:i+batch_size]
            batch_size_actual = len(batch)
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_files+batch_size-1)//batch_size} ({batch_size_actual} files)")
            
            # Process batch
            batch_results = self._process_batch_safe(batch, output_dir)
            results.extend(batch_results)
            
            processed_count += batch_size_actual
            self.logger.info(f"Completed {processed_count}/{total_files} files")
            
            # Save metrics after each batch
            self._save_metrics(output_dir)
            
            # Cooldown period between batches to free memory
            if i + batch_size < total_files:
                self.logger.info(f"Cooling down for {self.config.cooldown_seconds} seconds to free memory...")
                gc.collect()
                time.sleep(self.config.cooldown_seconds)
        
        return results
    
    def _process_batch_safe(self, files: List[Path], output_dir: str):
        """Process a batch of files safely with limited workers"""
        results = []
        
        # Use ProcessPoolExecutor for parallel processing with limited workers
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            for file_path in files:
                subject_id = self._extract_subject_id(str(file_path))
                output_path = Path(output_dir) / f"{subject_id}_brain.nii.gz"
                
                # Submit task
                future = executor.submit(
                    self._process_single_file_safe, str(file_path), str(output_path)
                )
                futures[future] = file_path
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.results.append(result)
                    
                    if result.get('success', False):
                        self.logger.info(f"✓ Completed: {file_path.name}")
                    else:
                        self.logger.error(f"✗ Failed: {file_path.name} - {result.get('error_message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {file_path.name}: {e}")
                    error_result = {
                        'subject_id': self._extract_subject_id(str(file_path)),
                        'input_file': str(file_path),
                        'output_file': str(Path(output_dir) / f"{self._extract_subject_id(str(file_path))}_brain.nii.gz"),
                        'success': False,
                        'error_message': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
                    self.results.append(error_result)
        
        return results
    
    def _process_single_file_safe(self, input_path: str, output_path: str) -> Dict:
        """Process a single file with retries and error handling"""
        start_time = datetime.now()
        
        for attempt in range(self.config.max_retries):
            try:
                success = self.extract_brain_lightweight(input_path, output_path)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if success:
                    return {
                        'subject_id': self._extract_subject_id(input_path),
                        'input_file': input_path,
                        'output_file': output_path,
                        'processing_step': 'brain_extraction',
                        'method': 'lightweight',
                        'timestamp': datetime.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'success': True,
                        'error_message': '',
                        'attempt': attempt + 1
                    }
                
                # If not successful but no exception, try again
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"Retry {attempt+1}/{self.config.max_retries} for {Path(input_path).name}")
                    time.sleep(5)  # Short delay before retry
                    gc.collect()  # Force garbage collection
                    
            except Exception as e:
                # Log exception and retry if attempts remain
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"Error on attempt {attempt+1}/{self.config.max_retries} for {Path(input_path).name}: {e}")
                    time.sleep(5)  # Short delay before retry
                    gc.collect()  # Force garbage collection
                else:
                    # Final attempt failed
                    processing_time = (datetime.now() - start_time).total_seconds()
                    return {
                        'subject_id': self._extract_subject_id(input_path),
                        'input_file': input_path,
                        'output_file': output_path,
                        'processing_step': 'brain_extraction',
                        'method': 'lightweight',
                        'timestamp': datetime.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'success': False,
                        'error_message': str(e),
                        'attempt': attempt + 1
                    }
        
        # If we got here, all attempts failed but no exception was raised
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'subject_id': self._extract_subject_id(input_path),
            'input_file': input_path,
            'output_file': output_path,
            'processing_step': 'brain_extraction',
            'method': 'lightweight',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'success': False,
            'error_message': 'All attempts failed without exception',
            'attempt': self.config.max_retries
        }
    
    def _save_metrics(self, output_dir: str) -> str:
        """Save metrics to CSV file"""
        if not self.results:
            return None
            
        csv_path = Path(output_dir) / "brain_extraction_metrics.csv"
        
        # Save metrics to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _extract_subject_id(self, file_path: str) -> str:
        """Extract subject ID from filename"""
        return Path(file_path).stem.split('_')[0]

def main():
    """Ultra-lightweight main function"""
    parser = argparse.ArgumentParser(description='Ultra-Lightweight Brain Extraction')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing NIfTI files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for extracted brains')
    parser.add_argument('--threshold', '-t', type=float, default=0.25, 
                      help='Threshold for brain extraction (0-1, lower is more aggressive)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                      help='Maximum number of parallel processes (default: 1)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                      help='Number of files to process before cooldown (default: 1)')
    parser.add_argument('--cooldown', '-c', type=int, default=30,
                      help='Cooldown period in seconds between batches (default: 30)')
    parser.add_argument('--no-mask', action='store_true',
                      help='Do not save brain mask for quality control')
    parser.add_argument('--retries', '-r', type=int, default=3,
                      help='Maximum number of retries for failed files (default: 3)')
    
    args = parser.parse_args()
    
    # Configure extractor
    config = BrainExtractionConfig(
        output_root=args.output,
        threshold=args.threshold,
        max_workers=args.workers,
        batch_size=args.batch_size,
        cooldown_seconds=args.cooldown,
        save_mask=not args.no_mask,
        max_retries=args.retries
    )
    
    # Print configuration
    print("Ultra-Lightweight Brain Extraction")
    print("=" * 35)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Max workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Save masks: {not args.no_mask}")
    print(f"Max retries: {args.retries}")
    print()
    
    # Initialize and run extractor
    extractor = UltraLightBrainExtractor(config)
    
    try:
        start_time = datetime.now()
        
        # Process files
        results = extractor.process_batch(args.input, args.output, args.batch_size)
        
        # Print summary
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        print("\nProcessing Summary:")
        print(f"Total files: {total}")
        print(f"Successfully processed: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {total - successful}")
        print(f"Total time: {total_time:.1f}s")
        if total > 0:
            print(f"Average time per file: {total_time/total:.1f}s")
        
        # Save final metrics
        metrics_path = extractor._save_metrics(args.output)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        # Save partial metrics
        metrics_path = extractor._save_metrics(args.output)
        print(f"Partial metrics saved to: {metrics_path}")
        return 1
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        # Try to save partial metrics
        try:
            metrics_path = extractor._save_metrics(args.output)
            print(f"Partial metrics saved to: {metrics_path}")
        except:
            print("Could not save metrics")
        return 1

if __name__ == "__main__":
    sys.exit(main())