#!/usr/bin/env python3
"""
T1w Anatomical MRI Denoising Module
===================================

Denoises T1-weighted anatomical scans using nibabel + skimage filters.
Restricts processing to anat/T1w scans in BIDS-like directory structure.
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
from scipy.stats import pearsonr
from dataclasses import dataclass
from skimage.restoration import denoise_nl_means, estimate_sigma
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DenoisingConfig:
    """Configuration for T1w denoising"""
    output_root: str = "D:\data_NIMHANS\t1w_denoising_outputs"
    patch_size: int = 5
    patch_distance: int = 6
    h_factor: float = 1.15   # strength of denoising
    min_snr_improvement: float = 0.05  # require ≥5% SNR improvement
    fast_mode: bool = True   # use fast mode for NL means
    downscale_factor: int = 1  # downscale images before processing (1 = no downscaling)
    max_workers: int = None  # None = use CPU count
    chunk_size: int = 1      # process images in chunks for memory efficiency


class QualityMetrics:
    """Quality metrics for denoising"""

    @staticmethod
    def calculate_snr(image_data: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        if mask is None:
            mask = image_data > (0.1 * np.max(image_data))
        if np.sum(mask) == 0:
            return 0.0
        signal = np.mean(image_data[mask])
        background_mask = image_data < (0.05 * np.max(image_data))
        noise = np.std(image_data[background_mask]) if np.sum(background_mask) > 0 else np.std(image_data[mask])
        return signal / noise if noise > 0 else 0.0


class T1wDenoising:
    """Performs denoising on T1-weighted anatomical images"""

    def __init__(self, config: DenoisingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger("t1w_denoising")

    def _extract_subject_info(self, file_path: str) -> Dict:
        """Extract subject, session, group info from BIDS-like path"""
        path_parts = Path(file_path).parts
        subject_id, session_id, group = "unknown", "01", "unknown"
        for part in path_parts:
            if part.startswith("sub-"):
                subject_id = part
            elif part.startswith("ses-"):
                session_id = part.replace("ses-", "")
        try:
            sub_idx = next(i for i, part in enumerate(path_parts) if part.startswith("sub-"))
            if sub_idx > 0:
                group = path_parts[sub_idx - 1]
        except (StopIteration, IndexError):
            pass
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group,
            "full_id": f"{subject_id}_ses-{session_id}" if session_id != "01" else subject_id
        }

    def _find_t1w_images(self, root_dir: Path) -> List[Path]:
        """Find all T1w anatomical scans in BIDS-like anat folders"""
        patterns = [
            "**/anat/*T1w.nii.gz", "**/anat/*T1w.nii",
            "**/anat/*T1.nii.gz",  "**/anat/*T1.nii",
        ]
        files = []
        for pattern in patterns:
            files.extend(list(root_dir.glob(pattern)))
        return sorted(set(files))

    def _denoise_data(self, data: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means denoising with optimizations"""
        # Downscale for faster processing if needed
        start_time = time.time()
        original_shape = data.shape
        
        if self.config.downscale_factor > 1:
            from scipy.ndimage import zoom
            # Downscale for processing
            small_data = zoom(data, 1/self.config.downscale_factor, order=1)
            self.logger.info(f"Downscaled from {original_shape} to {small_data.shape} for faster processing")
            
            # Estimate sigma on smaller data (faster)
            sigma_est = np.mean(estimate_sigma(small_data, channel_axis=None))
            
            # Apply denoising on smaller data
            denoised_small = denoise_nl_means(
                small_data,
                h=self.config.h_factor * sigma_est,
                patch_size=self.config.patch_size,
                patch_distance=self.config.patch_distance,
                fast_mode=self.config.fast_mode,
                channel_axis=None
            )
            
            # Upscale back to original size
            denoised = zoom(denoised_small, self.config.downscale_factor, order=1)
        else:
            # Process at full resolution
            sigma_est = np.mean(estimate_sigma(data, channel_axis=None))
            
            denoised = denoise_nl_means(
                data,
                h=self.config.h_factor * sigma_est,
                patch_size=self.config.patch_size,
                patch_distance=self.config.patch_distance,
                fast_mode=self.config.fast_mode,
                channel_axis=None
            )
        
        elapsed = time.time() - start_time
        self.logger.info(f"Denoising completed in {elapsed:.1f} seconds")
        return denoised

    def process(self, input_path: str, output_path: str) -> Dict:
        self.logger.info(f"Processing T1w: {os.path.basename(input_path)}")
        try:
            info = self._extract_subject_info(input_path)
            img = nib.load(input_path)
            data_orig = img.get_fdata()

            # Brain mask
            brain_mask = data_orig > (0.1 * np.max(data_orig))

            # Original metrics
            orig_snr = QualityMetrics.calculate_snr(data_orig, brain_mask)
            orig_std = np.std(data_orig[brain_mask]) if np.sum(brain_mask) > 0 else 0

            # Denoise
            data_denoised = self._denoise_data(data_orig)

            # Save
            denoised_img = nib.Nifti1Image(data_denoised, img.affine, img.header)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            nib.save(denoised_img, output_path)

            # New metrics
            denoised_snr = QualityMetrics.calculate_snr(data_denoised, brain_mask)
            denoised_std = np.std(data_denoised[brain_mask]) if np.sum(brain_mask) > 0 else 0
            snr_improvement = (denoised_snr - orig_snr) / orig_snr if orig_snr > 0 else 0
            noise_reduction = (orig_std - denoised_std) / orig_std if orig_std > 0 else 0

            metrics = {
                **info,
                "input_file": input_path,
                "output_file": output_path,
                "processing_step": "t1w_denoising",
                "timestamp": datetime.now().isoformat(),
                "original_snr": float(orig_snr),
                "denoised_snr": float(denoised_snr),
                "snr_improvement": float(snr_improvement),
                "snr_improvement_percent": float(snr_improvement * 100),
                "original_noise_std": float(orig_std),
                "denoised_noise_std": float(denoised_std),
                "noise_reduction": float(noise_reduction),
                "noise_reduction_percent": float(noise_reduction * 100),
                "correlation_with_original": float(pearsonr(data_orig.flatten(), data_denoised.flatten())[0]),
                "success": True,
                "error_message": ""
            }
            self.results.append(metrics)

            if snr_improvement >= self.config.min_snr_improvement:
                self.logger.info(f"✓ Denoising improved SNR by {snr_improvement*100:.1f}%")
            else:
                self.logger.warning(f"⚠ Denoising SNR improvement below threshold ({snr_improvement*100:.1f}%)")

            return metrics
        except Exception as e:
            metrics = {
                "input_file": input_path,
                "output_file": output_path,
                "processing_step": "t1w_denoising",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error_message": str(e)
            }
            self.results.append(metrics)
            self.logger.error(f"✗ Failed: {e}")
            return metrics

    def process_batch(self, input_dir: str, output_dir: str) -> List[Dict]:
        self.logger.info(f"Searching for T1w images in: {input_dir}")
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t1w_files = self._find_t1w_images(input_dir)
        if not t1w_files:
            self.logger.error("No T1w anatomical images found")
            return []
        
        self.logger.info(f"Found {len(t1w_files)} T1w anatomical images")
        
        # Prepare processing tasks
        tasks = []
        for f in t1w_files:
            info = self._extract_subject_info(str(f))
            group_dir = output_dir / info["group"]
            group_dir.mkdir(exist_ok=True)
            out_file = group_dir / f"{info['full_id']}_T1w_denoised.nii.gz"
            tasks.append((str(f), str(out_file)))
        
        # Determine number of workers
        max_workers = self.config.max_workers or min(multiprocessing.cpu_count(), len(tasks))
        self.logger.info(f"Processing with {max_workers} parallel workers")
        
        # Process images in parallel
        start_time = time.time()
        all_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process in chunks to avoid loading too many images at once
            for i in range(0, len(tasks), self.config.chunk_size):
                chunk = tasks[i:i+self.config.chunk_size]
                futures = {executor.submit(self._process_wrapper, input_path, output_path): 
                          (input_path, output_path) for input_path, output_path in chunk}
                
                for future in tqdm(as_completed(futures), total=len(futures),
                                  desc=f"Processing batch {i//self.config.chunk_size + 1}/{(len(tasks)-1)//self.config.chunk_size + 1}"):
                    input_path, output_path = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed on {os.path.basename(input_path)}: {str(e)}")
                        all_results.append({
                            "input_file": input_path,
                            "output_file": output_path,
                            "success": False,
                            "error_message": str(e)
                        })
        
        elapsed = time.time() - start_time
        self.results = all_results
        
        # Calculate success rate
        success_count = sum(1 for r in all_results if r.get('success', False))
        self.logger.info(f"Processed {len(all_results)} images in {elapsed:.1f} seconds "
                         f"({len(all_results)/elapsed:.2f} img/sec)")
        self.logger.info(f"Success rate: {success_count}/{len(all_results)} "
                         f"({100*success_count/len(all_results):.1f}%)")
        
        # Save metrics
        metrics_file = self.save_metrics(str(output_dir))
        if metrics_file:
            self.logger.info(f"Saved detailed metrics to: {metrics_file}")
        
        return all_results
    
    def _process_wrapper(self, input_path: str, output_path: str) -> Dict:
        """Wrapper for process to use with parallel execution"""
        return self.process(input_path, output_path)

    def save_metrics(self, output_dir: str):
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = Path(output_dir) / "t1w_denoising_metrics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved metrics: {csv_file}")
            return str(csv_file)
        return None


def main():
    parser = argparse.ArgumentParser(description="T1w Anatomical MRI Denoising")
    parser.add_argument("--input", "-i", required=True, help="Input directory or T1w image file")
    parser.add_argument("--output", "-o", default="./t1w_denoising_outputs", help="Output directory")
    parser.add_argument("--single", action="store_true", help="Process single T1w instead of batch")
    
    # Basic denoising parameters
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size for non-local means")
    parser.add_argument("--patch-distance", type=int, default=6, help="Patch distance for non-local means")
    parser.add_argument("--h", type=float, default=1.15, help="Filtering strength factor")
    
    # Optimization parameters
    parser.add_argument("--fast", action="store_true", default=True, 
                       help="Use fast mode for denoising (default: True)")
    parser.add_argument("--downscale", type=int, default=1, 
                       help="Downscale factor for faster processing (1=no downscaling, 2=half size)")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=1, 
                       help="Process images in chunks of this size (for memory efficiency)")

    args = parser.parse_args()

    config = DenoisingConfig(
        output_root=args.output,
        patch_size=args.patch_size,
        patch_distance=args.patch_distance,
        h_factor=args.h,
        fast_mode=args.fast,
        downscale_factor=args.downscale,
        max_workers=args.workers,
        chunk_size=args.chunk_size
    )

    print("\nT1w Anatomical MRI Denoising")
    print("============================")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Single image' if args.single else 'Batch processing'}")
    print(f"Optimization: {'Enabled' if args.fast else 'Disabled'} "
          f"(Downscale={args.downscale}x, Workers={args.workers or 'auto'}, Chunk size={args.chunk_size})")
    print("")

    processor = T1wDenoising(config)

    try:
        if args.single:
            info = processor._extract_subject_info(args.input)
            out_file = Path(args.output) / f"{info['full_id']}_T1w_denoised.nii.gz"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            start_time = time.time()
            result = processor.process(args.input, str(out_file))
            elapsed = time.time() - start_time
            processor.save_metrics(args.output)
            
            print(f"\nProcessing completed in {elapsed:.1f} seconds")
            print("✓ Done" if result["success"] else f"✗ Failed: {result['error_message']}")
        else:
            start_time = time.time()
            results = processor.process_batch(args.input, args.output)
            elapsed = time.time() - start_time
            
            success_count = sum(1 for r in results if r.get('success', False))
            print(f"\nBatch processing completed in {elapsed:.1f} seconds "
                  f"({len(results)/elapsed:.2f} images/sec)")
            print(f"Processed {len(results)} scans: {success_count} successful, "
                  f"{len(results)-success_count} failed")
            print(f"Metrics saved in {args.output}/t1w_denoising_metrics.csv")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
