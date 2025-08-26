#!/usr/bin/env python3
"""
Tissue Segmentation Module
=========================

This module handles tissue segmentation using DIPY.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import warnings
from scipy.ndimage import label
warnings.filterwarnings('ignore')

@dataclass
class TissueSegmentationConfig:
    """Configuration for tissue segmentation"""
    output_root: str = "./tissue_segmentation_outputs"
    median_radius: int = 2  # Median filter radius
    num_pass: int = 4       # Number of median filter passes
    threshold: float = 0.5  # Threshold for brain/non-brain segmentation

class TissueSegmentation:
    """Tissue Segmentation using DIPY"""
    
    def __init__(self, config: TissueSegmentationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('tissue_segmentation')
    
    def process(self, input_path: str, output_prefix: str, **kwargs) -> Dict:
        """Perform tissue segmentation using DIPY's median_otsu method"""
        self.logger.info(f"Segmenting tissues: {os.path.basename(input_path)}")
        
        try:
            # Load image data
            data, affine = load_nifti(input_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_prefix)
            os.makedirs(output_dir, exist_ok=True)
            
            # Perform segmentation
            self.logger.info("Running DIPY tissue segmentation...")
            
            # Get the brain mask using median_otsu
            _, brain_mask = median_otsu(data, 
                                      median_radius=self.config.median_radius, 
                                      numpass=self.config.num_pass, 
                                      autocrop=False)
            
            # Simple threshold-based tissue segmentation
            # Normalize the data to range 0-1 inside the brain mask
            brain_data = data * brain_mask
            brain_min = np.min(brain_data[brain_data > 0])
            brain_max = np.max(brain_data)
            norm_data = (brain_data - brain_min) / (brain_max - brain_min) * brain_mask
            
            # Create segmentation: 0=background, 1=CSF, 2=GM, 3=WM
            # Thresholds based on typical T1 intensities (CSF is dark, GM is intermediate, WM is bright)
            csf_mask = (norm_data > 0) & (norm_data <= 0.33)
            gm_mask = (norm_data > 0.33) & (norm_data <= 0.66) 
            wm_mask = norm_data > 0.66
            
            # Combine into a single segmentation
            seg_data = np.zeros_like(data, dtype=np.int16)
            seg_data[csf_mask] = 1  # CSF
            seg_data[gm_mask] = 2   # GM
            seg_data[wm_mask] = 3   # WM
            
            # Save segmentation output files
            seg_file = f"{output_prefix}_seg.nii.gz"
            
            # Create and save segmentation image
            seg_img = nib.Nifti1Image(seg_data, affine)
            nib.save(seg_img, seg_file)
            
            # Create probability maps (simplified - just binary maps)
            csf_prob = f"{output_prefix}_pve_0.nii.gz"
            gm_prob = f"{output_prefix}_pve_1.nii.gz"
            wm_prob = f"{output_prefix}_pve_2.nii.gz"
            
            # Save probability maps
            csf_img = nib.Nifti1Image(csf_mask.astype(np.float32), affine)
            gm_img = nib.Nifti1Image(gm_mask.astype(np.float32), affine)
            wm_img = nib.Nifti1Image(wm_mask.astype(np.float32), affine)
            
            nib.save(csf_img, csf_prob)
            nib.save(gm_img, gm_prob)
            nib.save(wm_img, wm_prob)
            
            metrics = {
                'subject_id': self._extract_subject_id(input_path),
                'input_file': input_path,
                'segmentation_file': seg_file,
                'csf_probability_map': csf_prob,
                'gm_probability_map': gm_prob,
                'wm_probability_map': wm_prob,
                'processing_step': 'tissue_segmentation',
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'error_message': ''
            }
            
            # Calculate tissue volumes
            try:
                # Get voxel dimensions
                img = nib.load(input_path)
                voxel_volume = np.prod(img.header.get_zooms()[:3]) / 1000  # ml per voxel
                
                # Count voxels for each tissue class
                csf_volume = np.sum(seg_data == 1) * voxel_volume  # CSF = 1
                gm_volume = np.sum(seg_data == 2) * voxel_volume   # GM = 2
                wm_volume = np.sum(seg_data == 3) * voxel_volume   # WM = 3
                total_brain_volume = csf_volume + gm_volume + wm_volume
                
                metrics.update({
                    'csf_volume_ml': float(csf_volume),
                    'gm_volume_ml': float(gm_volume),
                    'wm_volume_ml': float(wm_volume),
                    'total_brain_volume_ml': float(total_brain_volume),
                    'csf_percent': float(csf_volume / total_brain_volume * 100) if total_brain_volume > 0 else 0,
                    'gm_percent': float(gm_volume / total_brain_volume * 100) if total_brain_volume > 0 else 0,
                    'wm_percent': float(wm_volume / total_brain_volume * 100) if total_brain_volume > 0 else 0,
                    'gm_wm_ratio': float(gm_volume / wm_volume) if wm_volume > 0 else 0
                })
                
                self.logger.info(f"  Tissue volumes: CSF={csf_volume:.1f}ml, GM={gm_volume:.1f}ml, WM={wm_volume:.1f}ml")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate tissue volumes: {e}")
                metrics.update({
                    'volume_calculation_error': str(e)
                })
            
            self.results.append(metrics)
            self.logger.info(f"✓ Tissue segmentation completed successfully")
            return metrics
            
        except Exception as e:
            error_metrics = {
                'subject_id': self._extract_subject_id(input_path),
                'input_file': input_path,
                'processing_step': 'tissue_segmentation',
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error_message': str(e)
            }
            self.results.append(error_metrics)
            self.logger.error(f"✗ Tissue segmentation failed: {e}")
            return error_metrics
    
    def process_batch(self, input_directory: str, output_directory: str) -> List[Dict]:
        """Process batch of images"""
        self.logger.info(f"Starting batch tissue segmentation: {input_directory}")
        
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
            
            # Create subject-specific output prefix
            subject_id = self._extract_subject_id(str(image_file))
            output_prefix = str(output_dir / f"{subject_id}_tissues")
            
            try:
                self.process(str(image_file), output_prefix)
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
        
        # Save results
        self.save_metrics(str(output_dir))
        return self.results
    
    def save_metrics(self, output_dir: str):
        """Save metrics to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = Path(output_dir) / "tissue_segmentation_metrics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved metrics to {csv_file}")
            return str(csv_file)
        return None
    
    def _extract_subject_id(self, file_path: str) -> str:
        return Path(file_path).stem.split('_')[0]

def main():
    """Main function for tissue segmentation"""
    parser = argparse.ArgumentParser(description='MRI Tissue Segmentation using DIPY')
    parser.add_argument('--input', '-i', required=True, help='Input directory or single image file')
    parser.add_argument('--output', '-o', default='./tissue_segmentation_outputs', help='Output directory')
    parser.add_argument('--single', action='store_true', help='Process single image instead of batch')
    parser.add_argument('--median-radius', type=int, default=2, help='Median filter radius (default: 2)')
    parser.add_argument('--num-pass', type=int, default=4, help='Number of median filter passes (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for segmentation (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TissueSegmentationConfig(
        output_root=args.output,
        median_radius=args.median_radius,
        num_pass=args.num_pass,
        threshold=args.threshold
    )
    
    # Initialize processor
    processor = TissueSegmentation(config)
    
    print("MRI Tissue Segmentation using DIPY")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Single image' if args.single else 'Batch processing'}")
    print(f"Settings: median radius={args.median_radius}, num pass={args.num_pass}, threshold={args.threshold}")
    print()
    
    try:
        if args.single:
            # Process single image
            subject_id = processor._extract_subject_id(args.input)
            output_prefix = str(Path(args.output) / f"{subject_id}_tissues")
            Path(args.output).mkdir(parents=True, exist_ok=True)
            result = processor.process(args.input, output_prefix)
            processor.save_metrics(args.output)
            
            if result['success']:
                print(f"✓ Successfully processed: {result['input_file']}")
                if 'total_brain_volume_ml' in result:
                    print(f"  Total brain volume: {result['total_brain_volume_ml']:.1f} ml")
                    print(f"  GM: {result['gm_volume_ml']:.1f} ml ({result['gm_percent']:.1f}%)")
                    print(f"  WM: {result['wm_volume_ml']:.1f} ml ({result['wm_percent']:.1f}%)")
                    print(f"  CSF: {result['csf_volume_ml']:.1f} ml ({result['csf_percent']:.1f}%)")
            else:
                print(f"✗ Processing failed: {result['error_message']}")
        else:
            # Process batch
            results = processor.process_batch(args.input, args.output)
            successful = sum(1 for r in results if r['success'])
            print(f"\nBatch processing completed:")
            print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")
        
        print(f"\nResults saved to: {Path(args.output).absolute()}")
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())