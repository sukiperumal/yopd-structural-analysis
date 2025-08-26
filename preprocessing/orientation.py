#!/usr/bin/env python3
"""
Orientation Correction Module using nibabel
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from nilearn import plotting
warnings.filterwarnings('ignore')

@dataclass
class OrientationConfig:
    """Configuration for orientation correction"""
    output_root: str = "./orientation_outputs"
    target_orientation: str = "RAS"  # Standard neurological orientation
    generate_qc_images: bool = True  # Generate quality control images
    qc_image_dir: str = "qc_images"  # Subdirectory for QC images
    pd_regions_check: bool = True    # Check PD-specific regions (basal ganglia, substantia nigra)
    pd_groups: List[str] = None      # PD-specific groups (set to None to auto-detect)
    
class OrientationCorrection:    
    def __init__(self, config: OrientationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = []
        self.target_ornt = axcodes2ornt(self.config.target_orientation)
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('orientation_correction')
    
    def _get_orientation_info(self, img: nib.Nifti1Image) -> Dict:
        """Get orientation information from image"""
        orig_ornt = io_orientation(img.affine)
        orig_labels = nib.orientations.ornt2axcodes(orig_ornt)
        
        return {
            'orientation_matrix': orig_ornt,
            'orientation_labels': ''.join(orig_labels),
            'voxel_sizes': img.header.get_zooms()[:3],
            'shape': img.shape[:3]
        }
    
    def _reorient_image(self, img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict]:
        """Reorient image to target orientation using nibabel (optimized)"""
        orig_ornt = io_orientation(img.affine)
        orig_labels = nib.orientations.ornt2axcodes(orig_ornt)
        
        # Quick orientation check
        if ''.join(orig_labels) == self.config.target_orientation:
            return img, {'reorientation_needed': False}
        
        # Apply reorientation
        transform = ornt_transform(orig_ornt, self.target_ornt)
        reoriented_img = img.as_reoriented(transform)
        
        return reoriented_img, {'reorientation_needed': True}
    
    def _calculate_similarity_metrics(self, data_orig: np.ndarray, data_reoriented: np.ndarray) -> Dict:
        """Calculate basic similarity metrics (optimized)"""
        # Use subset for large images to speed up correlation
        if data_orig.size > 1000000:  # 1M voxels
            # Sample every 8th voxel for correlation
            orig_sample = data_orig.flatten()[::8]
            reor_sample = data_reoriented.flatten()[::8]
        else:
            orig_sample = data_orig.flatten()
            reor_sample = data_reoriented.flatten()
        
        # Quick finite check on sample
        valid_mask = np.isfinite(orig_sample) & np.isfinite(reor_sample)
        if not np.any(valid_mask):
            return {'data_correlation': 0.0, 'data_range_match': True}
        
        orig_valid = orig_sample[valid_mask]
        reor_valid = reor_sample[valid_mask]
        
        # Fast correlation calculation
        correlation = np.corrcoef(orig_valid, reor_valid)[0,1] if len(orig_valid) > 1 else 1.0
        
        return {
            'data_correlation': float(correlation) if not np.isnan(correlation) else 1.0,
            'data_range_match': np.allclose(np.percentile(orig_valid, [5, 95]), 
                                          np.percentile(reor_valid, [5, 95]), rtol=0.01)
        }
        
    def _generate_qc_image(self, img: nib.Nifti1Image, output_path: str, subject_info: Dict, 
                          is_pd_subject: bool = False) -> Optional[str]:
        """
        Generate quality control images showing mid-axial, sagittal and coronal slices
        With focus on Parkinson's disease relevant regions
        """
        if not self.config.generate_qc_images:
            return None
            
        try:
            # Create QC output directory
            qc_dir = Path(self.config.output_root) / self.config.qc_image_dir
            qc_dir.mkdir(exist_ok=True, parents=True)
            
            # Determine subject group and create folder
            group = subject_info.get('group', 'unknown')
            group_dir = qc_dir / group
            group_dir.mkdir(exist_ok=True)
            
            # Create a figure with 3 views: axial, coronal, sagittal
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Get image dimensions
            data = img.get_fdata()
            x, y, z = data.shape
            
            # Key slices for Parkinson's Disease neuroimaging
            # For PD subjects, focus on basal ganglia and substantia nigra regions
            # These coordinates are approximate and based on standard brain
            if is_pd_subject:
                # For PD subjects, we want to focus on basal ganglia and substantia nigra
                axial_slice = int(z * 0.55)      # Slightly above mid-axial to show basal ganglia
                coronal_slice = int(y * 0.45)    # Anterior portion to show caudate and putamen
                sagittal_slice = int(x * 0.5)    # Mid-sagittal
            else:
                # For controls, standard central slices
                axial_slice = int(z * 0.5)
                coronal_slice = int(y * 0.5)
                sagittal_slice = int(x * 0.5)
            
            # Plot the slices
            axes[0].imshow(np.rot90(data[:, :, axial_slice]), cmap='gray')
            axes[0].set_title(f'Axial (z={axial_slice})')
            axes[0].axis('off')
            
            axes[1].imshow(np.rot90(data[:, coronal_slice, :]), cmap='gray')
            axes[1].set_title(f'Coronal (y={coronal_slice})')
            axes[1].axis('off')
            
            axes[2].imshow(np.rot90(data[sagittal_slice, :, :]), cmap='gray')
            axes[2].set_title(f'Sagittal (x={sagittal_slice})')
            axes[2].axis('off')
            
            # Set a title for the whole figure
            fig.suptitle(f"{subject_info.get('subject_id')} - {subject_info.get('group')} - Orientation: {nib.orientations.ornt2axcodes(io_orientation(img.affine))}")
            
            # Save the figure
            qc_filename = f"{subject_info.get('subject_id')}_orientation_qc.png"
            qc_path = group_dir / qc_filename
            plt.savefig(str(qc_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return str(qc_path)
        except Exception as e:
            self.logger.warning(f"Could not generate QC image: {e}")
            return None
            
    def _check_pd_specific_regions(self, img: nib.Nifti1Image) -> Dict:
        """
        Check orientation of PD-specific regions of interest
        This helps validate that structures important for PD analysis are properly oriented
        """
        if not self.config.pd_regions_check:
            return {}
            
        try:
            # Simple check to ensure key coordinates are within expected intensity ranges
            # This is a basic check to see if bright/dark regions are where we expect them
            data = img.get_fdata()
            x, y, z = data.shape
            
            # Coordinates for key structures (normalized by image dimensions)
            # These are approximate locations in RAS orientation
            
            # Left and right putamen areas (checking symmetry)
            left_putamen = data[int(x*0.35), int(y*0.55), int(z*0.55)]
            right_putamen = data[int(x*0.65), int(y*0.55), int(z*0.55)]
            
            # Check intensity values in substantia nigra region
            substantia_nigra = data[int(x*0.5), int(y*0.35), int(z*0.4)]
            
            # Check for left-right symmetry (key for proper orientation)
            # If the image is incorrectly flipped, this ratio would deviate from 1.0
            lr_symmetry = left_putamen / right_putamen if right_putamen != 0 else 0
            
            # Calculate intensity statistics for normalization
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            # Return normalized measures
            # This helps detect if regions have expected intensities for their anatomical location
            return {
                'pd_left_putamen_intensity': float(left_putamen / data_mean) if data_mean != 0 else 0,
                'pd_right_putamen_intensity': float(right_putamen / data_mean) if data_mean != 0 else 0,
                'pd_substantia_nigra_intensity': float(substantia_nigra / data_mean) if data_mean != 0 else 0,
                'pd_lr_symmetry': float(lr_symmetry) if not np.isnan(lr_symmetry) else 0,
                'pd_regions_check_passed': abs(lr_symmetry - 1.0) < 0.5 if not np.isnan(lr_symmetry) else False
            }
        except Exception as e:
            self.logger.warning(f"Could not check PD-specific regions: {e}")
            return {
                'pd_regions_check_passed': False,
                'pd_regions_check_error': str(e)
            }
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict:
        self.logger.info(f"Processing T1w: {os.path.basename(input_path)}")
        
        try:
            # Extract subject information
            subject_info = self._extract_subject_info(input_path)
            
            # Determine if this is a PD subject
            is_pd_subject = 'PIGD' in subject_info['group'] or 'TDPD' in subject_info['group']
            
            # Load original image (header only first for speed)
            img_orig = nib.load(input_path)
            
            # Get original orientation info (fast - no data loading)
            orig_ornt = io_orientation(img_orig.affine)
            orig_labels = nib.orientations.ornt2axcodes(orig_ornt)
            orig_orientation = ''.join(orig_labels)
            
            # Quick check if reorientation is needed
            if orig_orientation == self.config.target_orientation:
                # No reorientation needed - just copy file
                import shutil
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(input_path, output_path)
                
                # For PD subjects or if QC is enabled, we generate QC images and check PD regions
                pd_metrics = {}
                qc_path = None
                
                # Only load data if we need to perform PD-specific checks or generate QC images
                if (self.config.pd_regions_check and is_pd_subject) or self.config.generate_qc_images:
                    # Generate QC image showing key slices
                    qc_path = self._generate_qc_image(img_orig, output_path, subject_info, is_pd_subject)
                    
                    # For PD subjects, check PD-specific regions
                    if self.config.pd_regions_check and is_pd_subject:
                        pd_metrics = self._check_pd_specific_regions(img_orig)
                
                metrics = {
                    **subject_info,
                    'input_file': input_path,
                    'output_file': output_path,
                    'processing_step': 'orientation_correction',
                    'timestamp': datetime.now().isoformat(),
                    'original_orientation': orig_orientation,
                    'final_orientation': orig_orientation,
                    'reorientation_needed': False,
                    'data_correlation': 1.0,
                    'scan_type': 'T1w',
                    'is_pd_subject': is_pd_subject,
                    'qc_image_path': qc_path,
                    **pd_metrics,
                    'success': True,
                    'error_message': ''
                }
                
                self.results.append(metrics)
                self.logger.info(f"✓ Already correct orientation ({orig_orientation})")
                return metrics
            
            # Reorientation needed - load data and process
            data_orig = img_orig.get_fdata()
            
            # Calculate transformation
            transform = ornt_transform(orig_ornt, self.target_ornt)
            
            # Apply reorientation
            img_reoriented = img_orig.as_reoriented(transform)
            
            # Save reoriented image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(img_reoriented, output_path)
            
            # Quick similarity check (load data only once)
            data_reoriented = img_reoriented.get_fdata()
            similarity_metrics = self._calculate_similarity_metrics(data_orig, data_reoriented)
            
            # Get final orientation
            final_ornt = io_orientation(img_reoriented.affine)
            final_labels = nib.orientations.ornt2axcodes(final_ornt)
            final_orientation = ''.join(final_labels)
            
            # For PD subjects, check PD-specific regions and generate QC image
            pd_metrics = {}
            qc_path = None
            
            if (self.config.pd_regions_check and is_pd_subject) or self.config.generate_qc_images:
                # Generate QC image showing key slices (after reorientation)
                qc_path = self._generate_qc_image(img_reoriented, output_path, subject_info, is_pd_subject)
                
                # For PD subjects, check PD-specific regions
                if self.config.pd_regions_check and is_pd_subject:
                    pd_metrics = self._check_pd_specific_regions(img_reoriented)
            
            # Compile metrics
            metrics = {
                **subject_info,
                'input_file': input_path,
                'output_file': output_path,
                'processing_step': 'orientation_correction',
                'timestamp': datetime.now().isoformat(),
                'original_orientation': orig_orientation,
                'final_orientation': final_orientation,
                'reorientation_needed': True,
                'original_shape': str(data_orig.shape),
                'reoriented_shape': str(data_reoriented.shape),
                'scan_type': 'T1w',
                'is_pd_subject': is_pd_subject,
                'qc_image_path': qc_path,
                **similarity_metrics,
                **pd_metrics,
                'success': True,
                'error_message': ''
            }
            
            self.results.append(metrics)
            self.logger.info(f"✓ Reoriented {orig_orientation} → {final_orientation}")
            return metrics
            
        except Exception as e:
            subject_info = self._extract_subject_info(input_path)
            error_metrics = {
                **subject_info,
                'input_file': input_path,
                'output_file': output_path,
                'processing_step': 'orientation_correction',
                'timestamp': datetime.now().isoformat(),
                'scan_type': 'T1w',
                'success': False,
                'error_message': str(e)
            }
            self.results.append(error_metrics)
            self.logger.error(f"✗ Failed: {e}")
            return error_metrics
    
    def process_batch(self, input_directory: str, output_directory: str) -> List[Dict]:
        self.logger.info(f"Searching for T1w images in: {input_directory}")
        
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all T1w anatomical images
        t1w_files = self._find_t1w_images(input_dir)
        
        if not t1w_files:
            self.logger.error("No T1w anatomical images found in anat directories")
            self.logger.info("Expected structure: */anat/*T1w.nii.gz")
            return []
        
        self.logger.info(f"Processing {len(t1w_files)} T1w images")
        
        # Process with progress updates
        total = len(t1w_files)
        progress_step = max(1, total // 10)
        
        for i, t1w_file in enumerate(t1w_files, 1):
            if i % progress_step == 0 or i == total:
                percent = (i / total) * 100
                self.logger.info(f"Progress: {i}/{total} ({percent:.0f}%)")
            
            # Extract subject info for output naming
            subject_info = self._extract_subject_info(str(t1w_file))
            
            # Create structured output path
            group_dir = output_dir / subject_info['group']
            group_dir.mkdir(exist_ok=True)
            
            output_file = group_dir / f"{subject_info['full_id']}_T1w_oriented.nii.gz"
            
            try:
                self.process(str(t1w_file), str(output_file))
            except Exception as e:
                self.logger.error(f"Failed {t1w_file.name}: {e}")
        
        # Save results with enhanced summary
        self._save_t1w_metrics(str(output_dir))
        return self.results
    
    def _save_t1w_metrics(self, output_dir: str):
        """Save metrics with T1w-specific summary and PD-specific metrics"""
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results)
        csv_file = Path(output_dir) / "T1w_orientation_metrics.csv"
        df.to_csv(csv_file, index=False)
        
        # Enhanced summary stats for T1w data
        successful = sum(1 for r in self.results if r.get('success', False))
        reoriented = sum(1 for r in self.results if r.get('reorientation_needed', False) and r.get('success', False))
        
        print(f"\nT1w PROCESSING SUMMARY")
        print("=" * 30)
        print(f"Total T1w scans: {len(self.results)}")
        print(f"Successfully processed: {successful}")
        print(f"Required reorientation: {reoriented}")
        print(f"Already correct: {successful - reoriented}")
        
        # Group-wise summary
        if successful > 0:
            groups = {}
            orientations = {}
            pd_metrics = {}
            
            for r in self.results:
                if r.get('success'):
                    group = r.get('group', 'unknown')
                    groups[group] = groups.get(group, 0) + 1
                    
                    orig_orient = r.get('original_orientation', 'unknown')
                    orientations[orig_orient] = orientations.get(orig_orient, 0) + 1
                    
                    # Track PD-specific metrics
                    if r.get('is_pd_subject', False):
                        pd_check = r.get('pd_regions_check_passed', None)
                        if pd_check is not None:
                            pd_key = 'pd_regions_passed' if pd_check else 'pd_regions_failed'
                            pd_metrics[pd_key] = pd_metrics.get(pd_key, 0) + 1
            
            if len(groups) > 1:
                print(f"\nBy group:")
                for group, count in sorted(groups.items()):
                    print(f"  {group}: {count} subjects")
            
            if len(orientations) > 1:
                print(f"\nOriginal orientations:")
                for orient, count in sorted(orientations.items()):
                    print(f"  {orient}: {count} scans")
                    
            # Print PD-specific metrics if available
            if pd_metrics:
                print(f"\nParkinson's Disease Specific Metrics:")
                passed = pd_metrics.get('pd_regions_passed', 0)
                failed = pd_metrics.get('pd_regions_failed', 0)
                total_pd_checks = passed + failed
                if total_pd_checks > 0:
                    pass_rate = (passed / total_pd_checks) * 100
                    print(f"  PD Region Checks: {passed}/{total_pd_checks} passed ({pass_rate:.1f}%)")
                    
                # If QC images were generated, show their location
                if any('qc_image_path' in r for r in self.results):
                    qc_dir = Path(output_dir) / self.config.qc_image_dir
                    print(f"  QC Images: {qc_dir}")
        
        print(f"\nDetailed metrics: {csv_file}")
        return str(csv_file)
    
    def save_metrics(self, output_dir: str):
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results)
        csv_file = Path(output_dir) / "orientation_metrics.csv"
        df.to_csv(csv_file, index=False)
        
        # Quick summary stats
        successful = sum(1 for r in self.results if r.get('success', False))
        reoriented = sum(1 for r in self.results if r.get('reorientation_needed', False) and r.get('success', False))
        
        print(f"\nProcessed: {len(self.results)} | Success: {successful} | Reoriented: {reoriented}")
        print(f"Metrics: {csv_file}")
        return str(csv_file)
    
    def _find_t1w_images(self, root_dir: Path) -> List[Path]:
        """Find all T1-weighted anatomical images in BIDS-like structure"""
        t1w_files = []
        
        # Search pattern for T1w images in anat directories
        # Supports both BIDS naming and variations
        patterns = [
            '**/anat/*T1w.nii.gz',
            '**/anat/*T1w.nii',
            '**/anat/*t1w.nii.gz', 
            '**/anat/*t1w.nii',
            '**/anat/*T1.nii.gz',
            '**/anat/*T1.nii'
        ]
        
        for pattern in patterns:
            t1w_files.extend(list(root_dir.glob(pattern)))
        
        # Remove duplicates and sort
        t1w_files = sorted(list(set(t1w_files)))
        
        # Log found structure
        if t1w_files:
            self.logger.info(f"Found {len(t1w_files)} T1w images")
            
            # Show directory structure summary
            subjects = {}
            for f in t1w_files:
                # Extract subject info from path
                path_parts = f.parts
                subject_part = None
                for part in path_parts:
                    if part.startswith('sub-'):
                        subject_part = part
                        break
                
                if subject_part:
                    # Get group (parent directory of subject)
                    try:
                        sub_idx = path_parts.index(subject_part)
                        if sub_idx > 0:
                            group = path_parts[sub_idx - 1]
                            if group not in subjects:
                                subjects[group] = []
                            subjects[group].append(subject_part)
                    except ValueError:
                        pass
            
            # Log structure summary
            for group, subs in subjects.items():
                self.logger.info(f"  {group}: {len(subs)} subjects")
                
        return t1w_files
    
    def _extract_subject_info(self, file_path: str) -> Dict[str, str]:
        """Extract subject, session, and group info from BIDS-like path"""
        path_parts = Path(file_path).parts
        
        # Extract subject ID
        subject_id = "unknown"
        session_id = "01"  # default
        group = "unknown"
        
        for part in path_parts:
            if part.startswith('sub-'):
                subject_id = part
            elif part.startswith('ses-'):
                session_id = part.replace('ses-', '')
        
        # Extract group (directory containing subject)
        try:
            sub_idx = next(i for i, part in enumerate(path_parts) if part.startswith('sub-'))
            if sub_idx > 0:
                group = path_parts[sub_idx - 1]
        except (StopIteration, IndexError):
            pass
        
        return {
            'subject_id': subject_id,
            'session_id': session_id,
            'group': group,
            'full_id': f"{subject_id}_ses-{session_id}" if session_id != "01" else subject_id
        }

def main():
    """Main function for orientation correction"""
    parser = argparse.ArgumentParser(
        description='T1w Anatomical MRI Orientation Correction using nibabel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, 
                       help='Input directory (will search for T1w in anat folders) or single T1w image file')
    parser.add_argument('--output', '-o', default='./t1w_orientation_outputs', 
                       help='Output directory')
    parser.add_argument('--single', action='store_true', 
                       help='Process single T1w image instead of batch')
    parser.add_argument('--orientation', default='RAS', 
                       choices=['RAS', 'LAS', 'RPS', 'LPS', 'RAI', 'LAI', 'RPI', 'LPI'],
                       help='Target orientation (default: RAS - neurological standard)')
    parser.add_argument('--generate-qc', action='store_true', 
                       help='Generate quality control images for each processed scan')
    parser.add_argument('--pd-regions-check', action='store_true', 
                       help='Check PD-specific regions (basal ganglia, substantia nigra)')
    parser.add_argument('--pd-groups', nargs='+', default=['PIGD', 'TDPD'],
                       help='Specify Parkinson\'s disease groups (default: PIGD TDPD)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OrientationConfig(
        output_root=args.output,
        target_orientation=args.orientation,
        generate_qc_images=args.generate_qc,
        pd_regions_check=args.pd_regions_check,
        pd_groups=args.pd_groups
    )
    
    # Initialize processor
    processor = OrientationCorrection(config)
    
    print("T1w Anatomical MRI Orientation Correction")
    print("=" * 45)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target orientation: {args.orientation}")
    print(f"Mode: {'Single T1w image' if args.single else 'Batch T1w processing'}")
    print(f"Scan type: T1-weighted anatomical only")
    
    # Add PD-specific options to header
    if args.generate_qc:
        print(f"QC Images: Enabled")
    if args.pd_regions_check:
        print(f"PD Regions Check: Enabled")
        print(f"PD Groups: {', '.join(args.pd_groups)}")
    print()
    
    try:
        if args.single:
            # Process single T1w image
            if not os.path.exists(args.input):
                print(f"✗ Input file not found: {args.input}")
                return 1
            
            # Validate it's a T1w image
            if 'T1w' not in args.input and 'T1' not in args.input:
                print(f"Warning: File doesn't appear to be T1w: {args.input}")
            
            subject_info = processor._extract_subject_info(args.input)
            output_file = Path(args.output) / f"{subject_info['full_id']}_T1w_oriented.nii.gz"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            
            result = processor.process(args.input, str(output_file))
            processor._save_t1w_metrics(args.output)
            
            if result['success']:
                print(f"✓ Successfully processed T1w: {result['output_file']}")
                if result.get('reorientation_needed'):
                    print(f"  Reoriented from {result.get('original_orientation', 'unknown')} to {result.get('final_orientation', 'unknown')}")
                else:
                    print(f"  Image was already in correct orientation ({result.get('original_orientation', 'unknown')})")
                
                # Display PD-specific metrics
                if args.pd_regions_check and result.get('is_pd_subject', False):
                    if result.get('pd_regions_check_passed', False):
                        print(f"  PD regions check: PASSED")
                    else:
                        print(f"  PD regions check: FAILED")
                        
                # Display QC image path
                if args.generate_qc and result.get('qc_image_path'):
                    print(f"  QC Image: {result.get('qc_image_path')}")
            else:
                print(f"✗ Processing failed: {result['error_message']}")
        else:
            # Process batch of T1w images
            if not os.path.exists(args.input):
                print(f"✗ Input directory not found: {args.input}")
                return 1
                
            results = processor.process_batch(args.input, args.output)
            if not results:
                print("✗ No T1w images found to process")
                print("Expected directory structure:")
                print("  <input_dir>/HC/sub-*/ses-*/anat/*T1w.nii.gz")
                print("  <input_dir>/PIGD/sub-*/ses-*/anat/*T1w.nii.gz") 
                print("  <input_dir>/TDPD/sub-*/ses-*/anat/*T1w.nii.gz")
                return 1
        
        print(f"\nResults saved to: {Path(args.output).absolute()}")
        
    except KeyboardInterrupt:
        print("\n✗ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())