#!/usr/bin/env python3
"""
YOPD Tissue Segmentation Quality Control
=======================================

This script analyzes tissue segmentation quality and provides
comprehensive metrics for Gray Matter, White Matter, and CSF segmentation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats
from scipy.stats import entropy
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TissueSegmentationConfig:
    """Configuration for tissue segmentation QC"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    # Literature-based normal tissue volume fractions
    normal_gm_fraction_range: Tuple[float, float] = (0.35, 0.65)
    normal_wm_fraction_range: Tuple[float, float] = (0.25, 0.50)
    normal_csf_fraction_range: Tuple[float, float] = (0.05, 0.25)
    normal_total_brain_volume_range: Tuple[int, int] = (1200, 1800)  # ml
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

class TissueSegmentationQC:
    """Quality control for tissue segmentation"""
    
    def __init__(self, config: TissueSegmentationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results_dir = Path(config.output_root) / "tissue_segmentation_qc"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('tissue_segmentation_qc')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = Path(self.config.output_root) / "logs" / "tissue_segmentation_qc.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def validate_tissue_volumes(self, gm_volume: int, wm_volume: int, csf_volume: int, 
                              voxel_size: Tuple[float, float, float]) -> Dict:
        """Validate tissue volumes against normative values"""
        try:
            total_volume_voxels = gm_volume + wm_volume + csf_volume
            voxel_volume_ml = np.prod(voxel_size) / 1000  # Convert mm³ to ml
            total_volume_ml = total_volume_voxels * voxel_volume_ml
            
            # Calculate tissue fractions
            gm_fraction = gm_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            wm_fraction = wm_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            csf_fraction = csf_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            
            # Validation flags
            validation_flags = {
                'total_volume_ml': total_volume_ml,
                'gm_volume_ml': gm_volume * voxel_volume_ml,
                'wm_volume_ml': wm_volume * voxel_volume_ml,
                'csf_volume_ml': csf_volume * voxel_volume_ml,
                'gm_fraction': gm_fraction,
                'wm_fraction': wm_fraction,
                'csf_fraction': csf_fraction,
                'volume_plausible': (
                    self.config.normal_total_brain_volume_range[0] <= 
                    total_volume_ml <= 
                    self.config.normal_total_brain_volume_range[1]
                ),
                'gm_fraction_normal': (
                    self.config.normal_gm_fraction_range[0] <= 
                    gm_fraction <= 
                    self.config.normal_gm_fraction_range[1]
                ),
                'wm_fraction_normal': (
                    self.config.normal_wm_fraction_range[0] <= 
                    wm_fraction <= 
                    self.config.normal_wm_fraction_range[1]
                ),
                'csf_fraction_normal': (
                    self.config.normal_csf_fraction_range[0] <= 
                    csf_fraction <= 
                    self.config.normal_csf_fraction_range[1]
                )
            }
            
            return validation_flags
            
        except Exception as e:
            self.logger.warning(f"Tissue volume validation failed: {e}")
            return {'error': str(e)}
    
    def calculate_segmentation_entropy(self, segmentation: np.ndarray) -> float:
        """Calculate entropy of tissue segmentation"""
        try:
            # Get unique tissue labels and their counts
            unique_labels, counts = np.unique(segmentation, return_counts=True)
            
            # Remove background (label 0)
            mask = unique_labels > 0
            if np.sum(mask) == 0:
                return 0.0
            
            counts = counts[mask]
            probabilities = counts / np.sum(counts)
            
            # Calculate Shannon entropy
            seg_entropy = entropy(probabilities, base=2)
            
            return seg_entropy
            
        except Exception as e:
            self.logger.warning(f"Segmentation entropy calculation failed: {e}")
            return np.nan
    
    def assess_tissue_boundary_smoothness(self, segmentation: np.ndarray) -> Dict:
        """Assess smoothness of tissue boundaries"""
        try:
            smoothness_metrics = {}
            
            # Define tissue labels (assuming 1=CSF, 2=GM, 3=WM)
            tissue_labels = {1: 'CSF', 2: 'GM', 3: 'WM'}
            
            for label, tissue_name in tissue_labels.items():
                tissue_mask = segmentation == label
                
                if np.sum(tissue_mask) > 1000:  # Only assess if sufficient tissue present
                    # Calculate boundary roughness using gradient magnitude
                    gradient_mag = np.sqrt(np.sum([
                        ndimage.gaussian_gradient_magnitude(tissue_mask.astype(float), sigma=1, axis=i)**2
                        for i in range(3)
                    ], axis=0))
                    
                    boundary_roughness = np.mean(gradient_mag[gradient_mag > 0])
                    smoothness_metrics[f'{tissue_name.lower()}_boundary_roughness'] = boundary_roughness
                else:
                    smoothness_metrics[f'{tissue_name.lower()}_boundary_roughness'] = np.nan
            
            return smoothness_metrics
            
        except Exception as e:
            self.logger.warning(f"Boundary smoothness assessment failed: {e}")
            return {'error': str(e)}
    
    def analyze_tissue_segmentation_quality(self, subject_id: str, 
                                          brain_image_path: str, segmentation_path: str) -> Dict:
        """Comprehensive analysis of tissue segmentation quality"""
        try:
            self.logger.info(f"Analyzing tissue segmentation for subject: {subject_id}")
            
            # Load images
            brain_img = nib.load(brain_image_path)
            brain_data = brain_img.get_fdata()
            
            seg_img = nib.load(segmentation_path)
            seg_data = seg_img.get_fdata().astype(int)
            
            voxel_size = brain_img.header.get_zooms()[:3]
            
            # Basic metrics
            metrics = {
                'subject_id': subject_id,
                'brain_image_path': brain_image_path,
                'segmentation_path': segmentation_path,
                'voxel_size': voxel_size,
            }
            
            # Calculate tissue volumes (assuming 1=CSF, 2=GM, 3=WM)
            csf_volume = np.sum(seg_data == 1)
            gm_volume = np.sum(seg_data == 2)
            wm_volume = np.sum(seg_data == 3)
            
            metrics.update({
                'csf_volume_voxels': csf_volume,
                'gm_volume_voxels': gm_volume,
                'wm_volume_voxels': wm_volume,
                'total_tissue_voxels': csf_volume + gm_volume + wm_volume
            })
            
            # Validate tissue volumes
            volume_validation = self.validate_tissue_volumes(gm_volume, wm_volume, csf_volume, voxel_size)
            metrics.update(volume_validation)
            
            # Segmentation entropy
            metrics['segmentation_entropy'] = self.calculate_segmentation_entropy(seg_data)
            
            # Tissue boundary smoothness
            boundary_metrics = self.assess_tissue_boundary_smoothness(seg_data)
            metrics.update(boundary_metrics)
            
            # Intensity statistics for each tissue type
            brain_mask = brain_data > 0
            for tissue_label, tissue_name in [(1, 'CSF'), (2, 'GM'), (3, 'WM')]:
                tissue_mask = (seg_data == tissue_label) & brain_mask
                
                if np.sum(tissue_mask) > 0:
                    tissue_intensities = brain_data[tissue_mask]
                    metrics.update({
                        f'{tissue_name.lower()}_mean_intensity': np.mean(tissue_intensities),
                        f'{tissue_name.lower()}_std_intensity': np.std(tissue_intensities),
                        f'{tissue_name.lower()}_median_intensity': np.median(tissue_intensities)
                    })
                else:
                    metrics.update({
                        f'{tissue_name.lower()}_mean_intensity': np.nan,
                        f'{tissue_name.lower()}_std_intensity': np.nan,
                        f'{tissue_name.lower()}_median_intensity': np.nan
                    })
            
            # Tissue contrast assessment
            if all(not np.isnan(metrics.get(f'{tissue}_mean_intensity', np.nan)) for tissue in ['gm', 'wm']):
                gm_intensity = metrics['gm_mean_intensity']
                wm_intensity = metrics['wm_mean_intensity']
                csf_intensity = metrics.get('csf_mean_intensity', 0)
                
                # Calculate tissue contrasts
                metrics['gm_wm_contrast'] = abs(gm_intensity - wm_intensity)
                metrics['gm_csf_contrast'] = abs(gm_intensity - csf_intensity) if not np.isnan(csf_intensity) else np.nan
                metrics['wm_csf_contrast'] = abs(wm_intensity - csf_intensity) if not np.isnan(csf_intensity) else np.nan
            
            # Overall segmentation quality score (0-10)
            score_components = []
            
            # Volume plausibility (0-3 points)
            volume_score = 0
            if metrics.get('volume_plausible', False):
                volume_score += 1
            if metrics.get('gm_fraction_normal', False):
                volume_score += 1
            if metrics.get('wm_fraction_normal', False):
                volume_score += 1
            score_components.append(volume_score)
            
            # Tissue contrast (0-4 points)
            if 'gm_wm_contrast' in metrics and not np.isnan(metrics['gm_wm_contrast']):
                # Normalize contrast to 0-4 scale (assuming good contrast > 50 intensity units)
                contrast_score = min(4, metrics['gm_wm_contrast'] / 50 * 4)
                score_components.append(contrast_score)
            
            # Segmentation entropy (0-3 points)
            if not np.isnan(metrics['segmentation_entropy']):
                # Good segmentation should have moderate entropy (around 1.5 for 3 tissues)
                entropy_score = min(3, (1 - abs(metrics['segmentation_entropy'] - 1.5) / 1.5) * 3)
                entropy_score = max(0, entropy_score)
                score_components.append(entropy_score)
            
            metrics['segmentation_quality_score'] = sum(score_components)
            metrics['segmentation_quality_good'] = metrics['segmentation_quality_score'] >= 6
            
            # Overall quality assessment
            metrics['all_fractions_normal'] = all([
                metrics.get('gm_fraction_normal', False),
                metrics.get('wm_fraction_normal', False),
                metrics.get('csf_fraction_normal', False)
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze tissue segmentation for {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'error': str(e),
                'analysis_failed': True
            }
    
    def find_segmentation_files(self) -> List[Tuple[str, str, str]]:
        """Find brain images and corresponding tissue segmentations"""
        self.logger.info("Searching for brain images and tissue segmentations...")
        
        pairs = []
        preproc_dir = Path(self.config.output_root) / "01_preprocessed"
        
        if not preproc_dir.exists():
            self.logger.error(f"Preprocessing directory not found: {preproc_dir}")
            return pairs
        
        for subject_dir in preproc_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                subject_id = subject_dir.name
                
                # Look for brain images and segmentations
                brain_patterns = ['*brain.nii.gz', '*T1w_brain.nii.gz', '*corrected.nii.gz']
                seg_patterns = ['*segmentation.nii.gz', '*tissues.nii.gz', '*seg.nii.gz']
                
                brain_file = None
                seg_file = None
                
                # Find brain image
                for pattern in brain_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        brain_file = str(files[0])
                        break
                
                # Find segmentation
                for pattern in seg_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        seg_file = str(files[0])
                        break
                
                if brain_file and seg_file:
                    pairs.append((subject_id, brain_file, seg_file))
                else:
                    self.logger.warning(f"Could not find segmentation pair for {subject_id}")
        
        self.logger.info(f"Found {len(pairs)} tissue segmentation pairs for analysis")
        return pairs
    
    def analyze_all_subjects(self) -> pd.DataFrame:
        """Analyze tissue segmentation quality for all subjects"""
        pairs = self.find_segmentation_files()
        
        if not pairs:
            self.logger.error("No tissue segmentation pairs found for analysis!")
            return pd.DataFrame()
        
        results = []
        for i, (subject_id, brain_path, seg_path) in enumerate(pairs, 1):
            self.logger.info(f"Processing {i}/{len(pairs)}: {subject_id}")
            
            metrics = self.analyze_tissue_segmentation_quality(subject_id, brain_path, seg_path)
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.results_dir / "tissue_segmentation_qc.csv"
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        
        return results_df
    
    def create_tissue_segmentation_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations for tissue segmentation QC"""
        if results_df.empty:
            self.logger.warning("No data available for visualization!")
            return
        
        self.logger.info("Creating tissue segmentation QC visualizations...")
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False).fillna(False)]
        
        if successful_df.empty:
            self.logger.warning("No successful analyses for visualization!")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle('Tissue Segmentation Quality Control Dashboard', fontsize=16, fontweight='bold')
        
        try:
            # Tissue volume fractions
            tissue_fractions = ['gm_fraction', 'wm_fraction', 'csf_fraction']
            tissue_names = ['Gray Matter', 'White Matter', 'CSF']
            
            for i, (fraction, name) in enumerate(zip(tissue_fractions, tissue_names)):
                if fraction in successful_df.columns:
                    fraction_data = successful_df[fraction].dropna()
                    if len(fraction_data) > 0:
                        axes[0, i].hist(fraction_data, bins=20, alpha=0.7, edgecolor='black')
                        
                        # Add normal range
                        if fraction == 'gm_fraction':
                            normal_range = self.config.normal_gm_fraction_range
                        elif fraction == 'wm_fraction':
                            normal_range = self.config.normal_wm_fraction_range
                        else:
                            normal_range = self.config.normal_csf_fraction_range
                        
                        axes[0, i].axvline(normal_range[0], color='green', linestyle='--', alpha=0.7)
                        axes[0, i].axvline(normal_range[1], color='green', linestyle='--', alpha=0.7)
                        axes[0, i].axvspan(normal_range[0], normal_range[1], alpha=0.2, color='green')
                        
                        axes[0, i].set_title(f'{name} Fraction Distribution')
                        axes[0, i].set_xlabel('Fraction')
                        axes[0, i].set_ylabel('Count')
            
            # Total brain volume
            if 'total_volume_ml' in successful_df.columns:
                volume_data = successful_df['total_volume_ml'].dropna()
                if len(volume_data) > 0:
                    axes[1, 0].hist(volume_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(self.config.normal_total_brain_volume_range[0], 
                                     color='green', linestyle='--', alpha=0.7)
                    axes[1, 0].axvline(self.config.normal_total_brain_volume_range[1], 
                                     color='green', linestyle='--', alpha=0.7)
                    axes[1, 0].axvspan(self.config.normal_total_brain_volume_range[0], 
                                     self.config.normal_total_brain_volume_range[1], 
                                     alpha=0.2, color='green')
                    axes[1, 0].set_title('Total Brain Volume Distribution')
                    axes[1, 0].set_xlabel('Volume (ml)')
                    axes[1, 0].set_ylabel('Count')
            
            # GM vs WM fraction scatter
            if all(col in successful_df.columns for col in ['gm_fraction', 'wm_fraction']):
                scatter_data = successful_df[['gm_fraction', 'wm_fraction']].dropna()
                if len(scatter_data) > 1:
                    axes[1, 1].scatter(scatter_data['gm_fraction'], scatter_data['wm_fraction'], alpha=0.6)
                    axes[1, 1].set_title('GM vs WM Fraction')
                    axes[1, 1].set_xlabel('GM Fraction')
                    axes[1, 1].set_ylabel('WM Fraction')
                    
                    # Add correlation coefficient
                    corr_coef = scatter_data['gm_fraction'].corr(scatter_data['wm_fraction'])
                    axes[1, 1].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[1, 1].transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Segmentation entropy
            if 'segmentation_entropy' in successful_df.columns:
                entropy_data = successful_df['segmentation_entropy'].dropna()
                if len(entropy_data) > 0:
                    axes[1, 2].hist(entropy_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 2].axvline(1.5, color='orange', linestyle='--', label='Ideal entropy (~1.5)')
                    axes[1, 2].set_title('Segmentation Entropy Distribution')
                    axes[1, 2].set_xlabel('Entropy (bits)')
                    axes[1, 2].set_ylabel('Count')
                    axes[1, 2].legend()
            
            # Quality flags summary
            quality_flags = ['gm_fraction_normal', 'wm_fraction_normal', 'csf_fraction_normal', 
                           'volume_plausible', 'segmentation_quality_good']
            available_flags = [col for col in quality_flags if col in successful_df.columns]
            
            if available_flags:
                flag_counts = successful_df[available_flags].sum()
                axes[2, 0].bar(range(len(flag_counts)), flag_counts.values)
                axes[2, 0].set_title('Quality Flags Summary')
                axes[2, 0].set_ylabel('Count Passing')
                axes[2, 0].set_xticks(range(len(flag_counts)))
                axes[2, 0].set_xticklabels([col.replace('_', ' ').title() for col in flag_counts.index], 
                                         rotation=45, ha='right')
            
            # Tissue intensity contrasts
            if 'gm_wm_contrast' in successful_df.columns:
                contrast_data = successful_df['gm_wm_contrast'].dropna()
                if len(contrast_data) > 0:
                    axes[2, 1].hist(contrast_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 1].set_title('GM-WM Contrast Distribution')
                    axes[2, 1].set_xlabel('Intensity Difference')
                    axes[2, 1].set_ylabel('Count')
            
            # Segmentation quality score
            if 'segmentation_quality_score' in successful_df.columns:
                score_data = successful_df['segmentation_quality_score'].dropna()
                if len(score_data) > 0:
                    axes[2, 2].hist(score_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 2].axvline(6, color='green', linestyle='--', label='Good threshold (6)')
                    axes[2, 2].set_title('Segmentation Quality Score')
                    axes[2, 2].set_xlabel('Score (0-10)')
                    axes[2, 2].set_ylabel('Count')
                    axes[2, 2].legend()
            
            # Tissue mean intensities comparison
            tissue_intensities = ['csf_mean_intensity', 'gm_mean_intensity', 'wm_mean_intensity']
            available_intensities = [col for col in tissue_intensities if col in successful_df.columns]
            
            if available_intensities:
                intensity_data = []
                labels = []
                for col in available_intensities:
                    data = successful_df[col].dropna()
                    if len(data) > 0:
                        intensity_data.append(data)
                        labels.append(col.replace('_mean_intensity', '').upper())
                
                if intensity_data:
                    axes[3, 0].boxplot(intensity_data, labels=labels)
                    axes[3, 0].set_title('Tissue Mean Intensities')
                    axes[3, 0].set_ylabel('Intensity')
            
            # Volume correlation matrix
            volume_cols = ['gm_volume_ml', 'wm_volume_ml', 'csf_volume_ml']
            available_volume_cols = [col for col in volume_cols if col in successful_df.columns]
            
            if len(available_volume_cols) >= 2:
                corr_matrix = successful_df[available_volume_cols].corr()
                im = axes[3, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[3, 1].set_title('Tissue Volume Correlations')
                axes[3, 1].set_xticks(range(len(available_volume_cols)))
                axes[3, 1].set_yticks(range(len(available_volume_cols)))
                axes[3, 1].set_xticklabels([col.replace('_volume_ml', '') for col in available_volume_cols])
                axes[3, 1].set_yticklabels([col.replace('_volume_ml', '') for col in available_volume_cols])
                
                # Add correlation values
                for i in range(len(available_volume_cols)):
                    for j in range(len(available_volume_cols)):
                        axes[3, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                       ha='center', va='center')
                
                plt.colorbar(im, ax=axes[3, 1])
            
            # Overall quality summary
            if 'segmentation_quality_good' in successful_df.columns:
                quality_counts = successful_df['segmentation_quality_good'].value_counts()
                colors = ['red' if not good else 'green' for good in quality_counts.index]
                axes[3, 2].bar(range(len(quality_counts)), quality_counts.values, color=colors, alpha=0.7)
                axes[3, 2].set_title('Overall Segmentation Quality')
                axes[3, 2].set_ylabel('Count')
                axes[3, 2].set_xticks(range(len(quality_counts)))
                axes[3, 2].set_xticklabels(['Poor', 'Good'])
            
            # Remove empty subplots
            for i in range(4):
                for j in range(3):
                    if not axes[i, j].has_data():
                        axes[i, j].text(0.5, 0.5, 'No Data Available', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
        
        except Exception as e:
            self.logger.warning(f"Some visualizations failed: {e}")
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / "tissue_segmentation_qc_dashboard.png"
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"QC dashboard saved to: {output_file}")
    
    def generate_tissue_segmentation_report(self, results_df: pd.DataFrame):
        """Generate comprehensive tissue segmentation QC report"""
        if results_df.empty:
            self.logger.warning("No data available for report generation!")
            return
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False).fillna(False)]
        
        if successful_df.empty:
            self.logger.warning("No successful analyses#!/usr/bin/env python3
"""
YOPD Tissue Segmentation Quality Control
=======================================

This script analyzes tissue segmentation quality and provides
comprehensive metrics for Gray Matter, White Matter, and CSF segmentation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats
from scipy.stats import entropy
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TissueSegmentationConfig:
    """Configuration for tissue segmentation QC"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    # Literature-based normal tissue volume fractions
    normal_gm_fraction_range: Tuple[float, float] = (0.35, 0.65)
    normal_wm_fraction_range: Tuple[float, float] = (0.25, 0.50)
    normal_csf_fraction_range: Tuple[float, float] = (0.05, 0.25)
    normal_total_brain_volume_range: Tuple[int, int] = (1200, 1800)  # ml
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

class TissueSegmentationQC:
    """Quality control for tissue segmentation"""
    
    def __init__(self, config: TissueSegmentationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results_dir = Path(config.output_root) / "tissue_segmentation_qc"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('tissue_segmentation_qc')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = Path(self.config.output_root) / "logs" / "tissue_segmentation_qc.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def validate_tissue_volumes(self, gm_volume: int, wm_volume: int, csf_volume: int, 
                              voxel_size: Tuple[float, float, float]) -> Dict:
        """Validate tissue volumes against normative values"""
        try:
            total_volume_voxels = gm_volume + wm_volume + csf_volume
            voxel_volume_ml = np.prod(voxel_size) / 1000  # Convert mm³ to ml
            total_volume_ml = total_volume_voxels * voxel_volume_ml
            
            # Calculate tissue fractions
            gm_fraction = gm_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            wm_fraction = wm_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            csf_fraction = csf_volume / total_volume_voxels if total_volume_voxels > 0 else 0
            
            # Validation flags
            validation_flags = {
                'total_volume_ml': total_volume_ml,
                'gm_volume_ml': gm_volume * voxel_volume_ml,
                'wm_volume_ml': wm_volume * voxel_volume_ml,
                'csf_volume_ml': csf_volume * voxel_volume_ml,
                'gm_fraction': gm_fraction,
                'wm_fraction': wm_fraction,
                'csf_fraction': csf_fraction,
                'volume_plausible': (
                    self.config.normal_total_brain_volume_range[0] <= 
                    total_volume_ml <= 
                    self.config.normal_total_brain_volume_range[1]
                ),
                'gm_fraction_normal': (
                    self.config.normal_gm_fraction_range[0] <= 
                    gm_fraction <= 
                    self.config.normal_gm_fraction_range[1]
                ),
                'wm_fraction_normal': (
                    self.config.normal_wm_fraction_range[0] <= 
                    wm_fraction <= 
                    self.config.normal_wm_fraction_range[1]
                ),
                'csf_fraction_normal': (
                    self.config.normal_csf_fraction_range[0] <= 
                    csf_fraction <= 
                    self.config.normal_csf_fraction_range[1]
                )
            }
            
            return validation_flags
            
        except Exception as e:
            self.logger.warning(f"Tissue volume validation failed: {e}")
            return {'error': str(e)}
    
    def calculate_segmentation_entropy(self, segmentation: np.ndarray) -> float:
        """Calculate entropy of tissue segmentation"""
        try:
            # Get unique tissue labels and their counts
            unique_labels, counts = np.unique(segmentation, return_counts=True)
            
            # Remove background (label 0)
            mask = unique_labels > 0
            if np.sum(mask) == 0:
                return 0.0
            
            counts = counts[mask]
            probabilities = counts / np.sum(counts)
            
            # Calculate Shannon entropy
            seg_entropy = entropy(probabilities, base=2)
            
            return seg_entropy
            
        except Exception as e:
            self.logger.warning(f"Segmentation entropy calculation failed: {e}")
            return np.nan
    
    def assess_tissue_boundary_smoothness(self, segmentation: np.ndarray) -> Dict:
        """Assess smoothness of tissue boundaries"""
        try:
            smoothness_metrics = {}
            
            # Define tissue labels (assuming 1=CSF, 2=GM, 3=WM)
            tissue_labels = {1: 'CSF', 2: 'GM', 3: 'WM'}
            
            for label, tissue_name in tissue_labels.items():
                tissue_mask = segmentation == label
                
                if np.sum(tissue_mask) > 1000:  # Only assess if sufficient tissue present
                    # Calculate boundary roughness using gradient magnitude
                    gradient_mag = np.sqrt(np.sum([
                        ndimage.gaussian_gradient_magnitude(tissue_mask.astype(float), sigma=1, axis=i)**2
                        for i in range(3)
                    ], axis=0))
                    
                    boundary_roughness = np.mean(gradient_mag[gradient_mag > 0])
                    smoothness_metrics[f'{tissue_name.lower()}_boundary_roughness'] = boundary_roughness
                else:
                    smoothness_metrics[f'{tissue_name.lower()}_boundary_roughness'] = np.nan
            
            return smoothness_metrics
            
        except Exception as e:
            self.logger.warning(f"Boundary smoothness assessment failed: {e}")
            return {'error': str(e)}
    
    def analyze_tissue_segmentation_quality(self, subject_id: str, 
                                          brain_image_path: str, segmentation_path: str) -> Dict:
        """Comprehensive analysis of tissue segmentation quality"""
        try:
            self.logger.info(f"Analyzing tissue segmentation for subject: {subject_id}")
            
            # Load images
            brain_img = nib.load(brain_image_path)
            brain_data = brain_img.get_fdata()
            
            seg_img = nib.load(segmentation_path)
            seg_data = seg_img.get_fdata().astype(int)
            
            voxel_size = brain_img.header.get_zooms()[:3]
            
            # Basic metrics
            metrics = {
                'subject_id': subject_id,
                'brain_image_path': brain_image_path,
                'segmentation_path': segmentation_path,
                'voxel_size': voxel_size,
            }
            
            # Calculate tissue volumes (assuming 1=CSF, 2=GM, 3=WM)
            csf_volume = np.sum(seg_data == 1)
            gm_volume = np.sum(seg_data == 2)
            wm_volume = np.sum(seg_data == 3)
            
            metrics.update({
                'csf_volume_voxels': csf_volume,
                'gm_volume_voxels': gm_volume,
                'wm_volume_voxels': wm_volume,
                'total_tissue_voxels': csf_volume + gm_volume + wm_volume
            })
            
            # Validate tissue volumes
            volume_validation = self.validate_tissue_volumes(gm_volume, wm_volume, csf_volume, voxel_size)
            metrics.update(volume_validation)
            
            # Segmentation entropy
            metrics['segmentation_entropy'] = self.calculate_segmentation_entropy(seg_data)
            
            # Tissue boundary smoothness
            boundary_metrics = self.assess_tissue_boundary_smoothness(seg_data)
            metrics.update(boundary_metrics)
            
            # Intensity statistics for each tissue type
            brain_mask = brain_data > 0
            for tissue_label, tissue_name in [(1, 'CSF'), (2, 'GM'), (3, 'WM')]:
                tissue_mask = (seg_data == tissue_label) & brain_mask
                
                if np.sum(tissue_mask) > 0:
                    tissue_intensities = brain_data[tissue_mask]
                    metrics.update({
                        f'{tissue_name.lower()}_mean_intensity': np.mean(tissue_intensities),
                        f'{tissue_name.lower()}_std_intensity': np.std(tissue_intensities),
                        f'{tissue_name.lower()}_median_intensity': np.median(tissue_intensities)
                    })
                else:
                    metrics.update({
                        f'{tissue_name.lower()}_mean_intensity': np.nan,
                        f'{tissue_name.lower()}_std_intensity': np.nan,
                        f'{tissue_name.lower()}_median_intensity': np.nan
                    })
            
            # Tissue contrast assessment
            if all(not np.isnan(metrics.get(f'{tissue}_mean_intensity', np.nan)) for tissue in ['gm', 'wm']):
                gm_intensity = metrics['gm_mean_intensity']
                wm_intensity = metrics['wm_mean_intensity']
                csf_intensity = metrics.get('csf_mean_intensity', 0)
                
                # Calculate tissue contrasts
                metrics['gm_wm_contrast'] = abs(gm_intensity - wm_intensity)
                metrics['gm_csf_contrast'] = abs(gm_intensity - csf_intensity) if not np.isnan(csf_intensity) else np.nan
                metrics['wm_csf_contrast'] = abs(wm_intensity - csf_intensity) if not np.isnan(csf_intensity) else np.nan
            
            # Overall segmentation quality score (0-10)
            score_components = []
            
            # Volume plausibility (0-3 points)
            volume_score = 0
            if metrics.get('volume_plausible', False):
                volume_score += 1
            if metrics.get('gm_fraction_normal', False):
                volume_score += 1
            if metrics.get('wm_fraction_normal', False):
                volume_score += 1
            score_components.append(volume_score)
            
            # Tissue contrast (0-4 points)
            if 'gm_wm_contrast' in metrics and not np.isnan(metrics['gm_wm_contrast']):
                # Normalize contrast to 0-4 scale (assuming good contrast > 50 intensity units)
                contrast_score = min(4, metrics['gm_wm_contrast'] / 50 * 4)
                score_components.append(contrast_score)
            
            # Segmentation entropy (0-3 points)
            if not np.isnan(metrics['segmentation_entropy']):
                # Good segmentation should have moderate entropy (around 1.5 for 3 tissues)
                entropy_score = min(3, (1 - abs(metrics['segmentation_entropy'] - 1.5) / 1.5) * 3)
                entropy_score = max(0, entropy_score)
                score_components.append(entropy_score)
            
            metrics['segmentation_quality_score'] = sum(score_components)
            metrics['segmentation_quality_good'] = metrics['segmentation_quality_score'] >= 6
            
            # Overall quality assessment
            metrics['all_fractions_normal'] = all([
                metrics.get('gm_fraction_normal', False),
                metrics.get('wm_fraction_normal', False),
                metrics.get('csf_fraction_normal', False)
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze tissue segmentation for {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'error': str(e),
                'analysis_failed': True
            }
    
    def find_segmentation_files(self) -> List[Tuple[str, str, str]]:
        """Find brain images and corresponding tissue segmentations"""
        self.logger.info("Searching for brain images and tissue segmentations...")
        
        pairs = []
        preproc_dir = Path(self.config.output_root) / "01_preprocessed"
        
        if not preproc_dir.exists():
            self.logger.error(f"Preprocessing directory not found: {preproc_dir}")
            return pairs
        
        for subject_dir in preproc_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                subject_id = subject_dir.name
                
                # Look for brain images and segmentations
                brain_patterns = ['*brain.nii.gz', '*T1w_brain.nii.gz', '*corrected.nii.gz']
                seg_patterns = ['*segmentation.nii.gz', '*tissues.nii.gz', '*seg.nii.gz']
                
                brain_file = None
                seg_file = None
                
                # Find brain image
                for pattern in brain_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        brain_file = str(files[0])
                        break
                
                # Find segmentation
                for pattern in seg_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        seg_file = str(files[0])
                        break
                
                if brain_file and seg_file:
                    pairs.append((subject_id, brain_file, seg_file))
                else:
                    self.logger.warning(f"Could not find segmentation pair for {subject_id}")
        
        self.logger.info(f"Found {len(pairs)} tissue segmentation pairs for analysis")
        return pairs
    
    def analyze_all_subjects(self) -> pd.DataFrame:
        """Analyze tissue segmentation quality for all subjects"""
        pairs = self.find_segmentation_files()
        
        if not pairs:
            self.logger.error("No tissue segmentation pairs found for analysis!")
            return pd.DataFrame()
        
        results = []
        for i, (subject_id, brain_path, seg_path) in enumerate(pairs, 1):
            self.logger.info(f"Processing {i}/{len(pairs)}: {subject_id}")
            
            metrics = self.analyze_tissue_segmentation_quality(subject_id, brain_path, seg_path)
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.results_dir / "tissue_segmentation_qc.csv"
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        
        return results_df
    
    def create_tissue_segmentation_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations for tissue segmentation QC"""
        if results_df.empty:
            self.logger.warning("No data available for visualization!")
            return
        
        self.logger.info("Creating tissue segmentation QC visualizations...")
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False).fillna(False)]
        
        if successful_df.empty:
            self.logger.warning("No successful analyses for visualization!")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle('Tissue Segmentation Quality Control Dashboard', fontsize=16, fontweight='bold')
        
        try:
            # Tissue volume fractions
            tissue_fractions = ['gm_fraction', 'wm_fraction', 'csf_fraction']
            tissue_names = ['Gray Matter', 'White Matter', 'CSF']
            
            for i, (fraction, name) in enumerate(zip(tissue_fractions, tissue_names)):
                if fraction in successful_df.columns:
                    fraction_data = successful_df[fraction].dropna()
                    if len(fraction_data) > 0:
                        axes[0, i].hist(fraction_data, bins=20, alpha=0.7, edgecolor='black')
                        
                        # Add normal range
                        if fraction == 'gm_fraction':
                            normal_range = self.config.normal_gm_fraction_range
                        elif fraction == 'wm_fraction':
                            normal_range = self.config.normal_wm_fraction_range
                        else:
                            normal_range = self.config.normal_csf_fraction_range
                        
                        axes[0, i].axvline(normal_range[0], color='green', linestyle='--', alpha=0.7)
                        axes[0, i].axvline(normal_range[1], color='green', linestyle='--', alpha=0.7)
                        axes[0, i].axvspan(normal_range[0], normal_range[1], alpha=0.2, color='green')
                        
                        axes[0, i].set_title(f'{name} Fraction Distribution')
                        axes[0, i].set_xlabel('Fraction')
                        axes[0, i].set_ylabel('Count')
            
            # Total brain volume
            if 'total_volume_ml' in successful_df.columns:
                volume_data = successful_df['total_volume_ml'].dropna()
                if len(volume_data) > 0:
                    axes[1, 0].hist(volume_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(self.config.normal_total_brain_volume_range[0], 
                                     color='green', linestyle='--', alpha=0.7)
                    axes[1, 0].axvline(self.config.normal_total_brain_volume_range[1], 
                                     color='green', linestyle='--', alpha=0.7)
                    axes[1, 0].axvspan(self.config.normal_total_brain_volume_range[0], 
                                     self.config.normal_total_brain_volume_range[1], 
                                     alpha=0.2, color='green')
                    axes[1, 0].set_title('Total Brain Volume Distribution')
                    axes[1, 0].set_xlabel('Volume (ml)')
                    axes[1, 0].set_ylabel('Count')
            
            # GM vs WM fraction scatter
            if all(col in successful_df.columns for col in ['gm_fraction', 'wm_fraction']):
                scatter_data = successful_df[['gm_fraction', 'wm_fraction']].dropna()
                if len(scatter_data) > 1:
                    axes[1, 1].scatter(scatter_data['gm_fraction'], scatter_data['wm_fraction'], alpha=0.6)
                    axes[1, 1].set_title('GM vs WM Fraction')
                    axes[1, 1].set_xlabel('GM Fraction')
                    axes[1, 1].set_ylabel('WM Fraction')
                    
                    # Add correlation coefficient
                    corr_coef = scatter_data['gm_fraction'].corr(scatter_data['wm_fraction'])
                    axes[1, 1].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[1, 1].transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Segmentation entropy
            if 'segmentation_entropy' in successful_df.columns:
                entropy_data = successful_df['segmentation_entropy'].dropna()
                if len(entropy_data) > 0:
                    axes[1, 2].hist(entropy_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 2].axvline(1.5, color='orange', linestyle='--', label='Ideal entropy (~1.5)')
                    axes[1, 2].set_title('Segmentation Entropy Distribution')
                    axes[1, 2].set_xlabel('Entropy (bits)')
                    axes[1, 2].set_ylabel('Count')
                    axes[1, 2].legend()
            
            # Quality flags summary
            quality_flags = ['gm_fraction_normal', 'wm_fraction_normal', 'csf_fraction_normal', 
                           'volume_plausible', 'segmentation_quality_good']
            available_flags = [col for col in quality_flags if col in successful_df.columns]
            
            if available_flags:
                flag_counts = successful_df[available_flags].sum()
                axes[2, 0].bar(range(len(flag_counts)), flag_counts.values)
                axes[2, 0].set_title('Quality Flags Summary')
                axes[2, 0].set_ylabel('Count Passing')
                axes[2, 0].set_xticks(range(len(flag_counts)))
                axes[2, 0].set_xticklabels([col.replace('_', ' ').title() for col in flag_counts.index], 
                                         rotation=45, ha='right')
            
            # Tissue intensity contrasts
            if 'gm_wm_contrast' in successful_df.columns:
                contrast_data = successful_df['gm_wm_contrast'].dropna()
                if len(contrast_data) > 0:
                    axes[2, 1].hist(contrast_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 1].set_title('GM-WM Contrast Distribution')
                    axes[2, 1].set_xlabel('Intensity Difference')
                    axes[2, 1].set_ylabel('Count')
            
            # Segmentation quality score
            if 'segmentation_quality_score' in successful_df.columns:
                score_data = successful_df['segmentation_quality_score'].dropna()
                if len(score_data) > 0:
                    axes[2, 2].hist(score_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 2].axvline(6, color='green', linestyle='--', label='Good threshold (6)')
                    axes[2, 2].set_title('Segmentation Quality Score')
                    axes[2, 2].set_xlabel('Score (0-10)')
                    axes[2, 2].set_ylabel('Count')
                    axes[2, 2].legend()
            
            # Tissue mean intensities comparison
            tissue_intensities = ['csf_mean_intensity', 'gm_mean_intensity', 'wm_mean_intensity']
            available_intensities = [col for col in tissue_intensities if col in successful_df.columns]
            
            if available_intensities:
                intensity_data = []
                labels = []
                for col in available_intensities:
                    data = successful_df[col].dropna()
                    if len(data) > 0:
                        intensity_data.append(data)
                        labels.append(col.replace('_mean_intensity', '').upper())
                
                if intensity_data:
                    axes[3, 0].boxplot(intensity_data, labels=labels)
                    axes[3, 0].set_title('Tissue Mean Intensities')
                    axes[3, 0].set_ylabel('Intensity')
            
            # Volume correlation matrix
            volume_cols = ['gm_volume_ml', 'wm_volume_ml', 'csf_volume_ml']
            available_volume_cols = [col for col in volume_cols if col in successful_df.columns]
            
            if len(available_volume_cols) >= 2:
                corr_matrix = successful_df[available_volume_cols].corr()
                im = axes[3, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[3, 1].set_title('Tissue Volume Correlations')
                axes[3, 1].set_xticks(range(len(available_volume_cols)))
                axes[3, 1].set_yticks(range(len(available_volume_cols)))
                axes[3, 1].set_xticklabels([col.replace('_volume_ml', '') for col in available_volume_cols])
                axes[3, 1].set_yticklabels([col.replace('_volume_ml', '') for col in available_volume_cols])
                
                # Add correlation values
                for i in range(len(available_volume_cols)):
                    for j in range(len(available_volume_cols)):
                        axes[3, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                       ha='center', va='center')
                
                plt.colorbar(im, ax=axes[3, 1])
            
            # Overall quality summary
            if 'segmentation_quality_good' in successful_df.columns:
                quality_counts = successful_df['segmentation_quality_good'].value_counts()
                colors = ['red' if not good else 'green' for good in quality_counts.index]
                axes[3, 2].bar(range(len(quality_counts)), quality_counts.values, color=colors, alpha=0.7)
                axes[3, 2].set_title('Overall Segmentation Quality')
                axes[3, 2].set_ylabel('Count')
                axes[3, 2].set_xticks(range(len(quality_counts)))
                axes[3, 2].set_xticklabels(['Poor', 'Good'])
            
            # Remove empty subplots
            for i in range(4):
                for j in range(3):
                    if
                    
# ...existing code...

    def generate_tissue_segmentation_report(self, results_df: pd.DataFrame):
        """Generate comprehensive tissue segmentation QC report"""
        if results_df.empty:
            self.logger.warning("No data available for report generation!")
            return
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False).fillna(False)]
        
        if successful_df.empty:
            self.logger.warning("No successful analyses for reporting!")
            return
        
        # Calculate summary statistics
        summary_stats = {}
        numeric_columns = successful_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if successful_df[col].notna().sum() > 0:
                summary_stats[col] = {
                    'mean': successful_df[col].mean(),
                    'std': successful_df[col].std(),
                    'median': successful_df[col].median(),
                    'min': successful_df[col].min(),
                    'max': successful_df[col].max(),
                    'count': successful_df[col].count()
                }
        
        # Generate text report
        report_file = self.results_dir / "tissue_segmentation_qc_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("YOPD TISSUE SEGMENTATION QUALITY CONTROL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Subjects Analyzed: {len(successful_df)}\n")
            if len(results_df) > len(successful_df):
                f.write(f"Failed Analyses: {len(results_df) - len(successful_df)}\n")
            f.write("\n")
            
            f.write("TISSUE VOLUME SUMMARY\n")
            f.write("-" * 21 + "\n")
            
            # Tissue fraction metrics
            tissue_metrics = {
                'gm_fraction': 'Gray Matter Fraction',
                'wm_fraction': 'White Matter Fraction', 
                'csf_fraction': 'CSF Fraction',
                'total_volume_ml': 'Total Brain Volume (ml)',
                'segmentation_entropy': 'Segmentation Entropy',
                'segmentation_quality_score': 'Quality Score (0-10)'
            }
            
            for metric_key, metric_name in tissue_metrics.items():
                if metric_key in summary_stats:
                    stats = summary_stats[metric_key]
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
            
            # Normative ranges assessment
            f.write("NORMATIVE RANGES ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            
            for tissue, range_attr in [('gm', 'normal_gm_fraction_range'), 
                                     ('wm', 'normal_wm_fraction_range'),
                                     ('csf', 'normal_csf_fraction_range')]:
                fraction_col = f'{tissue}_fraction'
                if fraction_col in summary_stats:
                    mean_fraction = summary_stats[fraction_col]['mean']
                    normal_range = getattr(self.config, range_attr)
                    
                    if normal_range[0] <= mean_fraction <= normal_range[1]:
                        f.write(f"✓ {tissue.upper()} fraction within normal range "
                               f"(mean={mean_fraction:.3f}, normal={normal_range[0]:.2f}-{normal_range[1]:.2f})\n")
                    else:
                        f.write(f"⚠ {tissue.upper()} fraction outside normal range "
                               f"(mean={mean_fraction:.3f}, normal={normal_range[0]:.2f}-{normal_range[1]:.2f})\n")
            
            # Total brain volume assessment
            if 'total_volume_ml' in summary_stats:
                mean_volume = summary_stats['total_volume_ml']['mean']
                normal_volume_range = self.config.normal_total_brain_volume_range
                
                if normal_volume_range[0] <= mean_volume <= normal_volume_range[1]:
                    f.write(f"✓ Total brain volume within normal range "
                           f"(mean={mean_volume:.0f}ml, normal={normal_volume_range[0]}-{normal_volume_range[1]}ml)\n")
                else:
                    f.write(f"⚠ Total brain volume outside normal range "
                           f"(mean={mean_volume:.0f}ml, normal={normal_volume_range[0]}-{normal_volume_range[1]}ml)\n")
            
            f.write("\n")
            
            # Quality flags summary
            quality_flags = ['gm_fraction_normal', 'wm_fraction_normal', 'csf_fraction_normal', 
                           'volume_plausible', 'segmentation_quality_good', 'all_fractions_normal']
            
            f.write("QUALITY FLAGS SUMMARY\n")
            f.write("-" * 25 + "\n")
            
            for flag in quality_flags:
                if flag in successful_df.columns:
                    count = successful_df[flag].sum()
                    total = len(successful_df)
                    percentage = count/total*100 if total > 0 else 0
                    f.write(f"{flag.replace('_', ' ').title()}: {count}/{total} ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Tissue contrast assessment
            if 'gm_wm_contrast' in summary_stats:
                mean_contrast = summary_stats['gm_wm_contrast']['mean']
                f.write("TISSUE CONTRAST ASSESSMENT\n")
                f.write("-" * 26 + "\n")
                f.write(f"GM-WM Contrast: {mean_contrast:.2f} intensity units\n")
                
                if mean_contrast > 50:
                    f.write("✓ Good tissue contrast for segmentation\n")
                elif mean_contrast > 25:
                    f.write("⚠ Moderate tissue contrast - acceptable but could be improved\n")
                else:
                    f.write("⚠ Poor tissue contrast - may affect segmentation quality\n")
                f.write("\n")
            
            # Overall quality assessment
            f.write("OVERALL QUALITY ASSESSMENT\n")
            f.write("-" * 27 + "\n")
            
            if 'segmentation_quality_score' in summary_stats:
                mean_score = summary_stats['segmentation_quality_score']['mean']
                if mean_score >= 7:
                    f.write(f"✓ High quality tissue segmentation (score={mean_score:.1f}/10)\n")
                elif mean_score >= 5:
                    f.write(f"⚠ Moderate quality tissue segmentation (score={mean_score:.1f}/10)\n")
                else:
                    f.write(f"⚠ Poor quality tissue segmentation (score={mean_score:.1f}/10)\n")
            
            # Segmentation entropy assessment
            if 'segmentation_entropy' in summary_stats:
                mean_entropy = summary_stats['segmentation_entropy']['mean']
                f.write(f"Segmentation Entropy: {mean_entropy:.3f} bits\n")
                
                if 1.2 <= mean_entropy <= 1.8:
                    f.write("✓ Optimal entropy for 3-tissue segmentation\n")
                else:
                    f.write("⚠ Suboptimal entropy - may indicate segmentation issues\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Check for systematic issues
            if 'all_fractions_normal' in successful_df.columns:
                normal_fraction_rate = successful_df['all_fractions_normal'].mean()
                if normal_fraction_rate < 0.7:  # Less than 70% normal
                    f.write("• Low rate of normal tissue fractions detected:\n")
                    f.write("  - Review segmentation algorithm parameters\n")
                    f.write("  - Check preprocessing quality (bias correction, brain extraction)\n")
                    f.write("  - Consider manual quality control\n")
                    f.write("  - Validate against expert annotations if available\n\n")
            
            if 'gm_wm_contrast' in summary_stats and summary_stats['gm_wm_contrast']['mean'] < 30:
                f.write("• Poor tissue contrast detected:\n")
                f.write("  - Review image acquisition parameters\n")
                f.write("  - Check bias field correction effectiveness\n")
                f.write("  - Consider different segmentation approach\n")
                f.write("  - Evaluate image quality before segmentation\n\n")
            
            if 'segmentation_quality_score' in summary_stats and summary_stats['segmentation_quality_score']['mean'] < 6:
                f.write("• Overall poor segmentation quality:\n")
                f.write("  - Review entire preprocessing pipeline\n")
                f.write("  - Consider subject-specific parameter tuning\n")
                f.write("  - Evaluate alternative segmentation methods\n")
                f.write("  - Implement manual correction workflow\n\n")
            
            # Volume-specific recommendations
            for tissue in ['gm', 'wm', 'csf']:
                fraction_col = f'{tissue}_fraction'
                normal_col = f'{tissue}_fraction_normal'
                
                if fraction_col in summary_stats and normal_col in successful_df.columns:
                    normal_rate = successful_df[normal_col].mean()
                    if normal_rate < 0.8:  # Less than 80% normal
                        f.write(f"• {tissue.upper()} fraction frequently abnormal:\n")
                        f.write(f"  - Review {tissue.upper()} segmentation specifically\n")
                        f.write(f"  - Check for systematic bias in {tissue.upper()} classification\n")
                        f.write(f"  - Consider {tissue.upper()}-specific preprocessing\n\n")
            
            f.write("TECHNICAL NOTES\n")
            f.write("-" * 15 + "\n")
            f.write("• Tissue fractions calculated relative to total tissue volume\n")
            f.write("• Normative ranges based on literature values for healthy adults\n")
            f.write("• Segmentation entropy measures tissue distribution balance\n")
            f.write("• Quality score combines volume plausibility, contrast, and entropy\n")
            f.write("• Boundary smoothness assessed using gradient magnitude\n")
            f.write("• All metrics should be interpreted in context of study population\n")
        
        # Export JSON summary
        json_stats = {}
        for key, value in summary_stats.items():
            json_stats[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for k, v in value.items()}
        
        # Add configuration parameters to JSON
        json_stats['analysis_config'] = {
            'normal_gm_fraction_range': self.config.normal_gm_fraction_range,
            'normal_wm_fraction_range': self.config.normal_wm_fraction_range,
            'normal_csf_fraction_range': self.config.normal_csf_fraction_range,
            'normal_total_brain_volume_range': self.config.normal_total_brain_volume_range
        }
        
        json_file = self.results_dir / "tissue_segmentation_qc_summary.json"
        with open(json_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_file}")
        self.logger.info(f"JSON summary saved to: {json_file}")

def main():
    """Main function to run tissue segmentation QC"""
    print("YOPD Tissue Segmentation Quality Control")
    print("=" * 45)
    
    # Initialize configuration
    config = TissueSegmentationConfig()
    
    # Verify output directory exists
    if not Path(config.output_root).exists():
        print(f"Error: Output directory not found: {config.output_root}")
        print("Please ensure the YOPD pipeline has been run first.")
        return
    
    # Initialize QC analyzer
    qc_analyzer = TissueSegmentationQC(config)
    
    try:
        # Analyze all subjects
        print("Starting tissue segmentation quality control...")
        results_df = qc_analyzer.analyze_all_subjects()
        
        if results_df.empty:
            print("No tissue segmentation pairs found for analysis!")
            return
        
        print(f"Successfully analyzed {len(results_df)} subjects")
        
        # Create visualizations
        print("Creating visualizations...")
        qc_analyzer.create_tissue_segmentation_visualizations(results_df)
        
        # Generate report
        print("Generating comprehensive report...")
        qc_analyzer.generate_tissue_segmentation_report(results_df)
        
        print("\nTissue Segmentation QC Complete!")
        print(f"Results saved to: {qc_analyzer.results_dir}")
        print("\nGenerated files:")
        print("  - tissue_segmentation_qc.csv: Detailed metrics")
        print("  - tissue_segmentation_qc_dashboard.png: Visualization")
        print("  - tissue_segmentation_qc_report.txt: Summary report")
        print("  - tissue_segmentation_qc_summary.json: JSON summary")
        
    except Exception as e:
        print(f"Error during tissue segmentation QC: {e}")
        qc_analyzer.logger.error(f"Tissue segmentation QC failed: {e}")
        raise

if __name__ == "__main__":
    main()