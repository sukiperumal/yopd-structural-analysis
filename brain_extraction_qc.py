#!/usr/bin/env python3
"""
YOPD Brain Extraction Quality Control
====================================

This script analyzes brain extraction (skull stripping) effectiveness
and provides comprehensive quality control metrics.
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
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BrainExtractionConfig:
    """Configuration for brain extraction analysis"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    min_solidity: float = 0.85  # Minimum solidity for good brain boundary
    min_brain_volume: int = 800000  # Minimum brain volume in voxels
    max_brain_volume: int = 2000000  # Maximum brain volume in voxels
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

class BrainExtractionAnalyzer:
    """Dedicated class for brain extraction quality analysis"""
    
    def __init__(self, config: BrainExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results_dir = Path(config.output_root) / "brain_extraction_qc"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('brain_extraction_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = Path(self.config.output_root) / "logs" / "brain_extraction_qc.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def calculate_brain_boundary_smoothness(self, brain_mask: np.ndarray) -> float:
        """Calculate brain boundary smoothness using solidity measure"""
        try:
            # Work with the largest connected component
            labeled_mask = ndimage.label(brain_mask)[0]
            props = regionprops(labeled_mask)
            
            if not props:
                return 0.0
            
            # Get the largest region
            largest_region = max(props, key=lambda x: x.area)
            
            # Calculate solidity for each slice and average
            solidity_scores = []
            
            for slice_idx in range(brain_mask.shape[2]):
                slice_mask = largest_region.image[:, :, slice_idx] if len(brain_mask.shape) == 3 else largest_region.image
                
                if np.sum(slice_mask) > 100:  # Only process slices with sufficient brain tissue
                    convex_area = np.sum(convex_hull_image(slice_mask))
                    actual_area = np.sum(slice_mask)
                    
                    if convex_area > 0:
                        solidity = actual_area / convex_area
                        solidity_scores.append(solidity)
            
            return np.mean(solidity_scores) if solidity_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Solidity calculation failed: {e}")
            return np.nan
    
    def detect_residual_skull(self, original_data: np.ndarray, brain_extracted_data: np.ndarray) -> Dict:
        """Detect residual skull tissue in brain-extracted image"""
        try:
            # Create masks
            brain_threshold = 0.1 * np.max(brain_extracted_data)
            brain_mask = brain_extracted_data > brain_threshold
            
            orig_threshold = 0.05 * np.max(original_data)
            orig_mask = original_data > orig_threshold
            
            # Estimate skull region (in original but not in extracted brain)
            skull_region = orig_mask & (~brain_mask)
            
            # Check for intensity in skull region within brain-extracted image
            residual_intensity = np.mean(brain_extracted_data[skull_region]) if np.sum(skull_region) > 0 else 0
            
            # Calculate metrics
            skull_volume = np.sum(skull_region)
            brain_volume = np.sum(brain_mask)
            
            metrics = {
                'residual_skull_intensity': residual_intensity,
                'estimated_skull_volume': skull_volume,
                'brain_volume': brain_volume,
                'skull_to_brain_ratio': skull_volume / brain_volume if brain_volume > 0 else 0,
                'residual_skull_detected': residual_intensity > (0.01 * np.max(brain_extracted_data))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Residual skull detection failed: {e}")
            return {'error': str(e)}
    
    def analyze_brain_extraction_quality(self, subject_id: str, 
                                       original_path: str, brain_extracted_path: str) -> Dict:
        """Comprehensive analysis of brain extraction quality"""
        try:
            self.logger.info(f"Analyzing brain extraction for subject: {subject_id}")
            
            # Load images
            orig_img = nib.load(original_path)
            orig_data = orig_img.get_fdata()
            
            brain_img = nib.load(brain_extracted_path)
            brain_data = brain_img.get_fdata()
            
            # Create brain mask
            brain_threshold = 0.1 * np.max(brain_data)
            brain_mask = brain_data > brain_threshold
            
            # Basic metrics
            metrics = {
                'subject_id': subject_id,
                'original_path': original_path,
                'brain_extracted_path': brain_extracted_path,
                'brain_volume_voxels': np.sum(brain_mask),
                'voxel_size': brain_img.header.get_zooms()[:3],
            }
            
            # Calculate brain volume in ml
            voxel_volume = np.prod(metrics['voxel_size'])
            metrics['brain_volume_ml'] = metrics['brain_volume_voxels'] * voxel_volume / 1000
            
            # Brain boundary smoothness
            metrics['boundary_smoothness'] = self.calculate_brain_boundary_smoothness(brain_mask)
            
            # Residual skull detection
            skull_metrics = self.detect_residual_skull(orig_data, brain_data)
            metrics.update(skull_metrics)
            
            # Intensity statistics within brain
            brain_intensities = brain_data[brain_mask]
            if len(brain_intensities) > 0:
                metrics.update({
                    'brain_mean_intensity': np.mean(brain_intensities),
                    'brain_std_intensity': np.std(brain_intensities),
                    'brain_median_intensity': np.median(brain_intensities),
                    'brain_intensity_range': np.max(brain_intensities) - np.min(brain_intensities)
                })
            
            # Quality flags
            metrics['volume_plausible'] = (
                self.config.min_brain_volume <= metrics['brain_volume_voxels'] <= self.config.max_brain_volume
            )
            metrics['boundary_smooth'] = metrics['boundary_smoothness'] >= self.config.min_solidity
            metrics['minimal_residual_skull'] = not metrics.get('residual_skull_detected', True)
            
            # Overall extraction quality score (0-10)
            score_components = []
            
            # Volume plausibility (0-3 points)
            if metrics['volume_plausible']:
                score_components.append(3)
            else:
                # Partial credit based on how close to acceptable range
                vol = metrics['brain_volume_voxels']
                if vol < self.config.min_brain_volume:
                    score_components.append(max(0, 3 * vol / self.config.min_brain_volume))
                else:  # vol > max_brain_volume
                    excess_ratio = (vol - self.config.max_brain_volume) / self.config.max_brain_volume
                    score_components.append(max(0, 3 * (1 - excess_ratio)))
            
            # Boundary smoothness (0-4 points)
            if not np.isnan(metrics['boundary_smoothness']):
                smoothness_score = min(4, metrics['boundary_smoothness'] / self.config.min_solidity * 4)
                score_components.append(smoothness_score)
            
            # Residual skull (0-3 points)
            if metrics['minimal_residual_skull']:
                score_components.append(3)
            else:
                # Partial credit based on residual intensity
                residual_ratio = metrics.get('residual_skull_intensity', 0) / np.max(brain_data) if np.max(brain_data) > 0 else 0
                score_components.append(max(0, 3 * (1 - residual_ratio * 10)))  # Penalize high residual intensity
            
            metrics['extraction_quality_score'] = sum(score_components)
            metrics['extraction_quality_good'] = metrics['extraction_quality_score'] >= 7
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze brain extraction for {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'error': str(e),
                'analysis_failed': True
            }
    
    def find_brain_extraction_pairs(self) -> List[Tuple[str, str, str]]:
        """Find original and brain-extracted image pairs"""
        self.logger.info("Searching for original and brain-extracted image pairs...")
        
        pairs = []
        preproc_dir = Path(self.config.output_root) / "01_preprocessed"
        
        if not preproc_dir.exists():
            self.logger.error(f"Preprocessing directory not found: {preproc_dir}")
            return pairs
        
        for subject_dir in preproc_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                subject_id = subject_dir.name
                
                # Look for original and brain-extracted files
                original_patterns = ['*T1w.nii.gz', '*T1w_corrected.nii.gz']
                brain_extracted_patterns = ['*brain.nii.gz', '*T1w_brain.nii.gz', '*skull_stripped.nii.gz']
                
                original_file = None
                brain_extracted_file = None
                
                # Find original file
                for pattern in original_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        original_file = str(files[0])
                        break
                
                # Find brain-extracted file
                for pattern in brain_extracted_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        brain_extracted_file = str(files[0])
                        break
                
                if original_file and brain_extracted_file:
                    pairs.append((subject_id, original_file, brain_extracted_file))
                else:
                    self.logger.warning(f"Could not find brain extraction pair for {subject_id}")
        
        self.logger.info(f"Found {len(pairs)} brain extraction pairs for analysis")
        return pairs
    
    def analyze_all_subjects(self) -> pd.DataFrame:
        """Analyze brain extraction quality for all subjects"""
        pairs = self.find_brain_extraction_pairs()
        
        if not pairs:
            self.logger.error("No brain extraction pairs found for analysis!")
            return pd.DataFrame()
        
        results = []
        for i, (subject_id, original_path, brain_extracted_path) in enumerate(pairs, 1):
            self.logger.info(f"Processing {i}/{len(pairs)}: {subject_id}")
            
            metrics = self.analyze_brain_extraction_quality(subject_id, original_path, brain_extracted_path)
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.results_dir / "brain_extraction_qc.csv"
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        
        return results_df
    
    def create_brain_extraction_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations for brain extraction QC"""
        if results_df.empty:
            self.logger.warning("No data available for visualization!")
            return
        
        self.logger.info("Creating brain extraction QC visualizations...")
        
        # Filter successful analyses
        if 'analysis_failed' in results_df.columns:
            successful_df = results_df[~results_df['analysis_failed'].fillna(False)]
        else:
            successful_df = results_df.copy()
        
        if successful_df.empty:
            self.logger.warning("No successful analyses for visualization!")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Brain Extraction Quality Control Dashboard', fontsize=16, fontweight='bold')
        
        try:
            # Brain volume distribution
            if 'brain_volume_ml' in successful_df.columns:
                volume_data = successful_df['brain_volume_ml'].dropna()
                if len(volume_data) > 0:
                    axes[0, 0].hist(volume_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].axvline(self.config.min_brain_volume * np.prod([1, 1, 1]) / 1000, 
                                     color='red', linestyle='--', label='Min threshold')
                    axes[0, 0].axvline(self.config.max_brain_volume * np.prod([1, 1, 1]) / 1000, 
                                     color='red', linestyle='--', label='Max threshold')
                    axes[0, 0].set_title('Brain Volume Distribution')
                    axes[0, 0].set_xlabel('Volume (ml)')
                    axes[0, 0].set_ylabel('Count')
                    axes[0, 0].legend()
            
            # Boundary smoothness distribution
            if 'boundary_smoothness' in successful_df.columns:
                smoothness_data = successful_df['boundary_smoothness'].dropna()
                if len(smoothness_data) > 0:
                    axes[0, 1].hist(smoothness_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 1].axvline(self.config.min_solidity, color='red', linestyle='--', 
                                     label=f'Min threshold ({self.config.min_solidity})')
                    axes[0, 1].set_title('Boundary Smoothness Distribution')
                    axes[0, 1].set_xlabel('Solidity')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].legend()
            
            # Extraction quality score distribution
            if 'extraction_quality_score' in successful_df.columns:
                score_data = successful_df['extraction_quality_score'].dropna()
                if len(score_data) > 0:
                    axes[0, 2].hist(score_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 2].axvline(7, color='green', linestyle='--', label='Good threshold (7)')
                    axes[0, 2].set_title('Extraction Quality Score')
                    axes[0, 2].set_xlabel('Score (0-10)')
                    axes[0, 2].set_ylabel('Count')
                    axes[0, 2].legend()
            
            # Quality flags summary
            quality_flags = ['volume_plausible', 'boundary_smooth', 'minimal_residual_skull', 'extraction_quality_good']
            available_flags = [col for col in quality_flags if col in successful_df.columns]
            
            if available_flags:
                flag_counts = successful_df[available_flags].sum()
                axes[1, 0].bar(range(len(flag_counts)), flag_counts.values)
                axes[1, 0].set_title('Quality Flags Summary')
                axes[1, 0].set_ylabel('Count Passing')
                axes[1, 0].set_xticks(range(len(flag_counts)))
                axes[1, 0].set_xticklabels([col.replace('_', ' ').title() for col in flag_counts.index], 
                                         rotation=45, ha='right')
            
            # Volume vs Smoothness correlation
            if all(col in successful_df.columns for col in ['brain_volume_ml', 'boundary_smoothness']):
                corr_data = successful_df[['brain_volume_ml', 'boundary_smoothness']].dropna()
                if len(corr_data) > 1:
                    axes[1, 1].scatter(corr_data['brain_volume_ml'], corr_data['boundary_smoothness'], alpha=0.6)
                    axes[1, 1].set_title('Volume vs Boundary Smoothness')
                    axes[1, 1].set_xlabel('Brain Volume (ml)')
                    axes[1, 1].set_ylabel('Boundary Smoothness')
                    
                    # Add correlation coefficient
                    corr_coef = corr_data['brain_volume_ml'].corr(corr_data['boundary_smoothness'])
                    axes[1, 1].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[1, 1].transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Residual skull detection
            if 'residual_skull_detected' in successful_df.columns:
                residual_counts = successful_df['residual_skull_detected'].value_counts()
                axes[1, 2].pie(residual_counts.values, labels=['Clean', 'Residual Skull'], 
                             autopct='%1.1f%%', colors=self.config.color_palette[:2])
                axes[1, 2].set_title('Residual Skull Detection')
            
            # Brain intensity statistics
            if 'brain_mean_intensity' in successful_df.columns:
                intensity_data = successful_df['brain_mean_intensity'].dropna()
                if len(intensity_data) > 0:
                    axes[2, 0].hist(intensity_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 0].set_title('Brain Mean Intensity')
                    axes[2, 0].set_xlabel('Mean Intensity')
                    axes[2, 0].set_ylabel('Count')
            
            # Skull-to-brain ratio
            if 'skull_to_brain_ratio' in successful_df.columns:
                ratio_data = successful_df['skull_to_brain_ratio'].dropna()
                if len(ratio_data) > 0:
                    axes[2, 1].hist(ratio_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 1].set_title('Skull-to-Brain Ratio')
                    axes[2, 1].set_xlabel('Ratio')
                    axes[2, 1].set_ylabel('Count')
            
            # Overall quality overview
            if 'extraction_quality_good' in successful_df.columns:
                quality_counts = successful_df['extraction_quality_good'].value_counts()
                colors = ['red' if not good else 'green' for good in quality_counts.index]
                axes[2, 2].bar(range(len(quality_counts)), quality_counts.values, color=colors, alpha=0.7)
                axes[2, 2].set_title('Overall Extraction Quality')
                axes[2, 2].set_ylabel('Count')
                axes[2, 2].set_xticks(range(len(quality_counts)))
                axes[2, 2].set_xticklabels(['Poor', 'Good'])
            
            # Remove empty subplots
            for i in range(3):
                for j in range(3):
                    if not axes[i, j].has_data():
                        axes[i, j].text(0.5, 0.5, 'No Data Available', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
        
        except Exception as e:
            self.logger.warning(f"Some visualizations failed: {e}")
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / "brain_extraction_qc_dashboard.png"
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"QC dashboard saved to: {output_file}")
    
    def generate_brain_extraction_report(self, results_df: pd.DataFrame):
        """Generate comprehensive brain extraction QC report"""
        if results_df.empty:
            self.logger.warning("No data available for report generation!")
            return
        
        # Filter successful analyses
        if 'analysis_failed' in results_df.columns:
            successful_df = results_df[~results_df['analysis_failed'].fillna(False)]
        else:
            successful_df = results_df.copy()
        
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
        report_file = self.results_dir / "brain_extraction_qc_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("YOPD BRAIN EXTRACTION QUALITY CONTROL REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Subjects Analyzed: {len(successful_df)}\n")
            if len(results_df) > len(successful_df):
                f.write(f"Failed Analyses: {len(results_df) - len(successful_df)}\n")
            f.write("\n")
            
            f.write("BRAIN EXTRACTION QUALITY SUMMARY\n")
            f.write("-" * 35 + "\n")
            
            # Key metrics
            key_metrics = {
                'brain_volume_ml': 'Brain Volume (ml)',
                'boundary_smoothness': 'Boundary Smoothness (Solidity)',
                'extraction_quality_score': 'Overall Quality Score (0-10)',
                'skull_to_brain_ratio': 'Skull-to-Brain Ratio'
            }
            
            for metric_key, metric_name in key_metrics.items():
                if metric_key in summary_stats:
                    stats = summary_stats[metric_key]
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
            
            # Quality flags analysis
            quality_flags = ['volume_plausible', 'boundary_smooth', 'minimal_residual_skull', 'extraction_quality_good']
            f.write("QUALITY FLAGS ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            for flag in quality_flags:
                if flag in successful_df.columns:
                    count = successful_df[flag].sum()
                    total = len(successful_df)
                    percentage = count / total * 100
                    f.write(f"{flag.replace('_', ' ').title()}: {count}/{total} ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Quality assessment
            f.write("QUALITY ASSESSMENT\n")
            f.write("-" * 18 + "\n")
            
            if 'brain_volume_ml' in summary_stats:
                mean_volume = summary_stats['brain_volume_ml']['mean']
                expected_range = (1200, 1800)  # ml
                if expected_range[0] <= mean_volume <= expected_range[1]:
                    f.write(f"✓ Brain volumes are within expected range (mean={mean_volume:.0f}ml)\n")
                else:
                    f.write(f"⚠ Brain volumes may be outside expected range (mean={mean_volume:.0f}ml)\n")
            
            if 'boundary_smoothness' in summary_stats:
                mean_smoothness = summary_stats['boundary_smoothness']['mean']
                if mean_smoothness >= self.config.min_solidity:
                    f.write(f"✓ Good boundary smoothness (mean solidity={mean_smoothness:.3f})\n")
                else:
                    f.write(f"⚠ Poor boundary smoothness detected (mean solidity={mean_smoothness:.3f})\n")
            
            if 'residual_skull_detected' in successful_df.columns:
                residual_rate = successful_df['residual_skull_detected'].mean()
                if residual_rate < 0.1:  # < 10%
                    f.write(f"✓ Low residual skull rate ({residual_rate*100:.1f}%)\n")
                else:
                    f.write(f"⚠ High residual skull rate detected ({residual_rate*100:.1f}%)\n")
            
            if 'extraction_quality_score' in summary_stats:
                mean_score = summary_stats['extraction_quality_score']['mean']
                if mean_score >= 7:
                    f.write(f"✓ High quality brain extraction (mean score={mean_score:.1f}/10)\n")
                elif mean_score >= 5:
                    f.write(f"⚠ Moderate quality brain extraction (mean score={mean_score:.1f}/10)\n")
                else:
                    f.write(f"⚠ Poor quality brain extraction (mean score={mean_score:.1f}/10)\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if 'boundary_smoothness' in summary_stats and summary_stats['boundary_smoothness']['mean'] < self.config.min_solidity:
                f.write("• Improve brain boundary smoothness:\n")
                f.write("  - Review brain extraction parameters\n")
                f.write("  - Consider morphological post-processing\n")
                f.write("  - Validate input image quality\n\n")
            
            if 'residual_skull_detected' in successful_df.columns and successful_df['residual_skull_detected'].mean() > 0.2:
                f.write("• Address residual skull tissue:\n")
                f.write("  - Review skull stripping algorithm parameters\n")
                f.write("  - Consider manual quality control\n")
                f.write("  - Validate brain extraction masks\n\n")
            
            if 'extraction_quality_good' in successful_df.columns:
                poor_extraction_rate = (len(successful_df) - successful_df['extraction_quality_good'].sum()) / len(successful_df)
                if poor_extraction_rate > 0.3:  # > 30% poor
                    f.write("• High rate of poor extractions detected:\n")
                    f.write("  - Review overall brain extraction pipeline\n")
                    f.write("  - Consider alternative skull stripping methods\n")
                    f.write("  - Implement manual quality control steps\n\n")
            
            f.write("TECHNICAL NOTES\n")
            f.write("-" * 15 + "\n")
            f.write("• Boundary smoothness measured using solidity (actual area / convex hull area)\n")
            f.write("• Residual skull detected by comparing intensities in skull regions\n")
            f.write("• Quality score combines volume plausibility, boundary smoothness, and skull removal\n")
            f.write("• Volume thresholds: 800k - 2M voxels (dataset dependent)\n")
        
        # Export JSON summary
        json_stats = {}
        for key, value in summary_stats.items():
            json_stats[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for k, v in value.items()}
        
        json_file = self.results_dir / "brain_extraction_qc_summary.json"
        with open(json_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_file}")
        self.logger.info(f"JSON summary saved to: {json_file}")

def main():
    """Main function to run brain extraction quality control"""
    print("YOPD Brain Extraction Quality Control")
    print("=" * 45)
    
    # Initialize configuration
    config = BrainExtractionConfig()
    
    # Verify output directory exists
    if not Path(config.output_root).exists():
        print(f"Error: Output directory not found: {config.output_root}")
        print("Please ensure the YOPD pipeline has been run first.")
        return
    
    # Initialize analyzer
    analyzer = BrainExtractionAnalyzer(config)
    
    try:
        # Analyze all subjects
        print("Starting brain extraction quality control...")
        results_df = analyzer.analyze_all_subjects()
        
        if results_df.empty:
            print("No brain extraction pairs found for analysis!")
            return
        
        print(f"Successfully analyzed {len(results_df)} subjects")
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.create_brain_extraction_visualizations(results_df)
        
        # Generate report
        print("Generating comprehensive report...")
        analyzer.generate_brain_extraction_report(results_df)
        
        print("\nBrain Extraction Quality Control Complete!")
        print(f"Results saved to: {analyzer.results_dir}")
        print("\nGenerated files:")
        print("  - brain_extraction_qc.csv: Detailed metrics")
        print("  - brain_extraction_qc_dashboard.png: Visualization")
        print("  - brain_extraction_qc_report.txt: Summary report")
        print("  - brain_extraction_qc_summary.json: JSON summary")
        
    except Exception as e:
        print(f"Error during brain extraction QC: {e}")
        analyzer.logger.error(f"Brain extraction QC failed: {e}")
        raise

if __name__ == "__main__":
    main()