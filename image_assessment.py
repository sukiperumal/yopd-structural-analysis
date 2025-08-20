#!/usr/bin/env python3
"""
YOPD Quality Assessment - Image Quality Analysis
===============================================

This script analyzes raw MRI image quality before processing and calculates
comprehensive quality metrics including SNR, CNR, and uniformity measures.
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
from scipy.stats import median_abs_deviation
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityConfig:
    """Configuration for quality assessment"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    min_snr: float = 15.0
    max_intensity_nonuniformity: float = 0.3
    min_brain_volume: int = 800000  # voxels
    figure_dpi: int = 300
    color_palette: List[str] = field(default_factory=lambda: ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'])

class QualityMetricsCalculator:
    """Dedicated class for quality metrics calculation"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('quality_assessment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_snr(self, image_data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> float:
        """Calculate SNR with robust estimation"""
        try:
            if brain_mask is None:
                brain_mask = image_data > (0.1 * np.max(image_data))
            
            if np.sum(brain_mask) == 0:
                return 0.0
            
            # Use median for robustness
            signal = np.median(image_data[brain_mask])
            
            # Noise estimation from edge regions
            noise = self._estimate_noise_from_edges(image_data)
            
            return signal / noise if noise > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"SNR calculation failed: {e}")
            return np.nan
    
    def _estimate_noise_from_edges(self, image_data: np.ndarray) -> float:
        """Estimate noise from edge regions of the image"""
        try:
            edge_slices = [
                image_data[:10, :, :],   # Front
                image_data[-10:, :, :],  # Back
                image_data[:, :10, :],   # Left
                image_data[:, -10:, :],  # Right
                image_data[:, :, :10],   # Bottom
                image_data[:, :, -10:]   # Top
            ]
            
            edge_data = np.concatenate([edge.flatten() for edge in edge_slices])
            return np.std(edge_data)
            
        except Exception as e:
            self.logger.warning(f"Noise estimation failed: {e}")
            return 1.0  # Fallback value
    
    def calculate_cnr(self, image_data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> float:
        """Calculate Contrast-to-Noise Ratio"""
        try:
            if brain_mask is None:
                brain_mask = image_data > (0.1 * np.max(image_data))
            
            if np.sum(brain_mask) == 0:
                return 0.0
            
            brain_signal = np.median(image_data[brain_mask])
            background_signal = np.median(image_data[~brain_mask])
            noise = self._estimate_noise_from_edges(image_data)
            
            return (brain_signal - background_signal) / noise if noise > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"CNR calculation failed: {e}")
            return np.nan
    
    def assess_uniformity(self, image_data: np.ndarray, brain_mask: np.ndarray) -> float:
        """Assess intensity uniformity using robust statistics"""
        try:
            brain_intensities = image_data[brain_mask]
            
            if len(brain_intensities) == 0:
                return 1.0  # Maximum non-uniformity
            
            # Use median absolute deviation for robustness
            median_intensity = np.median(brain_intensities)
            mad_intensity = median_abs_deviation(brain_intensities)
            
            return mad_intensity / median_intensity if median_intensity > 0 else 1.0
            
        except Exception as e:
            self.logger.warning(f"Uniformity assessment failed: {e}")
            return np.nan
    
    def analyze_original_image(self, image_path: str) -> Dict:
        """Comprehensive analysis of original image quality"""
        try:
            self.logger.info(f"Analyzing image: {os.path.basename(image_path)}")
            
            # Load image
            img = nib.load(image_path)
            data = img.get_fdata()
            
            # Create brain mask
            brain_threshold = 0.1 * np.max(data)
            brain_mask = data > brain_threshold
            
            # Calculate metrics
            metrics = {
                'file_path': image_path,
                'subject_id': self._extract_subject_id(image_path),
                'image_shape': data.shape,
                'voxel_size': img.header.get_zooms()[:3],
                'orig_mean_intensity': np.mean(data[data > 0]),
                'orig_std_intensity': np.std(data[data > 0]),
                'orig_min_intensity': np.min(data),
                'orig_max_intensity': np.max(data),
                'brain_volume_voxels': np.sum(brain_mask),
                'brain_volume_ml': np.sum(brain_mask) * np.prod(img.header.get_zooms()[:3]) / 1000,
                'orig_snr': self.calculate_snr(data, brain_mask),
                'orig_cnr': self.calculate_cnr(data, brain_mask),
                'orig_noise_level': self._estimate_noise_from_edges(data),
                'orig_uniformity': self.assess_uniformity(data, brain_mask),
                'intensity_range': np.max(data) - np.min(data)
            }
            
            # Quality flags
            metrics['snr_adequate'] = metrics['orig_snr'] >= self.config.min_snr
            metrics['uniformity_good'] = metrics['orig_uniformity'] <= self.config.max_intensity_nonuniformity
            metrics['volume_adequate'] = metrics['brain_volume_voxels'] >= self.config.min_brain_volume
            metrics['overall_quality_good'] = all([
                metrics['snr_adequate'],
                metrics['uniformity_good'],
                metrics['volume_adequate']
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze image {image_path}: {e}")
            return self._create_error_metrics(image_path, str(e))
    
    def _extract_subject_id(self, file_path: str) -> str:
        """Extract subject ID from file path"""
        filename = os.path.basename(file_path)
        # Assume filename starts with subject ID
        if filename.startswith('sub-'):
            return filename.split('_')[0]
        return filename.split('.')[0]
    
    def _create_error_metrics(self, image_path: str, error_msg: str) -> Dict:
        """Create error metrics for failed analyses"""
        return {
            'file_path': image_path,
            'subject_id': self._extract_subject_id(image_path),
            'error': error_msg,
            'analysis_failed': True
        }

class ImageQualityAnalyzer:
    """Main class for image quality analysis"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.calculator = QualityMetricsCalculator(config)
        self.logger = self._setup_logging()
        self.results_dir = Path(config.output_root) / "quality_assessment"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger('image_quality_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = Path(self.config.output_root) / "logs" / "quality_assessment.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def find_original_images(self) -> List[str]:
        """Find original T1 images in the dataset"""
        self.logger.info("Searching for original T1 images...")
        
        # Check inventory file first
        inventory_path = Path(self.config.output_root) / "02_quality_control" / "data_inventory.csv"
        
        image_paths = []
        
        if inventory_path.exists():
            self.logger.info(f"Loading image paths from inventory: {inventory_path}")
            try:
                inventory_df = pd.read_csv(inventory_path)
                if 't1_file_path' in inventory_df.columns:
                    image_paths = inventory_df['t1_file_path'].dropna().tolist()
                    self.logger.info(f"Found {len(image_paths)} images from inventory")
                else:
                    self.logger.warning("No t1_file_path column in inventory")
            except Exception as e:
                self.logger.error(f"Failed to load inventory: {e}")
        
        # Fallback: search preprocessing directory
        if not image_paths:
            self.logger.info("Searching preprocessing directory for original images...")
            preproc_dir = Path(self.config.output_root) / "01_preprocessed"
            
            if preproc_dir.exists():
                for subject_dir in preproc_dir.iterdir():
                    if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                        # Look for original or brain-extracted T1
                        for pattern in ['*T1w.nii.gz', '*T1w_brain.nii.gz', '*corrected.nii.gz']:
                            files = list(subject_dir.glob(pattern))
                            if files:
                                image_paths.append(str(files[0]))
                                break
        
        self.logger.info(f"Total images found: {len(image_paths)}")
        return image_paths
    
    def analyze_all_images(self) -> pd.DataFrame:
        """Analyze all images and return results DataFrame"""
        image_paths = self.find_original_images()
        
        if not image_paths:
            self.logger.error("No images found for analysis!")
            return pd.DataFrame()
        
        self.logger.info(f"Starting analysis of {len(image_paths)} images")
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            if not os.path.exists(image_path):
                self.logger.warning(f"Image not found: {image_path}")
                continue
            
            try:
                metrics = self.calculator.analyze_original_image(image_path)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append(self.calculator._create_error_metrics(image_path, str(e)))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.results_dir / "image_quality_metrics.csv"
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        
        return results_df
    
    def generate_quality_report(self, results_df: pd.DataFrame):
        """Generate comprehensive quality assessment report"""
        if results_df.empty:
            self.logger.error("No results to generate report from!")
            return
        
        self.logger.info("Generating quality assessment report...")
        
        # Filter successful analyses
        if 'analysis_failed' in results_df.columns:
            successful_df = results_df[~results_df['analysis_failed'].fillna(False)]
        else:
            successful_df = results_df.copy()
        
        if successful_df.empty:
            self.logger.warning("No successful analyses to report!")
            return
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_statistics(successful_df)
        
        # Create visualizations
        self._create_quality_visualizations(successful_df)
        
        # Generate text report
        self._generate_text_report(successful_df, summary_stats)
        
        # Export detailed metrics
        self._export_detailed_metrics(successful_df, summary_stats)
        
        self.logger.info("Quality assessment report completed!")
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive summary statistics"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {}
        for col in numeric_columns:
            if df[col].notna().sum() > 0:  # Only process columns with valid data
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'count': df[col].count()
                }
        
        # Quality flags summary
        quality_flags = {
            'snr_adequate_count': df.get('snr_adequate', pd.Series([])).sum(),
            'uniformity_good_count': df.get('uniformity_good', pd.Series([])).sum(),
            'volume_adequate_count': df.get('volume_adequate_count', pd.Series([])).sum(),
            'overall_quality_good_count': df.get('overall_quality_good', pd.Series([])).sum(),
            'total_subjects': len(df)
        }
        
        stats_dict['quality_summary'] = quality_flags
        
        return stats_dict
    
    def _create_quality_visualizations(self, df: pd.DataFrame):
        """Create comprehensive quality visualizations"""
        self.logger.info("Creating quality visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create main quality dashboard
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Image Quality Assessment Dashboard', fontsize=16, fontweight='bold')
        
        try:
            # SNR distribution
            if 'orig_snr' in df.columns and df['orig_snr'].notna().sum() > 0:
                axes[0, 0].hist(df['orig_snr'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(self.config.min_snr, color='red', linestyle='--', 
                                  label=f'Min threshold ({self.config.min_snr})')
                axes[0, 0].set_title('Signal-to-Noise Ratio Distribution')
                axes[0, 0].set_xlabel('SNR')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].legend()
            
            # CNR distribution
            if 'orig_cnr' in df.columns and df['orig_cnr'].notna().sum() > 0:
                axes[0, 1].hist(df['orig_cnr'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Contrast-to-Noise Ratio Distribution')
                axes[0, 1].set_xlabel('CNR')
                axes[0, 1].set_ylabel('Count')
            
            # Uniformity distribution
            if 'orig_uniformity' in df.columns and df['orig_uniformity'].notna().sum() > 0:
                axes[0, 2].hist(df['orig_uniformity'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[0, 2].axvline(self.config.max_intensity_nonuniformity, color='red', linestyle='--',
                                  label=f'Max threshold ({self.config.max_intensity_nonuniformity})')
                axes[0, 2].set_title('Intensity Uniformity Distribution')
                axes[0, 2].set_xlabel('Non-uniformity')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].legend()
            
            # Brain volume distribution
            if 'brain_volume_ml' in df.columns and df['brain_volume_ml'].notna().sum() > 0:
                axes[1, 0].hist(df['brain_volume_ml'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Brain Volume Distribution')
                axes[1, 0].set_xlabel('Volume (ml)')
                axes[1, 0].set_ylabel('Count')
            
            # SNR vs CNR correlation
            if all(col in df.columns for col in ['orig_snr', 'orig_cnr']):
                valid_data = df[['orig_snr', 'orig_cnr']].dropna()
                if len(valid_data) > 1:
                    axes[1, 1].scatter(valid_data['orig_snr'], valid_data['orig_cnr'], alpha=0.6)
                    axes[1, 1].set_title('SNR vs CNR Correlation')
                    axes[1, 1].set_xlabel('SNR')
                    axes[1, 1].set_ylabel('CNR')
                    
                    # Add correlation coefficient
                    corr_coef = valid_data['orig_snr'].corr(valid_data['orig_cnr'])
                    axes[1, 1].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[1, 1].transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Quality flags summary
            quality_columns = ['snr_adequate', 'uniformity_good', 'volume_adequate', 'overall_quality_good']
            available_quality_cols = [col for col in quality_columns if col in df.columns]
            
            if available_quality_cols:
                quality_counts = df[available_quality_cols].sum()
                axes[1, 2].bar(range(len(quality_counts)), quality_counts.values)
                axes[1, 2].set_title('Quality Criteria Summary')
                axes[1, 2].set_xlabel('Quality Criteria')
                axes[1, 2].set_ylabel('Count Passing')
                axes[1, 2].set_xticks(range(len(quality_counts)))
                axes[1, 2].set_xticklabels([col.replace('_', ' ').title() for col in quality_counts.index], 
                                          rotation=45, ha='right')
            
            # Intensity statistics
            if 'orig_mean_intensity' in df.columns and df['orig_mean_intensity'].notna().sum() > 0:
                axes[2, 0].hist(df['orig_mean_intensity'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[2, 0].set_title('Mean Intensity Distribution')
                axes[2, 0].set_xlabel('Mean Intensity')
                axes[2, 0].set_ylabel('Count')
            
            # Noise level distribution
            if 'orig_noise_level' in df.columns and df['orig_noise_level'].notna().sum() > 0:
                axes[2, 1].hist(df['orig_noise_level'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[2, 1].set_title('Noise Level Distribution')
                axes[2, 1].set_xlabel('Noise Level')
                axes[2, 1].set_ylabel('Count')
            
            # Overall quality score (if calculable)
            if all(col in df.columns for col in ['orig_snr', 'orig_uniformity']):
                # Calculate a simple quality score
                valid_data = df[['orig_snr', 'orig_uniformity']].dropna()
                if len(valid_data) > 0:
                    quality_score = (
                        np.minimum(valid_data['orig_snr'], 30) / 30 * 5 +  # SNR component (0-5)
                        (1 - np.minimum(valid_data['orig_uniformity'], 1)) * 5  # Uniformity component (0-5)
                    )
                    axes[2, 2].hist(quality_score, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 2].set_title('Composite Quality Score')
                    axes[2, 2].set_xlabel('Quality Score (0-10)')
                    axes[2, 2].set_ylabel('Count')
            
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
        output_file = self.results_dir / "quality_assessment_dashboard.png"
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Quality dashboard saved to: {output_file}")
    
    def _generate_text_report(self, df: pd.DataFrame, summary_stats: Dict):
        """Generate comprehensive text report"""
        report_file = self.results_dir / "quality_assessment_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("YOPD IMAGE QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Analyzed: {len(df)}\n\n")
            
            f.write("QUALITY METRICS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            # Key metrics
            key_metrics = ['orig_snr', 'orig_cnr', 'orig_uniformity', 'brain_volume_ml']
            for metric in key_metrics:
                if metric in summary_stats and summary_stats[metric]['count'] > 0:
                    stats = summary_stats[metric]
                    f.write(f"{metric.replace('orig_', '').replace('_', ' ').title()}:\n")
                    f.write(f"  Mean +/- SD: {stats['mean']:.3f} +/- {stats['std']:.3f}\n")
                    f.write(f"  Median (IQR): {stats['median']:.3f} ({stats['q25']:.3f}-{stats['q75']:.3f})\n")
                    f.write(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}\n\n")
            
            f.write("QUALITY FLAGS SUMMARY\n")
            f.write("-" * 25 + "\n")
            if 'quality_summary' in summary_stats:
                quality = summary_stats['quality_summary']
                total = quality.get('total_subjects', len(df))
                
                f.write(f"SNR Adequate (>={self.config.min_snr}): {quality.get('snr_adequate_count', 0)}/{total} "
                       f"({quality.get('snr_adequate_count', 0)/total*100:.1f}%)\n")
                f.write(f"Good Uniformity (<={self.config.max_intensity_nonuniformity}): "
                       f"{quality.get('uniformity_good_count', 0)}/{total} "
                       f"({quality.get('uniformity_good_count', 0)/total*100:.1f}%)\n")
                f.write(f"Adequate Volume (>={self.config.min_brain_volume} voxels): "
                       f"{quality.get('volume_adequate_count', 0)}/{total} "
                       f"({quality.get('volume_adequate_count', 0)/total*100:.1f}%)\n")
                f.write(f"Overall Good Quality: {quality.get('overall_quality_good_count', 0)}/{total} "
                       f"({quality.get('overall_quality_good_count', 0)/total*100:.1f}%)\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if 'orig_snr' in summary_stats and summary_stats['orig_snr']['count'] > 0:
                mean_snr = summary_stats['orig_snr']['mean']
                if mean_snr < self.config.min_snr:
                    f.write(f"⚠ Low SNR detected (mean={mean_snr:.1f}). Consider:\n")
                    f.write("  - Reviewing acquisition parameters\n")
                    f.write("  - Checking for motion artifacts\n")
                    f.write("  - Increasing acquisition time if possible\n\n")
                else:
                    f.write(f"✓ SNR is adequate (mean={mean_snr:.1f})\n\n")
            
            if 'orig_uniformity' in summary_stats and summary_stats['orig_uniformity']['count'] > 0:
                mean_uniformity = summary_stats['orig_uniformity']['mean']
                if mean_uniformity > self.config.max_intensity_nonuniformity:
                    f.write(f"⚠ Poor intensity uniformity detected (mean={mean_uniformity:.3f}). Consider:\n")
                    f.write("  - Bias field correction\n")
                    f.write("  - Checking coil sensitivity\n")
                    f.write("  - Reviewing shimming procedures\n\n")
                else:
                    f.write(f"✓ Intensity uniformity is good (mean={mean_uniformity:.3f})\n\n")
        
        self.logger.info(f"Text report saved to: {report_file}")
    
    def _export_detailed_metrics(self, df: pd.DataFrame, summary_stats: Dict):
        """Export detailed metrics in multiple formats"""
        # JSON export
        json_file = self.results_dir / "quality_metrics_summary.json"
        
        # Convert numpy types to Python native types for JSON serialization
        json_stats = {}
        for key, value in summary_stats.items():
            if isinstance(value, dict):
                json_stats[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                  for k, v in value.items()}
            else:
                json_stats[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=2)
        
        self.logger.info(f"JSON summary saved to: {json_file}")
        
        # Detailed CSV export (already done in analyze_all_images)
        self.logger.info("Quality assessment analysis completed successfully!")

def main():
    """Main function to run image quality assessment"""
    print("YOPD Image Quality Assessment")
    print("=" * 40)
    
    # Initialize configuration
    config = QualityConfig()
    
    # Verify output directory exists
    if not Path(config.output_root).exists():
        print(f"Error: Output directory not found: {config.output_root}")
        print("Please ensure the YOPD pipeline has been run first.")
        return
    
    # Initialize analyzer
    analyzer = ImageQualityAnalyzer(config)
    
    try:
        # Analyze all images
        print("Starting image quality analysis...")
        results_df = analyzer.analyze_all_images()
        
        if results_df.empty:
            print("No images found or analyzed successfully!")
            return
        
        print(f"Successfully analyzed {len(results_df)} images")
        
        # Generate comprehensive report
        print("Generating quality assessment report...")
        analyzer.generate_quality_report(results_df)
        
        print("\nQuality Assessment Complete!")
        print(f"Results saved to: {analyzer.results_dir}")
        print("\nGenerated files:")
        print(f"  - image_quality_metrics.csv: Detailed metrics")
        print(f"  - quality_assessment_dashboard.png: Visualization")
        print(f"  - quality_assessment_report.txt: Summary report")
        print(f"  - quality_metrics_summary.json: JSON summary")
        
    except Exception as e:
        print(f"Error during quality assessment: {e}")
        analyzer.logger.error(f"Quality assessment failed: {e}")
        raise

if __name__ == "__main__":
    main()