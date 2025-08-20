#!/usr/bin/env python3
"""
YOPD Bias Field Correction Analysis
==================================

This script analyzes the effectiveness of bias field correction
and provides comprehensive metrics and visualizations.
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
from skimage.metrics import structural_similarity as ssim
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BiasAnalysisConfig:
    """Configuration for bias field analysis"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

class BiasFieldAnalyzer:
    """Dedicated class for bias field analysis"""
    
    def __init__(self, config: BiasAnalysisConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results_dir = Path(config.output_root) / "bias_field_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('bias_field_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = Path(self.config.output_root) / "logs" / "bias_field_analysis.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def assess_uniformity_improvement(self, original_data: np.ndarray, corrected_data: np.ndarray, 
                                    brain_mask: np.ndarray) -> Dict:
        """Assess improvement in intensity uniformity after bias correction"""
        try:
            # Original uniformity
            orig_brain_intensities = original_data[brain_mask]
            orig_median = np.median(orig_brain_intensities)
            orig_mad = median_abs_deviation(orig_brain_intensities)
            orig_uniformity = orig_mad / orig_median if orig_median > 0 else 1.0
            
            # Corrected uniformity
            corr_brain_intensities = corrected_data[brain_mask]
            corr_median = np.median(corr_brain_intensities)
            corr_mad = median_abs_deviation(corr_brain_intensities)
            corr_uniformity = corr_mad / corr_median if corr_median > 0 else 1.0
            
            # Improvement metrics
            uniformity_improvement = orig_uniformity - corr_uniformity
            improvement_percentage = (uniformity_improvement / orig_uniformity * 100) if orig_uniformity > 0 else 0
            
            return {
                'orig_uniformity': orig_uniformity,
                'corrected_uniformity': corr_uniformity,
                'uniformity_improvement': uniformity_improvement,
                'improvement_percentage': improvement_percentage,
                'uniformity_improved': uniformity_improvement > 0
            }
            
        except Exception as e:
            self.logger.warning(f"Uniformity assessment failed: {e}")
            return {'error': str(e)}
    
    def calculate_structural_similarity(self, original_data: np.ndarray, corrected_data: np.ndarray) -> float:
        """Calculate structural similarity between original and corrected images"""
        try:
            # Normalize images to [0, 1] range
            orig_norm = (original_data - np.min(original_data)) / (np.max(original_data) - np.min(original_data))
            corr_norm = (corrected_data - np.min(corrected_data)) / (np.max(corrected_data) - np.min(corrected_data))
            
            # Calculate SSIM for each slice and take mean
            ssim_scores = []
            for i in range(original_data.shape[2]):  # Assuming axial slices
                if np.std(orig_norm[:, :, i]) > 0 and np.std(corr_norm[:, :, i]) > 0:
                    score = ssim(orig_norm[:, :, i], corr_norm[:, :, i], data_range=1.0)
                    ssim_scores.append(score)
            
            return np.mean(ssim_scores) if ssim_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {e}")
            return np.nan
    
    def analyze_bias_correction_effectiveness(self, subject_id: str, 
                                            original_path: str, corrected_path: str) -> Dict:
        """Comprehensive analysis of bias correction effectiveness"""
        try:
            self.logger.info(f"Analyzing bias correction for subject: {subject_id}")
            
            # Load images
            orig_img = nib.load(original_path)
            orig_data = orig_img.get_fdata()
            
            corr_img = nib.load(corrected_path)
            corr_data = corr_img.get_fdata()
            
            # Create brain mask from corrected image
            brain_threshold = 0.1 * np.max(corr_data)
            brain_mask = corr_data > brain_threshold
            
            # Basic metrics
            metrics = {
                'subject_id': subject_id,
                'original_path': original_path,
                'corrected_path': corrected_path,
                'brain_volume_voxels': np.sum(brain_mask),
            }
            
            # Intensity statistics
            orig_brain = orig_data[brain_mask]
            corr_brain = corr_data[brain_mask]
            
            metrics.update({
                'orig_mean_intensity': np.mean(orig_brain),
                'orig_std_intensity': np.std(orig_brain),
                'orig_median_intensity': np.median(orig_brain),
                'corr_mean_intensity': np.mean(corr_brain),
                'corr_std_intensity': np.std(corr_brain),
                'corr_median_intensity': np.median(corr_brain),
            })
            
            # Uniformity assessment
            uniformity_metrics = self.assess_uniformity_improvement(orig_data, corr_data, brain_mask)
            metrics.update(uniformity_metrics)
            
            # Structural similarity
            metrics['structural_similarity'] = self.calculate_structural_similarity(orig_data, corr_data)
            
            # Bias correction score (0-10 scale)
            score_components = []
            
            # Uniformity improvement component (0-4 points)
            if 'improvement_percentage' in metrics and not np.isnan(metrics['improvement_percentage']):
                uniformity_score = min(4, metrics['improvement_percentage'] / 25 * 4)  # 25% improvement = 4 points
                score_components.append(uniformity_score)
            
            # Structural similarity component (0-3 points)
            if not np.isnan(metrics['structural_similarity']):
                similarity_score = metrics['structural_similarity'] * 3  # SSIM of 1 = 3 points
                score_components.append(similarity_score)
            
            # Overall intensity consistency component (0-3 points)
            if 'corr_uniformity' in metrics and not np.isnan(metrics['corr_uniformity']):
                consistency_score = max(0, (0.5 - metrics['corr_uniformity']) / 0.5 * 3)  # Lower uniformity = higher score
                score_components.append(consistency_score)
            
            metrics['bias_correction_score'] = sum(score_components) if score_components else 0
            metrics['correction_effective'] = metrics['bias_correction_score'] >= 6  # Arbitrary threshold
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze bias correction for {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'error': str(e),
                'analysis_failed': True
            }
    
    def find_image_pairs(self) -> List[Tuple[str, str, str]]:
        """Find original and bias-corrected image pairs"""
        self.logger.info("Searching for original and bias-corrected image pairs...")
        
        pairs = []
        preproc_dir = Path(self.config.output_root) / "01_preprocessed"
        
        if not preproc_dir.exists():
            self.logger.error(f"Preprocessing directory not found: {preproc_dir}")
            return pairs
        
        for subject_dir in preproc_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                subject_id = subject_dir.name
                
                # Look for original and corrected files
                original_patterns = ['*T1w.nii.gz', '*T1w_brain.nii.gz']
                corrected_patterns = ['*corrected.nii.gz', '*T1w_corrected.nii.gz']
                
                original_file = None
                corrected_file = None
                
                # Find original file
                for pattern in original_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        original_file = str(files[0])
                        break
                
                # Find corrected file
                for pattern in corrected_patterns:
                    files = list(subject_dir.glob(pattern))
                    if files:
                        corrected_file = str(files[0])
                        break
                
                if original_file and corrected_file:
                    pairs.append((subject_id, original_file, corrected_file))
                else:
                    self.logger.warning(f"Could not find image pair for {subject_id}")
                    self.logger.debug(f"  Original: {original_file}")
                    self.logger.debug(f"  Corrected: {corrected_file}")
        
        self.logger.info(f"Found {len(pairs)} image pairs for analysis")
        return pairs
    
    def analyze_all_subjects(self) -> pd.DataFrame:
        """Analyze bias correction effectiveness for all subjects"""
        pairs = self.find_image_pairs()
        
        if not pairs:
            self.logger.error("No image pairs found for analysis!")
            return pd.DataFrame()
        
        results = []
        for i, (subject_id, original_path, corrected_path) in enumerate(pairs, 1):
            self.logger.info(f"Processing {i}/{len(pairs)}: {subject_id}")
            
            metrics = self.analyze_bias_correction_effectiveness(subject_id, original_path, corrected_path)
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.results_dir / "bias_correction_analysis.csv"
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to: {output_file}")
        
        return results_df
    
    def create_bias_correction_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations for bias correction analysis"""
        if results_df.empty:
            self.logger.warning("No data available for visualization!")
            return
        
        self.logger.info("Creating bias correction visualizations...")
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False)]
        
        if successful_df.empty:
            self.logger.warning("No successful analyses for visualization!")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Bias Field Correction Analysis Dashboard', fontsize=16, fontweight='bold')
        
        try:
            # Uniformity improvement histogram
            if 'improvement_percentage' in successful_df.columns:
                improvement_data = successful_df['improvement_percentage'].dropna()
                if len(improvement_data) > 0:
                    axes[0, 0].hist(improvement_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].axvline(0, color='red', linestyle='--', label='No improvement')
                    axes[0, 0].set_title('Uniformity Improvement (%)')
                    axes[0, 0].set_xlabel('Improvement Percentage')
                    axes[0, 0].set_ylabel('Count')
                    axes[0, 0].legend()
            
            # Before vs After uniformity comparison
            if all(col in successful_df.columns for col in ['orig_uniformity', 'corrected_uniformity']):
                uniform_data = successful_df[['orig_uniformity', 'corrected_uniformity']].dropna()
                if len(uniform_data) > 0:
                    axes[0, 1].scatter(uniform_data['orig_uniformity'], uniform_data['corrected_uniformity'], 
                                     alpha=0.6)
                    # Add diagonal line
                    min_val = min(uniform_data.min())
                    max_val = max(uniform_data.max())
                    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No change')
                    axes[0, 1].set_title('Uniformity: Before vs After')
                    axes[0, 1].set_xlabel('Original Uniformity')
                    axes[0, 1].set_ylabel('Corrected Uniformity')
                    axes[0, 1].legend()
            
            # Structural similarity distribution
            if 'structural_similarity' in successful_df.columns:
                ssim_data = successful_df['structural_similarity'].dropna()
                if len(ssim_data) > 0:
                    axes[0, 2].hist(ssim_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 2].axvline(0.95, color='green', linestyle='--', label='Excellent (0.95)')
                    axes[0, 2].set_title('Structural Similarity Distribution')
                    axes[0, 2].set_xlabel('SSIM')
                    axes[0, 2].set_ylabel('Count')
                    axes[0, 2].legend()
            
            # Bias correction score distribution
            if 'bias_correction_score' in successful_df.columns:
                score_data = successful_df['bias_correction_score'].dropna()
                if len(score_data) > 0:
                    axes[1, 0].hist(score_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(6, color='orange', linestyle='--', label='Effective threshold (6)')
                    axes[1, 0].set_title('Bias Correction Score Distribution')
                    axes[1, 0].set_xlabel('Score (0-10)')
                    axes[1, 0].set_ylabel('Count')
                    axes[1, 0].legend()
            
            # Intensity change analysis
            if all(col in successful_df.columns for col in ['orig_mean_intensity', 'corr_mean_intensity']):
                intensity_data = successful_df[['orig_mean_intensity', 'corr_mean_intensity']].dropna()
                if len(intensity_data) > 0:
                    axes[1, 1].scatter(intensity_data['orig_mean_intensity'], intensity_data['corr_mean_intensity'], 
                                     alpha=0.6)
                    # Add diagonal line
                    min_val = min(intensity_data.min())
                    max_val = max(intensity_data.max())
                    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No change')
                    axes[1, 1].set_title('Mean Intensity: Before vs After')
                    axes[1, 1].set_xlabel('Original Mean Intensity')
                    axes[1, 1].set_ylabel('Corrected Mean Intensity')
                    axes[1, 1].legend()
            
            # Effectiveness summary
            if 'correction_effective' in successful_df.columns:
                effectiveness_counts = successful_df['correction_effective'].value_counts()
                axes[1, 2].pie(effectiveness_counts.values, labels=['Ineffective', 'Effective'], 
                             autopct='%1.1f%%', colors=self.config.color_palette[:2])
                axes[1, 2].set_title('Correction Effectiveness Summary')
            
            # Brain volume consistency
            if 'brain_volume_voxels' in successful_df.columns:
                volume_data = successful_df['brain_volume_voxels'].dropna()
                if len(volume_data) > 0:
                    axes[2, 0].hist(volume_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[2, 0].set_title('Brain Volume Distribution')
                    axes[2, 0].set_xlabel('Volume (voxels)')
                    axes[2, 0].set_ylabel('Count')
            
            # Correlation between improvement and original uniformity
            if all(col in successful_df.columns for col in ['orig_uniformity', 'improvement_percentage']):
                corr_data = successful_df[['orig_uniformity', 'improvement_percentage']].dropna()
                if len(corr_data) > 1:
                    axes[2, 1].scatter(corr_data['orig_uniformity'], corr_data['improvement_percentage'], alpha=0.6)
                    axes[2, 1].set_title('Original Uniformity vs Improvement')
                    axes[2, 1].set_xlabel('Original Uniformity')
                    axes[2, 1].set_ylabel('Improvement (%)')
                    
                    # Add correlation coefficient
                    corr_coef = corr_data['orig_uniformity'].corr(corr_data['improvement_percentage'])
                    axes[2, 1].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[2, 1].transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Overall quality metrics
            quality_metrics = ['bias_correction_score', 'structural_similarity', 'improvement_percentage']
            available_metrics = [col for col in quality_metrics if col in successful_df.columns]
            
            if available_metrics:
                metric_means = successful_df[available_metrics].mean()
                axes[2, 2].bar(range(len(metric_means)), metric_means.values)
                axes[2, 2].set_title('Average Quality Metrics')
                axes[2, 2].set_ylabel('Score')
                axes[2, 2].set_xticks(range(len(metric_means)))
                axes[2, 2].set_xticklabels([col.replace('_', ' ').title() for col in metric_means.index], 
                                         rotation=45, ha='right')
            
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
        output_file = self.results_dir / "bias_correction_dashboard.png"
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Dashboard saved to: {output_file}")
    
    def generate_bias_correction_report(self, results_df: pd.DataFrame):
        """Generate comprehensive bias correction report"""
        if results_df.empty:
            self.logger.warning("No data available for report generation!")
            return
        
        # Filter successful analyses
        successful_df = results_df[~results_df.get('analysis_failed', False)]
        
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
        report_file = self.results_dir / "bias_correction_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("YOPD BIAS FIELD CORRECTION ANALYSIS REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Subjects Analyzed: {len(successful_df)}\n")
            if len(results_df) > len(successful_df):
                f.write(f"Failed Analyses: {len(results_df) - len(successful_df)}\n")
            f.write("\n")
            
            f.write("BIAS CORRECTION EFFECTIVENESS SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Key metrics
            key_metrics = {
                'improvement_percentage': 'Uniformity Improvement (%)',
                'structural_similarity': 'Structural Similarity (SSIM)',
                'bias_correction_score': 'Overall Correction Score (0-10)',
                'orig_uniformity': 'Original Uniformity',
                'corrected_uniformity': 'Corrected Uniformity'
            }
            
            for metric_key, metric_name in key_metrics.items():
                if metric_key in summary_stats:
                    stats = summary_stats[metric_key]
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
            
            # Effectiveness analysis
            if 'correction_effective' in successful_df.columns:
                effective_count = successful_df['correction_effective'].sum()
                total_count = len(successful_df)
                f.write("CORRECTION EFFECTIVENESS\n")
                f.write("-" * 25 + "\n")
                f.write(f"Effective Corrections: {effective_count}/{total_count} ({effective_count/total_count*100:.1f}%)\n")
                f.write(f"Ineffective Corrections: {total_count-effective_count}/{total_count} ({(total_count-effective_count)/total_count*100:.1f}%)\n\n")
            
            # Quality assessment
            f.write("QUALITY ASSESSMENT\n")
            f.write("-" * 18 + "\n")
            
            if 'improvement_percentage' in summary_stats:
                mean_improvement = summary_stats['improvement_percentage']['mean']
                if mean_improvement > 10:
                    f.write(f"✓ Good uniformity improvement (mean={mean_improvement:.1f}%)\n")
                elif mean_improvement > 5:
                    f.write(f"⚠ Moderate uniformity improvement (mean={mean_improvement:.1f}%)\n")
                else:
                    f.write(f"⚠ Limited uniformity improvement (mean={mean_improvement:.1f}%)\n")
            
            if 'structural_similarity' in summary_stats:
                mean_ssim = summary_stats['structural_similarity']['mean']
                if mean_ssim > 0.95:
                    f.write(f"✓ Excellent structural preservation (SSIM={mean_ssim:.3f})\n")
                elif mean_ssim > 0.90:
                    f.write(f"✓ Good structural preservation (SSIM={mean_ssim:.3f})\n")
                else:
                    f.write(f"⚠ Structural preservation needs attention (SSIM={mean_ssim:.3f})\n")
            
            if 'bias_correction_score' in summary_stats:
                mean_score = summary_stats['bias_correction_score']['mean']
                if mean_score >= 7:
                    f.write(f"✓ High quality bias correction (score={mean_score:.1f}/10)\n")
                elif mean_score >= 5:
                    f.write(f"⚠ Moderate quality bias correction (score={mean_score:.1f}/10)\n")
                else:
                    f.write(f"⚠ Poor quality bias correction (score={mean_score:.1f}/10)\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if 'improvement_percentage' in summary_stats and summary_stats['improvement_percentage']['mean'] < 5:
                f.write("• Consider adjusting bias correction parameters:\n")
                f.write("  - Increase number of iterations\n")
                f.write("  - Adjust convergence threshold\n")
                f.write("  - Review B-spline grid resolution\n\n")
            
            if 'structural_similarity' in summary_stats and summary_stats['structural_similarity']['mean'] < 0.90:
                f.write("• Review structural preservation:\n")
                f.write("  - Check for over-correction\n")
                f.write("  - Validate brain extraction quality\n")
                f.write("  - Consider alternative bias correction methods\n\n")
            
            if 'correction_effective' in successful_df.columns:
                ineffective_rate = (len(successful_df) - successful_df['correction_effective'].sum()) / len(successful_df)
                if ineffective_rate > 0.2:  # > 20% ineffective
                    f.write("• High rate of ineffective corrections detected:\n")
                    f.write("  - Review input image quality\n")
                    f.write("  - Validate preprocessing steps\n")
                    f.write("  - Consider subject-specific parameters\n\n")
            
            f.write("TECHNICAL NOTES\n")
            f.write("-" * 15 + "\n")
            f.write("• Uniformity measured using Median Absolute Deviation (MAD)\n")
            f.write("• Structural similarity calculated using SSIM index\n")
            f.write("• Correction score combines uniformity improvement, structural preservation, and final uniformity\n")
            f.write("• Effectiveness threshold set at score ≥ 6/10\n")
        
        # Export JSON summary
        json_stats = {}
        for key, value in summary_stats.items():
            json_stats[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for k, v in value.items()}
        
        json_file = self.results_dir / "bias_correction_summary.json"
        with open(json_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_file}")
        self.logger.info(f"JSON summary saved to: {json_file}")

def main():
    """Main function to run bias field correction analysis"""
    print("YOPD Bias Field Correction Analysis")
    print("=" * 45)
    
    # Initialize configuration
    config = BiasAnalysisConfig()
    
    # Verify output directory exists
    if not Path(config.output_root).exists():
        print(f"Error: Output directory not found: {config.output_root}")
        print("Please ensure the YOPD pipeline has been run first.")
        return
    
    # Initialize analyzer
    analyzer = BiasFieldAnalyzer(config)
    
    try:
        # Analyze all subjects
        print("Starting bias field correction analysis...")
        results_df = analyzer.analyze_all_subjects()
        
        if results_df.empty:
            print("No image pairs found for analysis!")
            return
        
        print(f"Successfully analyzed {len(results_df)} subjects")
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.create_bias_correction_visualizations(results_df)
        
        # Generate report
        print("Generating comprehensive report...")
        analyzer.generate_bias_correction_report(results_df)
        
        print("\nBias Field Correction Analysis Complete!")
        print(f"Results saved to: {analyzer.results_dir}")
        print("\nGenerated files:")
        print("  - bias_correction_analysis.csv: Detailed metrics for each subject")
        print("  - bias_correction_dashboard.png: Comprehensive visualization dashboard")
        print("  - bias_correction_report.txt: Summary report with recommendations")
        print("  - bias_correction_summary.json: Machine-readable summary statistics")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        analyzer.logger.error(f"Analysis failed: {e}")
        return

