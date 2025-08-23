#!/usr/bin/env python3
"""
Enhanced Image Quality Assessment - Main Analysis Script
=======================================================

This is the main script for running enhanced image quality assessment
on your MRI data. It provides comprehensive, step-by-step analysis with
detailed technical reporting.

Usage:
    python enhanced_quality_assessment.py [image_path]
    python enhanced_quality_assessment.py --batch [directory_path]
    python enhanced_quality_assessment.py --config [config_file]

Features:
- Stage-by-stage analysis (raw → brain extraction → quality metrics)
- Multiple brain extraction methods (threshold, BET-style)
- Multiple noise estimation methods (edge regions, MAD, background ROI)
- Comprehensive technical documentation
- JSON output for detailed analysis
- CSV summary for quick overview
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Enhanced Image Quality Assessment for MRI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single image
    python enhanced_quality_assessment.py /path/to/image.nii.gz
    
    # Batch analyze directory
    python enhanced_quality_assessment.py --batch /path/to/preprocessed/
    
    # Use custom configuration
    python enhanced_quality_assessment.py --config my_config.json image.nii.gz
    
    # Use specific methods
    python enhanced_quality_assessment.py --brain-method bet_style --noise-method mad image.nii.gz
        """
    )
    
    parser.add_argument('image_path', nargs='?', help='Path to single image file')
    parser.add_argument('--batch', '-b', help='Directory containing images to analyze')
    parser.add_argument('--output', '-o', default='./image_analysis_outputs', 
                       help='Output directory (default: ./image_analysis_outputs)')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    
    # Method selection
    parser.add_argument('--brain-method', choices=['threshold_based', 'bet_style'],
                       default='threshold_based', help='Brain extraction method')
    parser.add_argument('--noise-method', choices=['edge_regions', 'mad', 'background_roi'],
                       default='edge_regions', help='Noise estimation method')
    
    # Quality thresholds
    parser.add_argument('--min-snr', type=float, default=10.0, help='Minimum SNR threshold')
    parser.add_argument('--max-uniformity', type=float, default=0.3, help='Maximum uniformity threshold')
    parser.add_argument('--min-volume', type=int, default=800000, help='Minimum brain volume (voxels)')
    
    # Processing options
    parser.add_argument('--compare-methods', action='store_true', 
                       help='Compare different noise estimation methods')
    parser.add_argument('--save-masks', action='store_true', 
                       help='Save brain extraction masks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser

def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config_data
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return {}

def create_config_from_args(args) -> 'QualityConfig':
    """Create configuration from command line arguments"""
    from modules.config import QualityConfig
    
    # Load base config from file if specified
    config_data = {}
    if args.config:
        config_data = load_config_from_file(args.config)
    
    # Override with command line arguments
    config = QualityConfig(
        output_root=args.output,
        brain_extraction_method=args.brain_method,
        noise_estimation_method=args.noise_method,
        min_snr=args.min_snr,
        max_intensity_nonuniformity=args.max_uniformity,
        min_brain_volume=args.min_volume
    )
    
    # Apply any config file overrides
    for key, value in config_data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def find_images_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """Find APPROPRIATE image files in directory (T1w only, excludes masks/segmentations)"""
    if extensions is None:
        extensions = ['.nii', '.nii.gz', '.img', '.hdr']
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find all image files first
    all_image_files = []
    for ext in extensions:
        all_image_files.extend(list(directory.rglob(f'*{ext}')))
    
    # File patterns to EXCLUDE (masks, segmentations, etc.)
    exclude_patterns = [
        "_mask.nii.gz",         # Any mask files
        "_CSF_",                # CSF segmentation
        "_GM_",                 # Gray matter segmentation  
        "_WM_",                 # White matter segmentation
        "_seg.nii.gz",          # Segmentation files
        "_prob_",               # Probability maps
        "_space-",              # Template space files
        "_desc-",               # Specific preprocessing outputs
        "jacobian",             # Jacobian maps
        "warp",                 # Warp fields
    ]
    
    # Filter out inappropriate files
    appropriate_files = []
    excluded_files = []
    
    for file_path in all_image_files:
        filename = file_path.name
        
        # Check if should be excluded
        should_exclude = any(pattern in filename for pattern in exclude_patterns)
        
        if should_exclude:
            excluded_files.append(file_path)
        elif "T1w" in filename:  # Keep T1w files
            appropriate_files.append(file_path)
        else:
            excluded_files.append(file_path)  # Exclude others by default
    
    # Sort for consistent processing order
    appropriate_files.sort()
    
    print(f"Found {len(all_image_files)} total image files in {directory}")
    print(f"Appropriate for analysis: {len(appropriate_files)} (T1w images)")
    print(f"Excluded: {len(excluded_files)} (masks/segmentations)")
    
    if len(excluded_files) > 0:
        print(f"\nSample excluded files:")
        for f in excluded_files[:5]:
            print(f"  ✗ {f.name}")
        if len(excluded_files) > 5:
            print(f"  ... and {len(excluded_files) - 5} more excluded files")
    
    return [str(f) for f in appropriate_files]

def analyze_single_image(image_path: str, config: 'QualityConfig', args) -> Dict:
    """Analyze a single image"""
    from modules.main_analyzer import EnhancedQualityAssessment
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Initialize analyzer
    analyzer = EnhancedQualityAssessment(config)
    
    # Enable method comparison if requested
    if args.compare_methods:
        analyzer._compare_methods = True
    
    # Run comprehensive analysis
    results = analyzer.analyze_single_image(image_path)
    
    # Save results
    output_dir = Path(config.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON results
    json_file = analyzer.save_results(results, str(output_dir), "detailed_analysis")
    
    # Save brain mask if requested
    if args.save_masks and 'analysis_failed' not in results:
        try:
            # This would require nibabel to save the mask
            # For now, just indicate where it would be saved
            mask_file = output_dir / f"{results.get('subject_id', 'unknown')}_brain_mask.nii.gz"
            print(f"  Brain mask would be saved to: {mask_file}")
        except Exception as e:
            print(f"  Could not save brain mask: {e}")
    
    return results

def create_summary_csv(all_results: List[Dict], output_dir: str) -> str:
    """Create CSV summary of all results"""
    summary_data = []
    
    for result in all_results:
        if 'analysis_failed' in result:
            # Failed analysis
            summary_row = {
                'subject_id': result.get('subject_id', 'unknown'),
                'file_path': result.get('file_path', ''),
                'analysis_status': 'FAILED',
                'error_message': result.get('error', ''),
                'snr': np.nan,
                'cnr': np.nan,
                'uniformity': np.nan,
                'brain_volume_ml': np.nan,
                'quality_score': np.nan,
                'snr_adequate': False,
                'uniformity_good': False,
                'volume_adequate': False,
                'overall_quality_good': False
            }
        else:
            # Successful analysis
            summary = result.get('stage_5_quality_summary', {})
            summary_row = {
                'subject_id': result.get('subject_id', 'unknown'),
                'file_path': result.get('file_path', ''),
                'analysis_status': 'SUCCESS',
                'error_message': '',
                'snr': result.get('orig_snr', np.nan),
                'cnr': result.get('orig_cnr', np.nan),
                'uniformity': result.get('orig_uniformity', np.nan),
                'brain_volume_ml': result.get('brain_volume_ml', np.nan),
                'quality_score': summary.get('quality_score', np.nan),
                'snr_adequate': summary.get('snr_adequate', False),
                'uniformity_good': summary.get('uniformity_good', False),
                'volume_adequate': summary.get('volume_adequate', False),
                'overall_quality_good': summary.get('overall_quality_good', False),
                'brain_extraction_method': result.get('config_used', {}).get('brain_extraction_method', ''),
                'noise_estimation_method': result.get('config_used', {}).get('noise_estimation_method', ''),
                'analysis_timestamp': result.get('analysis_timestamp', '')
            }
        
        summary_data.append(summary_row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_file = Path(output_dir) / "quality_assessment_summary.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)

def print_batch_summary(all_results: List[Dict]):
    """Print summary of batch analysis"""
    total_images = len(all_results)
    successful_analyses = sum(1 for r in all_results if 'analysis_failed' not in r)
    failed_analyses = total_images - successful_analyses
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Success rate: {successful_analyses/total_images*100:.1f}%")
    
    if successful_analyses > 0:
        # Calculate quality statistics
        successful_results = [r for r in all_results if 'analysis_failed' not in r]
        
        snr_values = [r.get('orig_snr') for r in successful_results if not pd.isna(r.get('orig_snr'))]
        cnr_values = [r.get('orig_cnr') for r in successful_results if not pd.isna(r.get('orig_cnr'))]
        uniformity_values = [r.get('orig_uniformity') for r in successful_results if not pd.isna(r.get('orig_uniformity'))]
        volume_values = [r.get('brain_volume_ml') for r in successful_results if not pd.isna(r.get('brain_volume_ml'))]
        
        if snr_values:
            print(f"\nSNR Statistics:")
            print(f"  Mean: {np.mean(snr_values):.2f}")
            print(f"  Std:  {np.std(snr_values):.2f}")
            print(f"  Range: {np.min(snr_values):.2f} - {np.max(snr_values):.2f}")
        
        if cnr_values:
            print(f"\nCNR Statistics:")
            print(f"  Mean: {np.mean(cnr_values):.2f}")
            print(f"  Std:  {np.std(cnr_values):.2f}")
            print(f"  Range: {np.min(cnr_values):.2f} - {np.max(cnr_values):.2f}")
        
        if uniformity_values:
            print(f"\nUniformity Statistics:")
            print(f"  Mean: {np.mean(uniformity_values):.4f}")
            print(f"  Std:  {np.std(uniformity_values):.4f}")
            print(f"  Range: {np.min(uniformity_values):.4f} - {np.max(uniformity_values):.4f}")
        
        if volume_values:
            print(f"\nBrain Volume Statistics:")
            print(f"  Mean: {np.mean(volume_values):.1f} ml")
            print(f"  Std:  {np.std(volume_values):.1f} ml")
            print(f"  Range: {np.min(volume_values):.1f} - {np.max(volume_values):.1f} ml")
        
        # Quality flags summary
        quality_flags = [r.get('stage_5_quality_summary', {}) for r in successful_results]
        snr_adequate_count = sum(1 for f in quality_flags if f.get('snr_adequate', False))
        uniformity_good_count = sum(1 for f in quality_flags if f.get('uniformity_good', False))
        volume_adequate_count = sum(1 for f in quality_flags if f.get('volume_adequate', False))
        overall_good_count = sum(1 for f in quality_flags if f.get('overall_quality_good', False))
        
        print(f"\nQuality Flags Summary:")
        print(f"  SNR adequate: {snr_adequate_count}/{successful_analyses} ({snr_adequate_count/successful_analyses*100:.1f}%)")
        print(f"  Uniformity good: {uniformity_good_count}/{successful_analyses} ({uniformity_good_count/successful_analyses*100:.1f}%)")
        print(f"  Volume adequate: {volume_adequate_count}/{successful_analyses} ({volume_adequate_count/successful_analyses*100:.1f}%)")
        print(f"  Overall quality good: {overall_good_count}/{successful_analyses} ({overall_good_count/successful_analyses*100:.1f}%)")

def main():
    """Main function"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.batch:
        parser.error("Must specify either image_path or --batch directory")
    
    if args.image_path and args.batch:
        parser.error("Cannot specify both image_path and --batch directory")
    
    # Import numpy here to avoid issues if not available during argument parsing
    import numpy as np
    
    print("Enhanced Image Quality Assessment")
    print("=" * 50)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        print(f"Configuration created:")
        config_summary = config.get_config_summary()
        for key, value in config_summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Create output directory
        output_dir = Path(config.output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Determine images to process
        if args.batch:
            image_paths = find_images_in_directory(args.batch)
            if not image_paths:
                print("No image files found in specified directory")
                return 1
        else:
            image_paths = [args.image_path]
        
        print(f"Processing {len(image_paths)} image(s)")
        
        # Process images
        all_results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                result = analyze_single_image(image_path, config, args)
                all_results.append(result)
                
                # Print quick summary
                if 'analysis_failed' not in result:
                    summary = result.get('stage_5_quality_summary', {})
                    print(f"  ✓ Analysis completed - Quality: {'GOOD' if summary.get('overall_quality_good', False) else 'POOR'}")
                    print(f"    SNR: {result.get('orig_snr', 'N/A'):.2f}, Volume: {result.get('brain_volume_ml', 'N/A'):.1f} ml")
                else:
                    print(f"  ✗ Analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {image_path}: {e}")
                error_result = {
                    'file_path': image_path,
                    'subject_id': os.path.splitext(os.path.basename(image_path))[0],
                    'analysis_failed': True,
                    'error': str(e)
                }
                all_results.append(error_result)
        
        # Create summary CSV
        if all_results:
            csv_file = create_summary_csv(all_results, str(output_dir))
            print(f"\n✓ Summary CSV saved: {csv_file}")
        
        # Print batch summary if multiple images
        if len(image_paths) > 1:
            print_batch_summary(all_results)
        
        # Final summary
        successful_count = sum(1 for r in all_results if 'analysis_failed' not in r)
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {len(all_results)} images")
        print(f"Successful: {successful_count}")
        print(f"Failed: {len(all_results) - successful_count}")
        print(f"Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
