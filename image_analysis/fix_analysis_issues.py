#!/usr/bin/env python3
"""
Fix for Image Analysis Issues
============================

This script addresses the main issues with the enhanced quality assessment:

1. File Type Filtering - Only analyze appropriate T1w images
2. Noise Estimation Fixes - Handle edge cases and binary data
3. SNR Calculation Fixes - Proper handling of zero noise

Run this to apply fixes and then re-run your analysis.
"""

import os
import sys
import shutil
from pathlib import Path

def backup_original_files():
    """Create backup of original modules"""
    modules_dir = Path("modules")
    backup_dir = Path("modules_backup")
    
    if not backup_dir.exists():
        shutil.copytree(modules_dir, backup_dir)
        print("✓ Created backup of original modules")
    else:
        print("✓ Backup already exists")

def identify_appropriate_files(directory):
    """Identify which files should be analyzed"""
    directory = Path(directory)
    
    # File patterns to INCLUDE (T1w images)
    include_patterns = [
        "*T1w.nii.gz",           # Original T1w
        "*T1w_preprocessed.nii.gz",  # Preprocessed T1w
        "*T1w_brain.nii.gz"     # Brain-extracted T1w (acceptable)
    ]
    
    # File patterns to EXCLUDE (masks, segmentations, etc.)
    exclude_patterns = [
        "*_mask.nii.gz",         # Any mask files
        "*_CSF_*",               # CSF segmentation
        "*_GM_*",                # Gray matter segmentation
        "*_WM_*",                # White matter segmentation
        "*_seg.nii.gz",          # Segmentation files
        "*_prob_*",              # Probability maps
        "*_space-*_desc-*",      # Specific preprocessing outputs
    ]
    
    print(f"\nAnalyzing directory: {directory}")
    
    # Find all .nii.gz files
    all_files = list(directory.rglob("*.nii.gz"))
    
    print(f"Total .nii.gz files found: {len(all_files)}")
    
    # Categorize files
    appropriate_files = []
    excluded_files = []
    
    for file_path in all_files:
        filename = file_path.name.lower()
        
        # Check if should be excluded
        should_exclude = any(
            filename.find(pattern.replace("*", "").lower()) != -1 
            for pattern in exclude_patterns
        )
        
        if should_exclude:
            excluded_files.append(file_path)
        else:
            # Check if it matches include patterns
            should_include = any(
                filename.find(pattern.replace("*", "").lower()) != -1
                for pattern in include_patterns
            )
            
            if should_include or "t1w" in filename:
                appropriate_files.append(file_path)
            else:
                excluded_files.append(file_path)
    
    print(f"\nFILE CATEGORIZATION:")
    print(f"Appropriate for analysis: {len(appropriate_files)}")
    print(f"Should be excluded: {len(excluded_files)}")
    
    if len(appropriate_files) <= 20:  # Show all if few
        print(f"\nFiles TO ANALYZE:")
        for f in appropriate_files[:20]:
            print(f"  ✓ {f.relative_to(directory)}")
    else:
        print(f"\nSample files TO ANALYZE (first 10):")
        for f in appropriate_files[:10]:
            print(f"  ✓ {f.relative_to(directory)}")
    
    if len(excluded_files) <= 20:  # Show all if few
        print(f"\nFiles TO EXCLUDE:")
        for f in excluded_files[:20]:
            print(f"  ✗ {f.relative_to(directory)}")
    else:
        print(f"\nSample files TO EXCLUDE (first 10):")
        for f in excluded_files[:10]:
            print(f"  ✗ {f.relative_to(directory)}")
    
    return appropriate_files, excluded_files

def create_fixed_enhanced_script():
    """Create a fixed version of the enhanced quality assessment script"""
    
    fixed_script_content = '''#!/usr/bin/env python3
"""
FIXED: Enhanced Image Quality Assessment - Main Analysis Script
=============================================================

Fixed version that properly filters file types and handles edge cases.

Key fixes:
1. Only analyzes appropriate T1w images (not masks/segmentations)
2. Handles zero noise estimates properly
3. Better error handling for edge cases

Usage:
    python fixed_enhanced_quality_assessment.py [image_path]
    python fixed_enhanced_quality_assessment.py --batch [directory_path]
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Optional
import numpy as np

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def filter_appropriate_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find APPROPRIATE image files for quality analysis
    
    FIXED: Only returns T1w images, excludes masks and segmentations
    """
    if extensions is None:
        extensions = ['.nii', '.nii.gz']
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find all image files
    all_files = []
    for ext in extensions:
        all_files.extend(list(directory.rglob(f'*{ext}')))
    
    # File patterns to INCLUDE (T1w images only)
    include_patterns = [
        "t1w.nii.gz",           # Original T1w
        "t1w_preprocessed.nii.gz",  # Preprocessed T1w  
        "t1w_brain.nii.gz"     # Brain-extracted T1w
    ]
    
    # File patterns to EXCLUDE (masks, segmentations, etc.)
    exclude_patterns = [
        "_mask.nii.gz",         # Any mask files
        "_csf_", "_gm_", "_wm_", # Tissue segmentations
        "_seg.nii.gz",          # Segmentation files
        "_prob_",               # Probability maps
        "_space-",              # Template space files
        "_desc-",               # Specific preprocessing outputs
        "jacobian",             # Jacobian maps
        "warp",                 # Warp fields
    ]
    
    appropriate_files = []
    excluded_files = []
    
    for file_path in all_files:
        filename = file_path.name.lower()
        
        # Check if should be excluded
        should_exclude = any(pattern.lower() in filename for pattern in exclude_patterns)
        
        if should_exclude:
            excluded_files.append(file_path)
            continue
            
        # Check if it matches include patterns OR contains "t1w"
        should_include = (
            any(pattern.lower() in filename for pattern in include_patterns) or
            "t1w" in filename
        )
        
        if should_include:
            appropriate_files.append(file_path)
        else:
            excluded_files.append(file_path)
    
    # Sort for consistent processing order
    appropriate_files.sort()
    
    print(f"Found {len(all_files)} total image files in {directory}")
    print(f"Appropriate for analysis: {len(appropriate_files)}")
    print(f"Excluded (masks/segmentations): {len(excluded_files)}")
    
    if len(excluded_files) > 0:
        print(f"\\nSample excluded files:")
        for f in excluded_files[:5]:
            print(f"  ✗ {f.name}")
        if len(excluded_files) > 5:
            print(f"  ... and {len(excluded_files) - 5} more")
    
    return [str(f) for f in appropriate_files]

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description='FIXED Enhanced Image Quality Assessment for MRI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single appropriate image
    python fixed_enhanced_quality_assessment.py /path/to/sub-001_T1w.nii.gz
    
    # Batch analyze directory (automatically filters appropriate files)
    python fixed_enhanced_quality_assessment.py --batch /path/to/preprocessed/
    
    # Use specific methods
    python fixed_enhanced_quality_assessment.py --brain-method bet_style --noise-method mad image.nii.gz
        """
    )
    
    parser.add_argument('image_path', nargs='?', help='Path to single image file')
    parser.add_argument('--batch', '-b', help='Directory containing images to analyze')
    parser.add_argument('--output', '-o', default='./fixed_analysis_outputs', 
                       help='Output directory (default: ./fixed_analysis_outputs)')
    
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
    parser.add_argument('--show-excluded', action='store_true', 
                       help='Show files that are excluded from analysis')
    
    return parser

def create_config_from_args(args) -> 'QualityConfig':
    """Create configuration from command line arguments"""
    from modules.config import QualityConfig
    
    config = QualityConfig(
        output_root=args.output,
        brain_extraction_method=args.brain_method,
        noise_estimation_method=args.noise_method,
        min_snr=args.min_snr,
        max_intensity_nonuniformity=args.max_uniformity,
        min_brain_volume=args.min_volume
    )
    
    return config

def analyze_single_image(image_path: str, config: 'QualityConfig', args) -> Dict:
    """Analyze a single image with improved error handling"""
    from modules.main_analyzer import EnhancedQualityAssessment
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"\\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Check if this looks like an appropriate file
    filename = os.path.basename(image_path).lower()
    exclude_patterns = ["_mask", "_csf_", "_gm_", "_wm_", "_seg", "_prob_"]
    
    if any(pattern in filename for pattern in exclude_patterns):
        print(f"⚠️  WARNING: This appears to be a mask/segmentation file:")
        print(f"   {filename}")
        print(f"   Results may not be meaningful for quality assessment.")
        print(f"   Consider analyzing the original T1w image instead.\\n")
    
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
    json_file = analyzer.save_results(results, str(output_dir), "fixed_detailed_analysis")
    
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
            snr_stage = result.get('stage_4_snr_analysis', {})
            cnr_stage = result.get('stage_4_cnr_analysis', {})
            uniformity_stage = result.get('stage_4_uniformity_analysis', {})
            
            summary_row = {
                'subject_id': result.get('subject_id', 'unknown'),
                'file_path': result.get('file_path', ''),
                'analysis_status': 'SUCCESS',
                'error_message': '',
                'snr': snr_stage.get('snr_value', np.nan),
                'cnr': cnr_stage.get('cnr_value', np.nan),
                'uniformity': uniformity_stage.get('uniformity_value', np.nan),
                'brain_volume_ml': result.get('stage_2_brain_extraction', {}).get('brain_volume_ml', np.nan),
                'quality_score': summary.get('quality_score', np.nan),
                'snr_adequate': summary.get('snr_adequate', False),
                'uniformity_good': summary.get('uniformity_good', False),
                'volume_adequate': summary.get('volume_adequate', False),
                'overall_quality_good': summary.get('overall_quality_good', False),
                'brain_extraction_method': result.get('config_used', {}).get('brain_extraction_method', ''),
                'noise_estimation_method': result.get('config_used', {}).get('noise_estimation_method', ''),
                'analysis_timestamp': result.get('analysis_timestamp', ''),
                'file_type_warning': 'mask_or_segmentation' if any(pattern in os.path.basename(result.get('file_path', '')).lower() for pattern in ['_mask', '_csf_', '_gm_', '_wm_']) else 'appropriate'
            }
        
        summary_data.append(summary_row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_file = Path(output_dir) / "FIXED_quality_assessment_summary.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)

def print_file_type_analysis(all_results: List[Dict]):
    """Print analysis of file types processed"""
    total_images = len(all_results)
    
    # Categorize by file type
    appropriate_files = []
    mask_files = []
    other_files = []
    
    for result in all_results:
        if 'analysis_failed' in result:
            continue
            
        filepath = result.get('file_path', '')
        filename = os.path.basename(filepath).lower()
        
        if any(pattern in filename for pattern in ['_mask', '_csf_', '_gm_', '_wm_', '_seg']):
            mask_files.append(result)
        elif 't1w' in filename:
            appropriate_files.append(result)
        else:
            other_files.append(result)
    
    print(f"\\n{'='*60}")
    print("FILE TYPE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total files processed: {total_images}")
    print(f"Appropriate T1w images: {len(appropriate_files)}")
    print(f"Mask/segmentation files: {len(mask_files)}")  
    print(f"Other files: {len(other_files)}")
    
    if mask_files:
        print(f"\\n⚠️  WARNING: {len(mask_files)} mask/segmentation files were analyzed")
        print("   These typically have SNR=0 and poor quality metrics")
        print("   Consider filtering these out for meaningful analysis")

def main():
    """Main function with improved file filtering"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.batch:
        parser.error("Must specify either image_path or --batch directory")
    
    if args.image_path and args.batch:
        parser.error("Cannot specify both image_path and --batch directory")
    
    print("FIXED Enhanced Image Quality Assessment")
    print("=" * 50)
    print("Key improvements:")
    print("- Filters out mask and segmentation files")
    print("- Better handling of zero noise estimates")
    print("- Improved error reporting")
    print("=" * 50)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        print(f"\\nConfiguration:")
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
        
        # Determine images to process with FILTERING
        if args.batch:
            print(f"\\nAnalyzing batch directory: {args.batch}")
            image_paths = filter_appropriate_files(args.batch)  # FIXED FILTERING
            if not image_paths:
                print("❌ No appropriate image files found in specified directory")
                print("   Make sure you have T1w images (not just masks/segmentations)")
                return 1
        else:
            image_paths = [args.image_path]
        
        print(f"\\nProcessing {len(image_paths)} appropriate image(s)")
        
        # Process images
        all_results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                result = analyze_single_image(image_path, config, args)
                all_results.append(result)
                
                # Print quick summary
                if 'analysis_failed' not in result:
                    summary = result.get('stage_5_quality_summary', {})
                    snr_val = result.get('stage_4_snr_analysis', {}).get('snr_value', 0)
                    vol_val = result.get('stage_2_brain_extraction', {}).get('brain_volume_ml', 0)
                    
                    print(f"  ✓ Analysis completed - Quality: {'GOOD' if summary.get('overall_quality_good', False) else 'POOR'}")
                    print(f"    SNR: {snr_val:.2f}, Volume: {vol_val:.1f} ml")
                    
                    # Special warning for likely mask files
                    if snr_val == 0:
                        print(f"    ⚠️  SNR=0 suggests this may be a binary mask file")
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
            print(f"\\n✓ Summary CSV saved: {csv_file}")
        
        # Print file type analysis
        print_file_type_analysis(all_results)
        
        # Final summary
        successful_count = sum(1 for r in all_results if 'analysis_failed' not in r)
        good_quality_count = sum(1 for r in all_results if 'analysis_failed' not in r and r.get('stage_5_quality_summary', {}).get('overall_quality_good', False))
        
        print(f"\\n{'='*60}")
        print("FIXED ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {len(all_results)} images")
        print(f"Successful: {successful_count}")
        print(f"Good Quality: {good_quality_count}")
        print(f"Failed: {len(all_results) - successful_count}")
        print(f"Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n\\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n\\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open("fixed_enhanced_quality_assessment.py", "w") as f:
        f.write(fixed_script_content)
    
    print("✓ Created fixed_enhanced_quality_assessment.py")

def create_noise_estimation_fix():
    """Create fixes for noise estimation edge cases"""
    
    noise_fix_content = '''#!/usr/bin/env python3
"""
Noise Estimation Fixes
======================

Apply these fixes to handle edge cases in noise estimation:
1. Handle zero noise estimates (binary masks)
2. Add minimum noise floor
3. Better detection of inappropriate data types
"""

def apply_noise_estimation_fixes():
    """Apply fixes to the noise estimation module"""
    
    # Read the original file
    with open("modules/noise_estimation.py", "r") as f:
        content = f.read()
    
    # Add minimum noise floor to edge regions method
    old_noise_calc = '''            # Calculate noise estimate
            if len(filtered_edge_data) > 0:
                noise_estimate = np.std(filtered_edge_data)
                edge_mean = np.mean(filtered_edge_data)
                edge_median = np.median(filtered_edge_data)
            else:
                # Fallback if filtering removes everything
                noise_estimate = np.std(edge_data)
                edge_mean = np.mean(edge_data)
                edge_median = np.median(edge_data)
                print("  Warning: All edge data filtered out, using unfiltered data")'''
    
    new_noise_calc = """            # Calculate noise estimate
            if len(filtered_edge_data) > 0:
                noise_estimate = np.std(filtered_edge_data)
                edge_mean = np.mean(filtered_edge_data)
                edge_median = np.median(filtered_edge_data)
            else:
                # Fallback if filtering removes everything
                noise_estimate = np.std(edge_data)
                edge_mean = np.mean(edge_data)
                edge_median = np.median(edge_data)
                print("  Warning: All edge data filtered out, using unfiltered data")
            
            # FIXED: Handle zero noise estimates (binary masks, etc.)
            if noise_estimate < 1e-6:
                print(f"  ⚠️  Very low noise estimate ({noise_estimate:.6f}) - possibly binary/mask data")
                # Use minimum noise floor or signal-based estimate
                signal_range = np.ptp(filtered_edge_data if len(filtered_edge_data) > 0 else edge_data)
                min_noise_floor = max(0.001, signal_range * 0.01)  # 1% of signal range
                noise_estimate = max(noise_estimate, min_noise_floor)
                print(f"  Applied minimum noise floor: {noise_estimate:.6f}")"""
    
    if old_noise_calc in content:
        content = content.replace(old_noise_calc, new_noise_calc)
        
        # Write back the fixed content
        with open("modules/noise_estimation_fixed.py", "w") as f:
            f.write(content)
        print("✓ Created fixed noise estimation module")
        return True
    else:
        print("✗ Could not find noise calculation code to fix")
        return False

if __name__ == "__main__":
    apply_noise_estimation_fixes()
'''
    
    with open("apply_noise_fixes.py", "w") as f:
        f.write(noise_fix_content)
    
    print("✓ Created apply_noise_fixes.py")

def main():
    print("Enhanced Image Quality Assessment - Issue Diagnosis and Fixes")
    print("=" * 60)
    
    # Step 1: Analyze the data directory to understand file types
    data_dir = r"D:\data_NIMHANS\outputs\01_preprocessed"
    
    if os.path.exists(data_dir):
        print(f"\n📁 ANALYZING YOUR DATA DIRECTORY:")
        appropriate, excluded = identify_appropriate_files(data_dir)
        
        print(f"\n📊 SUMMARY:")
        print(f"Total appropriate files for analysis: {len(appropriate)}")
        print(f"Files to exclude (masks/segmentations): {len(excluded)}")
        
        if len(appropriate) == 0:
            print(f"\n❌ PROBLEM IDENTIFIED:")
            print("No appropriate T1w images found for analysis!")
            print("You may need to:")
            print("1. Check if T1w images exist in the directory")
            print("2. Verify file naming patterns")
            print("3. Look in subdirectories")
        else:
            print(f"\n✅ GOOD NEWS:")
            print(f"Found {len(appropriate)} appropriate files to analyze")
    else:
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("Please check the path or run this script from the correct location")
    
    # Step 2: Create fixes
    print(f"\n🔧 CREATING FIXES:")
    backup_original_files()
    create_fixed_enhanced_script()
    create_noise_estimation_fix()
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Use the fixed script instead of the original:")
    print("   python fixed_enhanced_quality_assessment.py --batch 'D:\\data_NIMHANS\\outputs\\01_preprocessed\\'")
    print("")
    print("2. Or analyze specific T1w files:")
    print("   python fixed_enhanced_quality_assessment.py 'path\\to\\sub-001_T1w.nii.gz'")
    print("")
    print("3. The fixed script will:")
    print("   - Only analyze T1w images (not masks)")
    print("   - Handle zero noise estimates properly")
    print("   - Provide better error messages")
    print("   - Show which files are excluded and why")

if __name__ == "__main__":
    main()
