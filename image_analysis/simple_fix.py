#!/usr/bin/env python3
"""
Simple Analysis Issue Fix
=========================

This script creates a fixed version that handles the main issues:
1. Only analyzes T1w images (not masks)  
2. Provides better filtering
"""

import os
import sys
from pathlib import Path

def identify_appropriate_files(directory):
    """Identify which files should be analyzed"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return [], []
    
    # Find all .nii.gz files
    all_files = list(directory.rglob("*.nii.gz"))
    
    print(f"Total .nii.gz files found: {len(all_files)}")
    
    # File patterns to EXCLUDE (masks, segmentations, etc.)
    exclude_patterns = [
        "_mask.nii.gz",         # Any mask files
        "_CSF_",                # CSF segmentation
        "_GM_",                 # Gray matter segmentation  
        "_WM_",                 # White matter segmentation
        "_seg.nii.gz",          # Segmentation files
        "_prob_",               # Probability maps
    ]
    
    # Categorize files
    appropriate_files = []
    excluded_files = []
    
    for file_path in all_files:
        filename = file_path.name
        
        # Check if should be excluded
        should_exclude = any(pattern in filename for pattern in exclude_patterns)
        
        if should_exclude:
            excluded_files.append(file_path)
        elif "T1w" in filename:  # Keep T1w files
            appropriate_files.append(file_path)
        else:
            excluded_files.append(file_path)  # Exclude others by default
    
    print(f"\nFILE ANALYSIS:")
    print(f"Appropriate for analysis: {len(appropriate_files)}")
    print(f"Should be excluded: {len(excluded_files)}")
    
    print(f"\nSample APPROPRIATE files (T1w images):")
    for f in appropriate_files[:10]:
        print(f"  ✓ {f.name}")
    
    if len(excluded_files) > 0:
        print(f"\nSample EXCLUDED files (masks/segmentations):")
        for f in excluded_files[:10]:
            print(f"  ✗ {f.name}")
    
    return [str(f) for f in appropriate_files], [str(f) for f in excluded_files]

def create_filtered_file_list(directory, output_file="appropriate_files.txt"):
    """Create a text file with appropriate files for analysis"""
    
    appropriate, excluded = identify_appropriate_files(directory)
    
    # Write appropriate files to text file
    with open(output_file, 'w') as f:
        f.write("# Appropriate files for image quality analysis\n")
        f.write("# These are T1w images (excluding masks/segmentations)\n\n")
        for filepath in appropriate:
            f.write(filepath + "\n")
    
    print(f"\n✓ Saved {len(appropriate)} appropriate file paths to: {output_file}")
    
    return appropriate

def main():
    print("Image Analysis Issue Diagnosis")
    print("=" * 40)
    
    # Analyze the data directory
    data_dir = r"D:\data_NIMHANS\outputs\01_preprocessed"
    
    if os.path.exists(data_dir):
        print(f"\nAnalyzing directory: {data_dir}")
        
        appropriate_files = create_filtered_file_list(data_dir, "appropriate_T1w_files.txt")
        
        print(f"\n🎯 SOLUTION:")
        print("1. Your original command was processing ALL .nii.gz files including:")
        print("   - CSF masks (binary 0-1 values)")  
        print("   - GM masks (binary 0-1 values)")
        print("   - WM masks (binary 0-1 values)")
        print("   - Other segmentation files")
        
        print(f"\n2. Binary mask files give SNR=0 because:")
        print("   - They only contain values 0 and 1")
        print("   - Noise estimation = 0 (no variation)")
        print("   - SNR = signal/0 = 0")
        
        print(f"\n3. Only {len(appropriate_files)} files are appropriate for quality analysis")
        
        if len(appropriate_files) > 0:
            print(f"\n✅ QUICK FIX - Run this command instead:")
            print(f'python enhanced_quality_assessment.py --batch "D:\\data_NIMHANS\\outputs\\01_preprocessed" --output "quality_analysis_T1w_only"')
            print(f"\nBut first, modify the enhanced script to filter files properly.")
            
            print(f"\n🔧 MANUAL SOLUTION:")
            print(f"1. Edit enhanced_quality_assessment.py")
            print(f"2. In the find_images_in_directory function, add file filtering")
            print(f"3. Or process individual T1w files:")
            
            for i, filepath in enumerate(appropriate_files[:3], 1):
                print(f'   {i}. python enhanced_quality_assessment.py "{filepath}"')
            
            if len(appropriate_files) > 3:
                print(f"   ... and {len(appropriate_files)-3} more T1w files")
        
        else:
            print(f"\n❌ NO APPROPRIATE FILES FOUND")
            print("Check if T1w images exist or have different naming patterns")
            
    else:
        print(f"Data directory not found: {data_dir}")

if __name__ == "__main__":
    main()
