#!/usr/bin/env python3
"""
Quick test script to validate the enhanced image quality assessment system
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import EnhancedQualityAssessment
import numpy as np

def main():
    print("🧠 Quick Test: Enhanced Image Quality Assessment")
    print("=" * 60)
    
    # Create a simple test analyzer
    analyzer = EnhancedQualityAssessment()
    
    print(f"✅ Analyzer created successfully")
    print(f"📋 Configuration: {analyzer.config}")
    print()
    
    # Create simple synthetic data
    print("🔬 Creating synthetic brain data...")
    data = np.zeros((64, 64, 64))
    
    # Create a simple brain-like structure
    center = (32, 32, 32)
    radius = 20
    
    # Add brain tissue
    for i in range(64):
        for j in range(64):
            for k in range(64):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < radius:
                    data[i, j, k] = 1000 + np.random.normal(0, 50)  # Brain with noise
                else:
                    data[i, j, k] = np.random.normal(0, 10)  # Background noise
    
    print(f"📊 Synthetic data created: shape={data.shape}, range={data.min():.1f}-{data.max():.1f}")
    print()
    
    # Save as temporary NIfTI file
    import nibabel as nib
    import tempfile
    
    temp_file = os.path.join(tempfile.gettempdir(), "quick_test.nii.gz")
    nii_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii_img, temp_file)
    
    print(f"💾 Temporary file saved: {temp_file}")
    print()
    
    # Run analysis
    print("🔍 Running quality analysis...")
    
    try:
        results = analyzer.analyze_single_image(temp_file)
        
        print("✅ Analysis completed successfully!")
        print()
        print("📈 Quality Metrics:")
        print(f"   • SNR: {results['stage_4_snr_analysis']['snr_value']:.2f}")
        print(f"   • CNR: {results['stage_4_cnr_analysis']['cnr_value']:.2f}")
        print(f"   • Uniformity: {results['stage_4_uniformity_analysis']['uniformity_mad_based']:.3f}")
        print(f"   • Brain Volume: {results['stage_2_brain_extraction']['brain_volume_voxels']:,} voxels")
        print()
        print(f"🎯 Overall Quality Good: {results['stage_5_quality_summary']['overall_quality_good']}")
        print(f"📊 Quality Score: {results['stage_5_quality_summary']['quality_score']:.1f}/10.0")
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        # Clean up on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

if __name__ == "__main__":
    success = main()
    print()
    if success:
        print("🎉 Quick test PASSED - System is ready!")
    else:
        print("💥 Quick test FAILED - Check system")
