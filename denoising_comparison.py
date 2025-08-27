#!/usr/bin/env python3
"""
Figure 2: Comparison of Brain MRI Slice Before and After Denoising
=================================================================
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

def load_nifti_data(filepath):
    """Load NIfTI data and return the image array."""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        return data, img.header
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def find_middle_slice_with_brain(data, axis=2):
    """Find a representative axial slice with brain tissue."""
    # Get the middle slice along the specified axis
    middle_idx = data.shape[axis] // 2
    
    # Look for slices around the middle with sufficient brain tissue
    for offset in range(0, 20):
        for direction in [0, 1, -1]:  # Try middle first, then above and below
            if direction == 0:
                slice_idx = middle_idx
            else:
                slice_idx = middle_idx + direction * offset
                
            if 0 <= slice_idx < data.shape[axis]:
                if axis == 2:  # axial
                    slice_data = data[:, :, slice_idx]
                elif axis == 1:  # coronal
                    slice_data = data[:, slice_idx, :]
                else:  # sagittal
                    slice_data = data[slice_idx, :, :]
                
                # Check if slice has sufficient brain tissue (non-zero values)
                non_zero_ratio = np.count_nonzero(slice_data) / slice_data.size
                if non_zero_ratio > 0.1:  # At least 10% non-zero
                    return slice_idx
    
    # Fallback to middle slice
    return middle_idx

def normalize_image(data, percentile=99):
    """Normalize image data to 0-1 range using percentile scaling."""
    # Remove zeros for percentile calculation
    non_zero_data = data[data > 0]
    if len(non_zero_data) == 0:
        return data
    
    # Calculate percentile-based normalization
    min_val = np.percentile(non_zero_data, 1)
    max_val = np.percentile(non_zero_data, percentile)
    
    # Normalize
    normalized = np.clip((data - min_val) / (max_val - min_val), 0, 1)
    return normalized

def create_denoising_comparison_figure():
    """Create Figure 2: Before and After Denoising Comparison."""
    
    # Define data paths
    base_dir = "D:/data_NIMHANS"
    denoised_dir = "D:/data_NIMHANS/t1w_denoised"
    
    # Representative subjects from each group
    subjects = {
        'HC': 'sub-YLOPDHC01',
        'PIGD': 'sub-YLOPD109', 
        'TDPD': 'sub-YLOPD100'
    }
    
    # Set up the figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.15)
    
    # Create subplots for each group
    for row, (group, subject) in enumerate(subjects.items()):
        print(f"\nProcessing {group}: {subject}")
        
        # Define file paths
        original_path = os.path.join(base_dir, group, subject, "ses-01", "anat", 
                                   f"{subject}_ses-01_run-1_T1w.nii.gz")
        denoised_path = os.path.join(denoised_dir, group, 
                                   f"{subject}_ses-01_run-1_T1w.nii.gz_T1w_denoised.nii.gz")
        
        # Check if files exist
        if not os.path.exists(original_path):
            print(f"Original file not found: {original_path}")
            continue
        if not os.path.exists(denoised_path):
            print(f"Denoised file not found: {denoised_path}")
            continue
        
        # Load the data
        print(f"Loading original: {original_path}")
        original_data, original_header = load_nifti_data(original_path)
        print(f"Loading denoised: {denoised_path}")
        denoised_data, denoised_header = load_nifti_data(denoised_path)
        
        if original_data is None or denoised_data is None:
            print(f"Failed to load data for {subject}")
            continue
        
        print(f"Original shape: {original_data.shape}, Denoised shape: {denoised_data.shape}")
        
        # Find a representative axial slice
        slice_idx = find_middle_slice_with_brain(original_data, axis=2)
        print(f"Using axial slice: {slice_idx}")
        
        # Extract the axial slices
        original_slice = original_data[:, :, slice_idx]
        denoised_slice = denoised_data[:, :, slice_idx]
        
        # Normalize the slices
        original_norm = normalize_image(original_slice)
        denoised_norm = normalize_image(denoised_slice)
        
        # Plot original image
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(np.rot90(original_norm), cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'{group}\nOriginal', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Plot denoised image
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(np.rot90(denoised_norm), cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'{group}\nDenoised', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Create difference image
        diff_image = np.abs(original_norm - denoised_norm)
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(np.rot90(diff_image), cmap='hot', vmin=0, vmax=0.3)
        ax3.set_title(f'{group}\nDifference', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Add colorbar for difference
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # ROI zoom (center region)
        h, w = original_norm.shape
        roi_size = min(h, w) // 3
        start_h, start_w = h//2 - roi_size//2, w//2 - roi_size//2
        
        # Extract ROI
        original_roi = original_norm[start_h:start_h+roi_size, start_w:start_w+roi_size]
        denoised_roi = denoised_norm[start_h:start_h+roi_size, start_w:start_w+roi_size]
        
        # Create side-by-side ROI comparison
        ax4 = fig.add_subplot(gs[row, 3])
        combined_roi = np.concatenate([original_roi, denoised_roi], axis=1)
        ax4.imshow(np.rot90(combined_roi), cmap='gray', vmin=0, vmax=1)
        ax4.set_title(f'{group}\nROI: Original | Denoised', fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # Add vertical line to separate ROIs
        ax4.axvline(x=roi_size/2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add main title
    fig.suptitle('Figure 2: Comparison of Brain MRI Slices Before and After Denoising\n' + 
                'Representative T1-weighted axial slices from each group',
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add group labels on the left
    fig.text(0.02, 0.83, 'Healthy Controls\n(HC)', fontsize=12, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.50, 'Postural Instability\nGait Difficulty\n(PIGD)', fontsize=12, 
             fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.02, 0.17, 'Tremor Dominant\nParkinson\'s Disease\n(TDPD)', fontsize=12, 
             fontweight='bold', rotation=90, va='center', ha='center')
    
    # Add column headers
    fig.text(0.15, 0.98, 'Original', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.33, 0.98, 'Denoised', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.51, 0.98, 'Difference', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.72, 0.98, 'ROI Comparison', fontsize=14, fontweight='bold', ha='center')
    
    # Save the figure
    output_path = 'figure2_denoising_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved as: {output_path}")
    
    # Also save as PDF
    pdf_path = 'figure2_denoising_comparison.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure also saved as: {pdf_path}")
    
    plt.show()
    
    return fig

def create_detailed_single_subject_comparison():
    """Create a detailed comparison for a single subject (sub-YLOPD109 as mentioned in caption)."""
    
    # Focus on PIGD subject sub-YLOPD109 as mentioned in the example caption
    base_dir = "D:/data_NIMHANS"
    denoised_dir = "D:/data_NIMHANS/t1w_denoised"
    subject = "sub-YLOPD109"
    group = "PIGD"
    
    # Define file paths
    original_path = os.path.join(base_dir, group, subject, "ses-01", "anat", 
                               f"{subject}_ses-01_run-1_T1w.nii.gz")
    denoised_path = os.path.join(denoised_dir, group, 
                               f"{subject}_ses-01_run-1_T1w.nii.gz_T1w_denoised.nii.gz")
    
    # Load the data
    print(f"Loading data for detailed comparison: {subject}")
    original_data, _ = load_nifti_data(original_path)
    denoised_data, _ = load_nifti_data(denoised_path)
    
    if original_data is None or denoised_data is None:
        print("Failed to load data for detailed comparison")
        return None
    
    # Find representative slice
    slice_idx = find_middle_slice_with_brain(original_data, axis=2)
    
    # Extract slices
    original_slice = original_data[:, :, slice_idx]
    denoised_slice = denoised_data[:, :, slice_idx]
    
    # Normalize
    original_norm = normalize_image(original_slice)
    denoised_norm = normalize_image(denoised_slice)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(np.rot90(original_norm), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('(A) Original Image\nwith visible background noise', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Denoised
    axes[1].imshow(np.rot90(denoised_norm), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('(B) Denoised Image\nshowing reduced noise and enhanced\ntissue contrast', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add main title
    plt.suptitle(f'Figure 2: Visual effect of non-local means denoising on a T1-weighted\n' +
                f'axial slice from subject {subject}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'figure2_detailed_denoising_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed figure saved as: {output_path}")
    
    # Also save as PDF
    pdf_path = 'figure2_detailed_denoising_comparison.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed figure also saved as: {pdf_path}")
    
    plt.show()
    
    return fig

def main():
    """Main function to generate both figures."""
    print("=" * 60)
    print("FIGURE 2: DENOISING COMPARISON")
    print("=" * 60)
    
    # Create comprehensive comparison figure
    print("\nCreating comprehensive denoising comparison figure...")
    fig1 = create_denoising_comparison_figure()
    
    print("\n" + "=" * 60)
    
    # Create detailed single subject figure
    print("\nCreating detailed single-subject comparison figure...")
    fig2 = create_detailed_single_subject_comparison()
    
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    
    print("""
    Generated files:
    1. figure2_denoising_comparison.png - Comprehensive comparison across all groups
    2. figure2_denoising_comparison.pdf - PDF version of comprehensive comparison
    3. figure2_detailed_denoising_comparison.png - Detailed single-subject comparison
    4. figure2_detailed_denoising_comparison.pdf - PDF version of detailed comparison
    
    These figures demonstrate the visual effect of non-local means denoising on T1-weighted
    brain MRI images, showing reduced noise and enhanced tissue contrast after processing.
    """)

if __name__ == "__main__":
    main()
