"""
Step 01: Data Inventory and Initial Quality Check
================================================

This script performs the initial data inventory and basic quality checks:
1. Discovers all T1w images in the BIDS dataset
2. Checks file integrity and basic image properties
3. Loads demographic data
4. Creates initial data summary
5. Identifies any missing or problematic files

This is the first step in the structural analysis pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (setup_logging, get_subject_list, find_t1_files, 
                   load_demographics, check_image_exists, basic_image_info,
                   print_analysis_summary)
from config import OUTPUT_DIRS, GROUPS


def create_data_inventory():
    """
    Create comprehensive data inventory.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing inventory of all subjects and their data
    """
    logger = setup_logging(__name__)
    logger.info("Starting data inventory creation...")
    
    # Get all subjects and their T1 files
    subjects_by_group = get_subject_list()
    t1_files = find_t1_files()
    
    # Load demographics
    demographics = load_demographics()
    
    # Create inventory list
    inventory_data = []
    
    for group_name, subjects in subjects_by_group.items():
        logger.info(f"Processing {group_name} group...")
        
        for subject in subjects:
            subject_data = {
                'subject_id': subject,
                'group': group_name,
                't1_file_found': False,
                't1_file_path': None,
                't1_file_readable': False,
                'file_size_mb': None,
                'image_shape': None,
                'voxel_size': None,
                'data_type': None,
                'demographics_available': False,
                'age': None,
                'sex': None,
                'notes': []
            }
            
            # Check T1 file
            if group_name in t1_files and subject in t1_files[group_name]:
                t1_path = t1_files[group_name][subject]
                subject_data['t1_file_found'] = True
                subject_data['t1_file_path'] = t1_path
                
                # Check file readability and get basic info
                if check_image_exists(t1_path):
                    subject_data['t1_file_readable'] = True
                    
                    # Get image info
                    img_info = basic_image_info(t1_path)
                    if 'error' not in img_info:
                        subject_data['file_size_mb'] = img_info['file_size_mb']
                        subject_data['image_shape'] = str(img_info['shape'])
                        subject_data['voxel_size'] = str(img_info['voxel_size'])
                        subject_data['data_type'] = str(img_info['data_type'])
                    else:
                        subject_data['notes'].append(f"Image info error: {img_info['error']}")
                else:
                    subject_data['notes'].append("T1 file not readable")
            else:
                subject_data['notes'].append("T1 file not found")
            
            # Check demographics
            if not demographics.empty:
                # Try to match subject in demographics
                # This might need adjustment based on the actual demographics file structure
                demo_match = demographics[demographics.columns[0]].astype(str).str.contains(subject.replace('sub-', ''), na=False)
                if demo_match.any():
                    subject_data['demographics_available'] = True
                    demo_row = demographics[demo_match].iloc[0]
                    
                    # Extract age and sex (adjust column names as needed)
                    for col in demographics.columns:
                        col_lower = col.lower()
                        if 'age' in col_lower:
                            subject_data['age'] = demo_row[col]
                        elif any(term in col_lower for term in ['sex', 'gender']):
                            subject_data['sex'] = demo_row[col]
            
            inventory_data.append(subject_data)
    
    # Convert to DataFrame
    inventory_df = pd.DataFrame(inventory_data)
    
    # Convert notes list to string
    inventory_df['notes'] = inventory_df['notes'].apply(lambda x: '; '.join(x) if x else '')
    
    logger.info(f"Inventory created for {len(inventory_df)} subjects")
    return inventory_df


def generate_quality_report(inventory_df: pd.DataFrame):
    """
    Generate quality control report from inventory.
    
    Parameters:
    -----------
    inventory_df : pd.DataFrame
        Data inventory DataFrame
    """
    logger = setup_logging(__name__)
    logger.info("Generating quality control report...")
    
    # Create QC report
    qc_report = {}
    
    # Overall statistics
    total_subjects = len(inventory_df)
    subjects_with_t1 = inventory_df['t1_file_found'].sum()
    subjects_with_readable_t1 = inventory_df['t1_file_readable'].sum()
    subjects_with_demographics = inventory_df['demographics_available'].sum()
    
    qc_report['overall'] = {
        'total_subjects': total_subjects,
        'subjects_with_t1': subjects_with_t1,
        'subjects_with_readable_t1': subjects_with_readable_t1,
        'subjects_with_demographics': subjects_with_demographics,
        't1_success_rate': subjects_with_readable_t1 / total_subjects * 100,
        'demographics_success_rate': subjects_with_demographics / total_subjects * 100
    }
    
    # Group-wise statistics
    qc_report['by_group'] = {}
    for group in GROUPS.keys():
        group_data = inventory_df[inventory_df['group'] == group]
        qc_report['by_group'][group] = {
            'n_subjects': len(group_data),
            'n_with_t1': group_data['t1_file_readable'].sum(),
            'success_rate': group_data['t1_file_readable'].sum() / len(group_data) * 100 if len(group_data) > 0 else 0
        }
    
    # File size statistics (for readable files only)
    readable_files = inventory_df[inventory_df['t1_file_readable']]
    if not readable_files.empty and readable_files['file_size_mb'].notna().any():
        file_sizes = readable_files['file_size_mb'].dropna()
        qc_report['file_sizes'] = {
            'mean_mb': file_sizes.mean(),
            'std_mb': file_sizes.std(),
            'min_mb': file_sizes.min(),
            'max_mb': file_sizes.max()
        }
    
    # Save QC report
    qc_report_path = os.path.join(OUTPUT_DIRS['qc'], 'data_inventory_qc_report.txt')
    with open(qc_report_path, 'w') as f:
        f.write("YOPD Structural Analysis - Data Inventory QC Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")
        for key, value in qc_report['overall'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nGROUP-WISE STATISTICS\n")
        f.write("-" * 30 + "\n")
        for group, stats in qc_report['by_group'].items():
            f.write(f"{group}: {stats['n_with_t1']}/{stats['n_subjects']} subjects ({stats['success_rate']:.1f}%)\n")
        
        if 'file_sizes' in qc_report:
            f.write("\nFILE SIZE STATISTICS\n")
            f.write("-" * 30 + "\n")
            for key, value in qc_report['file_sizes'].items():
                f.write(f"{key}: {value:.2f}\n")
    
    logger.info(f"QC report saved to: {qc_report_path}")
    return qc_report


def create_visualizations(inventory_df: pd.DataFrame):
    """
    Create visualization plots for the data inventory.
    
    Parameters:
    -----------
    inventory_df : pd.DataFrame
        Data inventory DataFrame
    """
    logger = setup_logging(__name__)
    logger.info("Creating visualization plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Data availability by group
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOPD Structural Data Inventory Summary', fontsize=16, fontweight='bold')
    
    # Subplot 1: Subject counts by group
    group_counts = inventory_df['group'].value_counts()
    axes[0, 0].bar(group_counts.index, group_counts.values)
    axes[0, 0].set_title('Total Subjects by Group')
    axes[0, 0].set_ylabel('Number of Subjects')
    for i, v in enumerate(group_counts.values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Subplot 2: T1 availability by group
    t1_by_group = inventory_df.groupby('group')['t1_file_readable'].agg(['sum', 'count'])
    axes[0, 1].bar(t1_by_group.index, t1_by_group['sum'], label='With T1')
    axes[0, 1].bar(t1_by_group.index, t1_by_group['count'] - t1_by_group['sum'], 
                   bottom=t1_by_group['sum'], label='Without T1', alpha=0.6)
    axes[0, 1].set_title('T1 Image Availability by Group')
    axes[0, 1].set_ylabel('Number of Subjects')
    axes[0, 1].legend()
    
    # Subplot 3: Demographics availability
    demo_by_group = inventory_df.groupby('group')['demographics_available'].agg(['sum', 'count'])
    axes[1, 0].bar(demo_by_group.index, demo_by_group['sum'], label='With Demographics')
    axes[1, 0].bar(demo_by_group.index, demo_by_group['count'] - demo_by_group['sum'], 
                   bottom=demo_by_group['sum'], label='Without Demographics', alpha=0.6)
    axes[1, 0].set_title('Demographics Availability by Group')
    axes[1, 0].set_ylabel('Number of Subjects')
    axes[1, 0].legend()
    
    # Subplot 4: File sizes (if available)
    readable_data = inventory_df[inventory_df['t1_file_readable'] & inventory_df['file_size_mb'].notna()]
    if not readable_data.empty:
        sns.boxplot(data=readable_data, x='group', y='file_size_mb', ax=axes[1, 1])
        axes[1, 1].set_title('T1 File Sizes by Group')
        axes[1, 1].set_ylabel('File Size (MB)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No file size data available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('T1 File Sizes by Group')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(OUTPUT_DIRS['qc'], 'data_inventory_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary plot saved to: {plot_path}")
    
    # Create age/sex distribution plots if demographics available
    demo_data = inventory_df[inventory_df['demographics_available'] & inventory_df['age'].notna()]
    if not demo_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Demographics Distribution', fontsize=14, fontweight='bold')
        
        # Age distribution
        sns.boxplot(data=demo_data, x='group', y='age', ax=axes[0])
        axes[0].set_title('Age Distribution by Group')
        axes[0].set_ylabel('Age (years)')
        
        # Sex distribution
        if demo_data['sex'].notna().any():
            sex_counts = demo_data.groupby(['group', 'sex']).size().unstack(fill_value=0)
            sex_counts.plot(kind='bar', ax=axes[1], stacked=True)
            axes[1].set_title('Sex Distribution by Group')
            axes[1].set_ylabel('Number of Subjects')
            axes[1].legend(title='Sex')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        demo_plot_path = os.path.join(OUTPUT_DIRS['qc'], 'demographics_distribution.png')
        plt.savefig(demo_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Demographics plot saved to: {demo_plot_path}")


def main():
    """
    Main function to run the data inventory and quality check.
    """
    logger = setup_logging(__name__)
    logger.info("="*60)
    logger.info("STARTING STEP 01: DATA INVENTORY AND QUALITY CHECK")
    logger.info("="*60)
    
    try:
        # Create data inventory
        logger.info("Creating comprehensive data inventory...")
        inventory_df = create_data_inventory()
        
        # Save inventory to CSV
        inventory_path = os.path.join(OUTPUT_DIRS['qc'], 'data_inventory.csv')
        inventory_df.to_csv(inventory_path, index=False)
        logger.info(f"Data inventory saved to: {inventory_path}")
        
        # Generate QC report
        qc_report = generate_quality_report(inventory_df)
        
        # Create visualizations
        create_visualizations(inventory_df)
        
        # Print summary
        readable_subjects = inventory_df['t1_file_readable'].sum()
        total_subjects = len(inventory_df)
        excluded_subjects = total_subjects - readable_subjects
        
        print_analysis_summary(
            analysis_name="Data Inventory and Quality Check",
            subjects_analyzed=readable_subjects,
            subjects_excluded=excluded_subjects,
            notes=f"Success rate: {readable_subjects/total_subjects*100:.1f}%"
        )
        
        # Display key findings
        print("\nKEY FINDINGS:")
        print(f"• Total subjects in dataset: {total_subjects}")
        print(f"• Subjects with readable T1 images: {readable_subjects}")
        print(f"• Demographics available for: {inventory_df['demographics_available'].sum()} subjects")
        
        for group in GROUPS.keys():
            group_data = inventory_df[inventory_df['group'] == group]
            n_readable = group_data['t1_file_readable'].sum()
            print(f"• {group}: {n_readable}/{len(group_data)} subjects ready for analysis")
        
        # Check for any critical issues
        if readable_subjects < total_subjects * 0.8:
            logger.warning("WARNING: Less than 80% of subjects have readable T1 images!")
        
        logger.info("Step 01 completed successfully!")
        return inventory_df
        
    except Exception as e:
        logger.error(f"Error in Step 01: {str(e)}")
        raise


if __name__ == "__main__":
    inventory_df = main()
