#!/usr/bin/env python3
"""
Group-wise Image Quality Analysis
================================

This script processes MRI image quality analysis by clinical groups:
- HC (Healthy Controls)
- PIGD (Postural Instability/Gait Difficulty) 
- TDPD (Tremor Dominant Parkinson's Disease)

It ensures all 75 subjects are analyzed and creates group-specific reports.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def load_subject_groups(excel_file):
    """Load subject group mappings from Excel file"""
    import pandas as pd
    
    # Read Excel file
    df = pd.read_excel(excel_file)
    df_clean = df.dropna(subset=['sub'])
    
    groups = {'HC': [], 'PIGD': [], 'TDPD': []}
    
    for _, row in df_clean.iterrows():
        subject_name = row['sub']
        
        # Convert YLOPD_XX format to sub-YLOPDXX format to match directory names
        if subject_name.startswith('YLOPD_HC_'):
            # HC subjects: YLOPD_HC_01 -> sub-YLOPDHC01
            subject_id = subject_name.replace('YLOPD_HC_', 'sub-YLOPDHC')
        else:
            # Other subjects: YLOPD_65 -> sub-YLOPD65
            subject_id = subject_name.replace('YLOPD_', 'sub-YLOPD')
        
        # Determine group based on binary flags
        if row['HC'] == 1:
            groups['HC'].append(subject_id)
        elif row['PIGD'] == 1:
            groups['PIGD'].append(subject_id)
        elif row['TDPD'] == 1:
            groups['TDPD'].append(subject_id)
    
    return groups, df_clean

def find_subject_files(data_dir, subject_id):
    """Find T1w files for a specific subject"""
    subject_dir = Path(data_dir) / subject_id
    
    if not subject_dir.exists():
        return []
    
    # Look for T1w files (both corrected and brain extracted)
    t1w_patterns = [
        f"{subject_id}_T1w_corrected.nii.gz",
        f"{subject_id}_T1w_brain.nii.gz"
    ]
    
    found_files = []
    for pattern in t1w_patterns:
        file_path = subject_dir / pattern
        if file_path.exists():
            found_files.append(str(file_path))
    
    return found_files

def analyze_single_subject(subject_files, output_dir, config):
    """Analyze all files for a single subject"""
    from modules.main_analyzer import EnhancedQualityAssessment
    
    if not subject_files:
        return []
    
    analyzer = EnhancedQualityAssessment(config)
    results = []
    
    for file_path in subject_files:
        try:
            print(f"  Analyzing: {os.path.basename(file_path)}")
            result = analyzer.analyze_single_image(file_path)
            results.append(result)
            
            # Save individual result
            json_file = analyzer.save_results(result, output_dir, "detailed_analysis")
            print(f"    ✓ Saved: {os.path.basename(json_file)}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            error_result = {
                'file_path': file_path,
                'subject_id': os.path.splitext(os.path.basename(file_path))[0],
                'analysis_failed': True,
                'error': str(e)
            }
            results.append(error_result)
    
    return results

def create_group_summary(group_results, group_name, output_dir, group_demographics=None):
    """Create summary statistics for a group"""
    
    # Extract successful analyses
    successful_results = []
    for subject_results in group_results.values():
        for result in subject_results:
            if 'analysis_failed' not in result:
                successful_results.append(result)
    
    if not successful_results:
        print(f"  No successful analyses for {group_name}")
        return None
    
    # Extract metrics
    snr_values = []
    cnr_values = []
    uniformity_values = []
    volume_values = []
    quality_scores = []
    
    for result in successful_results:
        # Extract SNR
        snr_stage = result.get('stage_4_snr_analysis', {})
        if 'snr_value' in snr_stage:
            snr_values.append(snr_stage['snr_value'])
        
        # Extract CNR
        cnr_stage = result.get('stage_4_cnr_analysis', {})
        if 'cnr_value' in cnr_stage:
            cnr_values.append(cnr_stage['cnr_value'])
        
        # Extract Uniformity
        uniformity_stage = result.get('stage_4_uniformity_analysis', {})
        if 'uniformity_value' in uniformity_stage:
            uniformity_values.append(uniformity_stage['uniformity_value'])
        
        # Extract Volume
        brain_stage = result.get('stage_2_brain_extraction', {})
        if 'brain_volume_ml' in brain_stage:
            volume_values.append(brain_stage['brain_volume_ml'])
        
        # Extract Quality Score
        quality_stage = result.get('stage_5_quality_summary', {})
        if 'quality_score' in quality_stage:
            quality_scores.append(quality_stage['quality_score'])
    
    # Calculate statistics
    def calc_stats(values, name):
        if not values:
            return {f'{name}_count': 0}
        return {
            f'{name}_count': len(values),
            f'{name}_mean': np.mean(values),
            f'{name}_std': np.std(values),
            f'{name}_min': np.min(values),
            f'{name}_max': np.max(values),
            f'{name}_median': np.median(values)
        }
    
    group_stats = {
        'group': group_name,
        'total_subjects': len(group_results),
        'successful_analyses': len(successful_results),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add demographic info if available
    if group_demographics is not None:
        group_demo = group_demographics[group_demographics[group_name] == 1]
        if len(group_demo) > 0:
            group_stats['demographics'] = {
                'mean_age': float(group_demo['age_assessment'].mean()),
                'std_age': float(group_demo['age_assessment'].std()),
                'male_count': int((group_demo['gender'] == 1).sum()),
                'female_count': int((group_demo['gender'] == 2).sum()),
                'total_subjects_in_demo': len(group_demo)
            }
    
    # Add metric statistics
    group_stats.update(calc_stats(snr_values, 'snr'))
    group_stats.update(calc_stats(cnr_values, 'cnr'))
    group_stats.update(calc_stats(uniformity_values, 'uniformity'))
    group_stats.update(calc_stats(volume_values, 'volume'))
    group_stats.update(calc_stats(quality_scores, 'quality_score'))
    
    # Quality flags summary
    quality_good_count = sum(1 for result in successful_results 
                           if result.get('stage_5_quality_summary', {}).get('overall_quality_good', False))
    
    group_stats['quality_good_count'] = quality_good_count
    group_stats['quality_good_percentage'] = (quality_good_count / len(successful_results) * 100) if successful_results else 0
    
    # Save group summary
    summary_file = Path(output_dir) / f"{group_name}_group_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(group_stats, f, indent=2, default=str)
    
    print(f"  ✓ Saved group summary: {summary_file}")
    return group_stats

def create_detailed_csv(all_results, output_dir):
    """Create detailed CSV with all results"""
    
    csv_data = []
    
    for group_name, group_results in all_results.items():
        for subject_id, subject_results in group_results.items():
            for result in subject_results:
                
                if 'analysis_failed' in result:
                    row = {
                        'group': group_name,
                        'subject_id': subject_id,
                        'file_name': os.path.basename(result.get('file_path', '')),
                        'file_type': 'unknown',
                        'analysis_status': 'FAILED',
                        'error_message': result.get('error', ''),
                        'snr': np.nan,
                        'cnr': np.nan,
                        'uniformity': np.nan,
                        'brain_volume_ml': np.nan,
                        'quality_score': np.nan,
                        'overall_quality_good': False
                    }
                else:
                    # Extract data from successful result
                    snr_stage = result.get('stage_4_snr_analysis', {})
                    cnr_stage = result.get('stage_4_cnr_analysis', {})
                    uniformity_stage = result.get('stage_4_uniformity_analysis', {})
                    brain_stage = result.get('stage_2_brain_extraction', {})
                    quality_stage = result.get('stage_5_quality_summary', {})
                    
                    file_path = result.get('file_path', '')
                    file_name = os.path.basename(file_path)
                    
                    # Determine file type
                    if '_corrected' in file_name:
                        file_type = 'T1w_corrected'
                    elif '_brain' in file_name:
                        file_type = 'T1w_brain'
                    else:
                        file_type = 'T1w_other'
                    
                    row = {
                        'group': group_name,
                        'subject_id': subject_id,
                        'file_name': file_name,
                        'file_type': file_type,
                        'analysis_status': 'SUCCESS',
                        'error_message': '',
                        'snr': snr_stage.get('snr_value', np.nan),
                        'cnr': cnr_stage.get('cnr_value', np.nan),
                        'uniformity': uniformity_stage.get('uniformity_value', np.nan),
                        'brain_volume_ml': brain_stage.get('brain_volume_ml', np.nan),
                        'quality_score': quality_stage.get('quality_score', np.nan),
                        'snr_adequate': quality_stage.get('snr_adequate', False),
                        'uniformity_good': quality_stage.get('uniformity_good', False),
                        'volume_adequate': quality_stage.get('volume_adequate', False),
                        'overall_quality_good': quality_stage.get('overall_quality_good', False),
                        'analysis_timestamp': result.get('analysis_timestamp', '')
                    }
                
                csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_file = Path(output_dir) / "complete_group_analysis.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"✓ Saved detailed CSV: {csv_file}")
    return csv_file

def print_overall_summary(group_summaries):
    """Print overall summary across all groups"""
    
    print(f"\n{'='*60}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    total_subjects = 0
    total_analyses = 0
    
    for group_name, stats in group_summaries.items():
        if stats:
            print(f"\n{group_name} GROUP:")
            print(f"  Subjects: {stats['total_subjects']}")
            print(f"  Successful analyses: {stats['successful_analyses']}")
            print(f"  Quality good: {stats['quality_good_count']}/{stats['successful_analyses']} ({stats['quality_good_percentage']:.1f}%)")
            
            # Add demographic info
            if 'demographics' in stats:
                demo = stats['demographics']
                print(f"  Demographics: Age {demo['mean_age']:.1f}±{demo['std_age']:.1f}, "
                      f"M/F: {demo['male_count']}/{demo['female_count']}")
            
            if stats.get('snr_count', 0) > 0:
                print(f"  SNR: {stats['snr_mean']:.1f} ± {stats['snr_std']:.1f}")
                print(f"  Volume: {stats['volume_mean']:.1f} ± {stats['volume_std']:.1f} ml")
                print(f"  Uniformity: {stats['uniformity_mean']:.3f} ± {stats['uniformity_std']:.3f}")
            
            total_subjects += stats['total_subjects']
            total_analyses += stats['successful_analyses']
    
    print(f"\nTOTAL ACROSS ALL GROUPS:")
    print(f"  Total subjects: {total_subjects}")
    print(f"  Total successful analyses: {total_analyses}")
    print(f"  Expected total subjects: 75 (25 HC + 25 PIGD + 25 TDPD)")

def main():
    """Main group-wise analysis function"""
    
    print("Group-wise Image Quality Analysis")
    print("=" * 50)
    
    # Configuration
    data_dir = r"D:\data_NIMHANS\outputs\01_preprocessed"
    excel_file = r"D:\data_NIMHANS\age_gender.xlsx"
    output_dir = "group_analysis_outputs"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load configuration
    from modules.config import QualityConfig
    config = QualityConfig(
        output_root=output_dir,
        brain_extraction_method='threshold_based',
        noise_estimation_method='edge_regions',
        min_snr=10.0,
        max_intensity_nonuniformity=0.3,
        min_brain_volume=800000
    )
    
    # Load subject groups from Excel file
    print(f"\nLoading subject groups from: {excel_file}")
    groups, demographics_df = load_subject_groups(excel_file)
    
    for group_name, subjects in groups.items():
        print(f"  {group_name}: {len(subjects)} subjects")
    
    # Process each group
    all_results = {}
    group_summaries = {}
    
    for group_name, subjects in groups.items():
        print(f"\n{'='*50}")
        print(f"PROCESSING {group_name} GROUP ({len(subjects)} subjects)")
        print(f"{'='*50}")
        
        group_results = {}
        group_output_dir = Path(output_dir) / group_name
        group_output_dir.mkdir(exist_ok=True)
        
        for i, subject_id in enumerate(subjects, 1):
            print(f"\n[{i}/{len(subjects)}] Processing {subject_id}")
            
            # Find subject files
            subject_files = find_subject_files(data_dir, subject_id)
            
            if not subject_files:
                print(f"  ⚠️  No T1w files found for {subject_id}")
                group_results[subject_id] = []
                continue
            
            print(f"  Found {len(subject_files)} files:")
            for f in subject_files:
                print(f"    - {os.path.basename(f)}")
            
            # Analyze subject
            subject_results = analyze_single_subject(subject_files, str(group_output_dir), config)
            group_results[subject_id] = subject_results
        
        all_results[group_name] = group_results
        
        # Create group summary
        print(f"\nCreating {group_name} group summary...")
        group_summary = create_group_summary(group_results, group_name, str(group_output_dir), demographics_df)
        group_summaries[group_name] = group_summary
    
    # Create overall CSV
    print(f"\nCreating complete analysis CSV...")
    create_detailed_csv(all_results, output_dir)
    
    # Print overall summary
    print_overall_summary(group_summaries)
    
    print(f"\n✅ Group-wise analysis complete!")
    print(f"Results saved to: {Path(output_dir).absolute()}")

if __name__ == "__main__":
    main()
