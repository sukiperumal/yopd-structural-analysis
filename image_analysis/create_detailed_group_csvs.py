import pandas as pd
import json
import os
from pathlib import Path
import glob

def extract_detailed_metrics_from_json(json_file_path):
    """Extract all detailed metrics from a single JSON analysis file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract file name from file path
        file_path = data.get('file_path', '')
        file_name = os.path.basename(file_path) if file_path else ''
        
        # Initialize the result dictionary with basic info
        result = {
            'subject_id': data.get('subject_id', ''),
            'file_name': file_name,
            'file_path': file_path,
            'file_type': 'T1w_brain' if 'brain' in file_name else 'T1w_corrected',
            'analysis_timestamp': data.get('analysis_timestamp', ''),
            'overall_quality': data.get('final_assessment', {}).get('overall_quality', ''),
            'quality_score': data.get('final_assessment', {}).get('composite_quality_score', 0)
        }
        
        # Stage 1: Raw image analysis metrics
        if 'stage_1_raw_analysis' in data:
            raw = data['stage_1_raw_analysis']
            result.update({
                'raw_mean_intensity': raw.get('raw_mean_intensity', 0),
                'raw_std_intensity': raw.get('raw_std_intensity', 0),
                'raw_min_intensity': raw.get('raw_min_intensity', 0),
                'raw_max_intensity': raw.get('raw_max_intensity', 0),
                'raw_intensity_range': raw.get('raw_intensity_range', 0),
                'raw_nonzero_voxels': raw.get('raw_nonzero_voxels', 0),
                'raw_zero_voxels': raw.get('raw_zero_voxels', 0),
                'image_shape_x': raw.get('image_shape', [0,0,0])[0] if raw.get('image_shape') else 0,
                'image_shape_y': raw.get('image_shape', [0,0,0])[1] if raw.get('image_shape') else 0,
                'image_shape_z': raw.get('image_shape', [0,0,0])[2] if raw.get('image_shape') else 0,
                'voxel_volume_mm3': raw.get('voxel_volume_mm3', 0),
                'total_image_volume_mm3': raw.get('total_image_volume_mm3', 0),
                'raw_intensity_skewness': raw.get('raw_intensity_skewness', 0),
                'raw_intensity_kurtosis': raw.get('raw_intensity_kurtosis', 0),
                'raw_median_intensity': raw.get('raw_intensity_percentiles', {}).get('50th', 0),
                'raw_q25_intensity': raw.get('raw_intensity_percentiles', {}).get('25th', 0),
                'raw_q75_intensity': raw.get('raw_intensity_percentiles', {}).get('75th', 0)
            })
        
        # Stage 1: Acquisition assessment
        if 'stage_1_acquisition_assessment' in data:
            acq = data['stage_1_acquisition_assessment']
            result.update({
                'edge_variance_mean': acq.get('edge_variance_mean', 0),
                'edge_variance_std': acq.get('edge_variance_std', 0),
                'histogram_peaks': acq.get('intensity_histogram_peaks', 0),
                'distribution_uniformity': acq.get('intensity_distribution_uniformity', 0),
                'potential_motion_artifacts': acq.get('potential_motion_artifacts', False)
            })
        
        # Stage 2: Brain extraction metrics
        if 'stage_2_brain_extraction' in data:
            brain = data['stage_2_brain_extraction']
            result.update({
                'brain_extraction_method': brain.get('method', ''),
                'brain_extraction_threshold_factor': brain.get('threshold_factor', 0),
                'brain_extraction_threshold_value': brain.get('threshold_value', 0),
                'initial_threshold_voxels': brain.get('initial_threshold_voxels', 0),
                'after_small_removal_voxels': brain.get('after_small_removal_voxels', 0),
                'after_hole_filling_voxels': brain.get('after_hole_filling_voxels', 0),
                'largest_component_voxels': brain.get('largest_component_voxels', 0),
                'final_brain_mask_voxels': brain.get('final_brain_mask_voxels', 0),
                'brain_coverage_percent': brain.get('brain_coverage_percent', 0)
            })
        
        # Stage 2: Brain extraction assessment
        if 'stage_2_extraction_assessment' in data:
            assess = data['stage_2_extraction_assessment']
            result.update({
                'reasonable_coverage': assess.get('reasonable_coverage', False),
                'good_contrast': assess.get('good_contrast', False),
                'sufficient_size': assess.get('sufficient_size', False)
            })
        
        # Stage 3: Noise estimation metrics
        if 'stage_3_noise_estimation' in data:
            noise = data['stage_3_noise_estimation']
            result.update({
                'noise_method': noise.get('method', ''),
                'edge_thickness_voxels': noise.get('edge_thickness_voxels', 0),
                'initial_edge_voxels': noise.get('initial_edge_voxels', 0),
                'filtered_edge_voxels': noise.get('filtered_edge_voxels', 0),
                'outliers_removed_count': noise.get('outliers_removed_count', 0),
                'outliers_removed_percent': noise.get('outliers_removed_percent', 0),
                'noise_estimate_std': noise.get('noise_estimate_std', 0),
                'edge_region_mean_intensity': noise.get('edge_region_mean_intensity', 0),
                'minimum_noise_floor_applied': noise.get('minimum_noise_floor_applied', False),
                'final_noise_estimate': noise.get('final_noise_estimate', 0)
            })
        
        # Stage 4: Quality metrics
        if 'stage_4_quality_metrics' in data:
            quality = data['stage_4_quality_metrics']
            
            # SNR metrics
            if 'snr_calculation' in quality:
                snr = quality['snr_calculation']
                result.update({
                    'snr_brain_voxels': snr.get('brain_voxels', 0),
                    'snr_signal_median': snr.get('signal_median', 0),
                    'snr_signal_mean': snr.get('signal_mean', 0),
                    'snr_noise_estimate': snr.get('noise_estimate', 0),
                    'snr_value': snr.get('snr', 0)
                })
            
            # CNR metrics
            if 'cnr_calculation' in quality:
                cnr = quality['cnr_calculation']
                result.update({
                    'cnr_brain_voxels': cnr.get('brain_voxels', 0),
                    'cnr_background_voxels': cnr.get('background_voxels', 0),
                    'cnr_brain_signal_median': cnr.get('brain_signal_median', 0),
                    'cnr_background_signal_median': cnr.get('background_signal_median', 0),
                    'cnr_contrast': cnr.get('contrast', 0),
                    'cnr_noise_estimate': cnr.get('noise_estimate', 0),
                    'cnr_value': cnr.get('cnr', 0)
                })
            
            # Uniformity metrics
            if 'uniformity_assessment' in quality:
                uniformity = quality['uniformity_assessment']
                result.update({
                    'uniformity_brain_voxels': uniformity.get('brain_voxels', 0),
                    'uniformity_outliers_removed_percent': uniformity.get('outliers_removed_percent', 0),
                    'uniformity_filtered_voxels': uniformity.get('filtered_voxels', 0),
                    'uniformity_median_intensity': uniformity.get('median_intensity', 0),
                    'uniformity_mad': uniformity.get('mad', 0),
                    'uniformity_ratio': uniformity.get('uniformity_ratio', 0),
                    'uniformity_cv': uniformity.get('cv', 0)
                })
        
        # Stage 5: Quality assessment flags
        if 'stage_5_quality_assessment' in data:
            assess = data['stage_5_quality_assessment']
            result.update({
                'snr_adequate': assess.get('snr_adequate', False),
                'uniformity_good': assess.get('uniformity_good', False),
                'volume_adequate': assess.get('volume_adequate', False),
                'overall_quality_good': assess.get('overall_quality_good', False),
                'snr_score_component': assess.get('snr_score', 0),
                'uniformity_score_component': assess.get('uniformity_score', 0),
                'volume_score_component': assess.get('volume_score', 0),
                'total_quality_score': assess.get('total_score', 0)
            })
        
        # Final assessment
        if 'final_assessment' in data:
            final = data['final_assessment']
            result.update({
                'final_snr': final.get('snr', 0),
                'final_cnr': final.get('cnr', 0),
                'final_uniformity': final.get('uniformity', 0),
                'final_volume_ml': final.get('volume_ml', 0),
                'final_quality_score': final.get('composite_quality_score', 0),
                'processing_stages_completed': final.get('processing_stages_completed', 0),
                'processing_stages_total': final.get('processing_stages_total', 5)
            })
        
        # Direct summary metrics (at root level of JSON)
        result.update({
            'summary_brain_volume_voxels': data.get('brain_volume_voxels', 0),
            'summary_brain_volume_ml': data.get('brain_volume_ml', 0),
            'summary_snr': data.get('orig_snr', 0),
            'summary_cnr': data.get('orig_cnr', 0),
            'summary_noise_level': data.get('orig_noise_level', 0),
            'summary_uniformity': data.get('orig_uniformity', 0),
            'summary_orig_mean_intensity': data.get('orig_mean_intensity', 0),
            'summary_orig_std_intensity': data.get('orig_std_intensity', 0),
            'summary_orig_min_intensity': data.get('orig_min_intensity', 0),
            'summary_orig_max_intensity': data.get('orig_max_intensity', 0),
            'summary_intensity_range': data.get('intensity_range', 0)
        })
        
        # Stage 5 quality summary
        if 'stage_5_quality_summary' in data:
            quality_summary = data['stage_5_quality_summary']
            result.update({
                'quality_summary_snr_adequate': quality_summary.get('snr_adequate', False),
                'quality_summary_uniformity_good': quality_summary.get('uniformity_good', False), 
                'quality_summary_volume_adequate': quality_summary.get('volume_adequate', False),
                'quality_summary_overall_good': quality_summary.get('overall_quality_good', False),
                'quality_summary_score': quality_summary.get('quality_score', 0)
            })
        
        # Processing summary
        if 'overall_processing_summary' in data:
            proc_summary = data['overall_processing_summary']
            result.update({
                'processing_total_steps': proc_summary.get('total_steps', 0),
                'processing_successful_steps': proc_summary.get('successful_steps', 0),
                'processing_failed_steps': proc_summary.get('failed_steps', 0),
                'processing_success_rate': proc_summary.get('success_rate', 0)
            })
        
        return result
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return None

def create_group_detailed_csv(group_name, output_dir):
    """Create detailed CSV for a specific group"""
    group_dir = os.path.join(output_dir, group_name)
    
    if not os.path.exists(group_dir):
        print(f"Group directory {group_dir} not found")
        return
    
    # Find all JSON files in the group directory
    json_files = glob.glob(os.path.join(group_dir, "detailed_analysis_*.json"))
    
    if not json_files:
        print(f"No JSON files found in {group_dir}")
        return
    
    print(f"Processing {len(json_files)} JSON files for {group_name} group...")
    
    all_data = []
    for json_file in json_files:
        data = extract_detailed_metrics_from_json(json_file)
        if data:
            data['group'] = group_name
            all_data.append(data)
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Sort by subject_id and file_type
        df = df.sort_values(['subject_id', 'file_type'])
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{group_name}_detailed_metrics.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Created detailed CSV for {group_name}: {output_file}")
        print(f"   - {len(df)} analyses")
        print(f"   - {len(df['subject_id'].unique())} subjects")
        print(f"   - {len(df.columns)} metrics per analysis")
        
        return df
    else:
        print(f"❌ No valid data found for {group_name}")
        return None

def main():
    """Create detailed CSV files for all groups"""
    output_dir = "group_analysis_outputs"
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found")
        return
    
    print("CREATING DETAILED GROUP METRICS CSVs")
    print("=" * 50)
    
    groups = ['HC', 'PIGD', 'TDPD']
    all_group_data = []
    
    for group in groups:
        print(f"\nProcessing {group} group...")
        df = create_group_detailed_csv(group, output_dir)
        if df is not None:
            all_group_data.append(df)
    
    # Create combined CSV with all groups
    if all_group_data:
        combined_df = pd.concat(all_group_data, ignore_index=True)
        combined_file = os.path.join(output_dir, "all_groups_detailed_metrics.csv")
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n✅ Created combined detailed CSV: {combined_file}")
        print(f"   - {len(combined_df)} total analyses")
        print(f"   - {len(combined_df['subject_id'].unique())} subjects")
        print(f"   - {len(combined_df.columns)} metrics per analysis")
        
        # Print summary statistics
        print(f"\nSUMMARY BY GROUP:")
        print("-" * 30)
        for group in groups:
            group_data = combined_df[combined_df.group == group]
            print(f"{group}: {len(group_data)} analyses, {len(group_data['subject_id'].unique())} subjects")
    
    print(f"\n🎉 DETAILED METRICS EXTRACTION COMPLETE!")

if __name__ == "__main__":
    main()
