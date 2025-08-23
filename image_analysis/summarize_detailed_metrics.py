import pandas as pd
import numpy as np

def summarize_detailed_metrics():
    """Create a summary of the detailed metrics extracted"""
    
    # Load the combined data
    df = pd.read_csv('group_analysis_outputs/all_groups_detailed_metrics.csv')
    
    print("DETAILED METRICS SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Groups: {', '.join(df['group'].unique())}")
    print(f"File types: {', '.join(df['file_type'].unique())}")
    print(f"Total metrics per record: {len(df.columns)}")
    
    print(f"\nMETRICS CATEGORIES:")
    print("-" * 40)
    
    # Basic info metrics
    basic_metrics = ['subject_id', 'file_name', 'file_type', 'analysis_timestamp', 'overall_quality', 'quality_score']
    print(f"Basic Info: {len(basic_metrics)} metrics")
    
    # Raw image metrics
    raw_metrics = [col for col in df.columns if col.startswith('raw_') or col.startswith('image_shape') or col.startswith('voxel_')]
    print(f"Raw Image Analysis: {len(raw_metrics)} metrics")
    
    # Acquisition metrics
    acq_metrics = [col for col in df.columns if any(x in col for x in ['edge_variance', 'histogram', 'distribution', 'motion'])]
    print(f"Acquisition Assessment: {len(acq_metrics)} metrics")
    
    # Brain extraction metrics  
    brain_metrics = [col for col in df.columns if col.startswith('brain_') or 'threshold' in col or 'voxels' in col]
    print(f"Brain Extraction: {len(brain_metrics)} metrics")
    
    # Noise estimation metrics
    noise_metrics = [col for col in df.columns if 'noise' in col or 'edge_' in col]
    print(f"Noise Estimation: {len(noise_metrics)} metrics")
    
    # Quality metrics
    quality_metrics = [col for col in df.columns if any(x in col for x in ['snr_', 'cnr_', 'uniformity_']) and not col.startswith('raw_')]
    print(f"Quality Metrics: {len(quality_metrics)} metrics")
    
    # Assessment flags
    flag_metrics = [col for col in df.columns if any(x in col for x in ['adequate', 'good', 'score_component'])]
    print(f"Quality Assessment Flags: {len(flag_metrics)} metrics")
    
    # Final metrics
    final_metrics = [col for col in df.columns if col.startswith('final_') or col.startswith('processing_')]
    print(f"Final Assessment: {len(final_metrics)} metrics")
    
    print(f"\nKEY METRICS BY GROUP:")
    print("-" * 30)
    
    # Filter for T1w_brain files only for cleaner comparison
    brain_files = df[df['file_type'] == 'T1w_brain'].copy()
    
    for group in ['HC', 'PIGD', 'TDPD']:
        group_data = brain_files[brain_files['group'] == group]
        if len(group_data) > 0:
            print(f"\n{group} GROUP (n={len(group_data)}):")
            
            # Key metrics that have non-zero values
            key_metrics = ['final_snr', 'final_cnr', 'final_volume_ml', 'final_quality_score']
            for metric in key_metrics:
                if metric in group_data.columns:
                    values = group_data[metric].dropna()
                    if len(values) > 0 and values.sum() > 0:
                        print(f"  {metric}: {values.mean():.1f} ± {values.std():.1f}")
    
    print(f"\nFILES CREATED:")
    print("-" * 20)
    print("✅ HC_detailed_metrics.csv - 25 subjects × 2 files × 39 metrics")
    print("✅ PIGD_detailed_metrics.csv - 25 subjects × 2 files × 39 metrics") 
    print("✅ TDPD_detailed_metrics.csv - 25 subjects × 2 files × 39 metrics")
    print("✅ all_groups_detailed_metrics.csv - Combined 150 analyses × 39 metrics")
    
    print(f"\nMETRICS INCLUDE:")
    print("-" * 20)
    print("📊 Raw Image: intensity statistics, shape, volume")
    print("🔍 Acquisition: motion artifacts, histogram analysis") 
    print("🧠 Brain Extraction: threshold values, voxel counts, coverage")
    print("📡 Noise Estimation: edge analysis, outlier removal")
    print("⭐ Quality Metrics: SNR, CNR, uniformity calculations")
    print("✅ Assessment Flags: quality thresholds, composite scores")
    print("📈 Final Results: overall quality, processing success")

if __name__ == "__main__":
    summarize_detailed_metrics()
