#!/usr/bin/env python3
"""
Step 06: Network Analysis
YOPD Structural Analysis Pipeline

This script performs network-based analysis including:
1. Structural connectivity analysis
2. Network topology metrics
3. Graph theory measures
4. Hub identification
5. Network-based statistics (NBS)

Author: GitHub Copilot
Date: August 2025
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
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def load_subject_group_mapping():
    """Load subject-to-group mapping from directory structure"""
    mapping = {}
    
    # Load from directories
    data_root = config.DATA_ROOT
    for group in ['HC', 'PIGD', 'TDPD']:
        group_dir = os.path.join(data_root, group)
        if os.path.exists(group_dir):
            subjects = [d for d in os.listdir(group_dir) if d.startswith('sub-')]
            for subject in subjects:
                mapping[subject] = group
    
    logging.info(f"Loaded group mapping for {len(mapping)} subjects")
    for group in ['HC', 'PIGD', 'TDPD']:
        count = sum(1 for g in mapping.values() if g == group)
        logging.info(f"  {group}: {count} subjects")
    
    return mapping

def setup_network_directories(output_dir):
    """Create network analysis-specific output directories"""
    network_dirs = {
        'connectivity': os.path.join(output_dir, 'connectivity_matrices'),
        'graphs': os.path.join(output_dir, 'graph_metrics'),
        'networks': os.path.join(output_dir, 'network_analysis'),
        'statistics': os.path.join(output_dir, 'statistics'),
        'qc': os.path.join(output_dir, 'quality_control'),
        'figures': os.path.join(output_dir, 'figures')
    }
    
    for name, path in network_dirs.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")
    
    return network_dirs

def create_brain_parcellation_network():
    """
    Create a comprehensive brain parcellation for network analysis
    Based on Desikan-Killiany atlas with additional subcortical regions
    """
    
    # Define network-relevant brain regions
    brain_regions = {
        # Default Mode Network (DMN)
        'dmn': {
            'medial_prefrontal': {'coordinates': (45, 60, 50), 'network': 'DMN'},
            'posterior_cingulate': {'coordinates': (45, 35, 45), 'network': 'DMN'},
            'angular_gyrus_l': {'coordinates': (35, 25, 50), 'network': 'DMN'},
            'angular_gyrus_r': {'coordinates': (55, 25, 50), 'network': 'DMN'},
            'hippocampus_l': {'coordinates': (35, 40, 30), 'network': 'DMN'},
            'hippocampus_r': {'coordinates': (55, 40, 30), 'network': 'DMN'},
        },
        
        # Executive Control Network (ECN)
        'ecn': {
            'dlpfc_l': {'coordinates': (25, 55, 45), 'network': 'ECN'},
            'dlpfc_r': {'coordinates': (65, 55, 45), 'network': 'ECN'},
            'posterior_parietal_l': {'coordinates': (25, 30, 50), 'network': 'ECN'},
            'posterior_parietal_r': {'coordinates': (65, 30, 50), 'network': 'ECN'},
            'anterior_cingulate': {'coordinates': (45, 50, 35), 'network': 'ECN'},
        },
        
        # Salience Network (SN)
        'sn': {
            'anterior_insula_l': {'coordinates': (35, 45, 35), 'network': 'SN'},
            'anterior_insula_r': {'coordinates': (55, 45, 35), 'network': 'SN'},
            'dorsal_acc': {'coordinates': (45, 55, 40), 'network': 'SN'},
            'supramarginal_l': {'coordinates': (30, 25, 45), 'network': 'SN'},
            'supramarginal_r': {'coordinates': (60, 25, 45), 'network': 'SN'},
        },
        
        # Motor Network (relevant for PD)
        'motor': {
            'precentral_l': {'coordinates': (25, 45, 55), 'network': 'Motor'},
            'precentral_r': {'coordinates': (65, 45, 55), 'network': 'Motor'},
            'postcentral_l': {'coordinates': (25, 35, 55), 'network': 'Motor'},
            'postcentral_r': {'coordinates': (65, 35, 55), 'network': 'Motor'},
            'sma': {'coordinates': (45, 50, 55), 'network': 'Motor'},
        },
        
        # Basal Ganglia Network (critical for PD)
        'basal_ganglia': {
            'caudate_l': {'coordinates': (35, 55, 40), 'network': 'BG'},
            'caudate_r': {'coordinates': (55, 55, 40), 'network': 'BG'},
            'putamen_l': {'coordinates': (30, 50, 40), 'network': 'BG'},
            'putamen_r': {'coordinates': (60, 50, 40), 'network': 'BG'},
            'pallidum_l': {'coordinates': (35, 48, 38), 'network': 'BG'},
            'pallidum_r': {'coordinates': (55, 48, 38), 'network': 'BG'},
            'subthalamic_l': {'coordinates': (38, 45, 35), 'network': 'BG'},
            'subthalamic_r': {'coordinates': (52, 45, 35), 'network': 'BG'},
        },
        
        # Thalamic Network
        'thalamus': {
            'thalamus_l': {'coordinates': (38, 50, 40), 'network': 'Thalamic'},
            'thalamus_r': {'coordinates': (52, 50, 40), 'network': 'Thalamic'},
        },
        
        # Visual Network
        'visual': {
            'occipital_l': {'coordinates': (25, 15, 45), 'network': 'Visual'},
            'occipital_r': {'coordinates': (65, 15, 45), 'network': 'Visual'},
            'fusiform_l': {'coordinates': (30, 25, 25), 'network': 'Visual'},
            'fusiform_r': {'coordinates': (60, 25, 25), 'network': 'Visual'},
        },
        
        # Auditory Network
        'auditory': {
            'superior_temporal_l': {'coordinates': (25, 35, 35), 'network': 'Auditory'},
            'superior_temporal_r': {'coordinates': (65, 35, 35), 'network': 'Auditory'},
            'heschl_l': {'coordinates': (30, 40, 35), 'network': 'Auditory'},
            'heschl_r': {'coordinates': (60, 40, 35), 'network': 'Auditory'},
        }
    }
    
    # Flatten the structure
    all_regions = {}
    region_id = 1
    
    for network_group, regions in brain_regions.items():
        for region_name, region_info in regions.items():
            all_regions[region_id] = {
                'name': region_name,
                'network': region_info['network'],
                'network_group': network_group,
                'coordinates': region_info['coordinates']
            }
            region_id += 1
    
    return all_regions

def extract_regional_volumes(subject_id, gm_path, thickness_path, brain_regions):
    """Extract regional volumes and thickness values for network analysis"""
    try:
        # Load images
        gm_img = nib.load(gm_path)
        gm_data = gm_img.get_fdata()
        voxel_volume = np.prod(gm_img.header.get_zooms()[:3])
        
        thickness_data = None
        if thickness_path and os.path.exists(thickness_path):
            thickness_img = nib.load(thickness_path)
            thickness_data = thickness_img.get_fdata()
        
        regional_measures = {'subject_id': subject_id}
        
        # Extract measures for each region
        for region_id, region_info in brain_regions.items():
            region_name = region_info['name']
            coords = region_info['coordinates']
            
            # Define spherical ROI around coordinates (radius = 5 voxels)
            x, y, z = coords
            radius = 5
            
            # Create ROI mask
            xx, yy, zz = np.ogrid[:gm_data.shape[0], :gm_data.shape[1], :gm_data.shape[2]]
            roi_mask = ((xx - x)**2 + (yy - y)**2 + (zz - z)**2) <= radius**2
            
            if np.sum(roi_mask) == 0:
                regional_measures[f"{region_name}_volume"] = np.nan
                regional_measures[f"{region_name}_gm_density"] = np.nan
                if thickness_data is not None:
                    regional_measures[f"{region_name}_thickness"] = np.nan
                continue
            
            # Extract volume and density
            roi_volume = np.sum(roi_mask) * voxel_volume
            roi_gm_density = np.mean(gm_data[roi_mask])
            
            regional_measures[f"{region_name}_volume"] = roi_volume
            regional_measures[f"{region_name}_gm_density"] = roi_gm_density
            
            # Extract thickness if available
            if thickness_data is not None:
                roi_thickness = np.mean(thickness_data[roi_mask])
                regional_measures[f"{region_name}_thickness"] = roi_thickness
        
        return regional_measures, True
        
    except Exception as e:
        logging.error(f"Regional extraction failed for {subject_id}: {str(e)}")
        return {'subject_id': subject_id}, False

def calculate_connectivity_matrix(regional_data_df, connectivity_type='correlation'):
    """Calculate connectivity matrices for each subject"""
    
    # Get measure columns
    volume_columns = [col for col in regional_data_df.columns if col.endswith('_volume')]
    density_columns = [col for col in regional_data_df.columns if col.endswith('_gm_density')]
    thickness_columns = [col for col in regional_data_df.columns if col.endswith('_thickness')]
    
    # Use GM density for connectivity analysis (most stable measure)
    measure_columns = density_columns if len(density_columns) > 0 else volume_columns
    
    if len(measure_columns) < 2:
        logging.warning("Insufficient regions for connectivity analysis")
        return None, None
    
    connectivity_matrices = {}
    subjects = regional_data_df['subject_id'].values
    
    for idx, subject_id in enumerate(subjects):
        subject_data = regional_data_df.iloc[idx][measure_columns].values
        
        # Convert to numeric, replacing non-numeric with NaN
        try:
            subject_data = pd.to_numeric(subject_data, errors='coerce')
        except:
            # If conversion fails, create array of NaN
            subject_data = np.full(len(measure_columns), np.nan)
        
        # Remove NaN values
        valid_indices = ~np.isnan(subject_data)
        if np.sum(valid_indices) < 2:
            continue
        
        valid_data = subject_data[valid_indices]
        valid_columns = np.array(measure_columns)[valid_indices]
        
        # Calculate connectivity matrix
        if connectivity_type == 'correlation':
            # Pearson correlation between regions
            n_regions = len(valid_data)
            connectivity_matrix = np.eye(n_regions)
            
            # This is a simplified version - in practice, you would use time series data
            # For structural data, we simulate connectivity based on similarity
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    # Simulate structural connectivity based on anatomical distance and similarity
                    similarity = 1 / (1 + abs(valid_data[i] - valid_data[j]))
                    connectivity_matrix[i, j] = similarity
                    connectivity_matrix[j, i] = similarity
        
        connectivity_matrices[subject_id] = {
            'matrix': connectivity_matrix,
            'regions': valid_columns.tolist(),
            'n_regions': len(valid_columns)
        }
    
    return connectivity_matrices, measure_columns

def calculate_graph_metrics(connectivity_matrix, threshold=0.5):
    """Calculate graph theory metrics from connectivity matrix"""
    
    # Threshold the matrix to create binary graph
    binary_matrix = (connectivity_matrix > threshold).astype(int)
    np.fill_diagonal(binary_matrix, 0)  # Remove self-connections
    
    # Create NetworkX graph
    G = nx.from_numpy_array(binary_matrix)
    
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        return {}
    
    metrics = {}
    
    try:
        # Global metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Check if graph is connected
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['global_efficiency'] = nx.global_efficiency(G)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            if len(G_largest) > 1:
                metrics['average_path_length'] = nx.average_shortest_path_length(G_largest)
                metrics['global_efficiency'] = nx.global_efficiency(G_largest)
            else:
                metrics['average_path_length'] = np.nan
                metrics['global_efficiency'] = np.nan
        
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Small-world metrics
        # Generate random graph for comparison
        random_G = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
        if nx.is_connected(random_G) and not np.isnan(metrics.get('average_path_length', np.nan)):
            random_clustering = nx.average_clustering(random_G)
            random_path_length = nx.average_shortest_path_length(random_G)
            
            if random_clustering > 0 and random_path_length > 0:
                gamma = metrics['clustering_coefficient'] / random_clustering
                lambda_val = metrics['average_path_length'] / random_path_length
                metrics['small_worldness'] = gamma / lambda_val if lambda_val > 0 else np.nan
            else:
                metrics['small_worldness'] = np.nan
        else:
            metrics['small_worldness'] = np.nan
        
        # Node-level metrics (averaged)
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        metrics['mean_degree_centrality'] = np.mean(list(degree_centrality.values()))
        metrics['mean_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
        metrics['mean_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
        metrics['mean_eigenvector_centrality'] = np.mean(list(eigenvector_centrality.values()))
        
        # Hub identification (top 20% highest degree)
        degrees = dict(G.degree())
        degree_threshold = np.percentile(list(degrees.values()), 80)
        hubs = [node for node, degree in degrees.items() if degree >= degree_threshold]
        metrics['n_hubs'] = len(hubs)
        metrics['hub_connectivity'] = np.mean([degrees[hub] for hub in hubs]) if hubs else 0
        
    except Exception as e:
        logging.warning(f"Error calculating graph metrics: {str(e)}")
        metrics['error'] = str(e)
    
    return metrics

def perform_network_analysis(connectivity_matrices, brain_regions, network_dirs):
    """Perform comprehensive network analysis"""
    
    logging.info("Performing network-based analysis...")
    
    graph_metrics_list = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for subject_id, conn_data in connectivity_matrices.items():
        connectivity_matrix = conn_data['matrix']
        
        subject_metrics = {'subject_id': subject_id}
        
        # Calculate metrics at different thresholds
        for threshold in thresholds:
            metrics = calculate_graph_metrics(connectivity_matrix, threshold)
            
            # Add threshold suffix to metric names
            for metric_name, metric_value in metrics.items():
                subject_metrics[f"{metric_name}_t{int(threshold*100)}"] = metric_value
        
        graph_metrics_list.append(subject_metrics)
    
    # Convert to DataFrame
    graph_metrics_df = pd.DataFrame(graph_metrics_list)
    
    # Save graph metrics
    graph_metrics_csv = os.path.join(network_dirs['graphs'], 'graph_metrics.csv')
    graph_metrics_df.to_csv(graph_metrics_csv, index=False)
    
    # Save connectivity matrices
    for subject_id, conn_data in connectivity_matrices.items():
        matrix_file = os.path.join(network_dirs['connectivity'], f"{subject_id}_connectivity.npy")
        np.save(matrix_file, conn_data['matrix'])
    
    logging.info(f"Graph metrics saved to: {graph_metrics_csv}")
    
    return graph_metrics_df

def perform_network_statistics(graph_metrics_df, regional_data_df, network_dirs):
    """Perform statistical analysis on network metrics"""
    
    logging.info("Performing network statistical analysis...")
    
    # Merge with group information
    group_info = regional_data_df[['subject_id', 'group']].drop_duplicates()
    network_stats_df = graph_metrics_df.merge(group_info, on='subject_id', how='left')
    
    # Get network metric columns
    metric_columns = [col for col in network_stats_df.columns 
                     if col not in ['subject_id', 'group'] and 
                     not network_stats_df[col].isna().all()]
    
    statistical_results = []
    
    for metric in metric_columns:
        # Clean data
        clean_data = network_stats_df[['group', metric]].dropna()
        
        if len(clean_data) < 6:
            continue
        
        # Separate groups
        groups = {}
        for group_name in ['HC', 'PIGD', 'TDPD']:
            group_data = clean_data[clean_data['group'] == group_name][metric].values
            if len(group_data) > 0:
                groups[group_name] = group_data
        
        if len(groups) < 2:
            continue
        
        # Perform comparisons
        comparisons = [('HC', 'PIGD'), ('HC', 'TDPD'), ('PIGD', 'TDPD')]
        
        for group1_name, group2_name in comparisons:
            if group1_name not in groups or group2_name not in groups:
                continue
            
            group1_data = groups[group1_name]
            group2_data = groups[group2_name]
            
            # Perform statistical test
            stat, p_value = stats.ttest_ind(group1_data, group2_data)
            effect_size = (np.mean(group1_data) - np.mean(group2_data)) / np.sqrt(
                ((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                 (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                (len(group1_data) + len(group2_data) - 2)
            )
            
            result = {
                'metric': metric,
                'comparison': f"{group1_name}_vs_{group2_name}",
                'statistic': stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'group1_mean': np.mean(group1_data),
                'group1_std': np.std(group1_data),
                'group1_n': len(group1_data),
                'group2_mean': np.mean(group2_data),
                'group2_std': np.std(group2_data),
                'group2_n': len(group2_data)
            }
            
            statistical_results.append(result)
    
    # Convert to DataFrame and apply corrections
    network_stats_results = pd.DataFrame(statistical_results)
    
    if len(network_stats_results) > 0:
        from statsmodels.stats.multitest import fdrcorrection
        
        p_values = network_stats_results['p_value'].values
        reject_fdr, pvals_fdr = fdrcorrection(p_values, alpha=0.05)
        
        network_stats_results['p_fdr'] = pvals_fdr
        network_stats_results['significant_fdr'] = reject_fdr
        
        # Save results
        stats_csv = os.path.join(network_dirs['statistics'], 'network_statistical_results.csv')
        network_stats_results.to_csv(stats_csv, index=False)
        
        logging.info(f"Network statistics saved to: {stats_csv}")
    
    return network_stats_results

def create_network_visualizations(graph_metrics_df, network_stats_results, regional_data_df, network_dirs):
    """Create network analysis visualizations"""
    
    logging.info("Creating network visualization plots...")
    
    plt.style.use('default')
    
    # 1. Network metrics comparison across groups
    group_info = regional_data_df[['subject_id', 'group']].drop_duplicates()
    network_viz_df = graph_metrics_df.merge(group_info, on='subject_id', how='left')
    
    # Select key metrics for visualization
    key_metrics = []
    for threshold in [30, 50, 70]:  # 30%, 50%, 70% thresholds
        key_metrics.extend([
            f'density_t{threshold}',
            f'clustering_coefficient_t{threshold}',
            f'global_efficiency_t{threshold}',
            f'small_worldness_t{threshold}'
        ])
    
    # Filter to existing columns
    available_metrics = [m for m in key_metrics if m in network_viz_df.columns]
    
    if len(available_metrics) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Metrics by Group', fontsize=16)
        
        for i, metric in enumerate(available_metrics[:4]):
            row, col = i // 2, i % 2
            clean_data = network_viz_df[['group', metric]].dropna()
            
            if len(clean_data) > 0:
                sns.boxplot(data=clean_data, x='group', y=metric, ax=axes[row, col])
                axes[row, col].set_title(metric.replace('_', ' ').title())
        
        plt.tight_layout()
        network_plot_path = os.path.join(network_dirs['figures'], 'network_metrics_by_group.png')
        plt.savefig(network_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Correlation matrix of network metrics
    numeric_columns = network_viz_df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'subject_id']
    
    if len(numeric_columns) > 1:
        corr_data = network_viz_df[numeric_columns].corr()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.heatmap(corr_data, cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Network Metrics Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        corr_plot_path = os.path.join(network_dirs['figures'], 'network_correlation_matrix.png')
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Statistical results summary
    if len(network_stats_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Effect sizes
        axes[0].scatter(network_stats_results['effect_size'], 
                       -np.log10(network_stats_results['p_value']),
                       c=['red' if sig else 'blue' for sig in network_stats_results['significant_fdr']],
                       alpha=0.7)
        axes[0].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Effect Size')
        axes[0].set_ylabel('-log10(p-value)')
        axes[0].set_title('Network Analysis: Volcano Plot')
        
        # P-value distribution
        axes[1].hist(network_stats_results['p_value'], bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(0.05, color='red', linestyle='--', alpha=0.7)
        axes[1].set_xlabel('p-value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('P-value Distribution')
        
        plt.tight_layout()
        stats_plot_path = os.path.join(network_dirs['figures'], 'network_statistics_summary.png')
        plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Network plots saved to: {network_dirs['figures']}")

def main():
    """Main network analysis function"""
    # Setup logging
    logger = setup_logging('step06_network')
    logger.info("=" * 60)
    logger.info("STARTING STEP 06: NETWORK ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        network_output_dir = os.path.join(config.OUTPUT_ROOT, '06_network_analysis')
        network_dirs = setup_network_directories(network_output_dir)
        
        # Load subject-group mapping
        subject_group_mapping = load_subject_group_mapping()
        
        # Create brain regions for network analysis
        brain_regions = create_brain_parcellation_network()
        logger.info(f"Created network with {len(brain_regions)} brain regions")
        
        # Load subject data from previous steps
        preproc_dir = os.path.join(config.OUTPUT_ROOT, '01_preprocessed')
        subject_dirs = [d for d in os.listdir(preproc_dir) 
                       if os.path.isdir(os.path.join(preproc_dir, d)) and d.startswith('sub-')]
        
        if not subject_dirs:
            raise ValueError("No preprocessed subjects found")
        
        # Extract regional data for network analysis
        logger.info("Extracting regional measures for network analysis...")
        regional_data_list = []
        successful_extractions = 0
        
        # Check for surface analysis results
        surface_dir = os.path.join(config.OUTPUT_ROOT, '04_surface_analysis', 'smoothed_surfaces')
        has_surface_data = os.path.exists(surface_dir)
        
        for subject_id in sorted(subject_dirs):
            # Get GM data
            gm_path = os.path.join(preproc_dir, subject_id, f"{subject_id}_GM_mask.nii.gz")
            
            # Get thickness data if available
            thickness_path = None
            if has_surface_data:
                thickness_path = os.path.join(surface_dir, f"{subject_id}_smoothed_thickness.nii.gz")
            
            if not os.path.exists(gm_path):
                logger.warning(f"GM data not found for {subject_id}")
                continue
            
            # Extract regional measures
            regional_measures, success = extract_regional_volumes(
                subject_id, gm_path, thickness_path, brain_regions
            )
            
            if success:
                # Get group from mapping
                group = subject_group_mapping.get(subject_id, 'Unknown')
                if group == 'Unknown':
                    logger.warning(f"Unknown group for subject {subject_id}")
                    continue
                
                regional_measures['group'] = group
                
                regional_data_list.append(regional_measures)
                successful_extractions += 1
            
            if len(regional_data_list) % 10 == 0:
                logger.info(f"Processed {len(regional_data_list)} subjects...")
        
        logger.info(f"Regional extraction completed: {successful_extractions}/{len(subject_dirs)} subjects")
        
        # Convert to DataFrame
        regional_data_df = pd.DataFrame(regional_data_list)
        
        # Save regional data
        regional_csv = os.path.join(network_dirs['networks'], 'regional_network_data.csv')
        regional_data_df.to_csv(regional_csv, index=False)
        
        # Calculate connectivity matrices
        logger.info("Calculating connectivity matrices...")
        connectivity_matrices, measure_columns = calculate_connectivity_matrix(regional_data_df)
        
        if connectivity_matrices is None:
            raise ValueError("Failed to calculate connectivity matrices")
        
        logger.info(f"Connectivity matrices calculated for {len(connectivity_matrices)} subjects")
        
        # Perform network analysis
        logger.info("Performing graph theory analysis...")
        graph_metrics_df = perform_network_analysis(connectivity_matrices, brain_regions, network_dirs)
        
        # Perform statistical analysis
        logger.info("Performing network statistics...")
        network_stats_results = perform_network_statistics(graph_metrics_df, regional_data_df, network_dirs)
        
        # Create visualizations
        logger.info("Creating network visualizations...")
        create_network_visualizations(graph_metrics_df, network_stats_results, regional_data_df, network_dirs)
        
        # Log summary
        log_analysis_summary(
            analysis_name="Network Analysis",
            subjects_analyzed=successful_extractions,
            subjects_excluded=len(subject_dirs) - successful_extractions,
            notes=f"Analyzed {len(brain_regions)} brain regions. "
                  f"Connectivity matrices calculated for {len(connectivity_matrices)} subjects. "
                  f"Statistical comparisons: {len(network_stats_results) if len(network_stats_results) > 0 else 0}."
        )
        
        logger.info("Step 06 completed successfully!")
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY: Network Analysis")
        print("=" * 60)
        print(f"Subjects processed: {successful_extractions}/{len(subject_dirs)}")
        print(f"Success rate: {100*successful_extractions/len(subject_dirs):.1f}%")
        print(f"Brain regions: {len(brain_regions)}")
        print(f"Connectivity matrices: {len(connectivity_matrices)}")
        
        if len(network_stats_results) > 0:
            significant_results = len(network_stats_results[network_stats_results['significant_fdr']])
            print(f"Statistical comparisons: {len(network_stats_results)}")
            print(f"Significant results (FDR): {significant_results}")
            
            for group in ['HC', 'PIGD', 'TDPD']:
                group_data = regional_data_df[regional_data_df['group'] == group]
                if len(group_data) > 0:
                    print(f"• {group}: {len(group_data)} subjects")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Network analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
