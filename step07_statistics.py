#!/usr/bin/env python3
"""
Step 07: Comprehensive Statistical Analysis
YOPD Structural Analysis Pipeline

This script performs comprehensive statistical analysis across all modalities:
1. Integrative statistical analysis across VBM, surface, ROI, and network data
2. Multi-modal correlations
3. Machine learning classification
4. Comprehensive multiple comparison corrections
5. Effect size and power analysis

Author: GitHub Copilot
Date: August 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.power import ttest_power
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def setup_statistics_directories(output_dir):
    """Create statistics-specific output directories"""
    stats_dirs = {
        'integrated': os.path.join(output_dir, 'integrated_analysis'),
        'multimodal': os.path.join(output_dir, 'multimodal_correlations'),
        'classification': os.path.join(output_dir, 'machine_learning'),
        'power_analysis': os.path.join(output_dir, 'power_analysis'),
        'meta_analysis': os.path.join(output_dir, 'meta_analysis'),
        'figures': os.path.join(output_dir, 'figures'),
        'reports': os.path.join(output_dir, 'final_reports')
    }
    
    for name, path in stats_dirs.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")
    
    return stats_dirs

def load_all_analysis_results():
    """Load results from all previous analysis steps"""
    results = {}
    
    try:
        # VBM results - check what's actually available
        vbm_stats_file = os.path.join(config.OUTPUT_ROOT, '03_vbm_analysis', 'statistics', 'vbm_statistics.csv')
        if os.path.exists(vbm_stats_file):
            results['vbm'] = pd.read_csv(vbm_stats_file)
            logging.info(f"Loaded VBM results: {len(results['vbm'])} subjects")
        
        # ROI results
        roi_file = os.path.join(config.OUTPUT_ROOT, '05_roi_analysis', 'extracted_data', 'roi_measures.csv')
        if os.path.exists(roi_file):
            results['roi'] = pd.read_csv(roi_file)
            logging.info(f"Loaded ROI results: {len(results['roi'])} subjects")
        
        # Network results - use correct file name
        network_file = os.path.join(config.OUTPUT_ROOT, '06_network_analysis', 'network_analysis', 'regional_network_data.csv')
        if os.path.exists(network_file):
            results['network'] = pd.read_csv(network_file)
            logging.info(f"Loaded network results: {len(results['network'])} subjects")
        
        # Surface results (if available)
        surface_file = os.path.join(config.OUTPUT_ROOT, '04_surface_analysis', 'surface_metrics.csv')
        if os.path.exists(surface_file):
            results['surface'] = pd.read_csv(surface_file)
            logging.info(f"Loaded surface results: {len(results['surface'])} subjects")
        
    except Exception as e:
        logging.error(f"Error loading analysis results: {str(e)}")
    
    return results

def create_integrated_dataset(results):
    """Create an integrated dataset combining all modalities"""
    
    if not results:
        logging.warning("No analysis results available for integration")
        return None
    
    # Start with the largest dataset as base
    base_dataset = None
    base_name = None
    max_subjects = 0
    
    for name, df in results.items():
        if len(df) > max_subjects:
            max_subjects = len(df)
            base_dataset = df[['subject_id', 'group']].copy()
            base_name = name
    
    if base_dataset is None:
        return None
    
    logging.info(f"Using {base_name} as base dataset with {len(base_dataset)} subjects")
    
    # Merge all datasets
    integrated_df = base_dataset.copy()
    
    for name, df in results.items():
        if name == base_name:
            # Add all columns except subject_id and group
            merge_cols = [col for col in df.columns if col not in ['subject_id', 'group']]
            if merge_cols:
                for col in merge_cols:
                    integrated_df[f"{name}_{col}"] = df.set_index('subject_id')[col]
        else:
            # Merge with suffix
            merge_df = df.copy()
            if 'group' in merge_df.columns:
                merge_df = merge_df.drop('group', axis=1)
            
            # Add prefix to column names
            rename_dict = {col: f"{name}_{col}" for col in merge_df.columns if col != 'subject_id'}
            merge_df = merge_df.rename(columns=rename_dict)
            
            integrated_df = integrated_df.merge(merge_df, on='subject_id', how='left')
    
    # Remove columns that are all NaN
    integrated_df = integrated_df.dropna(axis=1, how='all')
    
    logging.info(f"Integrated dataset created: {len(integrated_df)} subjects, {len(integrated_df.columns)} features")
    
    return integrated_df

def perform_multimodal_analysis(integrated_df, stats_dirs):
    """Perform comprehensive multimodal statistical analysis"""
    
    logging.info("Performing multimodal statistical analysis...")
    
    if integrated_df is None or len(integrated_df) == 0:
        logging.warning("No integrated data available for multimodal analysis")
        return None
    
    # Get feature columns (exclude subject_id and group)
    feature_columns = [col for col in integrated_df.columns 
                      if col not in ['subject_id', 'group'] and 
                      integrated_df[col].dtype in ['int64', 'float64']]
    
    if len(feature_columns) < 2:
        logging.warning("Insufficient features for multimodal analysis")
        return None
    
    # Clean data
    clean_df = integrated_df[['subject_id', 'group'] + feature_columns].dropna()
    
    logging.info(f"Clean dataset: {len(clean_df)} subjects, {len(feature_columns)} features")
    
    # Perform group comparisons for each feature
    statistical_results = []
    
    for feature in feature_columns:
        # Separate groups
        groups = {}
        for group_name in ['HC', 'PIGD', 'TDPD']:
            group_data = clean_df[clean_df['group'] == group_name][feature].values
            if len(group_data) > 0:
                groups[group_name] = group_data
        
        if len(groups) < 2:
            continue
        
        # Perform pairwise comparisons
        comparisons = [('HC', 'PIGD'), ('HC', 'TDPD'), ('PIGD', 'TDPD')]
        
        for group1_name, group2_name in comparisons:
            if group1_name not in groups or group2_name not in groups:
                continue
            
            group1_data = groups[group1_name]
            group2_data = groups[group2_name]
            
            if len(group1_data) < 3 or len(group2_data) < 3:
                continue
            
            # Perform statistical test
            stat, p_value = ttest_ind(group1_data, group2_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                 (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                                (len(group1_data) + len(group2_data) - 2))
            
            if pooled_std > 0:
                effect_size = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
            else:
                effect_size = 0
            
            # Calculate power
            try:
                power = ttest_power(effect_size, nobs=min(len(group1_data), len(group2_data)), alpha=0.05)
            except:
                power = np.nan
            
            result = {
                'feature': feature,
                'modality': feature.split('_')[0] if '_' in feature else 'unknown',
                'comparison': f"{group1_name}_vs_{group2_name}",
                'statistic': stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'power': power,
                'group1_mean': np.mean(group1_data),
                'group1_std': np.std(group1_data),
                'group1_n': len(group1_data),
                'group2_mean': np.mean(group2_data),
                'group2_std': np.std(group2_data),
                'group2_n': len(group2_data)
            }
            
            statistical_results.append(result)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(statistical_results)
    
    if len(stats_df) == 0:
        logging.warning("No statistical comparisons could be performed")
        return stats_df
    
    # Apply multiple comparison corrections
    p_values = stats_df['p_value'].values
    
    # FDR correction
    reject_fdr, pvals_fdr = fdrcorrection(p_values, alpha=0.05)
    stats_df['p_fdr'] = pvals_fdr
    stats_df['significant_fdr'] = reject_fdr
    
    # Bonferroni correction  
    pvals_bonf = p_values * len(p_values)  # Simple Bonferroni
    pvals_bonf = np.minimum(pvals_bonf, 1.0)  # Cap at 1.0
    stats_df['p_bonferroni'] = pvals_bonf
    stats_df['significant_bonferroni'] = pvals_bonf < 0.05
    
    # Save results
    multimodal_csv = os.path.join(stats_dirs['integrated'], 'multimodal_statistical_results.csv')
    stats_df.to_csv(multimodal_csv, index=False)
    
    logging.info(f"Multimodal analysis completed: {len(stats_df)} comparisons")
    logging.info(f"Results saved to: {multimodal_csv}")
    
    return stats_df

def perform_classification_analysis(integrated_df, stats_dirs):
    """Perform machine learning classification analysis"""
    
    logging.info("Performing classification analysis...")
    
    if integrated_df is None or len(integrated_df) == 0:
        return None
    
    # Prepare data for classification
    feature_columns = [col for col in integrated_df.columns 
                      if col not in ['subject_id', 'group'] and 
                      integrated_df[col].dtype in ['int64', 'float64']]
    
    # Clean data
    clean_df = integrated_df[['group'] + feature_columns].dropna()
    
    if len(clean_df) < 10:
        logging.warning("Insufficient data for classification analysis")
        return None
    
    X = clean_df[feature_columns].values
    y = clean_df['group'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    classification_results = {}
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for clf_name, clf in classifiers.items():
        logging.info(f"Training {clf_name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Fit on full data for feature importance
        clf.fit(X_scaled, y)
        
        # Get feature importance
        if hasattr(clf, 'feature_importances_'):
            feature_importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            feature_importance = np.abs(clf.coef_).mean(axis=0)
        else:
            feature_importance = np.zeros(len(feature_columns))
        
        classification_results[clf_name] = {
            'cv_accuracy': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'feature_importance': feature_importance,
            'feature_names': feature_columns
        }
        
        logging.info(f"{clf_name} - Mean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Save classification results
    class_results = []
    for clf_name, results in classification_results.items():
        for i, (feature, importance) in enumerate(zip(results['feature_names'], results['feature_importance'])):
            class_results.append({
                'classifier': clf_name,
                'feature': feature,
                'importance': importance,
                'mean_accuracy': results['mean_accuracy'],
                'std_accuracy': results['std_accuracy']
            })
    
    class_df = pd.DataFrame(class_results)
    class_csv = os.path.join(stats_dirs['classification'], 'classification_results.csv')
    class_df.to_csv(class_csv, index=False)
    
    logging.info(f"Classification results saved to: {class_csv}")
    
    return classification_results

def create_correlation_analysis(integrated_df, stats_dirs):
    """Perform comprehensive correlation analysis across modalities"""
    
    logging.info("Performing correlation analysis...")
    
    if integrated_df is None:
        return None
    
    # Get numeric columns by modality
    modalities = {}
    
    for col in integrated_df.columns:
        if col in ['subject_id', 'group']:
            continue
        
        if integrated_df[col].dtype in ['int64', 'float64']:
            modality = col.split('_')[0] if '_' in col else 'other'
            if modality not in modalities:
                modalities[modality] = []
            modalities[modality].append(col)
    
    # Calculate within and between modality correlations
    correlation_results = []
    
    # Clean data
    numeric_columns = [col for cols in modalities.values() for col in cols]
    clean_df = integrated_df[numeric_columns].dropna()
    
    if len(clean_df) < 10:
        logging.warning("Insufficient data for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = clean_df.corr()
    
    # Extract significant correlations
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
            if i >= j:  # Avoid duplicates and self-correlations
                continue
            
            correlation = corr_matrix.loc[col1, col2]
            
            if abs(correlation) > 0.3:  # Only report moderate to strong correlations
                mod1 = col1.split('_')[0] if '_' in col1 else 'other'
                mod2 = col2.split('_')[0] if '_' in col2 else 'other'
                
                # Calculate p-value
                n = len(clean_df)
                t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                
                correlation_results.append({
                    'feature1': col1,
                    'feature2': col2,
                    'modality1': mod1,
                    'modality2': mod2,
                    'correlation_type': 'within' if mod1 == mod2 else 'between',
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_subjects': n
                })
    
    # Convert to DataFrame and apply corrections
    corr_df = pd.DataFrame(correlation_results)
    
    if len(corr_df) > 0:
        p_values = corr_df['p_value'].values
        reject_fdr, pvals_fdr = fdrcorrection(p_values, alpha=0.05)
        
        corr_df['p_fdr'] = pvals_fdr
        corr_df['significant_fdr'] = reject_fdr
        
        # Save results
        corr_csv = os.path.join(stats_dirs['multimodal'], 'correlation_analysis.csv')
        corr_df.to_csv(corr_csv, index=False)
        
        logging.info(f"Correlation analysis completed: {len(corr_df)} correlations analyzed")
        logging.info(f"Results saved to: {corr_csv}")
    
    return corr_df

def generate_comprehensive_report(stats_df, classification_results, correlation_df, stats_dirs):
    """Generate comprehensive final report"""
    
    logging.info("Generating comprehensive analysis report...")
    
    report_file = os.path.join(stats_dirs['reports'], 'comprehensive_analysis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("YOPD STRUCTURAL ANALYSIS - COMPREHENSIVE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write("This report summarizes the comprehensive structural MRI analysis\n")
        f.write("of Young-Onset Parkinson's Disease (YOPD) including:\n")
        f.write("• Voxel-Based Morphometry (VBM)\n")
        f.write("• Surface-Based Cortical Thickness Analysis\n")
        f.write("• Region of Interest (ROI) Analysis\n")
        f.write("• Network Analysis\n")
        f.write("• Integrative Statistical Analysis\n\n")
        
        # Statistical results summary
        if stats_df is not None and len(stats_df) > 0:
            f.write("STATISTICAL ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total comparisons performed: {len(stats_df)}\n")
            f.write(f"Significant results (FDR corrected): {len(stats_df[stats_df['significant_fdr']])}\n")
            f.write(f"Significant results (Bonferroni): {len(stats_df[stats_df['significant_bonferroni']])}\n\n")
            
            # Effect size summary
            large_effects = len(stats_df[abs(stats_df['effect_size']) > 0.8])
            medium_effects = len(stats_df[(abs(stats_df['effect_size']) > 0.5) & (abs(stats_df['effect_size']) <= 0.8)])
            small_effects = len(stats_df[(abs(stats_df['effect_size']) > 0.2) & (abs(stats_df['effect_size']) <= 0.5)])
            
            f.write("EFFECT SIZES:\n")
            f.write(f"• Large effects (|d| > 0.8): {large_effects}\n")
            f.write(f"• Medium effects (0.5 < |d| ≤ 0.8): {medium_effects}\n")
            f.write(f"• Small effects (0.2 < |d| ≤ 0.5): {small_effects}\n\n")
            
            # Top significant results
            significant_results = stats_df[stats_df['significant_fdr']].sort_values('p_fdr').head(10)
            
            if len(significant_results) > 0:
                f.write("TOP SIGNIFICANT FINDINGS (FDR corrected):\n")
                f.write("-" * 40 + "\n")
                
                for _, row in significant_results.iterrows():
                    f.write(f"{row['feature']} ({row['comparison']}):\n")
                    f.write(f"  p-FDR: {row['p_fdr']:.6f}\n")
                    f.write(f"  Effect size: {row['effect_size']:.3f}\n")
                    f.write(f"  Modality: {row.get('modality', 'unknown')}\n\n")
        
        # Classification results
        if classification_results:
            f.write("MACHINE LEARNING CLASSIFICATION\n")
            f.write("-" * 35 + "\n")
            
            for clf_name, results in classification_results.items():
                f.write(f"{clf_name}:\n")
                f.write(f"  Cross-validation accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}\n")
                
                # Top features
                feature_importance = results['feature_importance']
                feature_names = results['feature_names']
                
                top_indices = np.argsort(feature_importance)[-5:][::-1]
                f.write("  Top 5 features:\n")
                for idx in top_indices:
                    f.write(f"    {feature_names[idx]}: {feature_importance[idx]:.3f}\n")
                f.write("\n")
        
        # Correlation results
        if correlation_df is not None and len(correlation_df) > 0:
            f.write("CROSS-MODAL CORRELATIONS\n")
            f.write("-" * 25 + "\n")
            
            between_modal = correlation_df[correlation_df['correlation_type'] == 'between']
            significant_between = between_modal[between_modal['significant_fdr']]
            
            f.write(f"Total correlations analyzed: {len(correlation_df)}\n")
            f.write(f"Between-modality correlations: {len(between_modal)}\n")
            f.write(f"Significant between-modality (FDR): {len(significant_between)}\n\n")
            
            if len(significant_between) > 0:
                f.write("SIGNIFICANT CROSS-MODAL CORRELATIONS:\n")
                f.write("-" * 40 + "\n")
                
                for _, row in significant_between.head(10).iterrows():
                    f.write(f"{row['modality1']} - {row['modality2']}:\n")
                    f.write(f"  r = {row['correlation']:.3f}, p-FDR = {row['p_fdr']:.6f}\n")
                    f.write(f"  Features: {row['feature1']} ~ {row['feature2']}\n\n")
        
        f.write("CONCLUSIONS\n")
        f.write("-" * 12 + "\n")
        f.write("This comprehensive analysis provides insights into structural\n")
        f.write("brain differences in YOPD across multiple imaging modalities.\n")
        f.write("The integrative approach allows for identification of consistent\n")
        f.write("patterns of neuroanatomical changes and their relationships.\n")
    
    logging.info(f"Comprehensive report saved to: {report_file}")

def main():
    """Main comprehensive statistics function"""
    # Setup logging
    logger = setup_logging('step07_statistics')
    logger.info("=" * 60)
    logger.info("STARTING STEP 07: COMPREHENSIVE STATISTICAL ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        stats_output_dir = os.path.join(config.OUTPUT_ROOT, '07_statistics')
        stats_dirs = setup_statistics_directories(stats_output_dir)
        
        # Load all analysis results
        logger.info("Loading analysis results from all modalities...")
        results = load_all_analysis_results()
        
        if not results:
            logger.warning("No analysis results found")
            return
        
        # Create integrated dataset
        logger.info("Creating integrated multimodal dataset...")
        integrated_df = create_integrated_dataset(results)
        
        if integrated_df is not None:
            # Save integrated dataset
            integrated_csv = os.path.join(stats_dirs['integrated'], 'integrated_dataset.csv')
            integrated_df.to_csv(integrated_csv, index=False)
            logger.info(f"Integrated dataset saved: {integrated_csv}")
        
        # Perform multimodal analysis
        logger.info("Performing multimodal statistical analysis...")
        stats_df = perform_multimodal_analysis(integrated_df, stats_dirs)
        
        # Perform classification analysis
        logger.info("Performing classification analysis...")
        classification_results = perform_classification_analysis(integrated_df, stats_dirs)
        
        # Perform correlation analysis
        logger.info("Performing correlation analysis...")
        correlation_df = create_correlation_analysis(integrated_df, stats_dirs)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        generate_comprehensive_report(stats_df, classification_results, correlation_df, stats_dirs)
        
        # Log summary
        subjects_analyzed = len(integrated_df) if integrated_df is not None else 0
        comparisons_performed = len(stats_df) if stats_df is not None else 0
        
        log_analysis_summary(
            analysis_name="Comprehensive Statistical Analysis",
            subjects_analyzed=subjects_analyzed,
            subjects_excluded=0,
            notes=f"Integrated {len(results)} modalities. "
                  f"Performed {comparisons_performed} statistical comparisons. "
                  f"Machine learning classification completed."
        )
        
        logger.info("Step 07 completed successfully!")
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY: Comprehensive Statistics")
        print("=" * 60)
        print(f"Modalities integrated: {len(results)}")
        print(f"Subjects analyzed: {subjects_analyzed}")
        print(f"Statistical comparisons: {comparisons_performed}")
        
        if stats_df is not None and len(stats_df) > 0:
            significant_fdr = len(stats_df[stats_df['significant_fdr']])
            print(f"Significant results (FDR): {significant_fdr}/{len(stats_df)}")
        
        if classification_results:
            for clf_name, results in classification_results.items():
                print(f"{clf_name} accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Comprehensive statistical analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
