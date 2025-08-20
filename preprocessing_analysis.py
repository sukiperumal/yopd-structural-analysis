#!/usr/bin/env python3
"""
YOPD Structural Analysis - Master Runner
=======================================

This script runs all YOPD preprocessing pipeline analysis components
and generates a comprehensive summary report.
"""

import os
import sys
import logging
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MasterAnalysisConfig:
    """Configuration for master analysis"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    run_image_quality: bool = True
    run_bias_correction: bool = True
    run_brain_extraction: bool = True
    run_tissue_segmentation: bool = True
    generate_master_report: bool = True
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']

class YOPDMasterAnalyzer:
    """Master analyzer for YOPD preprocessing pipeline"""
    
    def __init__(self, config: MasterAnalysisConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.master_results_dir = Path(config.output_root) / "00_master_analysis"
        self.master_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis components
        self.components = {
            'image_quality': 'image_assessment.py',
            'bias_correction': 'bias_field_analysis.py',
            'brain_extraction': 'brain_extraction_qc.py',
            'tissue_segmentation': 'tissue_segmentation_qc.py'
        }
        
        self.analysis_results = {}
    
    def _setup_logging(self):
        """Setup comprehensive logging for master analysis"""
        logger = logging.getLogger('yopd_master_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create log directory
            log_dir = Path(self.config.output_root) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler
            log_file = log_dir / "master_analysis.log"
            file_handler = logging.FileHandler(log_file)
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_analysis_component(self, component_name: str, script_path: str) -> bool:
        """Run an individual analysis component"""
        try:
            self.logger.info(f"Starting {component_name} analysis...")
            
            # Check if script exists
            if not Path(script_path).exists():
                self.logger.error(f"Script not found: {script_path}")
                return False
            
            # Run the script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"{component_name} analysis completed successfully")
                return True
            else:
                self.logger.error(f"{component_name} analysis failed:")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run {component_name} analysis: {e}")
            return False
    
    def load_analysis_results(self) -> Dict:
        """Load results from all analysis components"""
        results = {}
        
        # Load image quality results
        if self.config.run_image_quality:
            quality_file = Path(self.config.output_root) / "quality_assessment" / "image_quality_metrics.csv"
            if quality_file.exists():
                try:
                    results['image_quality'] = pd.read_csv(quality_file)
                    self.logger.info(f"Loaded image quality results: {len(results['image_quality'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load image quality results: {e}")
        
        # Load bias correction results
        if self.config.run_bias_correction:
            bias_file = Path(self.config.output_root) / "bias_field_analysis" / "bias_correction_analysis.csv"
            if bias_file.exists():
                try:
                    results['bias_correction'] = pd.read_csv(bias_file)
                    self.logger.info(f"Loaded bias correction results: {len(results['bias_correction'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load bias correction results: {e}")
        
        # Load brain extraction results
        if self.config.run_brain_extraction:
            brain_file = Path(self.config.output_root) / "brain_extraction_qc" / "brain_extraction_qc.csv"
            if brain_file.exists():
                try:
                    results['brain_extraction'] = pd.read_csv(brain_file)
                    self.logger.info(f"Loaded brain extraction results: {len(results['brain_extraction'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load brain extraction results: {e}")
        
        # Load tissue segmentation results
        if self.config.run_tissue_segmentation:
            tissue_file = Path(self.config.output_root) / "tissue_segmentation_qc" / "tissue_segmentation_qc.csv"
            if tissue_file.exists():
                try:
                    results['tissue_segmentation'] = pd.read_csv(tissue_file)
                    self.logger.info(f"Loaded tissue segmentation results: {len(results['tissue_segmentation'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load tissue segmentation results: {e}")
        
        return results
    
    def create_master_summary_statistics(self, results: Dict) -> Dict:
        """Create comprehensive summary statistics across all components"""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_subjects_by_component': {},
            'success_rates': {},
            'key_metrics': {}
        }
        
        for component_name, df in results.items():
            if df is not None and not df.empty:
                total_subjects = len(df)
                successful_subjects = len(df[~df.get('analysis_failed', False).fillna(False)])
                success_rate = successful_subjects / total_subjects if total_subjects > 0 else 0
                
                summary['total_subjects_by_component'][component_name] = total_subjects
                summary['success_rates'][component_name] = success_rate
                
                # Extract key metrics for each component
                component_metrics = {}
                
                if component_name == 'image_quality':
                    successful_df = df[~df.get('analysis_failed', False).fillna(False)]
                    if not successful_df.empty:
                        component_metrics = {
                            'mean_snr': successful_df.get('orig_snr', pd.Series()).mean(),
                            'mean_cnr': successful_df.get('orig_cnr', pd.Series()).mean(),
                            'mean_uniformity': successful_df.get('orig_uniformity', pd.Series()).mean(),
                            'snr_adequate_rate': successful_df.get('snr_adequate', pd.Series()).mean(),
                            'uniformity_good_rate': successful_df.get('uniform#!/usr/bin/env python3
"""
YOPD Structural Analysis - Master Runner
=======================================

This script runs all YOPD preprocessing pipeline analysis components
and generates a comprehensive summary report.
"""

import os
import sys
import logging
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MasterAnalysisConfig:
    """Configuration for master analysis"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    run_image_quality: bool = True
    run_bias_correction: bool = True
    run_brain_extraction: bool = True
    run_tissue_segmentation: bool = True
    generate_master_report: bool = True
    figure_dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']

class YOPDMasterAnalyzer:
    """Master analyzer for YOPD preprocessing pipeline"""
    
    def __init__(self, config: MasterAnalysisConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.master_results_dir = Path(config.output_root) / "00_master_analysis"
        self.master_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis components
        self.components = {
            'image_quality': 'image_assessment.py',
            'bias_correction': 'bias_field_analysis.py',
            'brain_extraction': 'brain_extraction_qc.py',
            'tissue_segmentation': 'tissue_segmentation_qc.py'
        }
        
        self.analysis_results = {}
    
    def _setup_logging(self):
        """Setup comprehensive logging for master analysis"""
        logger = logging.getLogger('yopd_master_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create log directory
            log_dir = Path(self.config.output_root) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler
            log_file = log_dir / "master_analysis.log"
            file_handler = logging.FileHandler(log_file)
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_analysis_component(self, component_name: str, script_path: str) -> bool:
        """Run an individual analysis component"""
        try:
            self.logger.info(f"Starting {component_name} analysis...")
            
            # Check if script exists
            if not Path(script_path).exists():
                self.logger.error(f"Script not found: {script_path}")
                return False
            
            # Run the script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"{component_name} analysis completed successfully")
                return True
            else:
                self.logger.error(f"{component_name} analysis failed:")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run {component_name} analysis: {e}")
            return False
    
    def load_analysis_results(self) -> Dict:
        """Load results from all analysis components"""
        results = {}
        
        # Load image quality results
        if self.config.run_image_quality:
            quality_file = Path(self.config.output_root) / "quality_assessment" / "image_quality_metrics.csv"
            if quality_file.exists():
                try:
                    results['image_quality'] = pd.read_csv(quality_file)
                    self.logger.info(f"Loaded image quality results: {len(results['image_quality'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load image quality results: {e}")
        
        # Load bias correction results
        if self.config.run_bias_correction:
            bias_file = Path(self.config.output_root) / "bias_field_analysis" / "bias_correction_analysis.csv"
            if bias_file.exists():
                try:
                    results['bias_correction'] = pd.read_csv(bias_file)
                    self.logger.info(f"Loaded bias correction results: {len(results['bias_correction'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load bias correction results: {e}")
        
        # Load brain extraction results
        if self.config.run_brain_extraction:
            brain_file = Path(self.config.output_root) / "brain_extraction_qc" / "brain_extraction_qc.csv"
            if brain_file.exists():
                try:
                    results['brain_extraction'] = pd.read_csv(brain_file)
                    self.logger.info(f"Loaded brain extraction results: {len(results['brain_extraction'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load brain extraction results: {e}")
        
        # Load tissue segmentation results
        if self.config.run_tissue_segmentation:
            tissue_file = Path(self.config.output_root) / "tissue_segmentation_qc" / "tissue_segmentation_qc.csv"
            if tissue_file.exists():
                try:
                    results['tissue_segmentation'] = pd.read_csv(tissue_file)
                    self.logger.info(f"Loaded tissue segmentation results: {len(results['tissue_segmentation'])} subjects")
                except Exception as e:
                    self.logger.warning(f"Failed to load tissue segmentation results: {e}")
        
        return results
    
    def create_master_summary_statistics(self, results: Dict) -> Dict:
        """Create comprehensive summary statistics across all components"""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_subjects_by_component': {},
            'success_rates': {},
            'key_metrics': {}
        }
        
        for component_name, df in results.items():
            if df is not None and not df.empty:
                total_subjects = len(df)
                successful_subjects = len(df[~df.get('analysis_failed', False).fillna(False)])
                success_rate = successful_subjects / total_subjects if total_subjects > 0 else 0
                
                summary['total_subjects_by_component'][component_name] = total_subjects
                summary['success_rates'][component_name] = success_rate
                
                # Extract key metrics for each component
                component_metrics = {}
                
                if component_name == 'image_quality':
                    successful_df =