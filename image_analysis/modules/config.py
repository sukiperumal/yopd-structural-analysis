#!/usr/bin/env python3
"""
Configuration module for enhanced image quality assessment
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class QualityConfig:
    """Configuration for quality assessment with technical specifications"""
    output_root: str = r"D:\data_NIMHANS\outputs"
    min_snr: float = 15.0
    max_intensity_nonuniformity: float = 0.3
    min_brain_volume: int = 800000  # voxels
    figure_dpi: int = 300
    color_palette: List[str] = field(default_factory=lambda: ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'])
    
    # Technical parameters
    brain_extraction_method: str = "threshold_based"  # or "bet_style"
    noise_estimation_method: str = "edge_regions"  # or "mad", "background_roi"
    uniformity_metric: str = "mad_based"  # or "coefficient_of_variation", "ratio_method"
    
    def __post_init__(self):
        """Validate configuration parameters"""
        valid_brain_methods = ["threshold_based", "bet_style"]
        valid_noise_methods = ["edge_regions", "mad", "background_roi"]
        valid_uniformity_methods = ["mad_based", "coefficient_of_variation", "ratio_method"]
        
        if self.brain_extraction_method not in valid_brain_methods:
            raise ValueError(f"brain_extraction_method must be one of {valid_brain_methods}")
        
        if self.noise_estimation_method not in valid_noise_methods:
            raise ValueError(f"noise_estimation_method must be one of {valid_noise_methods}")
        
        if self.uniformity_metric not in valid_uniformity_methods:
            raise ValueError(f"uniformity_metric must be one of {valid_uniformity_methods}")
    
    def get_config_summary(self):
        """Get a summary of current configuration"""
        return {
            'brain_extraction_method': self.brain_extraction_method,
            'noise_estimation_method': self.noise_estimation_method,
            'uniformity_metric': self.uniformity_metric,
            'quality_thresholds': {
                'min_snr': self.min_snr,
                'max_intensity_nonuniformity': self.max_intensity_nonuniformity,
                'min_brain_volume': self.min_brain_volume
            }
        }
