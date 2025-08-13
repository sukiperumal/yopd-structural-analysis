# YOPD Structural Analysis Pipeline

A comprehensive structural MRI analysis pipeline for Young-Onset Parkinson's Disease (YOPD) research.

## Overview

This pipeline implements a complete structural brain analysis workflow following current neuroimaging best practices. It includes:

- **Voxel-Based Morphometry (VBM)** analysis
- **Surface-based cortical thickness** analysis  
- **Region of Interest (ROI)** analysis
- **Network connectivity** analysis
- **Comprehensive statistical** analysis
- **Publication-ready figure** generation

## Dataset

- **Participants**: 75 subjects (25 HC, 25 PIGD, 25 TDPD)
- **Modality**: T1-weighted structural MRI
- **Format**: BIDS-compliant dataset
- **Groups**: 
  - HC: Healthy Controls
  - PIGD: Postural Instability/Gait Difficulty dominant PD
  - TDPD: Tremor Dominant PD

## Pipeline Structure

```
yopd-structural-analysis/
├── config.py                    # Configuration and parameters
├── utils.py                     # Shared utility functions
├── step01_data_inventory.py     # Data validation and inventory
├── step02_preprocessing.py      # Image preprocessing
├── step03_vbm_analysis.py      # Voxel-based morphometry
├── step04_surface_analysis.py  # Surface-based analysis
├── step05_roi_analysis.py      # Region of interest analysis
├── step06_network_analysis.py  # Network connectivity analysis
├── step07_statistics.py        # Comprehensive statistics
├── step08_figures.py           # Publication figures
└── outputs/                    # Analysis results
    ├── 01_preprocessed/
    ├── 02_quality_control/
    ├── 03_vbm_analysis/
    ├── 04_surface_analysis/
    ├── 05_roi_analysis/
    ├── 06_network_analysis/
    ├── 07_statistics/
    └── 08_figures/
```

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/yopd-structural-analysis.git
cd yopd-structural-analysis
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install nibabel numpy pandas matplotlib seaborn scipy scikit-learn statsmodels networkx
```

### Data Setup

1. **Prepare BIDS dataset**: Ensure your data follows BIDS format
2. **Update paths**: Modify `config.py` with your data paths:
```python
DATA_ROOT = "path/to/your/data"
OUTPUT_ROOT = "path/to/output/directory"
```

## Usage

### Step-by-Step Execution

Run each analysis step sequentially:

```bash
# Step 1: Data Inventory and Validation
python step01_data_inventory.py

# Step 2: Preprocessing
python step02_preprocessing.py

# Step 3: VBM Analysis
python step03_vbm_analysis.py

# Step 4: Surface Analysis
python step04_surface_analysis.py

# Step 5: ROI Analysis
python step05_roi_analysis.py

# Step 6: Network Analysis
python step06_network_analysis.py

# Step 7: Comprehensive Statistics
python step07_statistics.py

# Step 8: Publication Figures
python step08_figures.py
```

### Parallel Execution

Some steps can be run in parallel after Step 2:
- Steps 3, 4, 5, 6 can run simultaneously
- Step 7 requires completion of steps 3-6
- Step 8 can run after Step 7

## Analysis Steps Details

### Step 01: Data Inventory
- Validates BIDS structure
- Creates subject inventory
- Checks data completeness
- Generates quality metrics

**Outputs**: 
- `data_inventory.csv`
- `subject_demographics.csv`
- Quality control reports

### Step 02: Preprocessing
- Bias field correction (ANTs N4)
- Brain extraction (HD-BET simulation)
- Tissue segmentation (SPM-style)
- Total intracranial volume calculation

**Outputs**:
- Preprocessed T1 images
- GM/WM/CSF probability maps
- Brain masks
- TIV measurements

### Step 03: VBM Analysis
- Spatial normalization to MNI space
- Modulation for volume differences
- Gaussian smoothing (8mm FWHM)
- Group statistical comparisons

**Outputs**:
- Normalized GM images
- Statistical maps
- Group comparison results
- Quality metrics

### Step 04: Surface Analysis
- FreeSurfer recon-all simulation
- Cortical thickness extraction
- Surface-based smoothing
- Parcellation-based measures

**Outputs**:
- Cortical thickness maps
- Surface reconstructions
- Parcellation results
- Quality metrics

### Step 05: ROI Analysis
- Multi-atlas parcellation
- Regional volume extraction
- Tissue density measurements
- Statistical comparisons

**Outputs**:
- ROI measures
- Atlas parcellations
- Statistical results
- Visualization plots

### Step 06: Network Analysis
- Structural connectivity simulation
- Graph theory metrics
- Network topology analysis
- Hub identification

**Outputs**:
- Connectivity matrices
- Graph metrics
- Network statistics
- Topology measures

### Step 07: Comprehensive Statistics
- Multi-modal integration
- Machine learning classification
- Cross-modal correlations
- Effect size analysis

**Outputs**:
- Integrated dataset
- Classification results
- Correlation matrices
- Comprehensive reports

### Step 08: Publication Figures
- Main manuscript figures
- Supplementary figures
- Brain visualizations
- Statistical plots

**Outputs**:
- Publication-ready figures
- Brain maps
- Statistical summaries
- Pipeline diagrams

## Configuration

### Key Parameters

The `config.py` file contains all analysis parameters:

```python
# Data paths
DATA_ROOT = "D:/data_NIMHANS"
OUTPUT_ROOT = "D:/data_NIMHANS/outputs"

# Analysis parameters
VBM_CONFIG = {
    'smoothing_fwhm': 8,
    'template': 'MNI152',
    'modulation': True
}

SURFACE_CONFIG = {
    'smoothing_fwhm': 10,
    'thickness_threshold': 0.5
}

STATS_CONFIG = {
    'alpha': 0.05,
    'multiple_comparisons': 'FDR',
    'min_cluster_size': 10
}
```

### Quality Control

Each step includes comprehensive quality control:
- Automated QC metrics
- Visual inspection outputs
- Outlier detection
- Processing success rates

## Results

### Expected Outputs

1. **Preprocessed Data**: Brain-extracted, bias-corrected T1 images
2. **VBM Results**: Group difference maps, statistical reports
3. **Surface Results**: Cortical thickness maps, parcellation data
4. **ROI Results**: Regional measures, statistical comparisons
5. **Network Results**: Connectivity matrices, graph metrics
6. **Integrated Analysis**: Multi-modal statistical results
7. **Figures**: Publication-ready visualizations

### Quality Metrics

The pipeline tracks:
- Processing success rates (>95% target)
- Image quality scores
- Motion artifacts
- Statistical power
- Effect sizes

## Validation

### Statistical Approach

- **Multiple comparison correction**: FDR and Bonferroni
- **Effect size reporting**: Cohen's d
- **Power analysis**: Post-hoc power calculations
- **Cross-validation**: Machine learning models

### Quality Assurance

- **Data validation**: BIDS compliance, completeness
- **Processing QC**: Automated and manual checks
- **Statistical validation**: Assumption testing
- **Reproducibility**: Documented parameters, random seeds

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size in config
2. **Path issues**: Use absolute paths in config
3. **Missing dependencies**: Check virtual environment
4. **Data format**: Ensure BIDS compliance

### Support

- Check log files in `outputs/logs/`
- Review QC reports for failed subjects
- Verify data paths and permissions
- Ensure sufficient disk space

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline, please cite:

```bibtex
@software{yopd_structural_pipeline,
  title={YOPD Structural Analysis Pipeline},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-username/yopd-structural-analysis}
}
```

## Acknowledgments

- This pipeline follows guidelines from FSL, SPM, and FreeSurfer
- Implements best practices from neuroimaging literature
- Uses open-source neuroimaging tools and libraries

## Contact

For questions or support:
- Email: [your.email@institution.edu]
- GitHub Issues: [repository-url]/issues

---

**Status**: ✅ Complete pipeline implementation
**Last Updated**: August 2025
**Pipeline Version**: 1.0
