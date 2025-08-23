# Enhanced Image Quality Assessment

A comprehensive, modular system for MRI image quality assessment with detailed technical reporting and step-by-step analysis documentation.

## Overview

This system provides granular, stage-by-stage analysis of MRI image quality with comprehensive technical documentation. It addresses the limitations of basic quality assessment approaches by providing:

- **Clear stage separation** (raw analysis → brain extraction → quality metrics)
- **Multiple algorithmic options** with technical documentation
- **Robust statistical methods** with outlier handling
- **Comprehensive reporting** suitable for research publications

## Key Features

### Stage-by-Stage Analysis
1. **Raw Image Analysis**: Analyze unprocessed image properties
2. **Brain Extraction**: Multiple methods with quality assessment
3. **Noise Estimation**: Multiple robust methods 
4. **Quality Metrics**: SNR, CNR, uniformity with detailed calculations
5. **Quality Assessment**: Threshold-based flags and composite scoring

### Technical Documentation
- Every processing step is logged with method details
- Algorithm parameters are recorded
- Processing success/failure tracking
- Comprehensive JSON output for reproducibility

### Multiple Algorithm Options
- **Brain Extraction**: threshold-based, BET-style
- **Noise Estimation**: edge regions, MAD, background ROI
- **Quality Metrics**: Robust statistics with outlier removal

## Installation

### Requirements
```bash
pip install numpy scipy scikit-image nibabel pandas matplotlib seaborn
```

### Directory Structure
```
image_analysis/
├── modules/                    # Core analysis modules
│   ├── config.py              # Configuration management
│   ├── preprocessing_tracker.py # Step tracking
│   ├── raw_image_analyzer.py   # Raw image analysis
│   ├── brain_extraction.py     # Brain extraction methods
│   ├── noise_estimation.py     # Noise estimation methods
│   ├── quality_metrics.py      # Quality calculations
│   └── main_analyzer.py        # Main integration
├── tests/                      # Step-by-step validation tests
│   ├── test_01_configuration.py
│   ├── test_02_raw_analysis.py
│   ├── test_03_brain_extraction.py
│   ├── test_04_noise_estimation.py
│   ├── test_05_quality_metrics.py
│   └── test_06_complete_integration.py
├── enhanced_quality_assessment.py  # Main analysis script
└── README.md
```

## Quick Start

### 1. Run Validation Tests
Test each module step-by-step:

```powershell
# Test 1: Configuration
cd image_analysis\tests
python test_01_configuration.py

# Test 2: Raw image analysis
python test_02_raw_analysis.py

# Test 3: Brain extraction
python test_03_brain_extraction.py

# Test 4: Noise estimation
python test_04_noise_estimation.py

# Test 5: Quality metrics
python test_05_quality_metrics.py

# Test 6: Complete integration
python test_06_complete_integration.py
```

### 2. Analyze Single Image
```powershell
cd image_analysis
python enhanced_quality_assessment.py "path\to\your\image.nii.gz"
```

### 3. Batch Analysis
```powershell
python enhanced_quality_assessment.py --batch "path\to\preprocessed\directory"
```

## Usage Examples

### Basic Analysis
```powershell
python enhanced_quality_assessment.py sub-001_T1w.nii.gz
```

### Custom Configuration
```powershell
python enhanced_quality_assessment.py --brain-method bet_style --noise-method mad --min-snr 15 image.nii.gz
```

### Batch Analysis with Options
```powershell
python enhanced_quality_assessment.py --batch ./preprocessed/ --output ./quality_results/ --compare-methods --save-masks
```

## Output Structure

### JSON Results (Detailed)
```json
{
  "stage_1_raw_analysis": {
    "raw_mean_intensity": 87.5,
    "raw_intensity_percentiles": {...},
    "image_shape": [208, 240, 256],
    "voxel_size": [1.0, 1.0, 1.0]
  },
  "stage_2_brain_extraction": {
    "method": "threshold_based",
    "brain_volume_ml": 1542.3,
    "extraction_details": {...},
    "quality_assessment": {...}
  },
  "stage_3_noise_analysis": {
    "method": "edge_regions",
    "noise_estimate": 23.4,
    "noise_parameters": {...}
  },
  "stage_4_snr_analysis": {
    "snr_value": 18.7,
    "signal_median": 437.2,
    "noise_estimate": 23.4,
    "noise_method": "edge_regions"
  },
  "stage_5_quality_summary": {
    "snr_adequate": true,
    "uniformity_good": true,
    "volume_adequate": true,
    "overall_quality_good": true,
    "quality_score": 7.3
  }
}
```

### CSV Summary (Overview)
| subject_id | snr | cnr | uniformity | brain_volume_ml | quality_score | overall_quality_good |
|------------|-----|-----|------------|-----------------|---------------|---------------------|
| sub-001    | 18.7| 12.3| 0.089      | 1542.3         | 7.3           | true                |

## Technical Details

### Brain Extraction Methods

#### Threshold-Based
- **Algorithm**: Otsu thresholding + morphological operations
- **Steps**: Intensity threshold → small object removal → hole filling → largest component
- **Parameters**: threshold_factor, connectivity, min_object_size
- **Best for**: Standard T1w images with good contrast

#### BET-Style
- **Algorithm**: Gradient-based edge detection (simplified FSL BET)
- **Steps**: Intensity normalization → gradient calculation → edge thresholding → morphological cleanup
- **Parameters**: edge_threshold_percentile, min_object_size
- **Best for**: Images with complex intensity distributions

### Noise Estimation Methods

#### Edge Regions
- **Method**: Standard deviation of edge region intensities
- **Assumption**: Edge regions contain primarily noise
- **Robust features**: 6-sided sampling, IQR outlier removal
- **Best for**: Images with clear background regions

#### MAD (Median Absolute Deviation)
- **Method**: MAD of background regions (non-brain)
- **Robust features**: Outlier-resistant, suitable for Rician noise
- **Requires**: Brain mask for background identification
- **Best for**: Images with well-defined brain/background separation

#### Background ROI
- **Method**: Standard deviation of background regions
- **Robust features**: Percentile filtering (10th-90th)
- **Conservative**: Uses entire background, not just edges
- **Best for**: Images with large, uniform background regions

### Quality Metrics Calculations

#### SNR (Signal-to-Noise Ratio)
```
SNR = median(brain_intensities) / noise_estimate
```
- **Signal**: Median brain intensity (robust to outliers)
- **Noise**: From selected noise estimation method
- **Threshold**: Configurable (default: 15.0)

#### CNR (Contrast-to-Noise Ratio)
```
CNR = (median(brain_intensities) - median(background_intensities)) / noise_estimate
```
- **Contrast**: Difference between brain and background signals
- **Noise**: Same as SNR calculation
- **Interpretation**: Higher values indicate better tissue contrast

#### Uniformity
```
Uniformity = MAD(filtered_brain_intensities) / median(filtered_brain_intensities)
```
- **Method**: MAD-based with outlier removal (1st-99th percentile)
- **Interpretation**: Lower values indicate better uniformity
- **Threshold**: Configurable (default: 0.3)

## Configuration Options

### Quality Thresholds
```python
config = QualityConfig(
    min_snr=15.0,                    # Minimum SNR for adequate quality
    max_intensity_nonuniformity=0.3, # Maximum uniformity for good quality
    min_brain_volume=800000,         # Minimum brain volume (voxels)
    brain_extraction_method="threshold_based",
    noise_estimation_method="edge_regions"
)
```

### Method Selection
- **brain_extraction_method**: `"threshold_based"`, `"bet_style"`
- **noise_estimation_method**: `"edge_regions"`, `"mad"`, `"background_roi"`

## Troubleshooting

### Common Issues

#### Low SNR Values
- Check if analyzing correct image type (T1w vs other sequences)
- Verify brain mask quality - poor extraction affects SNR
- Consider different noise estimation method
- Check for acquisition issues (scanner calibration)

#### Brain Extraction Failures
- Try different extraction method (`bet_style` vs `threshold_based`)
- Adjust threshold_factor for threshold method
- Check image intensity range and normalization

#### Volume Assessment Issues
- Verify voxel size in image header
- Check brain mask coverage visually
- Consider adjusting min_brain_volume threshold

### Debugging Steps
1. Run individual test scripts to isolate issues
2. Use `--verbose` flag for detailed output
3. Check intermediate results in JSON output
4. Verify image loading and header information

## Research Applications

### Publication Reporting

#### Methods Section Example
*Image quality assessment was performed using a comprehensive, stage-by-stage analysis approach. Raw image properties were analyzed directly from unprocessed T1w images. Brain extraction was performed using [threshold-based/BET-style] method with morphological operations including connected components analysis and hole filling. Noise estimation was conducted using [edge regions/MAD] method with robust outlier removal using the IQR method. Signal-to-noise ratio (SNR) was calculated as the median brain signal divided by the noise estimate. Contrast-to-noise ratio (CNR) was calculated as brain-background contrast divided by noise. Intensity uniformity was assessed using median absolute deviation (MAD) with 1-99 percentile outlier filtering.*

#### Quality Criteria
- SNR ≥ 15: Adequate signal quality
- Uniformity ≤ 0.3: Good intensity uniformity  
- Volume ≥ 800,000 voxels: Adequate brain coverage
- Overall quality: All three criteria met

### Data Analysis
- Export CSV summary for statistical analysis
- Use JSON results for detailed quality parameter analysis
- Compare different preprocessing pipelines
- Quality control for large datasets

## Contributing

### Adding New Methods
1. Create new method in appropriate module
2. Add to method selection in config
3. Update tests
4. Document technical details

### Module Structure
Each module follows the pattern:
- Initialization with configuration
- Main processing method with error handling
- Detailed parameter logging
- Comprehensive error reporting

## License

This enhanced image quality assessment system is designed for research use. Please cite appropriately if used in publications.

## Contact

For questions about implementation or technical details, please refer to the comprehensive technical documentation in the JSON output files.
