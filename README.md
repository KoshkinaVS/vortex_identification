# Vortex Identification

Implementation of Eulerian vortex identification criteria for atmospheric coherent vortical structure analysis.

## Overview

This repository provides tools for analyzing mesoscale coherent vortical structures (CVS) in the atmosphere using Eulerian vortex identification methods. The implementation includes calculations of Q-criterion, Δ-criterion, λ₂-criterion, swirling strength (λ_ci), and Rortex criteria based on numerical atmospheric data (accepting 2D or 3D velocity fields). Additionally, functions for computing primary CVS statistics (size, shape, thermodynamic parameters) are provided.

## Features

- **Multiple Vortex Identification Criteria**:
  - Q-criterion
  - Δ-criterion
  - λ₂-criterion
  - Swirling strength (λ_ci)
  - Rortex criterion

- **Data Support**:
  - Loading capabilities for NAAD (North Atlantic Atmospheric Database) and ERA5 (ECMWF Reanalysis v5) datasets
  - Processing of 2D and 3D velocity fields

- **Post-processing**:
  - CVS clustering using DBSCAN algorithm
  - Statistical analysis of identified vortices
  - Visualization tools

## Repository Structure

```
vortex_identification/
├── examples/
│   └── compute_rortex_year.py          # Example: Applying Rortex criterion to NAAD HiRes data
└── vortex_dir/
    ├── load_data.py                    # Data loading functions (NAAD, ERA5)
    ├── compute_criteria.py             # Velocity gradient tensor computation and criteria calculations
    ├── vortex_processing.py            # Post-processing: clustering and statistics
    └── show_vortex.py                  # Visualization functions (requires refinement)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vortex_identification.git
cd vortex_identification

# Install required dependencies
pip install -r requirements.txt
```

Dependencies include: `numpy`, `scipy`, `xarray`, `netCDF4`, `scikit-learn`, `matplotlib`

## Usage

### Basic Workflow

1. **Load Data**:
```python
from vortex_dir.load_data import load_naad_data, load_era5_data

# Load NAAD data
data = load_naad_data("path/to/naad_file.nc")

# Load ERA5 data
data = load_era5_data("path/to/era5_file.nc")
```

2. **Compute Vortex Criteria**:
```python
from vortex_dir.compute_criteria import compute_q, compute_delta, compute_lambda2, compute_swirling_strength, compute_rortex

# Assuming u, v, w are velocity component arrays with shape (time, level, y, x)
q_criterion = compute_q(u, v, w)
delta_criterion = compute_delta(u, v, w)
rortex = compute_rortex(u, v, w)
```

**Note**: Functions are optimized for arrays with shape `(time, level, y, x)`.

3. **Post-process Results**:
```python
from vortex_dir.vortex_processing import clustering_DBSCAN, get_stat

# Cluster identified vortices using DBSCAN
labels = clustering_DBSCAN(vortex_mask, eps=0.5, min_samples=10)

# Calculate vortex statistics
statistics = get_stat(vortex_mask, labels, additional_fields=['temperature', 'vorticity'])
```

**Note**: Post-processing functions are optimized for arrays with shape `(y, x)`.

### Example Application

The `examples/compute_rortex_year.py` script demonstrates applying the Rortex criterion to NAAD HiRes data:

```bash
python examples/compute_rortex_year.py --input data/naad_hires.nc --output results/rortex_output.nc
```

This script produces NetCDF files containing computed criteria arrays for further analysis.

## Available Criteria

### 1. Q-Criterion
Identifies vortices as regions where the rotation rate tensor dominates the strain rate tensor:
\[ Q = \frac{1}{2}(||\Omega||^2 - ||S||^2) \]
where \(\Omega\) is the vorticity tensor and \(S\) is the strain rate tensor.

### 2. Δ-Criterion
Based on the discriminant of the velocity gradient tensor:
\[ \Delta = \left(\frac{Q}{3}\right)^3 + \left(\frac{\det(\nabla u)}{2}\right)^2 \]

### 3. λ₂-Criterion
Identifies vortices as connected regions with negative second eigenvalue of \(S^2 + \Omega^2\).

### 4. Swirling Strength (λ_ci)
The imaginary part of the complex eigenvalue pair of the velocity gradient tensor.

### 5. Rortex Criterion
Measures the local rigid rotation part of the velocity gradient, providing both magnitude and direction.

## Output Format

Results are saved in NetCDF format with the following structure:
- Dimensions: `time`, `level`, `lat`, `lon`
- Variables: Computed criteria arrays, vortex masks, cluster labels
- Attributes: Metadata including calculation parameters and data sources

## Visualization

Basic visualization functions are provided in `show_vortex.py`:

```python
from vortex_dir.show_vortex import plot_vortex_field

plot_vortex_field(vortex_mask, background_field, save_path='vortex_plot.png')
```

**Note**: Visualization functions require refinement and customization for specific use cases.

## Citing This Work

If you use this software in your research, please cite:

```
@software{vortex_identification,
  title = {Vortex Identification: Eulerian Methods for Atmospheric Coherent Vortical Structures},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/vortex_identification}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NAAD (North Atlantic Atmospheric Database) team for data access
- ECMWF for ERA5 reanalysis data
- Contributors and reviewers of the vortex identification methods

## Contact

For questions and support, please open an issue on GitHub or contact [your.email@example.com].
