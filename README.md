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
  - Loading capabilities for WRF, [NAAD](https://naad.ocean.ru/) and [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) datasets
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
```

Dependencies include: `numpy`, `scipy`, `xarray`, `netCDF4`, `scikit-learn`, `matplotlib`

## Usage


### Example Application

The `examples/compute_rortex_year.py` script demonstrates applying the Rortex criterion to NAAD HiRes data.

This script produces NetCDF files containing computed criteria arrays for further analysis.

## Available Criteria

### 1. Q-Criterion
Identifies vortices as regions where the rotation rate tensor dominates the strain rate tensor:
```math
\[ Q = \frac{1}{2}(||\Omega||^2 - ||S||^2) \]
```
where \(\Omega\) is the vorticity tensor and \(S\) is the strain rate tensor.

### 2. Δ-Criterion
Based on the discriminant of the velocity gradient tensor:
```math
\[ \Delta = \left(\frac{Q}{3}\right)^3 + \left(\frac{\det(\nabla u)}{2}\right)^2 \]
```

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


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For questions and support, please open an issue on GitHub or contact koshkina.vs@phystech.edu.
