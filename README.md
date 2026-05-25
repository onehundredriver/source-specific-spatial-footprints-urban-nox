# Source-specific Spatial Footprints of Urban NOX in Shanghai

## Summary

This repository provides data and code for analyzing source-specific spatial footprints of urban NOX in Shanghai. It includes figure source data, figure reproduction scripts, public processed NOX observations, model split files, feature lists, and model-reproduction pipelines.

Large model-input files are hosted on Zenodo and should be downloaded separately when full model reproduction is needed.

## Data DOI

Large model-input files are available from Zenodo:

https://doi.org/10.5281/zenodo.20092921

The Zenodo record includes the following model-input datasets:

- `modeling_dataset_1hour.parquet`
- `modeling_dataset_24hour.parquet`
- `modeling_dataset_168hour.parquet`
- `modeling_dataset_31day.parquet`

After downloading, place these files in:

```text
data/model_inputs/
```

## Repository Structure

```text
code/
  config/                  # Model configuration files
  pipelines/
    xgboost/               # XGBoost spatial and temporal CV pipelines
    rf/                    # Random Forest spatial and temporal CV pipelines
    lgbm/                  # LightGBM spatial and temporal CV pipelines
  scripts/                 # Figure reproduction scripts
  environment.yml
  requirements.txt
  check_release_package_integrity.py
  run_all_figure_reproductions.py

data/
  model_inputs/            # Model-input matrices; large files are hosted on Zenodo
  model_feature_lists/     # Feature-list files used by the model pipelines
  public/                  # Public processed NOX observations and station metadata
  source_data/
    main/                  # Source data for main figures
    supplementary/         # Source data for supplementary figures
  splits/                  # Spatial and temporal cross-validation split files
  supplementary_data/      # Supplementary derived data and station-clustering outputs

figures/
  main/                    # Main figure outputs
  supplementary/           # Supplementary figure outputs

docs/
  PROJECT_OVERVIEW.md
  MODEL_INPUTS_DESCRIPTION.md
  REPRODUCIBILITY_GUIDE.md
```

## What Is Included

This repository includes:

1. source data for released main, Extended Data and supplementary figures;
2. scripts for reproducing figures from source data;
3. processed public national-control NOX observations and station metadata;
4. NOX quality-control summaries;
5. spatial and temporal cross-validation split files;
6. model feature lists for the released modelling matrices;
7. XGBoost, Random Forest, and LightGBM spatial and temporal CV pipelines;
8. anomalous-station exclusion records and final station-clustering labels;
9. supplementary derived data used for result checking and interpretation.

## Quick Start

### 1. Create the software environment

Using Conda:

```bash
conda env create -f code/environment.yml
conda activate data_handle
```

Alternatively, install the required Python packages with pip:

```bash
pip install -r code/requirements.txt
```

### 2. Check package integrity

Run the integrity check from the repository root:

```bash
python code/check_release_package_integrity.py
```

### 3. Reproduce figures

Run all figure reproduction scripts:

```bash
python code/run_all_figure_reproductions.py
```

Individual figure scripts are located in:

```text
code/scripts/
```

Reproduced outputs are written to:

```text
figures/main/
figures/supplementary/
```

### 4. Download full model-input data

For full model reproduction, download the large model-input parquet files from Zenodo:

```text
https://doi.org/10.5281/zenodo.20092921
```

Place the downloaded files in:

```text
data/model_inputs/
```

### 5. Run model pipelines

Example XGBoost spatial CV:

```bash
python code/pipelines/xgboost/run_spatial_cv.py --config code/config/xgboost_spatial_1hour.json
```

Example XGBoost temporal CV:

```bash
python code/pipelines/xgboost/run_temporal_cv.py --config code/config/xgboost_temporal_1hour.json
```

Random Forest and LightGBM pipelines are available in:

```text
code/pipelines/rf/
code/pipelines/lgbm/
```

Their corresponding configuration files are located in:

```text
code/config/
```

## Public Data

Public processed NOX data and station metadata are provided in:

```text
data/public/
```

Key files include:

```text
national_control_station_metadata.csv
processed_national_control_NOx_hourly.csv
NOx_QC_overall_summary.csv
NOx_QC_summary_by_station.csv
```

## Model Inputs

Model-input matrices are organized by temporal aggregation scale:

```text
data/model_inputs/modeling_dataset_1hour.parquet
data/model_inputs/modeling_dataset_24hour.parquet
data/model_inputs/modeling_dataset_168hour.parquet
data/model_inputs/modeling_dataset_31day.parquet
```

Feature-list files are provided in:

```text
data/model_feature_lists/
```

The model-input matrices were filtered to public national-control stations and harmonized with the released feature-list files.

## Cross-validation Splits

Spatial and temporal split files are stored in:

```text
data/splits/
```

Important files include:

```text
spatial_5fold_station_assignment.csv
spatial_5fold_station_assignment_public_national_control.csv
temporal_5fold_window_assignment.csv
temporal_buffer_intervals.csv
anomalous_station_exclusion_list.csv
final_cluster_station_labels.csv
```

## Restricted Data Boundary

Some original input datasets cannot be redistributed because of third-party licenses, data-use agreements, or provider restrictions. These include, but are not limited to:

- raw Gaode traffic-speed records;
- raw Baidu Maps POI records;
- restricted monitoring records;
- licensed or agreement-based geospatial layers.

For these restricted components, this repository provides processed source data, figure-level data, derived summaries, split files, and documentation to support result verification.

## License

Data and example datasets are released under Creative Commons Attribution 4.0 International (CC-BY 4.0), unless otherwise stated.

Code is released under the MIT License, unless otherwise stated.

## Contact

Please contact the project maintainer for questions about data access, reproduction, or repository use.

## Last Updated

2026-05-09
