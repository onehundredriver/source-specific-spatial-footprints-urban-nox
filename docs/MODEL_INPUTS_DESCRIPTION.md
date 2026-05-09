# Model inputs description

## Purpose

This folder contains public model-input matrices for XGBoost spatial cross-validation experiments.

The model-input matrices were prepared by filtering the full modelling datasets to national-control stations and by harmonizing emission-density variable names.

## Files

| Time scale | Model input file | Feature list file | Target variable |
|---|---|---|---|
| 1hour | `data/model_inputs/modeling_dataset_1hour.parquet` | `data/model_feature_lists/columns_apply_1hour.xlsx` | `NOx` |
| 24hour | `data/model_inputs/modeling_dataset_24hour.parquet` | `data/model_feature_lists/columns_apply_24hour.xlsx` | `NOx_24hour` |
| 168hour | `data/model_inputs/modeling_dataset_168hour.parquet` | `data/model_feature_lists/columns_apply_168hour.xlsx` | `NOx_168hour` |
| 31day | `data/model_inputs/modeling_dataset_31day.parquet` | `data/model_feature_lists/columns_apply_31day.xlsx` | `NOx_31day` |

## Station filtering

The public model-input matrices include national-control stations only. The national-control station list was extracted from the 1-hour modelling matrix using `StationTypeName == 国控站`, and the same station-name list was applied to the 24-hour, 168-hour and 31-day smoothed matrices.

## Variable-name harmonization

All variables beginning with:

```text
emission_other_density_
```

were renamed to:

```text
emission_density_
```

The same replacement was applied to the corresponding `columns_apply` feature-list files.

## Manifest

The file-level manifest is available at:

```text
data/model_inputs/public_model_input_manifest.csv
```

The feature-list rename report is available at:

```text
data/model_feature_lists/feature_list_rename_report.csv
```

## Reproducibility note

These public model-input matrices are intended to support model-level reproducibility and auditability. If the files are too large for GitHub, they can be deposited on Zenodo and referenced from this repository using a DOI.

Last updated: 2026-05-08
**数据 DOI / Data DOI:** [10.5281/zenodo.20092921](https://doi.org/10.5281/zenodo.20092921)
