# Reproducibility guide

## Purpose

This guide explains how to use the release package to inspect the source data, reproduce figures, and understand the experimental reproducibility boundary.

## 1. Check package integrity

After Step 17C, run the following command from the repository root:

```bash
python code/check_release_package_integrity.py
```

This checks whether the expected source-data files, figure scripts, public data files, split files and documentation files are present.

## 2. Reproduce released figures

After Step 17C, run:

```bash
python code/run_all_figure_reproductions.py
```

The reproduced outputs are written to:

```text
figures/main/
figures/supplementary/
```

## 3. Inspect public NOX data

The processed national-control NOX data and quality-control summaries are located in:

```text
data/public/
```

Expected files include:

```text
national_control_station_metadata.csv
processed_national_control_NOx_hourly.csv
NOx_QC_overall_summary.csv
NOx_QC_summary_by_station.csv
```

## 4. Inspect model split files

The model split files are located in:

```text
data/splits/
```

These files document the spatial 5-fold station assignments and temporal 5-fold window assignments used in the modelling workflow.

## 5. Experimental reproducibility

The current release package directly supports figure-level reproduction and split-file auditing.

Full model retraining requires a complete modelling matrix containing the target variable, station identifier, timestamp and numerical predictors. Because some raw data sources are restricted, the full modelling matrix may not be publicly redistributed unless the data-use conditions allow it.

A generic experimental-reproducibility scaffold will be created in Step 17D.
**数据 DOI / Data DOI:** [10.5281/zenodo.20092921](https://doi.org/10.5281/zenodo.20092921)
