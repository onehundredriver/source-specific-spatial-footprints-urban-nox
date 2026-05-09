# Public NOX data

This folder contains processed national-control NOX data and quality-control summaries used to support the manuscript's reproducibility package.

## Files

```text
national_control_station_metadata.csv
processed_national_control_NOx_hourly.csv
NOx_QC_overall_summary.csv
NOx_QC_summary_by_station.csv
```

## File descriptions

- `national_control_station_metadata.csv`: monitoring-station metadata.
- `processed_national_control_NOx_hourly.csv`: processed hourly NOX observations after quality control.
- `NOx_QC_overall_summary.csv`: overall quality-control summary.
- `NOx_QC_summary_by_station.csv`: station-level quality-control summary.

## Quality-control logic

The NOX observations were screened for non-physical concentrations, negative values, sentinel-coded missing values and persistent unchanged readings.

The QC summaries allow users to inspect station-level missingness and usable-record counts.

## Link to manuscript

These files support the Data availability statement and the supplementary method section describing NOX observation preprocessing and quality control.
