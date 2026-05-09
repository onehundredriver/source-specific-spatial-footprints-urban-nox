# -*- coding: utf-8 -*-
"""
Configuration-driven XGBoost temporal 5-fold cross-validation.

This script uses the released temporal role table:

data/splits/temporal_5fold_window_assignment.csv

The temporal split file is interpreted as a role table, not as a one-to-one
Timestamp -> Temporal_Fold mapping.

Correct split logic:
- filter by Aggregation_Scale;
- within each Temporal_Fold:
    - Role == Training   -> training timestamps;
    - Role == Validation -> validation timestamps;
    - Role == Buffer     -> buffer-excluded timestamps.

Example commands, run from repository root:

python code/pipelines/xgboost/run_temporal_cv.py --config code/config/xgboost_temporal_1hour.json
python code/pipelines/xgboost/run_temporal_cv.py --config code/config/xgboost_temporal_24hour.json
python code/pipelines/xgboost/run_temporal_cv.py --config code/config/xgboost_temporal_168hour.json
python code/pipelines/xgboost/run_temporal_cv.py --config code/config/xgboost_temporal_31day.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap
from bayes_opt import BayesianOptimization


# =============================================================================
# Repository root
# =============================================================================

# run_temporal_cv.py is located at:
# code/pipelines/xgboost/run_temporal_cv.py
REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# Matplotlib settings
# =============================================================================

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Basic utilities
# =============================================================================

def resolve_repo_path(path_str: str) -> Path:
    """
    Resolve a path relative to repository root unless it is already absolute.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def load_config(config_path: str | Path) -> dict:
    """
    Load JSON config.
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(config_path)
    return cfg


def write_json(path: Path, obj: dict) -> None:
    """
    Write JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {path}")


def rmse(y_true, y_pred) -> float:
    """
    Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Outlier screening
# =============================================================================

def remove_outliers_iqr(
    data: pd.DataFrame,
    column: str,
    k: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove target outliers using IQR.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    n_outliers = int(outliers_mask.sum())

    info = {
        "column": column,
        "method": "IQR",
        "k": float(k),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "n_outliers": n_outliers,
        "outlier_percent": float(n_outliers / len(data) * 100) if len(data) > 0 else np.nan,
        "n_before": int(len(data)),
        "n_after": int((~outliers_mask).sum()),
    }

    print(f"  IQR异常值检测 - {column}:")
    print(f"    Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"    下界={lower_bound:.4f}, 上界={upper_bound:.4f}")
    print(f"    发现异常值: {n_outliers} 个 ({info['outlier_percent']:.2f}%)")

    return data.loc[~outliers_mask].copy(), info


def visualize_outlier_analysis(
    data: pd.DataFrame,
    column: str,
    output_dir: Path,
    k: float = 1.5,
) -> None:
    """
    Visualize target distribution and IQR boundaries.
    """
    print("正在生成异常值分析可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    box_plot = axes[0, 0].boxplot(data[column], vert=True, patch_artist=True)
    box_plot["boxes"][0].set_facecolor("lightblue")
    axes[0, 0].set_title("Boxplot - outlier screening", fontsize=14)
    axes[0, 0].set_ylabel(column)

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    axes[0, 0].axhline(
        y=lower_bound,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Lower: {lower_bound:.2f}",
    )
    axes[0, 0].axhline(
        y=upper_bound,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Upper: {upper_bound:.2f}",
    )
    axes[0, 0].legend()

    axes[0, 1].hist(data[column], bins=50, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(lower_bound, color="r", linestyle="--", linewidth=2, label="IQR lower")
    axes[0, 1].axvline(upper_bound, color="r", linestyle="--", linewidth=2, label="IQR upper")
    axes[0, 1].axvline(data[column].mean(), linestyle="-", linewidth=2, label="Mean")
    axes[0, 1].axvline(data[column].median(), linestyle="-", linewidth=2, label="Median")
    axes[0, 1].set_title(f"{column} distribution", fontsize=14)
    axes[0, 1].legend()

    stats.probplot(data[column], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot", fontsize=14)

    sorted_data = np.sort(data[column])
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, cdf, linewidth=2)
    axes[1, 1].axvline(lower_bound, color="r", linestyle="--", alpha=0.7)
    axes[1, 1].axvline(upper_bound, color="r", linestyle="--", alpha=0.7)
    axes[1, 1].set_title("CDF", fontsize=14)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "outlier_analysis_visualization.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_feature_list(feature_list_file: Path, feature_col: str) -> list[str]:
    """
    Load selected feature list from Excel.
    """
    if not feature_list_file.exists():
        raise FileNotFoundError(f"Feature list file not found: {feature_list_file}")

    selected_vars_df = pd.read_excel(feature_list_file)

    if feature_col not in selected_vars_df.columns:
        raise ValueError(
            f"Feature list file does not contain required column: {feature_col}\n"
            f"Available columns: {selected_vars_df.columns.tolist()}"
        )

    selected_features = (
        selected_vars_df[feature_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    return selected_features


def load_and_preprocess_data(
    cfg: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str], str, dict]:
    """
    Load modelling matrix and selected features.

    This function follows the logic of the original temporal XGBoost scripts:
    - convert Timestamp to datetime;
    - sort by time and reset index;
    - generate Station_ID from station names;
    - append Station_ID to selected features if it is not already listed;
    - remove missing target rows;
    - apply IQR target outlier screening;
    - reset index again after filtering;
    - median-impute missing predictor values.
    """
    paths = cfg["paths"]
    columns_cfg = cfg["columns"]
    preprocessing_cfg = cfg["preprocessing"]

    model_input_file = resolve_repo_path(paths["model_input_file"])
    feature_list_file = resolve_repo_path(paths["feature_list_file"])

    target_col = columns_cfg["target_column"]
    station_col = columns_cfg["station_column"]
    timestamp_col = columns_cfg["timestamp_column"]
    feature_list_col = columns_cfg["feature_list_column"]

    print("=" * 100)
    print("Loading model input")
    print(f"Model input : {model_input_file}")
    print(f"Feature list: {feature_list_file}")
    print(f"Target      : {target_col}")
    print("=" * 100)

    if not model_input_file.exists():
        raise FileNotFoundError(f"Model input file not found: {model_input_file}")

    df = pd.read_parquet(model_input_file)
    print(f"[INFO] 原始数据形状: {df.shape}")

    for required_col in [target_col, station_col, timestamp_col]:
        if required_col not in df.columns:
            raise ValueError(f"Required column not found: {required_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    if df[timestamp_col].isna().any():
        n_bad = int(df[timestamp_col].isna().sum())
        raise ValueError(f"{timestamp_col} contains unparseable timestamps: {n_bad}")

    # 原始时间 CV 代码中有严格的时间排序和 reset_index
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    print(f"[INFO] 数据已按 {timestamp_col} 排序并重置索引。")

    selected_features_raw = load_feature_list(feature_list_file, feature_list_col)

    # 原始时间 CV 代码中生成 Station_ID，并将其加入特征列表
    le = LabelEncoder()
    df["Station_ID"] = le.fit_transform(df[station_col].astype(str))
    station_mapping_path = output_dir / "station_id_mapping.csv"
    pd.DataFrame({
        station_col: le.classes_,
        "Station_ID": range(len(le.classes_)),
    }).to_csv(station_mapping_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {station_mapping_path}")

    if "Station_ID" not in selected_features_raw:
        selected_features_raw.append("Station_ID")

    selected_features = [var for var in selected_features_raw if var in df.columns]
    missing_features = [var for var in selected_features_raw if var not in df.columns]

    print(f"[INFO] Feature list 原始变量数: {len(selected_features_raw)}")
    print(f"[INFO] 在建模矩阵中匹配到变量数: {len(selected_features)}")
    print(f"[INFO] 在建模矩阵中缺失变量数: {len(missing_features)}")

    if missing_features:
        print(f"[WARNING] 缺失变量示例: {missing_features[:20]}")

    if len(selected_features) == 0:
        raise ValueError("No selected features are available in the model input matrix.")

    preprocessing_report = {
        "model_input_file": str(model_input_file),
        "feature_list_file": str(feature_list_file),
        "target_column": target_col,
        "station_column": station_col,
        "timestamp_column": timestamp_col,
        "raw_rows": int(len(df)),
        "raw_cols": int(len(df.columns)),
        "feature_list_count_after_station_id_append": int(len(selected_features_raw)),
        "matched_feature_count": int(len(selected_features)),
        "missing_feature_count": int(len(missing_features)),
        "missing_features": missing_features,
        "timestamp_min": str(df[timestamp_col].min()),
        "timestamp_max": str(df[timestamp_col].max()),
        "station_count": int(df[station_col].nunique()),
    }

    if preprocessing_cfg.get("drop_missing_target", True):
        original_rows = len(df)
        df = df.dropna(subset=[target_col]).copy()
        print(f"[INFO] 删除目标变量缺失行后: {original_rows} -> {len(df)}")
        preprocessing_report["rows_after_drop_missing_target"] = int(len(df))

    outlier_method = preprocessing_cfg.get("target_outlier_method", "IQR")
    outlier_k = float(preprocessing_cfg.get("target_outlier_k", 5.0))

    if outlier_method.upper() == "IQR":
        visualize_outlier_analysis(df, target_col, output_dir, k=outlier_k)
        print(f"\n使用 IQR 方法剔除 {target_col} 异常值 (k={outlier_k})...")
        df, outlier_info = remove_outliers_iqr(df, target_col, k=outlier_k)
        preprocessing_report["outlier_screening"] = outlier_info
        print(f"[INFO] 删除异常值后数据形状: {df.shape}")
    else:
        raise ValueError(f"Unsupported target outlier method: {outlier_method}")

    # 原始时间 CV 代码中，异常值剔除后必须再次 reset_index
    df = df.reset_index(drop=True)
    print("[INFO] IQR 过滤后已重置索引。")

    if preprocessing_cfg.get("predictor_missing_strategy", "median") != "median":
        raise ValueError("Only median predictor imputation is currently supported.")

    imputed_features = []
    for feature in selected_features:
        missing_count = int(df[feature].isnull().sum())
        if missing_count > 0:
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value)
            imputed_features.append({
                "feature": feature,
                "missing_count": missing_count,
                "median_value": float(median_value) if pd.notnull(median_value) else None,
            })

    preprocessing_report["imputed_feature_count"] = len(imputed_features)
    preprocessing_report["imputed_features"] = imputed_features
    preprocessing_report["rows_after_preprocessing"] = int(len(df))
    preprocessing_report["cols_after_preprocessing"] = int(len(df.columns))

    return df, selected_features, target_col, preprocessing_report


# =============================================================================
# Temporal split role table
# =============================================================================

def get_aggregation_scale_label(time_scale: str) -> str:
    """
    Map config time_scale to Aggregation_Scale label used in
    temporal_5fold_window_assignment.csv.
    """
    mapping = {
        "1hour": "1 h",
        "1h": "1 h",
        "24hour": "24 h",
        "24h": "24 h",
        "168hour": "7 d",
        "7day": "7 d",
        "7d": "7 d",
        "31day": "31 d",
        "31d": "31 d",
    }

    if time_scale not in mapping:
        raise ValueError(
            f"Unsupported time_scale: {time_scale}. "
            f"Expected one of: {list(mapping.keys())}"
        )

    return mapping[time_scale]


def normalize_role_value(value) -> str:
    """
    Normalize Role values from temporal role table.
    """
    s = str(value).strip().lower()

    if s in {"training", "train", "is_train"}:
        return "Training"
    if s in {"validation", "valid", "val", "test", "is_test"}:
        return "Validation"
    if s in {"buffer", "buffer_interval", "is_buffer_interval"}:
        return "Buffer"

    # Keep original if unexpected; caller can diagnose by value_counts.
    return str(value).strip()


def to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert common 0/1, true/false, yes/no values to bool.
    """
    if series.dtype == bool:
        return series.fillna(False)

    s = series.astype(str).str.strip().str.lower()

    return s.isin(["1", "true", "t", "yes", "y"])


def load_temporal_fold_role_table(cfg: dict) -> pd.DataFrame:
    """
    Load temporal_5fold_window_assignment.csv as a role table.

    Important:
    This file is not a one-to-one Timestamp -> Temporal_Fold mapping.
    It is a role table indexed by:
        Aggregation_Scale + Temporal_Fold + Timestamp + Role.
    """
    paths = cfg["paths"]
    columns_cfg = cfg["columns"]

    temporal_fold_file = resolve_repo_path(paths["temporal_fold_file"])

    if not temporal_fold_file.exists():
        raise FileNotFoundError(f"Temporal fold file not found: {temporal_fold_file}")

    split_df = pd.read_csv(temporal_fold_file)

    timestamp_col = columns_cfg["split_timestamp_column"]
    fold_col = columns_cfg["split_fold_column"]

    required_cols = {
        "Aggregation_Scale",
        timestamp_col,
        fold_col,
        "Role",
        "Is_Train",
        "Is_Test",
        "Is_Buffer_Interval",
    }

    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(
            f"Temporal fold role table missing required columns: {missing_cols}\n"
            f"Available columns: {split_df.columns.tolist()}"
        )

    scale_label = get_aggregation_scale_label(cfg["time_scale"])

    split_df = split_df[split_df["Aggregation_Scale"].astype(str) == scale_label].copy()

    if split_df.empty:
        raise ValueError(
            f"No rows found in temporal fold role table for Aggregation_Scale={scale_label}"
        )

    split_df[timestamp_col] = pd.to_datetime(split_df[timestamp_col], errors="coerce")

    if split_df[timestamp_col].isna().any():
        n_bad = int(split_df[timestamp_col].isna().sum())
        raise ValueError(f"Temporal fold role table contains unparseable timestamps: {n_bad}")

    split_df[fold_col] = split_df[fold_col].astype(int)
    split_df["Role_Normalized"] = split_df["Role"].map(normalize_role_value)

    print("=" * 100)
    print("Loaded temporal fold role table")
    print(f"Temporal fold file: {temporal_fold_file}")
    print(f"Aggregation_Scale: {scale_label}")
    print(f"Rows: {len(split_df)}")
    print(f"Timestamp range: {split_df[timestamp_col].min()} -> {split_df[timestamp_col].max()}")
    print(f"Folds: {sorted(split_df[fold_col].unique().tolist())}")
    print("Role counts:")
    print(split_df["Role_Normalized"].value_counts())
    print("=" * 100)

    return split_df


def load_buffer_rule(cfg: dict) -> int:
    """
    Load buffer length from temporal_buffer_intervals.csv.
    This is primarily recorded for reporting because the role table already
    defines Training / Validation / Buffer timestamps.
    """
    paths = cfg["paths"]
    time_scale = cfg["time_scale"]

    buffer_file = resolve_repo_path(paths["temporal_buffer_file"])

    if not buffer_file.exists():
        print("[WARNING] temporal_buffer_file not found. Using expected_gap_hours from config.")
        expected_gap = cfg.get("temporal_cv", {}).get("expected_gap_hours", 0)
        return 0 if expected_gap is None else int(expected_gap)

    buffer_df = pd.read_csv(buffer_file)

    required_cols = {"Aggregation_Scale", "Buffer_Length", "Buffer_Length_Unit"}
    missing_cols = required_cols - set(buffer_df.columns)

    if missing_cols:
        raise ValueError(
            f"Temporal buffer file missing required columns: {missing_cols}\n"
            f"Available columns: {buffer_df.columns.tolist()}"
        )

    subset = buffer_df[buffer_df["Aggregation_Scale"].astype(str) == str(time_scale)].copy()

    if subset.empty:
        # Also try mapped scale label, in case the file uses "1 h" style labels.
        scale_label = get_aggregation_scale_label(time_scale)
        subset = buffer_df[buffer_df["Aggregation_Scale"].astype(str) == scale_label].copy()

    if subset.empty:
        print("[WARNING] No matching Aggregation_Scale in temporal_buffer_file. Using config expected_gap_hours.")
        expected_gap = cfg.get("temporal_cv", {}).get("expected_gap_hours", 0)
        return 0 if expected_gap is None else int(expected_gap)

    if subset["Buffer_Length"].nunique() != 1:
        raise ValueError(
            f"Multiple Buffer_Length values found for {time_scale}: "
            f"{subset['Buffer_Length'].unique()}"
        )

    unit_values = subset["Buffer_Length_Unit"].astype(str).str.lower().unique().tolist()

    if not all(u in ["hour", "hours"] for u in unit_values):
        raise ValueError(f"Only hour-based buffers are supported. Found units: {unit_values}")

    buffer_hours = int(subset["Buffer_Length"].iloc[0])
    print(f"[INFO] Temporal buffer rule for {time_scale}: {buffer_hours} hours")

    return buffer_hours


def get_role_timestamps_for_fold(
    split_df: pd.DataFrame,
    fold: int,
    cfg: dict,
) -> tuple[set[pd.Timestamp], set[pd.Timestamp], set[pd.Timestamp], dict]:
    """
    Extract training, validation and buffer timestamps for one fold from
    the temporal role table.
    """
    timestamp_col = cfg["columns"]["split_timestamp_column"]
    fold_col = cfg["columns"]["split_fold_column"]

    fold_df = split_df[split_df[fold_col].astype(int) == int(fold)].copy()

    if fold_df.empty:
        raise ValueError(f"No temporal role rows found for fold {fold}")

    # Prefer Role_Normalized.
    training_ts = set(
        fold_df.loc[fold_df["Role_Normalized"] == "Training", timestamp_col]
    )
    validation_ts = set(
        fold_df.loc[fold_df["Role_Normalized"] == "Validation", timestamp_col]
    )
    buffer_ts = set(
        fold_df.loc[fold_df["Role_Normalized"] == "Buffer", timestamp_col]
    )

    # Fallback to explicit boolean columns if Role text is not standard.
    if len(training_ts) == 0:
        is_train = to_bool_series(fold_df["Is_Train"])
        training_ts = set(fold_df.loc[is_train, timestamp_col])

    if len(validation_ts) == 0:
        is_test = to_bool_series(fold_df["Is_Test"])
        validation_ts = set(fold_df.loc[is_test, timestamp_col])

    if len(buffer_ts) == 0:
        is_buffer = to_bool_series(fold_df["Is_Buffer_Interval"])
        buffer_ts = set(fold_df.loc[is_buffer, timestamp_col])

    if len(validation_ts) == 0:
        raise ValueError(f"Fold {fold} has zero validation timestamps.")

    if len(training_ts) == 0:
        raise ValueError(f"Fold {fold} has zero training timestamps.")

    info = {
        "fold": int(fold),
        "n_role_table_rows": int(len(fold_df)),
        "n_training_timestamps": int(len(training_ts)),
        "n_validation_timestamps": int(len(validation_ts)),
        "n_buffer_timestamps": int(len(buffer_ts)),
    }

    return training_ts, validation_ts, buffer_ts, info


def get_temporal_masks_for_fold(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    fold: int,
    cfg: dict,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """
    Build train / validation / buffer masks for one temporal fold using
    the role table.

    Training:
        Role == Training or Is_Train == 1

    Validation:
        Role == Validation or Is_Test == 1

    Buffer:
        Role == Buffer or Is_Buffer_Interval == 1
    """
    timestamp_col = cfg["columns"]["timestamp_column"]

    training_ts, validation_ts, buffer_ts, info = get_role_timestamps_for_fold(
        split_df=split_df,
        fold=fold,
        cfg=cfg,
    )

    train_mask = df[timestamp_col].isin(training_ts)
    val_mask = df[timestamp_col].isin(validation_ts)
    buffer_excluded_mask = df[timestamp_col].isin(buffer_ts)

    info.update({
        "n_train_records": int(train_mask.sum()),
        "n_validation_records": int(val_mask.sum()),
        "n_buffer_records_in_model_input": int(buffer_excluded_mask.sum()),
    })

    if info["n_train_records"] == 0:
        raise ValueError(f"Fold {fold} has zero training records.")

    if info["n_validation_records"] == 0:
        raise ValueError(f"Fold {fold} has zero validation records.")

    print(
        f"[INFO] Fold {fold}: "
        f"training_ts={info['n_training_timestamps']}, "
        f"validation_ts={info['n_validation_timestamps']}, "
        f"buffer_ts={info['n_buffer_timestamps']}"
    )
    print(
        f"[INFO] Fold {fold}: "
        f"train_records={info['n_train_records']}, "
        f"validation_records={info['n_validation_records']}, "
        f"buffer_records={info['n_buffer_records_in_model_input']}"
    )

    return train_mask, val_mask, buffer_excluded_mask, info


def save_temporal_fold_assignment_used_from_role_table(
    split_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save the actual temporal role table used for the current aggregation scale.
    """
    path = output_dir / "temporal_fold_assignment_used.csv"
    split_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path}")
    return path


# =============================================================================
# XGBoost optimization and training
# =============================================================================

def xgb_bayesian_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """
    Bayesian optimization for XGBoost hyperparameters.

    If cv_splits is provided, it is used in cross_val_score. This is useful for
    temporal CV because it respects the released fold assignment and buffer roles.
    """
    bo_cfg = cfg["bayesian_optimization"]
    fixed_params = cfg["xgboost_fixed_params"]

    if not bo_cfg.get("enabled", True):
        print("[INFO] Bayesian optimization disabled. Using default XGBoost parameters.")
        params = dict(fixed_params)
        params.update(cfg.get("default_xgboost_params_when_bo_disabled", {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }))
        return params

    print("\n[Step 1] 正在全局数据集上使用贝叶斯优化进行参数寻优...")

    bounds = bo_cfg["parameter_bounds"]
    pbounds = {key: tuple(value) for key, value in bounds.items()}

    def xgb_crossval(
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        reg_alpha,
        reg_lambda,
    ):
        params = dict(fixed_params)
        params.update({
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        })

        model = xgb.XGBRegressor(**params)

        if cv_splits is not None:
            cv = cv_splits
        else:
            cv = int(bo_cfg.get("cv", 3))

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=bo_cfg.get("scoring", "neg_mean_squared_error"),
        )

        return float(np.mean(scores))

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds=pbounds,
        random_state=int(fixed_params.get("random_state", 42)),
        verbose=2,
    )

    optimizer.maximize(
        init_points=int(bo_cfg.get("init_points", 5)),
        n_iter=int(bo_cfg.get("n_iter", 20)),
    )

    best_params = optimizer.max["params"]
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])

    final_params = dict(fixed_params)
    final_params.update(best_params)

    print(f"\n最佳全局参数: {final_params}")
    return final_params


# =============================================================================
# Main temporal CV workflow
# =============================================================================

def run_temporal_cv(cfg: dict) -> None:
    """
    Main workflow for temporal 5-fold XGBoost and SHAP extraction.
    """
    experiment_id = cfg["experiment_id"]
    n_folds = int(cfg["temporal_cv"].get("n_folds", 5))

    output_dir = resolve_repo_path(cfg["paths"]["output_dir"])
    ensure_output_dir(output_dir)

    print("=" * 100)
    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {output_dir}")
    print("=" * 100)

    write_json(output_dir / "run_config_used.json", cfg)

    # -------------------------------------------------------------------------
    # 1. Load and preprocess
    # -------------------------------------------------------------------------
    df, selected_features, target_var, preprocessing_report = load_and_preprocess_data(
        cfg=cfg,
        output_dir=output_dir,
    )

    station_col = cfg["columns"]["station_column"]
    timestamp_col = cfg["columns"]["timestamp_column"]

    write_json(output_dir / "preprocessing_report.json", preprocessing_report)

    # Keep full filtered context for SHAP mapping
    df_iqr_filtered = df.copy()
    df_iqr_filtered["IQR后索引"] = df_iqr_filtered.index

    feature_list_path = output_dir / "feature_list_used.csv"
    pd.DataFrame({"feature": selected_features}).to_csv(
        feature_list_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[WRITE] {feature_list_path}")

    # -------------------------------------------------------------------------
    # 2. Load temporal role table and buffer rule
    # -------------------------------------------------------------------------
    split_df = load_temporal_fold_role_table(cfg)
    save_temporal_fold_assignment_used_from_role_table(split_df, output_dir)

    buffer_hours = load_buffer_rule(cfg)

    # -------------------------------------------------------------------------
    # 3. Precompute fold masks and CV splits
    # -------------------------------------------------------------------------
    fold_split_records = []
    cv_splits_for_bo = []

    for fold_idx in range(1, n_folds + 1):
        train_mask, val_mask, buffer_mask, fold_info = get_temporal_masks_for_fold(
            df=df,
            split_df=split_df,
            fold=fold_idx,
            cfg=cfg,
        )

        train_idx = np.where(train_mask.values)[0]
        val_idx = np.where(val_mask.values)[0]

        cv_splits_for_bo.append((train_idx, val_idx))

        fold_info["buffer_hours_reported"] = int(buffer_hours)
        fold_split_records.append(fold_info)

    fold_split_summary_path = output_dir / "temporal_fold_split_summary.csv"
    pd.DataFrame(fold_split_records).to_csv(
        fold_split_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[WRITE] {fold_split_summary_path}")

    # -------------------------------------------------------------------------
    # 4. Global parameter optimization
    # -------------------------------------------------------------------------
    scaler_global = StandardScaler()
    X_global_scaled = pd.DataFrame(
        scaler_global.fit_transform(df[selected_features]),
        columns=selected_features,
        index=df.index,
    )
    y_global = df[target_var]

    best_params = xgb_bayesian_optimization(
        X_train=X_global_scaled,
        y_train=y_global,
        cfg=cfg,
        cv_splits=cv_splits_for_bo,
    )
    write_json(output_dir / "best_params.json", best_params)

    # -------------------------------------------------------------------------
    # 5. Fold-wise training and validation SHAP
    # -------------------------------------------------------------------------
    all_metrics = {
        "fold": [],
        "train_rmse": [],
        "train_r2": [],
        "train_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "val_mae": [],
        "n_train": [],
        "n_val": [],
        "n_buffer_records_in_model_input": [],
        "n_training_timestamps": [],
        "n_validation_timestamps": [],
        "n_buffer_timestamps": [],
    }

    all_built_in_importances = []
    all_perm_importances = []

    list_shap_val_dfs = []
    list_val_data_dfs = []
    list_prediction_dfs = []
    buffer_exclusion_rows = []

    print("\n[Step 2] 开始 5-Fold 时间交叉验证与 SHAP 抽取...")

    for fold_idx in range(1, n_folds + 1):
        print(f"\n{'-' * 20} Temporal Fold {fold_idx} {'-' * 20}")

        train_mask, val_mask, buffer_excluded_mask, fold_info = get_temporal_masks_for_fold(
            df=df,
            split_df=split_df,
            fold=fold_idx,
            cfg=cfg,
        )

        X_train = df.loc[train_mask, selected_features]
        X_val = df.loc[val_mask, selected_features]
        y_train = df.loc[train_mask, target_var]
        y_val = df.loc[val_mask, target_var]

        # Scaling: fit on training fold only
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_s = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train_s, y_train)

        y_train_pred = model.predict(X_train_s)
        y_val_pred = model.predict(X_val_s)

        fold_train_rmse = rmse(y_train, y_train_pred)
        fold_train_r2 = float(r2_score(y_train, y_train_pred))
        fold_train_mae = float(mean_absolute_error(y_train, y_train_pred))

        fold_val_rmse = rmse(y_val, y_val_pred)
        fold_val_r2 = float(r2_score(y_val, y_val_pred))
        fold_val_mae = float(mean_absolute_error(y_val, y_val_pred))

        all_metrics["fold"].append(fold_idx)
        all_metrics["train_rmse"].append(fold_train_rmse)
        all_metrics["train_r2"].append(fold_train_r2)
        all_metrics["train_mae"].append(fold_train_mae)
        all_metrics["val_rmse"].append(fold_val_rmse)
        all_metrics["val_r2"].append(fold_val_r2)
        all_metrics["val_mae"].append(fold_val_mae)
        all_metrics["n_train"].append(int(len(X_train)))
        all_metrics["n_val"].append(int(len(X_val)))
        all_metrics["n_buffer_records_in_model_input"].append(
            int(fold_info["n_buffer_records_in_model_input"])
        )
        all_metrics["n_training_timestamps"].append(
            int(fold_info["n_training_timestamps"])
        )
        all_metrics["n_validation_timestamps"].append(
            int(fold_info["n_validation_timestamps"])
        )
        all_metrics["n_buffer_timestamps"].append(
            int(fold_info["n_buffer_timestamps"])
        )

        print(
            f"Fold {fold_idx}: "
            f"val_R2={fold_val_r2:.4f}, "
            f"val_RMSE={fold_val_rmse:.4f}, "
            f"val_MAE={fold_val_mae:.4f}"
        )

        buffer_exclusion_rows.append({
            "fold": fold_idx,
            "buffer_hours": int(buffer_hours),
            "n_buffer_timestamps": int(fold_info["n_buffer_timestamps"]),
            "n_buffer_records_in_model_input": int(fold_info["n_buffer_records_in_model_input"]),
            "n_train_records": int(len(X_train)),
            "n_validation_records": int(len(X_val)),
        })

        # Built-in feature importance
        fold_imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        })
        all_built_in_importances.append(fold_imp.set_index("feature"))

        # Permutation importance
        perm_res = permutation_importance(
            model,
            X_train_s,
            y_train,
            n_repeats=5,
            random_state=int(cfg["xgboost_fixed_params"].get("random_state", 42)),
            n_jobs=-1,
        )
        fold_perm = pd.DataFrame({
            "feature": X_train.columns,
            "permutation_importance": perm_res.importances_mean,
        })
        all_perm_importances.append(fold_perm.set_index("feature"))

        # Predictions
        pred_df = pd.DataFrame({
            "IQR后索引": X_val_s.index,
            "fold": fold_idx,
            "station": df_iqr_filtered.loc[X_val_s.index, station_col].values,
            "timestamp": df_iqr_filtered.loc[X_val_s.index, timestamp_col].astype(str).values,
            "y_true": y_val.values,
            "y_pred": y_val_pred,
        })
        list_prediction_dfs.append(pred_df)

        # SHAP on validation fold
        if cfg.get("shap", {}).get("enabled", True):
            print(f"正在计算 Fold {fold_idx} 的验证集 SHAP 值...")

            explainer = shap.TreeExplainer(model)
            shap_values_val = explainer.shap_values(X_val_s)

            shap_val_df = pd.DataFrame(
                shap_values_val,
                columns=[f"shap_{col}" for col in X_val_s.columns],
                index=X_val_s.index,
            )

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = np.asarray(expected_value).ravel()[0]

            shap_val_df["base_value"] = expected_value
            shap_cols = [f"shap_{col}" for col in X_val_s.columns]
            shap_val_df["prediction"] = shap_val_df["base_value"] + shap_val_df[shap_cols].sum(axis=1)

            shap_val_df = shap_val_df.reset_index().rename(columns={"index": "IQR后索引"})
            shap_val_df.insert(1, "fold", fold_idx)
            list_shap_val_dfs.append(shap_val_df)

            val_data_for_shap = df_iqr_filtered.loc[
                X_val_s.index,
                selected_features + [target_var],
            ].copy()
            val_data_for_shap["IQR后索引"] = val_data_for_shap.index
            val_data_for_shap["fold"] = fold_idx
            val_data_for_shap["站点名称"] = df_iqr_filtered.loc[X_val_s.index, station_col].values
            val_data_for_shap["Timestamp"] = df_iqr_filtered.loc[X_val_s.index, timestamp_col].astype(str).values

            val_data_for_shap = val_data_for_shap[
                ["IQR后索引", "fold", "站点名称", "Timestamp"]
                + selected_features
                + [target_var]
            ]

            list_val_data_dfs.append(val_data_for_shap)

    # -------------------------------------------------------------------------
    # 6. Save metrics
    # -------------------------------------------------------------------------
    fold_metrics_df = pd.DataFrame(all_metrics)

    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {fold_metrics_path}")

    fold_metrics_xlsx_path = output_dir / "all_folds_detail.xlsx"
    fold_metrics_df.to_excel(fold_metrics_xlsx_path, index=False)
    print(f"[WRITE] {fold_metrics_xlsx_path}")

    buffer_exclusion_path = output_dir / "temporal_buffer_exclusion_summary.csv"
    pd.DataFrame(buffer_exclusion_rows).to_csv(
        buffer_exclusion_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[WRITE] {buffer_exclusion_path}")

    summary_metrics = {
        "experiment_id": experiment_id,
        "target_var": target_var,
        "n_folds": n_folds,
        "buffer_hours_reported": int(buffer_hours),
        "mean_train_rmse": float(fold_metrics_df["train_rmse"].mean()),
        "std_train_rmse": float(fold_metrics_df["train_rmse"].std()),
        "mean_train_r2": float(fold_metrics_df["train_r2"].mean()),
        "std_train_r2": float(fold_metrics_df["train_r2"].std()),
        "mean_train_mae": float(fold_metrics_df["train_mae"].mean()),
        "std_train_mae": float(fold_metrics_df["train_mae"].std()),
        "mean_val_rmse": float(fold_metrics_df["val_rmse"].mean()),
        "std_val_rmse": float(fold_metrics_df["val_rmse"].std()),
        "mean_val_r2": float(fold_metrics_df["val_r2"].mean()),
        "std_val_r2": float(fold_metrics_df["val_r2"].std()),
        "mean_val_mae": float(fold_metrics_df["val_mae"].mean()),
        "std_val_mae": float(fold_metrics_df["val_mae"].std()),
        "total_validation_records": int(fold_metrics_df["n_val"].sum()),
        "total_buffer_records_in_model_input": int(
            fold_metrics_df["n_buffer_records_in_model_input"].sum()
        ),
    }

    write_json(output_dir / "cv_summary_metrics.json", summary_metrics)

    summary_str = (
        f"\n{'=' * 20} 5-Fold Temporal CV 平均指标汇总 (XGBoost) {'=' * 20}\n"
        f"Experiment: {experiment_id}\n"
        f"基于已发布 temporal_5fold_window_assignment.csv 的 5-Fold 时间交叉验证结果:\n"
        f"Aggregation scale: {get_aggregation_scale_label(cfg['time_scale'])}\n"
        f"Reported temporal buffer: {buffer_hours} hours\n"
        f"训练集平均 RMSE: {summary_metrics['mean_train_rmse']:.4f} ± {summary_metrics['std_train_rmse']:.4f}\n"
        f"训练集平均 R^2:  {summary_metrics['mean_train_r2']:.4f} ± {summary_metrics['std_train_r2']:.4f}\n"
        f"训练集平均 MAE:  {summary_metrics['mean_train_mae']:.4f} ± {summary_metrics['std_train_mae']:.4f}\n"
        f"验证集平均 RMSE: {summary_metrics['mean_val_rmse']:.4f} ± {summary_metrics['std_val_rmse']:.4f}\n"
        f"验证集平均 R^2:  {summary_metrics['mean_val_r2']:.4f} ± {summary_metrics['std_val_r2']:.4f}\n"
        f"验证集平均 MAE:  {summary_metrics['mean_val_mae']:.4f} ± {summary_metrics['std_val_mae']:.4f}\n"
        f"{'=' * 50}\n"
    )

    print(summary_str)

    summary_txt_path = output_dir / "summary_metrics.txt"
    summary_txt_path.write_text(summary_str, encoding="utf-8")
    print(f"[WRITE] {summary_txt_path}")

    # -------------------------------------------------------------------------
    # 7. Save predictions
    # -------------------------------------------------------------------------
    prediction_df = pd.concat(list_prediction_dfs, ignore_index=True)

    prediction_path = output_dir / "oof_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {prediction_path}")

    # -------------------------------------------------------------------------
    # 8. Save SHAP and validation data
    # -------------------------------------------------------------------------
    if list_shap_val_dfs:
        final_shap_val_df = pd.concat(list_shap_val_dfs, ignore_index=True)
        final_val_data_df = pd.concat(list_val_data_dfs, ignore_index=True)

        shap_val_path = output_dir / "shap_values_val.parquet"
        val_data_path = output_dir / "val_data_for_shap.parquet"

        final_shap_val_df.to_parquet(shap_val_path, index=False)
        final_val_data_df.to_parquet(val_data_path, index=False)

        print(f"[WRITE] {shap_val_path}")
        print(f"[WRITE] {val_data_path}")

        # Compatibility files for downstream figure scripts
        if cfg.get("outputs", {}).get("save_combined_compatibility_files", True):
            shap_combined_df = final_shap_val_df.copy()

            if "dataset" not in shap_combined_df.columns:
                shap_combined_df.insert(1, "dataset", "val")

            shap_combined_path = output_dir / "shap_values_combined.parquet"
            shap_combined_df.to_parquet(shap_combined_path, index=False)
            print(f"[WRITE] {shap_combined_path}")

            combined_data_for_shap = final_val_data_df.copy()

            if "dataset" not in combined_data_for_shap.columns:
                combined_data_for_shap.insert(1, "dataset", "val")

            combined_data_path = output_dir / "combined_data_for_shap.parquet"
            combined_data_for_shap.to_parquet(combined_data_path, index=False)
            print(f"[WRITE] {combined_data_path}")

    # -------------------------------------------------------------------------
    # 9. Save feature importance
    # -------------------------------------------------------------------------
    avg_builtin_imp = (
        pd.concat(all_built_in_importances, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_builtin_imp.columns = ["feature", "importance"]

    avg_builtin_path = output_dir / "average_builtin_importance.csv"
    avg_builtin_imp.to_csv(avg_builtin_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {avg_builtin_path}")

    avg_perm_imp = (
        pd.concat(all_perm_importances, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_perm_imp.columns = ["feature", "permutation_importance"]

    avg_perm_path = output_dir / "average_permutation_importance.csv"
    avg_perm_imp.to_csv(avg_perm_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {avg_perm_path}")

    # -------------------------------------------------------------------------
    # 10. Run metadata
    # -------------------------------------------------------------------------
    metadata = {
        "experiment_id": experiment_id,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "config_path": cfg.get("_config_path", ""),
        "output_dir": str(output_dir),
        "aggregation_scale": get_aggregation_scale_label(cfg["time_scale"]),
        "n_rows_after_preprocessing": int(len(df)),
        "n_features_used": int(len(selected_features)),
        "n_stations": int(df[station_col].nunique()),
        "buffer_hours_reported": int(buffer_hours),
        "temporal_split_source": cfg["paths"]["temporal_fold_file"],
        "temporal_split_interpretation": "Aggregation_Scale + Temporal_Fold + Role table",
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("\n任务完成。")
    print(f"输出目录: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBoost temporal 5-fold CV from a JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config, relative to repository root or absolute path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_temporal_cv(cfg)


if __name__ == "__main__":
    main()