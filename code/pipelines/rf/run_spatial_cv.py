# -*- coding: utf-8 -*-
"""
Configuration-driven Random Forest spatial 5-fold cross-validation.

Example commands, run from repository root:

python code/pipelines/rf/run_spatial_cv.py --config code/config/rf_spatial_1hour.json
python code/pipelines/rf/run_spatial_cv.py --config code/config/rf_spatial_24hour.json
python code/pipelines/rf/run_spatial_cv.py --config code/config/rf_spatial_168hour.json
python code/pipelines/rf/run_spatial_cv.py --config code/config/rf_spatial_31day.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor
import shap


# -----------------------------------------------------------------------------
# Repository root
# -----------------------------------------------------------------------------

# run_spatial_cv.py is located at:
# code/pipelines/rf/run_spatial_cv.py
REPO_ROOT = Path(__file__).resolve().parents[3]


# -----------------------------------------------------------------------------
# Matplotlib font setting
# -----------------------------------------------------------------------------

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

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
    Write JSON with UTF-8.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rmse(y_true, y_pred) -> float:
    """
    Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def remove_outliers_iqr(data: pd.DataFrame, column: str, k: float = 1.5) -> tuple[pd.DataFrame, dict]:
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
        "k": k,
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


def visualize_outlier_analysis(data: pd.DataFrame, column: str, output_dir: Path, k: float = 1.5) -> None:
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

    axes[0, 0].axhline(y=lower_bound, color="r", linestyle="--", alpha=0.7, label=f"Lower: {lower_bound:.2f}")
    axes[0, 0].axhline(y=upper_bound, color="r", linestyle="--", alpha=0.7, label=f"Upper: {upper_bound:.2f}")
    axes[0, 0].legend()

    axes[0, 1].hist(data[column], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 1].axvline(lower_bound, color="r", linestyle="--", linewidth=2, label="IQR lower")
    axes[0, 1].axvline(upper_bound, color="r", linestyle="--", linewidth=2, label="IQR upper")
    axes[0, 1].axvline(data[column].mean(), color="orange", linestyle="-", linewidth=2, label="Mean")
    axes[0, 1].axvline(data[column].median(), color="green", linestyle="-", linewidth=2, label="Median")
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
    plt.savefig(output_dir / "outlier_analysis_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

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


def load_and_preprocess_data(cfg: dict, output_dir: Path) -> tuple[pd.DataFrame, list[str], str, dict]:
    """
    Load modelling matrix and selected features, then perform target filtering,
    outlier screening and predictor median imputation.
    """
    paths = cfg["paths"]
    columns_cfg = cfg["columns"]
    preprocessing_cfg = cfg["preprocessing"]

    model_input_file = resolve_repo_path(paths["model_input_file"])
    feature_list_file = resolve_repo_path(paths["feature_list_file"])

    target_col = columns_cfg["target_column"]
    station_col = columns_cfg["station_column"]
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

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    if station_col not in df.columns:
        raise ValueError(f"Station column not found: {station_col}")

    selected_features_raw = load_feature_list(feature_list_file, feature_list_col)

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
        "raw_rows": int(len(df)),
        "raw_cols": int(len(df.columns)),
        "feature_list_count": int(len(selected_features_raw)),
        "matched_feature_count": int(len(selected_features)),
        "missing_feature_count": int(len(missing_features)),
        "missing_features": missing_features,
    }

    if preprocessing_cfg.get("drop_missing_target", True):
        original_rows = len(df)
        df = df.dropna(subset=[target_col]).copy()
        print(f"[INFO] 删除目标变量缺失行后: {original_rows} -> {len(df)}")
        preprocessing_report["rows_after_drop_missing_target"] = int(len(df))

    # Outlier visualization before filtering
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

    # Reset index after filtering; this is crucial for SHAP index mapping.
    df = df.reset_index(drop=True)
    print("[INFO] IQR 过滤后已重置索引。")

    # Predictor missing-value imputation
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

    return df, selected_features, target_col, preprocessing_report


# -----------------------------------------------------------------------------
# Spatial split
# -----------------------------------------------------------------------------

def create_spatial_folds_from_assignment(
    df: pd.DataFrame,
    spatial_fold_file_path: Path,
    n_folds: int = 5,
    data_station_col: str = "Name",
    split_station_col: str = "Station_Name_Public",
    fold_col: str = "Spatial_Fold",
) -> dict[int, list[str]]:
    """
    Read released spatial_5fold_station_assignment.csv and construct
    validation-station lists for each spatial fold.
    """
    print("=" * 100)
    print(f"正在从已发布的空间五折划分文件读取 {n_folds} 折空间交叉验证...")
    print(f"空间划分文件: {spatial_fold_file_path}")
    print("=" * 100)

    if not spatial_fold_file_path.exists():
        raise FileNotFoundError(f"Spatial fold file not found: {spatial_fold_file_path}")

    split_df = pd.read_csv(spatial_fold_file_path)

    required_cols = {split_station_col, fold_col}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(
            f"空间划分文件缺少必要列: {missing_cols}\n"
            f"当前列名为: {split_df.columns.tolist()}"
        )

    if data_station_col not in df.columns:
        raise ValueError(
            f"建模数据中未找到站点列: {data_station_col}\n"
            f"当前列名为: {df.columns.tolist()[:30]} ..."
        )

    all_stations = set(df[data_station_col].dropna().astype(str).unique())

    split_df = split_df.copy()
    split_df[split_station_col] = split_df[split_station_col].astype(str)

    folds = {i: [] for i in range(n_folds)}

    for fold_number in sorted(split_df[fold_col].dropna().unique()):
        fold_number_int = int(fold_number)
        fold_idx = fold_number_int - 1

        if fold_idx < 0 or fold_idx >= n_folds:
            raise ValueError(f"发现非法 fold 编号: {fold_number_int}")

        fold_stations_raw = (
            split_df.loc[split_df[fold_col] == fold_number, split_station_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        valid_stations = [s for s in fold_stations_raw if s in all_stations]
        missing_stations = [s for s in fold_stations_raw if s not in all_stations]

        folds[fold_idx].extend(valid_stations)

        print(
            f"Fold {fold_number_int}: "
            f"split文件站点数={len(fold_stations_raw)}, "
            f"建模数据中匹配站点数={len(valid_stations)}, "
            f"未匹配站点数={len(missing_stations)}"
        )

        if missing_stations:
            print(f"  未匹配站点示例: {missing_stations[:10]}")

    assigned_stations = set()
    for station_list in folds.values():
        assigned_stations.update(station_list)

    unassigned_stations = sorted(all_stations - assigned_stations)
    if unassigned_stations:
        print(f"[WARNING] 建模数据中有 {len(unassigned_stations)} 个站点没有出现在空间五折划分文件中。")
        print(f"未分配站点示例: {unassigned_stations[:20]}")

    empty_folds = [i + 1 for i, stations in folds.items() if len(stations) == 0]
    if empty_folds:
        raise ValueError(f"以下 fold 没有任何有效验证站点: {empty_folds}")

    print("\n空间 5 折验证集站点分配完成：")
    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx + 1}: {len(folds[fold_idx])} 个验证集站点")

    return folds


def split_data_by_fold(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    fold_stations: list[str],
    station_col: str = "Name",
):
    """
    Split train/validation records according to validation-station list.
    """
    val_stations = fold_stations
    train_stations = [s for s in df[station_col].unique() if s not in val_stations]

    train_mask = df[station_col].isin(train_stations)
    val_mask = df[station_col].isin(val_stations)

    return (
        df.loc[train_mask, features],
        df.loc[val_mask, features],
        df.loc[train_mask, target],
        df.loc[val_mask, target],
        train_stations,
        val_stations,
        train_mask,
        val_mask,
    )


def save_spatial_fold_assignment_used(
    folds: dict[int, list[str]],
    output_dir: Path,
) -> Path:
    """
    Save station-level fold assignments actually used by this run.
    """
    rows = []
    for fold_idx, station_list in folds.items():
        for station_name in station_list:
            rows.append({
                "Spatial_Fold": fold_idx + 1,
                "Validation_Station": station_name,
            })

    path = output_dir / "spatial_fold_assignment_used.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path}")
    return path


# -----------------------------------------------------------------------------
# Random Forest optimization and training
# -----------------------------------------------------------------------------

def rf_parameter_selection(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict) -> dict:
    """
    Bayesian optimization for Random Forest hyperparameters.
    This follows the original legacy script design: global hyperparameter search
    is performed before the spatial folds are trained.
    """
    bo_cfg = cfg["bayesian_optimization"]
    fixed_params = cfg["rf_fixed_params"]

    if not bo_cfg.get("enabled", True):
        print("[INFO] Bayesian optimization disabled. Using default Random Forest parameters.")
        params = dict(fixed_params)
        params.update(cfg.get("default_rf_params_when_bo_disabled", {'n_estimators': 300,
 'max_depth': None,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 1.0,
 'bootstrap': True}))
        return params

    raise NotImplementedError(
        "Random Forest Bayesian optimization is not included in this public wrapper. "
        "Set bayesian_optimization.enabled=false, or implement model-specific parameter bounds."
    )


    bounds = bo_cfg["parameter_bounds"]

    pbounds = {
        key: tuple(value)
        for key, value in bounds.items()
    }

    def rf_crossval(n_estimators, max_depth, learning_rate, subsample,
                     colsample_bytree, reg_alpha, reg_lambda):
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

        model = RandomForestRegressor(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=int(bo_cfg.get("cv", 3)),
            scoring=bo_cfg.get("scoring", "neg_mean_squared_error"),
        )
        return float(np.mean(scores))

    optimizer = BayesianOptimization(
        f=rf_crossval,
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


def run_spatial_cv(cfg: dict) -> None:
    """
    Main workflow for spatial 5-fold Random Forest and SHAP extraction.
    """
    experiment_id = cfg["experiment_id"]
    n_folds = int(cfg.get("n_folds", 5))

    output_dir = resolve_repo_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {output_dir}")
    print("=" * 100)

    # Save run config first
    write_json(output_dir / "run_config_used.json", cfg)

    # 1. Load and preprocess
    df, selected_features, target_var, preprocessing_report = load_and_preprocess_data(cfg, output_dir)
    station_col = cfg["columns"]["station_column"]

    # Save preprocessing report
    write_json(output_dir / "preprocessing_report.json", preprocessing_report)

    # Keep full filtered context for SHAP mapping
    df_iqr_filtered = df.copy()
    df_iqr_filtered["IQR后索引"] = df_iqr_filtered.index

    # Save feature list actually used
    feature_list_path = output_dir / "feature_list_used.csv"
    pd.DataFrame({"feature": selected_features}).to_csv(feature_list_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {feature_list_path}")

    # 2. Global parameter optimization
    scaler_global = StandardScaler()
    X_global_scaled = pd.DataFrame(
        scaler_global.fit_transform(df[selected_features]),
        columns=selected_features,
    )
    y_global = df[target_var]

    best_params = rf_parameter_selection(X_global_scaled, y_global, cfg)
    write_json(output_dir / "best_params.json", best_params)

    # 3. Spatial folds
    spatial_fold_file = resolve_repo_path(cfg["paths"]["spatial_fold_file"])
    folds = create_spatial_folds_from_assignment(
        df=df,
        spatial_fold_file_path=spatial_fold_file,
        n_folds=n_folds,
        data_station_col=station_col,
        split_station_col=cfg["columns"]["split_station_column"],
        fold_col=cfg["columns"]["split_fold_column"],
    )
    save_spatial_fold_assignment_used(folds, output_dir)

    # 4. Fold-wise training and validation SHAP
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
        "n_train_stations": [],
        "n_val_stations": [],
    }

    all_built_in_importances = []
    all_perm_importances = []

    list_shap_val_dfs = []
    list_val_data_dfs = []
    list_prediction_dfs = []

    print("\n[Step 2] 开始 5-Fold 空间交叉验证与 SHAP 抽取...")

    for fold_idx in range(n_folds):
        print(f"\n{'-' * 20} Fold {fold_idx + 1} {'-' * 20}")

        (
            X_train,
            X_val,
            y_train,
            y_val,
            tr_st,
            val_st,
            train_mask,
            val_mask,
        ) = split_data_by_fold(
            df,
            selected_features,
            target_var,
            folds[fold_idx],
            station_col=station_col,
        )

        if len(X_val) == 0:
            raise ValueError(f"Fold {fold_idx + 1} has zero validation records.")

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

        model = RandomForestRegressor(**best_params)
        model.fit(X_train_s, y_train)

        y_train_pred = model.predict(X_train_s)
        y_val_pred = model.predict(X_val_s)

        fold_train_rmse = rmse(y_train, y_train_pred)
        fold_train_r2 = float(r2_score(y_train, y_train_pred))
        fold_train_mae = float(mean_absolute_error(y_train, y_train_pred))
        fold_val_rmse = rmse(y_val, y_val_pred)
        fold_val_r2 = float(r2_score(y_val, y_val_pred))
        fold_val_mae = float(mean_absolute_error(y_val, y_val_pred))

        all_metrics["fold"].append(fold_idx + 1)
        all_metrics["train_rmse"].append(fold_train_rmse)
        all_metrics["train_r2"].append(fold_train_r2)
        all_metrics["train_mae"].append(fold_train_mae)
        all_metrics["val_rmse"].append(fold_val_rmse)
        all_metrics["val_r2"].append(fold_val_r2)
        all_metrics["val_mae"].append(fold_val_mae)
        all_metrics["n_train"].append(int(len(X_train)))
        all_metrics["n_val"].append(int(len(X_val)))
        all_metrics["n_train_stations"].append(int(len(tr_st)))
        all_metrics["n_val_stations"].append(int(len(val_st)))

        print(
            f"Fold {fold_idx + 1}: "
            f"val_R2={fold_val_r2:.4f}, val_RMSE={fold_val_rmse:.4f}, val_MAE={fold_val_mae:.4f}"
        )

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
            random_state=int(cfg["rf_fixed_params"].get("random_state", 42)),
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
            "fold": fold_idx + 1,
            "station": df_iqr_filtered.loc[X_val_s.index, station_col].values,
            "y_true": y_val.values,
            "y_pred": y_val_pred,
        })
        list_prediction_dfs.append(pred_df)

        # SHAP on validation fold
        if cfg.get("shap", {}).get("enabled", True):
            print(f"正在计算 Fold {fold_idx + 1} 的验证集 SHAP 值...")
            explainer = shap.Explainer(model)
            shap_values_val = explainer(X_val_s)

            shap_val_df = pd.DataFrame(
                shap_values_val.values,
                columns=[f"shap_{col}" for col in X_val_s.columns],
                index=X_val_s.index,
            )
            shap_val_df["base_value"] = shap_values_val.base_values
            shap_cols = [f"shap_{col}" for col in X_val_s.columns]
            shap_val_df["prediction"] = shap_val_df["base_value"] + shap_val_df[shap_cols].sum(axis=1)

            shap_val_df = shap_val_df.reset_index().rename(columns={"index": "IQR后索引"})
            shap_val_df.insert(1, "fold", fold_idx + 1)
            list_shap_val_dfs.append(shap_val_df)

            val_data_for_shap = df_iqr_filtered.loc[X_val_s.index, selected_features + [target_var]].copy()
            val_data_for_shap["IQR后索引"] = val_data_for_shap.index
            val_data_for_shap["fold"] = fold_idx + 1
            val_data_for_shap["站点名称"] = df_iqr_filtered.loc[X_val_s.index, station_col].values
            val_data_for_shap = val_data_for_shap[
                ["IQR后索引", "fold", "站点名称"] + selected_features + [target_var]
            ]
            list_val_data_dfs.append(val_data_for_shap)

    # 5. Save fold metrics
    fold_metrics_df = pd.DataFrame(all_metrics)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {fold_metrics_path}")

    fold_metrics_xlsx_path = output_dir / "all_folds_detail.xlsx"
    fold_metrics_df.to_excel(fold_metrics_xlsx_path, index=False)
    print(f"[WRITE] {fold_metrics_xlsx_path}")

    summary_metrics = {
        "experiment_id": experiment_id,
        "target_var": target_var,
        "n_folds": n_folds,
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
    }

    write_json(output_dir / "cv_summary_metrics.json", summary_metrics)

    summary_str = (
        f"\n{'=' * 20} 5-Fold CV 平均指标汇总 (Random Forest) {'=' * 20}\n"
        f"Experiment: {experiment_id}\n"
        f"基于已发布 spatial_5fold_station_assignment.csv 的 5-Fold 空间交叉验证结果:\n"
        f"训练集平均 RMSE: {summary_metrics['mean_train_rmse']:.4f} ± {summary_metrics['std_train_rmse']:.4f}\n"
        f"训练集平均 R^2:  {summary_metrics['mean_train_r2']:.4f} ± {summary_metrics['std_train_r2']:.4f}\n"
        f"训练集平均 MAE:  {summary_metrics['mean_train_mae']:.4f} ± {summary_metrics['std_train_mae']:.4f}\n"
        f"验证集平均 RMSE: {summary_metrics['mean_val_rmse']:.4f} ± {summary_metrics['std_val_rmse']:.4f}\n"
        f"验证集平均 R^2:  {summary_metrics['mean_val_r2']:.4f} ± {summary_metrics['std_val_r2']:.4f}\n"
        f"验证集平均 MAE:  {summary_metrics['mean_val_mae']:.4f} ± {summary_metrics['std_val_mae']:.4f}\n"
        f"{'=' * 50}\n"
    )
    print(summary_str)
    (output_dir / "summary_metrics.txt").write_text(summary_str, encoding="utf-8")
    print(f"[WRITE] {output_dir / 'summary_metrics.txt'}")

    # 6. Save predictions
    prediction_df = pd.concat(list_prediction_dfs, ignore_index=True)
    prediction_path = output_dir / "oof_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {prediction_path}")

    # 7. Save SHAP and validation data
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

            combined_data_for_shap = final_val_data_df.copy()
            if "dataset" not in combined_data_for_shap.columns:
                combined_data_for_shap.insert(1, "dataset", "val")
            combined_data_path = output_dir / "combined_data_for_shap.parquet"
            combined_data_for_shap.to_parquet(combined_data_path, index=False)

            print(f"[WRITE] {shap_combined_path}")
            print(f"[WRITE] {combined_data_path}")

    # 8. Save feature importance
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

    # 9. Run metadata
    metadata = {
        "experiment_id": experiment_id,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "config_path": cfg.get("_config_path", ""),
        "output_dir": str(output_dir),
        "n_rows_after_preprocessing": int(len(df)),
        "n_features_used": int(len(selected_features)),
        "n_stations": int(df[station_col].nunique()),
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("\n任务完成。")
    print(f"输出目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Random Forest spatial 5-fold CV from a JSON config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config, relative to repository root or absolute path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_spatial_cv(cfg)


if __name__ == "__main__":
    main()
