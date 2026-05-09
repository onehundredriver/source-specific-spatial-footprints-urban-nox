# 项目总体概述 / Project Overview

## 中文版
本项目旨在研究上海市城市 NOx 排放源的空间和时间分布特征，通过机器学习方法（XGBoost / 随机森林 / LightGBM）分析不同来源对城市 NOx 的贡献。  
仓库提供核心 pipeline 脚本、特征列表、小型示例数据和论文图复现脚本。  
大型模型输入和衍生数据已上传至 Zenodo，可通过 DOI 下载。

### 仓库结构
- `code/`：模型 pipeline、配置文件
- `data/model_inputs/`：小型示例数据，parquet 文件上传 Zenodo
- `data/model_feature_lists/`：模型特征列表
- `data/public/`：公共 NOx 数据和站点元数据
- `data/supplementary_data/`：聚类结果和衍生数据
- `figures/`：复现论文图
- `docs/`：项目总体介绍和复现指南

## English Version
This project aims to analyze the spatial and temporal patterns of urban NOx sources in Shanghai using machine learning models (XGBoost / Random Forest / LightGBM).  
The repository contains core pipeline scripts, feature lists, small example datasets, and figure reproduction scripts.  
Large model inputs and derived datasets are hosted on Zenodo, accessible via DOI.

### Repository structure
- `code/` : pipelines and config files
- `data/model_inputs/` : small example datasets, large parquet files on Zenodo
- `data/model_feature_lists/` : feature lists
- `data/public/` : public NOx and station metadata
- `data/supplementary_data/` : clustering results and derived data
- `figures/` : figure reproduction
- `docs/` : project overview and reproducibility guide
**数据 DOI / Data DOI:** [10.5281/zenodo.20092921](https://doi.org/10.5281/zenodo.20092921)
