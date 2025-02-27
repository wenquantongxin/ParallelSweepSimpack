# ParallelSweepSimpack

- 使用并行化编程进行转向架设计参数的扫略，并行调用 SIMPACK 车辆模型计算临界速度等 7 种动力学指标。
- 使用 pymoo 以三种动力学指标为目标，包括曲线磨耗数之和、典型地铁直线运行工况 Sperling 均方根、临界速度，进行悬挂参数的迭代寻优。

## 工程简介

- **参数配置**：`config_opt.xlsx` 中保存的优化相关信息
- **关键 Python 脚本**：
  - `Opt_12vars_to_3dims.py`：使用 pymoo 群优化算法，对于车辆模型参数进行迭代寻优
  - `FindCrticalVelocity.py`：二分搜索算法求解临界速度
  - `PrepareBatchFiles.py`：生成批量仿真文件
  - `SweepLx1Lx2xL3x.py`：针对三个维度的扫频脚本
  - `CRVPerf.py`：曲线通过性能计算
  - `STRPerf.py`：直线运行计算
- **SIMPACK 文件**：如 `Vehicle4WDB_IRWCRV300m_OptBase.spck`、`subvars_OptBase.subvar` 等
- **输出文件**：仿真结果 `.sbr` / `.sir` / `.spf` / `.dat` 等

## 运行环境配置

在 pypack 环境之中，使用 Anaconda Prompt 安装

- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install pandas numpy ipywidgets matplotlib openpyxl ipykernel
- pip install -U pymoo ipympl scikit-learn seaborn SALib open3d plotly
