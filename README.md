# ParallelSweepSimpack

- 使用并行化编程进行转向架设计参数的扫略，并行调用 SIMPACK 车辆模型计算临界速度。
- 使用 pymoo 以三种动力学指标为目标，包括曲线磨耗数之和、典型地铁直线运行工况 Sperling 均方根、临界速度，进行悬挂参数的迭代寻优。

## 工程简介

- **参数配置**：如 `config_opt.xlsx` 中保存的优化相关信息
- **关键 Python 脚本**：
  - `FindCrticalVelocity`：二分搜索算法求解临界速度
  - `PrepareBatchFiles.py`：生成批量仿真文件
  - `SweepL1xL2xL3x.py`：针对三个维度的扫频脚本
  - `CRVPerf.py`：曲线通过性能计算
  - `STRPerf.py`：直线（主要是平稳性）计算
- **SIMPACK 文件**：如 `Vehicle4WDB_IRWCRV300m_OptBase.spck`、`subvars_OptBase.subvar` 等
- **输出文件**：仿真结果 `.sbr` / `.sir` / `.spf` / `.dat` 等

## 环境安装

在 pypack 环境之中，使用 Anaconda Prompt 安装

- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install pymoo pandas numpy ipywidgets matplotlib openpyxl ipykernel
- pip install -U pymoo ipympl
