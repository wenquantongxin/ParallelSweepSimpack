# ParallelSweepSimpack

## 项目简介
ParallelSweepSimpack 面向轨道交通车辆转向架悬挂参数的并行仿真与多目标优化。项目通过 Python 脚本批量生成 SIMPACK 所需的 .spck/.spf/.subvar 文件，自动调用 `simpack-cmd` 完成多体动力学仿真，并以临界速度 (critical velocity)、轮轨磨耗指标 (wear number)、Sperling 指标 (ride comfort) 等多项性能指标为目标进行优化。整体流程支持大规模参数敏感性分析、正交试验 (Orthogonal DoE) 以及多目标遗传算法 (NSGA-II/NSGA-III/R-NSGA-II) 的运行。

## 功能特性
- 并行批量仿真 (parallel sweep)，利用 `concurrent.futures` 同时调度多个 SIMPACK 任务。
- 自动临界速度搜索 (critical speed search)，结合二分法与 SIMPACK 后处理 (`simpack-post`)。
- 多目标进化算法优化 (multi-objective evolutionary optimization)，基于 `pymoo` 的 NSGA-II/NSGA-III/R-NSGA-II 等。
- DOE/扫频分析脚本，支持 Lx1/Lx2/Lx3 参数网格扫描与结果持久化。
- 参数相关性与敏感性分析的可视化。

## 目录结构
```
ParallelSweepSimpack
|-- BatchTmp/                     # 运行时生成的 SIMPACK 批量输入与输出 (自动创建)
|-- ChkPnt/                       # 优化过程中每一代的种群快照
|-- Results_OrthogonalDoE/        # 正交试验与扫频结果
|-- SweepLx1Lx2xL3x.py            # 三参数网格扫略脚本
|-- Opt_12vars_to_3dims.py        # 多目标优化主程序 (NSGA 系列)
|-- FindCrticalVelocity.py        # 临界速度自动搜索工具
|-- PrepareBatchFiles.py          # 批量生成 .spck/.spf/.subvar 辅助脚本
|-- CRVPerf.py / STRPerf.py       # 曲线通过与直线运行性能计算
|-- config_opt.xlsx               # 参数范围、batch 设置、权重等配置
|-- ref_files/                    # 模型基准文件和参考几何
|-- SbrExport_SPCKResult.qs       # SIMPACK post 处理脚本 (qs)
|-- Analysis_*.ipynb              # 数据分析、可视化与参考点生成
```

## 环境依赖
### 基础软件
- SIMPACK 2021（需包含 `simpack-cmd`, `simpack-slv`, `simpack-post` 命令行工具，并在系统 PATH 中可直接调用）。
- 安装 Anaconda / Miniconda 以方便管理 Python 环境。

### Python 环境 (Python 3.10 推荐)
```bash
conda create -n parallelsweep python=3.10
conda activate parallelsweep
conda install pandas numpy matplotlib ipykernel ipywidgets openpyxl seaborn
pip install pymoo scikit-learn SALib ipympl plotly open3d
# 如果需要进行深度学习辅助建模，可按需安装 PyTorch:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 配置步骤
1. **准备 Python 环境**  
2. **设置 SIMPACK 命令行路径**  
3. **配置优化/仿真参数**  
   根据当前研究任务编辑 `config_opt.xlsx`：
   - `Sheet1` 定义 12 个悬挂参数范围、批次标签 (tag)、速度扫描范围等。
   - `BatchSettings` 可调整并行批次数量、超时时间等。
4. **生成批量输入文件（可选）**  
   如果需要先单独准备 SIMPACK 批量，运行 `python PrepareBatchFiles.py`，或直接让主脚本自动创建。
5. **启动多目标优化**  
   修改 `Opt_12vars_to_3dims.py` 中的 `WorkingDir`、`tag`、`StartVel/EndVel`、`batch_size` 等参数后执行：  
   `python Opt_12vars_to_3dims.py`
   - 脚本会自动在 `BatchTmp/` 中生成并行任务并调用 SIMPACK 求解。
   - 每一代的种群保存在 `ChkPnt/generation_*.npz`，非支配解保存在 `generation_*_nondom.npz`。
   - 最终 Pareto 前沿写入 `final_solutions.csv` 和 `final_solutions.npz`。
6. **执行参数扫频或验证（按需）**  
   - 网格扫频：`python SweepLx1Lx2xL3x.py`。  
   - 单点临界速度验证：`python FindCrticalVelocity.py`，根据提示选择 `tag` 与 `idx`。
7. **结果分析与可视化**  
   - `Analysis_OptsResults.ipynb`：综合分析 NSGA 结果。  
   - `Analysis_Corr.ipynb`：参数-指标相关性与敏感性研究。  
   运行前请在 Jupyter Notebook / MATLAB 中切换至项目根目录。

## 主要工作流说明
- **并行仿真 (Parallel Batch Simulation)**：`opt_PrepareSpckFilesForEachBatch` 自动复制基准 `.spck/.spf`，并重写 `subvar`；`opt_WriteSubvarsFile_idx` 根据参数向量写入悬挂变量；`opt_XEvalPrl` 调用 SIMPACK 并返回三项性能指标。
- **临界速度搜索 (Critical Velocity Search)**：`FindCrticalVelocity.py` 通过二分法调整目标速度，并调用 `SbrExport_SPCKResult.qs` 解析 `.sbr` 输出，提取最大横移量，结果写入 `.dat/.csv` 与 `Overall_FinalSolutions.npz`。
- **多目标遗传算法 (Multi-objective Genetic Algorithm)**：支持 NSGA-II、NSGA-III、R-NSGA-II，切换 `selected_algorithm` 即可；`my_callback` 会把每代结果存档，便于中途恢复或后续分析。
- **设计空间探索与 DoE**：`SweepLx1Lx2xL3x.py` 针对关键参数网格扫描，结果位于 `Results_OrthogonalDoE/`；`Analysis_doe.py` 与 `Analysis_Doe.ipynb` 提供统计分析与可视化。

## 数据管理与版本控制建议
- `.gitignore` 已排除大体积仿真结果、`.mlx` 与 `.sbr` 等文件，避免仓库体积膨胀。
- 若需保留关键仿真数据，可将文件打包后放入 `Results_OrthogonalDoE/` 或外部数据仓库。
- 新的参数批次建议使用不同 `tag` 以示区分。
