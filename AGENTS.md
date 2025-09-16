# AGENTS 指南

## 快速认知
- 项目名称：ParallelSweepSimpack，目标是自动化铁路车辆在 SIMPACK 中的动力学仿真、参数扫略与多目标优化。
- 关键指标：临界速度（critical velocity）、磨耗数（wear number）、Sperling 骑乘舒适度指标。
- 工作方式：利用 Python 编排批量生成 `.spck/.spf/.subvar`，调用 `simpack-slv` 和 `simpack-post`，再通过 `pymoo` 等库执行 NSGA 系列算法与统计分析。

## 核心构件
- `PrepareBatchFiles.py`：按照 `config_opt.xlsx` 提供的基准/上下限生成批量仿真输入，重写 subvar 引用并复制 SPF 模板。
- `FindCrticalVelocity.py`：针对单一参数组合进行二分临界速度搜索，封装了 SIMPACK 求解、稳定性判定以及 `.dat` 结果解析。
- `CRVPerf.py` 与 `STRPerf.py`：分别完成 R300 曲线与 AAR5 直线工况的磨耗/横移量与 Sperling 指标提取。
- `SweepLx1Lx2xL3x.py`：在给定 Lx1/Lx2/Lx3 网格内批量启动上述流程，结果保存至 `ChkPnt/` 与 `Result_<tag>.npy`，并调用 `ProcessPoolExecutor` 并行。
- `Opt_12vars_to_3dims.py`：定义 12 维结构-悬挂参数的多目标问题，目标函数交由 `opt_XEvalPrl` 并发调用 SIMPACK；支持 NSGA-II/III 与参考点驱动的 R-NSGA-II。
- `Analysis_doe.py`：整合正交试验或扫略结果，产出 `Results_OrthogonalDoE/` 下的 tidy 数据、相关矩阵、回归、部分相关与随机森林特征重要度。
- 参考资源：`ref_files/` 存放标准转向架模型，`Vehicle4WDB_*.spck` 为仿真模板，`config_opt.xlsx`、`ParameterSweep_fromExcel.xlsx` 等用于录入变量范围。

## 推荐工作流
1. **环境准备**：满足 README 中的 Python 依赖与 SIMPACK 2021 命令行工具 (`simpack-cmd`, `simpack-slv`, `simpack-post`)；在 Windows 环境下建议预设 PATH。
2. **配置试验空间**：在 `config_opt.xlsx` 填写 12 个设计变量的基准值、上下限与标签；可通过 `ParameterSweep_fromExcel.xlsx` 或 `Overall_FinalSolutions.npz` 复现既有组合。
3. **生成批次文件**：执行 `python PrepareBatchFiles.py` 或由 `SweepLx1Lx2xL3x.py`/`Opt_12vars_to_3dims.py` 内部调用，自动写入 `BatchTmp/`。
4. **批量仿真/评估**：
   - 扫略：在 `SweepLx1Lx2xL3x.py` 中设定 `tag`, `BatchSize_parallel`, `StartVel/EndVel/N_depth`，执行后在 `ChkPnt/` 中获得阶段性 `myXvars_<tag>.npy` 与 `batch_result_<tag>_batch*.npy`。
   - 优化：在 `Opt_12vars_to_3dims.py` 修改 `WorkingDir`, `tag`, `batch_size`, 终止条件与算法选择；运行后输出 `final_solutions.csv/.npz` 与 `res_history.npz`。
5. **后处理**：
   - `FindCrticalVelocity.py` 单独验证某组参数；
   - `Analysis_*.ipynb`、`Analysis_doe.py`、`Analysis_L1L2L3Sweep.ipynb` 或 MATLAB `Analysis_L1L2L3Sweep.mlx` 对结果可视化；
   - 统计分析产物集中在 `Results_OrthogonalDoE/`，绘图在 `sweep_plots_output/` 与 `Results_OrthogonalDoE/fig/`。

## 数据与结果索引
- `ChkPnt/`：存放扫略或优化过程中的输入、输出检查点。
- `BatchTmp/`：每批仿真即时生成的 `.spck/.spf/.subvar` 及 `.output/` 结果，按执行完毕可清理。
- `Results_OrthogonalDoE/`：`tidy.csv`（整洁数据）、`corr_*.csv`（相关矩阵）、`ols_*.txt/csv`（回归结果）、`rf_perm_importance_*.csv`（随机森林重要度）等。
- `Overall_FinalSolutions.npz`、`Result_Sweep_manyNSGARlts.npy`：历史优化成果。
- `sweep_plots_output/`：Lx1/Lx2/Lx3 临界速度切片图 (`CriticalVel_L1L2_Lx3is0.png`) 等。

## 关键假设与限制
- 仿真依赖 SIMPACK 安装及许可；`run_simpack_cmd` 默认 10 分钟超时，必要时在脚本内调整。
- 目录结构默认脚本所在目录为根；跨磁盘执行需确保 `WorkingDir` 指向绝对路径并具有读写权限。
- 并行为进程级，Windows 默认同时启动 `BatchSize_parallel` 个 SIMPACK 实例，需关注 CPU/许可证上限。

## 面向 AI 的协作提示
- 如需扩展优化维度，可在 `config_opt.xlsx` 与 `Opt_12vars_to_3dims.MyBatchProblem` 内同步增加变量与约束，并留意 `Import_Subvars_To_File_idx` 的写入顺序。
- 批量分析脚本设计成模块化函数，可直接在新 Notebook 中 `from Analysis_doe import main`，或复用 `tidy_df` 产物进行机器学习。
- 在自动代理场景下，可通过检测 `BatchTmp/` 是否为空来判断仿真状态，或利用 `run.log`/`res_history.npz` 跟踪进度。
- 所有输出均为 CSV/NPZ/PNG，便于进一步整理入知识库或上链到 MLOps 流水线。

## 已获结论摘要（示例）
- 正交试验分析揭示 Lx3 与磨耗差 `DiffWear` 呈显著线性关系，`Results_OrthogonalDoE/corr_Lx3_vs_targets.csv` 与 `roi_slope_DiffWear_vs_Lx3.json` 给出相关系数与 ROI 斜率。
- 多目标优化结果保存在 `final_solutions.csv` 与 `Overall_FinalSolutions.npz`，可作为后续设计迭代的初始解集。

## 后续建议
1. 如需新增指标（例如噪声或能耗），在 `opt_XEvalPrl` 内补充调用逻辑，并同步扩充 `Problem.n_obj` 与约束定义。
2. 考虑将 `BatchTmp/` 输出改写到日期戳目录，避免并发任务互相覆盖。
3. 将 `Analysis_doe.py` 中的统计过程包装为 CLI `--report` 选项，便于持续集成自动生成 HTML 摘要。
