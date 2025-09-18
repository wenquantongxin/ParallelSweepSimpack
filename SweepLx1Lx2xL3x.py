# -*- coding: utf-8 -*-

"""
ParallelSweepSimpack · Lx1/Lx2/Lx3 扫略与批量评估入口

本脚本负责：
- 按不同来源（固定 Lx1/Lx2/Lx3 网格、Excel、NPZ）生成待评估的整车参数矩阵 `X_vars`；
- 为每一批组合生成 SIMPACK 的 .spck/.spf/.subvar 文件；
- 并发调用临界速度搜索（HalfSearch）、曲线工况（CRV）与直线工况（STR）的评估函数；
- 将批次结果保存到 `ChkPnt/` 下，最终汇总为 `Result_<tag>.npy` 与 `Xvars_<tag>.npy`。

相较于旧版本散落的“# 修改点”，本版本通过命令行参数统一配置：
- 修改点 0（实验标识 tag）→ `--tag`；
- 修改点 1（参数来源/输入文件）→ `--mode` + `--input` [以及 `--excel-sheet`]；
- 修改点 2（并发批量大小）→ `--batch-size`；
- 修改点 3（结果指标维度）→ 由 `METRIC_NAMES` 自动推导（无需再改）；
- 修改点 4（结果写入映射）→ 已集中于 `execute_batches` 内部，不再手动改写。

使用示例见文件末尾注释块。
"""

import argparse
import concurrent.futures
import itertools
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    # Fallback for headless environments (no GUI backend available)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback

from PrepareBatchFiles import (
    read_config_opt_excel,
    ClearBatchTmpFolder,
    prepare_SpckFiles_eachBatch,
)
from FindCrticalVelocity import HalfSearch_CrticalVelocity
from STRPerf import STRPerf_idx
from CRVPerf import CRVPerf_idx

SUPPORTED_MODES = ("JustSweepL123", "FromExcel", "FromNPZ")
# 结果指标名称（按行顺序写入 `final_results`）
METRIC_NAMES = (
    "critical_velocity",
    "rigid_wear_number",
    "rigid_max_lateral_displacement",
    "irw_wear_number",
    "irw_max_lateral_displacement",
    "sperling_y_aar5",
    "sperling_z_aar5",
)
RESULT_DIM = len(METRIC_NAMES)


@dataclass
class SimulationConfig:
    working_dir: Path
    tag: str
    mode: str
    input_path: Optional[Path]
    excel_sheet: str
    start_velocity: float
    end_velocity: float
    search_depth: int
    batch_size: int
    checkpoint_name: str

    @property
    def checkpoint_dir(self) -> Path:
        return self.working_dir / self.checkpoint_name


# 解析命令行参数 —— 见下方参数说明
def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Run SIMPACK sweeps with configurable parameter sources."
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="JustSweepL123",
        help="Parameter source mode."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file used when --mode is FromExcel or FromNPZ."
    )
    parser.add_argument(
        "--excel-sheet",
        default="Sheet1",
        help="Sheet name to read when --mode is FromExcel."
    )
    parser.add_argument(
        "--tag",
        default="reNS23",
        help="Identifier appended to generated batches and result files."
    )
    parser.add_argument(
        "--start-vel",
        type=float,
        default=100.0 / 3.6,
        help="Binary search start velocity in m/s (default: 100 km/h)."
    )
    parser.add_argument(
        "--end-vel",
        type=float,
        default=900.0 / 3.6,
        help="Binary search end velocity in m/s (default: 900 km/h)."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="Binary search depth for HalfSearch_CrticalVelocity."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=22,
        help="Number of worker processes to spawn per batch."
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory containing SIMPACK templates."
    )
    parser.add_argument(
        "--checkpoint",
        default="ChkPnt",
        help="Folder name (relative to working dir) used for checkpoint outputs."
    )

    args = parser.parse_args()
    working_dir = args.working_dir.resolve()

    input_path: Optional[Path] = None
    if args.input is not None:
        input_path = args.input
        if not input_path.is_absolute():
            input_path = (working_dir / input_path).resolve()

    if args.mode in {"FromExcel", "FromNPZ"} and input_path is None:
        parser.error("--input is required when --mode is FromExcel or FromNPZ.")
    if args.mode == "FromExcel" and input_path is not None and input_path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        parser.error("--input must be an Excel file when --mode is FromExcel.")
    if args.mode == "FromNPZ" and input_path is not None and input_path.suffix.lower() != ".npz":
        parser.error("--input must be an NPZ file when --mode is FromNPZ.")

    return SimulationConfig(
        working_dir=working_dir,
        tag=args.tag,
        mode=args.mode,
        input_path=input_path,
        excel_sheet=args.excel_sheet,
        start_velocity=args.start_vel,
        end_velocity=args.end_vel,
        search_depth=args.depth,
        batch_size=args.batch_size,
        checkpoint_name=args.checkpoint,
    )


# 生成 Lx1/Lx2/Lx3 的网格扫略矩阵；其他变量取基准值
# 返回 X_vars: shape=(N_params, N_cases)，按列为一个设计点
def generate_lx123_sweep(working_dir: Path) -> np.ndarray:
    print("[INFO] Sweeping Lx1/Lx2/Lx3 while keeping other variables at baseline.")
    opt_config = read_config_opt_excel(str(working_dir))
    x_base = opt_config["基准值"].to_list()

    lx1 = np.arange(0.0, 0.64 + 0.001, 0.04)
    lx2 = np.arange(0.0, 0.60 + 0.001, 0.04)
    lx3 = np.arange(-0.6, 0.40 + 0.001, 0.1)

    columns = []
    for lx1_val, lx2_val, lx3_val in itertools.product(lx1, lx2, lx3):
        x_temp = x_base.copy()
        x_temp[29] = lx1_val
        x_temp[30] = lx2_val
        x_temp[31] = lx3_val
        columns.append(x_temp)

    x_vars = np.column_stack(columns)
    print(f"[INFO] Generated sweep grid with shape {x_vars.shape}.")
    return x_vars


# 从 Excel 读取参数矩阵：按“是否优化”为 1 的行依次回填（从第2列起）
# 特殊耦合：第3行复制至第4行，第7行复制至第8行
def load_parameters_from_excel(config: SimulationConfig) -> np.ndarray:
    if config.input_path is None:
        raise ValueError("Excel mode requires --input to be set.")

    print(f"[INFO] Loading vehicle parameters from Excel: {config.input_path}")
    opt_config = read_config_opt_excel(str(config.working_dir), excel_name="config_opt.xlsx")
    x_base = opt_config["基准值"].to_list()
    is_to_opt = opt_config["是否优化"].to_list()

    param_sweep_df = pd.read_excel(config.input_path, sheet_name=config.excel_sheet, header=None)
    changing_vars = param_sweep_df.iloc[:, 1:]
    n_cases = changing_vars.shape[1]
    n_total = len(is_to_opt)
    n_opt = int(np.sum(is_to_opt))

    # 行数一致性检查，避免静默错配
    if changing_vars.shape[0] != n_opt:
        need, got = n_opt, changing_vars.shape[0]
        print(
            f"[WARN] Excel 行数({got}) 与 is_to_opt==1 计数({need}) 不一致；"
            f"将仅使用前 {min(got, need)} 行进行匹配。"
        )
        # 打印前后两行索引帮助对齐（若行数足够）
        head_idx = list(range(min(2, need)))
        tail_idx = list(range(max(0, need - 2), need))
        print(f"[HINT] 期望匹配的优化变量行索引样例（前/后）：{head_idx} ... {tail_idx}")
        changing_vars = changing_vars.iloc[:need, :]

    x_vars = np.zeros((n_total, n_cases))
    opt_row_idx = 0
    for line_idx, flag in enumerate(is_to_opt):
        if flag == 0:
            x_vars[line_idx] = float(x_base[line_idx])
        else:
            if opt_row_idx >= changing_vars.shape[0]:
                raise ValueError(
                    "Excel does not contain enough rows for optimized variables "
                    f"(needed {n_opt}, got {changing_vars.shape[0]})."
                )
            x_vars[line_idx] = changing_vars.iloc[opt_row_idx, :].to_numpy(dtype=float)
            opt_row_idx += 1

    x_vars[3] = x_vars[2]
    x_vars[7] = x_vars[6]
    print(f"[INFO] Loaded parameter grid with shape {x_vars.shape}.")
    return x_vars


# 从 NPZ 读取 final_X 或 X，并按“是否优化”为 1 的行依次回填
# 特殊耦合：第3行复制至第4行，第7行复制至第8行
def load_parameters_from_npz(config: SimulationConfig) -> np.ndarray:
    if config.input_path is None:
        raise ValueError("NPZ mode requires --input to be set.")

    print(f"[INFO] Loading vehicle parameters from NPZ: {config.input_path}")
    with np.load(config.input_path, allow_pickle=True) as data:
        if "final_X" in data:
            final_x = data["final_X"].T
            print("[INFO] Found 'final_X' in NPZ file.")
        elif "X" in data:
            final_x = data["X"].T
            print("[INFO] Falling back to 'X' in NPZ file.")
        else:
            keys = list(data.keys())
            raise KeyError(f"NPZ must contain 'final_X' or 'X'. Found keys: {keys}")

    opt_config = read_config_opt_excel(str(config.working_dir), excel_name="config_opt.xlsx")
    x_base = opt_config["基准值"].to_list()
    is_to_opt = opt_config["是否优化"].to_list()

    n_cases = final_x.shape[1]
    n_total = len(is_to_opt)

    x_vars = np.zeros((n_total, n_cases))
    opt_row_idx = 0
    for line_idx, flag in enumerate(is_to_opt):
        if flag == 0:
            x_vars[line_idx] = x_base[line_idx]
        else:
            if opt_row_idx >= final_x.shape[0]:
                raise ValueError(
                    "NPZ does not contain enough rows for optimized variables "
                    f"(needed {int(np.sum(is_to_opt))}, got {final_x.shape[0]} with shape {final_x.shape})."
                )
            x_vars[line_idx] = final_x[opt_row_idx, :]
            opt_row_idx += 1

    x_vars[3] = x_vars[2]
    x_vars[7] = x_vars[6]
    print(f"[INFO] Loaded parameter grid with shape {x_vars.shape}.")
    return x_vars


# 根据 --mode 选择参数来源（Lx123/Excel/NPZ）
def generate_vehicle_parameters(config: SimulationConfig) -> np.ndarray:
    if config.mode == "JustSweepL123":
        return generate_lx123_sweep(config.working_dir)
    if config.mode == "FromExcel":
        return load_parameters_from_excel(config)
    if config.mode == "FromNPZ":
        return load_parameters_from_npz(config)
    raise ValueError(f"Unsupported mode: {config.mode}")


# 并行子进程：评估单个设计点，返回 (批内列索引, 指标..)，指标顺序见 METRIC_NAMES
def parallel_worker(args):
    """Worker executed in a separate process to evaluate one design point."""
    (
        col_idx_in_batch,
        start_idx,
        working_dir,
        x_vars,
        tag,
        start_vel,
        end_vel,
        depth,
    ) = args

    actual_idx = start_idx + col_idx_in_batch

    critical_velocity = HalfSearch_CrticalVelocity(
        working_dir,
        x_vars,
        tag,
        actual_idx,
        start_vel,
        end_vel,
        depth,
    )
    time.sleep(1)

    rigid_wear, rigid_lat_disp, irw_wear, irw_lat_disp = CRVPerf_idx(
        working_dir,
        x_vars,
        tag,
        actual_idx,
    )
    time.sleep(1)

    sperling_y, sperling_z = STRPerf_idx(
        working_dir,
        x_vars,
        tag,
        actual_idx,
    )
    time.sleep(1)

    return (
        col_idx_in_batch,
        critical_velocity,
        rigid_wear,
        rigid_lat_disp,
        irw_wear,
        irw_lat_disp,
        sperling_y,
        sperling_z,
    )

# ========== 内存映射版本（减少进程间拷贝 + 容错） ==========
_XVARS_MM = None  # type: Optional[np.ndarray]

def _init_xvars_memmap(xvars_path_str: str):
    """ProcessPool initializer: 在子进程中打开只读内存映射的 X_vars。"""
    global _XVARS_MM
    _XVARS_MM = np.load(xvars_path_str, mmap_mode="r")

def parallel_worker_mm(args):
    """子进程评估单点（memmap 版本，避免传递大对象），包含容错返回。"""
    (
        col_idx_in_batch,
        start_idx,
        working_dir,
        tag,
        start_vel,
        end_vel,
        depth,
    ) = args

    global _XVARS_MM
    actual_idx = start_idx + col_idx_in_batch

    try:
        critical_velocity = HalfSearch_CrticalVelocity(
            working_dir, _XVARS_MM, tag, actual_idx, start_vel, end_vel, depth
        )
        time.sleep(1)

        rigid_wear, rigid_lat_disp, irw_wear, irw_lat_disp = CRVPerf_idx(
            working_dir, _XVARS_MM, tag, actual_idx
        )
        time.sleep(1)

        sperling_y, sperling_z = STRPerf_idx(
            working_dir, _XVARS_MM, tag, actual_idx
        )
        time.sleep(1)

        return (
            col_idx_in_batch,
            critical_velocity,
            rigid_wear,
            rigid_lat_disp,
            irw_wear,
            irw_lat_disp,
            sperling_y,
            sperling_z,
            "",  # err
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return (
            col_idx_in_batch,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            err,
        )


# 创建或清空检查点目录，并返回路径
def prepare_checkpoint_dir(config: SimulationConfig) -> Path:
    checkpoint_dir = config.checkpoint_dir
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


# 按批次并发执行评估；每批保存到 ChkPnt/batch_result_<tag>_batch<i>.npy
# 返回 final_results: shape=(RESULT_DIM, N_cases)
def execute_batches(
    config: SimulationConfig,
    x_vars: np.ndarray,
    checkpoint_dir: Path,
) -> np.ndarray:
    total_columns = x_vars.shape[1]
    if total_columns == 0:
        raise ValueError("No parameter combinations available for evaluation.")

    batch_size = max(1, config.batch_size)
    num_batches = math.ceil(total_columns / batch_size)

    print(f"[INFO] Total parameter combinations: {total_columns}")
    print(f"[INFO] Parallel workers per batch: {batch_size}")
    print(f"[INFO] Number of batches: {num_batches}")

    all_batch_results = []
    working_dir_str = str(config.working_dir)
    # 使用已保存到 checkpoint 的 X_vars 作为 memmap 源
    xvars_mm_path = str(checkpoint_dir / f"myXvars_{config.tag}.npy")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_columns)

        ClearBatchTmpFolder(working_dir_str)
        prepare_SpckFiles_eachBatch(working_dir_str, config.tag, start_idx, end_idx)

        batch_columns = end_idx - start_idx
        print(
            f"[INFO] Batch {batch_idx + 1}/{num_batches}: column range [{start_idx}:{end_idx}), size={batch_columns}"
        )

        batch_result = np.zeros((RESULT_DIM, batch_columns))
        worker_count = min(batch_size, batch_columns)
        if worker_count == 0:
            continue

        t0 = time.time()
        failed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_xvars_memmap,
            initargs=(xvars_mm_path,)
        ) as executor:
            futures = []
            for col_idx_in_batch in range(batch_columns):
                args = (
                    col_idx_in_batch,
                    start_idx,
                    working_dir_str,
                    config.tag,
                    config.start_velocity,
                    config.end_velocity,
                    config.search_depth,
                )
                futures.append(executor.submit(parallel_worker_mm, args))

            for future in concurrent.futures.as_completed(futures):
                (
                    col_idx,
                    critical_velocity,
                    rigid_wear,
                    rigid_lat_disp,
                    irw_wear,
                    irw_lat_disp,
                    sperling_y,
                    sperling_z,
                    err,
                ) = future.result()

                if err:
                    print(f"[WARN] idx={start_idx + col_idx} failed.\n{err}")
                    failed += 1

                metrics = (
                    critical_velocity,
                    rigid_wear,
                    rigid_lat_disp,
                    irw_wear,
                    irw_lat_disp,
                    sperling_y,
                    sperling_z,
                )
                for metric_idx, value in enumerate(metrics):
                    batch_result[metric_idx, col_idx] = value

        all_batch_results.append(batch_result)
        batch_file = checkpoint_dir / f"batch_result_{config.tag}_batch{batch_idx}.npy"
        np.save(batch_file, batch_result)
        print(f"[INFO] Batch {batch_idx + 1} finished in {time.time()-t0:.2f}s, failed={failed}/{batch_columns}")

    final_results = np.concatenate(all_batch_results, axis=1)
    return final_results


# 基于保存的 Result/Xvars 绘制 Lx1-Lx2-临界速度曲面图（需要 GUI 后端）
def show_meshgrid_from_result(
    tag: str,
    working_dir: Optional[Path] = None,
    lx3: Optional[float] = None,  # 指定 Lx3 切片；None 表示对 Lx3 聚合
    agg: str = "max",             # 当 lx3=None 时，对 (Lx1,Lx2) 维度的聚合方式：max/min/mean/median
    atol: float = 1e-9,            # Lx3 匹配容差
) -> None:
    base_dir = working_dir or Path.cwd()
    result_path = base_dir / f"Result_{tag}.npy"
    xvars_path = base_dir / f"Xvars_{tag}.npy"

    R = np.load(result_path)  # shape = (RESULT_DIM, N_cases)
    X = np.load(xvars_path)   # shape = (N_params, N_cases)
    cv = R[0, :]

    lx1_all = X[29, :]
    lx2_all = X[30, :]
    lx3_all = X[31, :]

    # 选择切片或对 Lx3 聚合
    if lx3 is not None:
        mask = np.isclose(lx3_all, lx3, atol=atol)
        if not np.any(mask):
            raise ValueError(f"No samples found for Lx3 ≈ {lx3} (atol={atol}).")
        lx1_vals = lx1_all[mask]
        lx2_vals = lx2_all[mask]
        z_vals = cv[mask]
        title_suffix = f"Lx3={lx3}"
    else:
        df = pd.DataFrame({"lx1": lx1_all, "lx2": lx2_all, "cv": cv})
        if agg == "max":
            grouped = df.groupby(["lx1", "lx2"])['cv'].max()
        elif agg == "min":
            grouped = df.groupby(["lx1", "lx2"])['cv'].min()
        elif agg == "mean":
            grouped = df.groupby(["lx1", "lx2"])['cv'].mean()
        elif agg == "median":
            grouped = df.groupby(["lx1", "lx2"])['cv'].median()
        else:
            raise ValueError("Unsupported agg. Choose from max/min/mean/median.")
        pairs = np.array(list(grouped.index))
        lx1_vals, lx2_vals = pairs[:, 0], pairs[:, 1]
        z_vals = grouped.values
        title_suffix = f"agg={agg}"

    # 生成网格，并以 NaN 填充缺失项
    lx1_unique = np.unique(lx1_vals)
    lx2_unique = np.unique(lx2_vals)
    Xg, Yg = np.meshgrid(lx1_unique, lx2_unique)
    Z = np.full_like(Xg, np.nan, dtype=float)
    idx1 = {v: j for j, v in enumerate(lx1_unique)}
    idx2 = {v: i for i, v in enumerate(lx2_unique)}
    for x1, x2, z in zip(lx1_vals, lx2_vals, z_vals):
        Z[idx2[x2], idx1[x1]] = z

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(Xg, Yg, Z, cmap="viridis")
    ax.set_xlabel("Lx1")
    ax.set_ylabel("Lx2")
    ax.invert_yaxis()
    ax.set_zlabel("Critical Velocity (m/s)")
    ax.set_title(f"Critical Velocity vs Lx1/Lx2 ({title_suffix})")
    fig.colorbar(surf, shrink=0.5)
    plt.show()


# 旧接口兼容封装
def ShowMeshgrid(
    tag: str = "reNS23",
    working_dir: Optional[Path] = None,
    lx3: Optional[float] = None,
    agg: str = "max",
    atol: float = 1e-9,
) -> None:
    """向后兼容封装：允许直接指定 Lx3 切片或聚合方式。"""
    show_meshgrid_from_result(tag, working_dir, lx3=lx3, agg=agg, atol=atol)

# 主流程入口
def main() -> None:
    config = parse_args()
    start_time = time.time()

    print(f"[INFO] Working directory: {config.working_dir}")
    print(f"[INFO] Sweep mode: {config.mode}")
    if config.input_path:
        print(f"[INFO] Input file: {config.input_path}")

    x_vars = generate_vehicle_parameters(config)

    checkpoint_dir = prepare_checkpoint_dir(config)
    np.save(checkpoint_dir / f"myXvars_{config.tag}.npy", x_vars)

    final_results = execute_batches(config, x_vars, checkpoint_dir)

    np.save(config.working_dir / f"Xvars_{config.tag}.npy", x_vars)
    np.save(config.working_dir / f"Result_{config.tag}.npy", final_results)

    if final_results.shape[1] > 0:
        preview = final_results[0, : min(10, final_results.shape[1])]
        print(f"[INFO] First {preview.size} critical velocities: {preview}")

    elapsed = time.time() - start_time
    print(f"[INFO] Completed sweep in {elapsed:.2f} s.")
    print(f"[INFO] Results saved to {config.working_dir / f'Result_{config.tag}.npy'}")


if __name__ == "__main__":
    main()

'''

 命令行调用示例（Windows）

# cmd.exe 多行（使用 ^ 续行）:
    F:  # 切换盘符                                                                                                             
    cd F:\ResearchMainStream\0.ResearchBySection\C.动力学模型\C23参数优化\参数优化实现\ParallelSweepSimpack

    python -X utf8 SweepLx1Lx2xL3x.py ^
       --mode FromExcel ^
       --input IRWnRW_OrthogonalDoE.xlsx ^
       --excel-sheet Sheet1 ^
       --tag ExcelRun ^
       --batch-size 5

# 从其他数据（内置Lx组合/Excel表格/npz数据包）来源读取并扫略计算:
   - 仅扫略 Lx1/Lx2/Lx3:
     python -X utf8 SweepLx1Lx2xL3x.py --mode JustSweepL123 --tag L123Grid --batch-size 22
   - 从 NPZ 导入:
     python -X utf8 SweepLx1Lx2xL3x.py --mode FromNPZ --input res_history.npz --tag NPZRun --batch-size 16
   - 自定义临界速度搜索区间与深度:
     python -X utf8 SweepLx1Lx2xL3x.py --mode FromExcel --input ParameterSweep_fromExcel.xlsx --tag TestVel --start-vel 27.78 --end-vel 250 --depth 7 --checkpoint ChkPnt

'''

