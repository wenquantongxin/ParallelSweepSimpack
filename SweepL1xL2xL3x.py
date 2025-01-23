# SweepL1xL2xL3x.py
# -*- coding: gbk -*-

import os
import pandas as pd
import numpy as np
import itertools
import math
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # 或者 "TkAgg"
import shutil

from PrepareBatchFiles import (
    read_config_opt_excel,
    ClearBatchTmpFolder,
    prepare_SpckFiles_eachBatch
)
from HalfSearch_CrticalVelocity import (HalfSearch_CrticalVelocity, CurvePerf_idx)


# 并行任务函数
def parallel_worker(args):
    """
    顶层作用域定义的并行任务函数，避免pickle错误。
    
    args 是一个元组，包含：
        (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth)
    我们在函数中解包后执行 HalfSearch_CrticalVelocity。
    返回 (col_idx_in_batch, cVel)，外部可由此了解各列的结果
    """
    (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth) = args
    
    # 实际上的全局列索引
    actual_idx = start_idx + col_idx_in_batch

    # 修改点 0
    
    # 并行任务1：调用半搜索函数，返回临界速度
    cVel = HalfSearch_CrticalVelocity(
        WorkingDir,
        X_vars,
        tag,
        actual_idx,  # 这个列对应 [start_idx, end_idx) 范围内的一列
        StartVel,
        EndVel,
        N_depth
    )
    
    # 并行任务2：调用曲线计算程序，返回曲线磨耗数、横移量
    WearNumber, LatDisp = CurvePerf_idx(WorkingDir, X_vars, tag, actual_idx)
    
    # 返回并行计算该 idx 的结果组向量
    return (col_idx_in_batch, cVel, WearNumber, LatDisp)

# 做三维图，显示 临界速度与 L1、L2 的关系
def ShowMeshgrid():
    CriticalVel = np.load("myCriticalVel.npy") 
    Xvars = np.load("myXvars.npy") 

    Z = CriticalVel  # Z.size = 169
    lx1_all = Xvars[29,:]  # 这 169 个值构成了若干重复/排列
    lx2_all = Xvars[30,:]  # 同理

    # 1) 找到唯一值并排序
    lx1_unique = np.unique(lx1_all)
    lx2_unique = np.unique(lx2_all)
    m = len(lx1_unique)
    n = len(lx2_unique)
    print("lx1_unique:", lx1_unique)
    print("lx2_unique:", lx2_unique)
    print("m,n =", m,n)
    # 这里 m*n 应该 = 169

    # 2) 做 meshgrid
    # 注意 meshgrid 的默认 (x, y) => X.shape = (n,m), Y.shape = (n,m)
    X, Y = np.meshgrid(lx1_unique, lx2_unique)
    print("X.shape = ", X.shape)  # (n,m)
    
    # 3) reshape Z
    Z_mat = Z.reshape(n, m)  # 因为 n 行, m 列 => shape=(len(lx2_unique), len(lx1_unique))
    
    # 4) 画 surf
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z_mat, cmap='viridis')
    ax.set_xlabel("Lx1")
    ax.set_ylabel("Lx2")
    ax.invert_yaxis()  # 使得 Y 轴反向递增
    ax.set_zlabel("Critical Velocity (m/s)")

    fig.colorbar(surf, shrink=0.5)
    plt.show()

# 主函数
def main():
    
    # 修改点 1    
    # 工作目录路径定义
    WorkingDir = r"F:\ResearchMainStream\0.ResearchBySection\C.动力学模型\参数优化\参数优化实现\并行化参数扫略计算SIMPACK模型临界速度"
    # 实验标识符
    tag="Test"  
    
    # 修改点 2
    # 二分法速度上下限、二分深度
    StartVel = 50/3.6 
    EndVel =  800/3.6
    N_depth = 7 
    
    # 1) 调用函数读取 config_opt.xlsx，获取基准值 X_base
    Opt_Config = read_config_opt_excel(WorkingDir)
    X_base = Opt_Config["基准值"].to_list()
    print("基准值 X_base:", X_base)

    # 2) 生成 X_vars (32 x N_combos)
    # 修改点 3
    Lx1_sweep = np.arange(0, 0.6 + 0.001, 0.05)  
    Lx2_sweep = np.arange(0, 0.6 + 0.001, 0.05)  
    Lx3_sweep = np.arange(0, 0.05, 0.05)  # 示例

    Lx123_combinations = list(itertools.product(Lx1_sweep, Lx2_sweep, Lx3_sweep))
    X_vars_columns = []

    for (lx1, lx2, lx3) in Lx123_combinations:
        x_temp = X_base.copy()
        x_temp[29] = lx1
        x_temp[30] = lx2
        x_temp[31] = lx3
        X_vars_columns.append(x_temp)

    X_vars = np.column_stack(X_vars_columns)
    print("X_vars的形状: ", X_vars.shape)
    
    # 保存整个 X_vars 
    # 创建子文件夹 ChkPnt（checkpoint (检查点)）如果不存在
    checkpoint_dir = os.path.join(WorkingDir, "ChkPnt")
     # 如果子文件夹已经存在且不为空，先删除它以及它包含的所有内容
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # 再次新建一个空白的 ChkPnt(checkpoint) 文件夹，便于后续写入
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    np.save(os.path.join(checkpoint_dir, f"myXvars_{tag}.npy"), X_vars)
    
    # 3) 设置批次和并行
    # 修改点 4
    BatchSize_parallel = 10
    total_columns = X_vars.shape[1]
    num_batches = math.ceil(total_columns / BatchSize_parallel)
    print("总的参数组合数：", total_columns)
    print("并行任务数：", BatchSize_parallel)
    print("批次数量：", num_batches)

    # 假设每个组合计算和保存的维度数
    # 修改点 5
    result_dim = 3  
    all_batch_results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BatchSize_parallel
        end_idx   = min((batch_idx + 1) * BatchSize_parallel, total_columns)

        # ============== (a) 清理 & 生成文件 ==============
        ClearBatchTmpFolder(WorkingDir)
        prepare_SpckFiles_eachBatch(WorkingDir, tag, start_idx, end_idx)

        X_vars_batch = X_vars[:, start_idx:end_idx]
        print(f"第 {batch_idx+1} 批：列索引范围 [{start_idx}:{end_idx}), 批量大小 = {X_vars_batch.shape[1]}")

        # 用来存储本批次的结果 (shape=(1, batch_size))
        batch_result = np.zeros((result_dim, X_vars_batch.shape[1]))

        # ============== (b) 并行处理 ==============
        with concurrent.futures.ProcessPoolExecutor(max_workers=BatchSize_parallel) as executor:
            future_list = []
            for col_idx_in_batch in range(X_vars_batch.shape[1]):
                args = (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth)
                future = executor.submit(parallel_worker, args)
                future_list.append(future)

            # 收集结果
            for future in concurrent.futures.as_completed(future_list):
                col_idx_in_batch, cVel, WearNumber, LatDisp = future.result()
                # 修改点 6
                batch_result[0, col_idx_in_batch] = cVel
                batch_result[1, col_idx_in_batch] = WearNumber
                batch_result[2, col_idx_in_batch] = LatDisp
        
        all_batch_results.append(batch_result)
        # 这里 all_batch_results 是一个 list，其中每个元素都是当前批次的结果 (batch_result)

        batch_result_toSave = batch_result # 保存数据解耦，避免保存“列表”时，NumPy 内部会尝试把 all_batch_results 转成一个统一的 ndarray
        np.save(os.path.join(checkpoint_dir, f"batch_result_{tag}_batch{batch_idx}.npy"), batch_result_toSave)

    # 拼接所有批次的结果
    final_results = np.concatenate(all_batch_results, axis=1)
    print("final_results shape:", final_results.shape)
    print("示例：前10列临界速度 = ", final_results[0, :10])
    
    # 最终结果保存
    np.save("myXvars.npy", X_vars)
    np.save("myCriticalVel.npy", final_results)
    
if __name__ == "__main__":
    main()
    # ShowMeshgrid()
    
"""
命令行调用：

    启动 pypack 环境的命令行
    F:  切换盘符
    cd F:\ResearchMainStream\0.ResearchBySection\C.动力学模型\参数优化\参数优化实现\并行化参数扫略计算SIMPACK模型临界速度
    python SweepL1xL2xL3x.py 执行本程序
    
环境安装：
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install  pymoo pandas numpy ipywidgets matplotlib openpyxl
    pip install -U pymoo
    
"""
