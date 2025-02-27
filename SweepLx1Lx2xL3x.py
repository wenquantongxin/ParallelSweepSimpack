# -*- coding: utf-8 -*-

# SweepLx1Lx2xL3x.py
import os
import pandas as pd
import numpy as np
import itertools
import math
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import shutil
import time

from PrepareBatchFiles import (
    read_config_opt_excel,
    ClearBatchTmpFolder,
    prepare_SpckFiles_eachBatch
)

from FindCrticalVelocity import (HalfSearch_CrticalVelocity)
from STRPerf import (STRPerf_idx)
from CRVPerf import (CRVPerf_idx)

# 做三维图，显示 临界速度与 L1、L2 的关系
def ShowMeshgrid():
    CriticalVel = np.load("myCriticalVel.npy") 
    Xvars = np.load("myXvars.npy") 
    Z = CriticalVel
    lx1_all = Xvars[29,:] 
    lx2_all = Xvars[30,:] 

    # 1) 找到唯一值并排序
    lx1_unique = np.unique(lx1_all)
    lx2_unique = np.unique(lx2_all)
    m = len(lx1_unique)
    n = len(lx2_unique)
    print("lx1_unique:", lx1_unique)
    print("lx2_unique:", lx2_unique)
    print("m,n =", m,n)
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

    # 并行任务组
    # 并行任务 1：调用半搜索函数，返回临界速度
    CrticalVelocity = HalfSearch_CrticalVelocity(WorkingDir, X_vars, tag, actual_idx, StartVel, EndVel, N_depth)
    time.sleep(1)
    # print("[INFO] 测试，临时跳过临界速度计算")  
    # CrticalVelocity = 666.66
    
    # 并行任务 2：调用曲线计算模型，返回曲线磨耗数、横移量
    SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = CRVPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # 并行任务 3：调用典型 AAR5 直线计算模型 性能评估，返回Sperling指标
    SperlingY_AAR5, SperlingZ_AAR5 = STRPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # 返回并行计算该 idx 的结果组向量
    return (col_idx_in_batch, CrticalVelocity, SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV, SperlingY_AAR5, SperlingZ_AAR5)

# 生成计算所需的完整车辆参数组合 X_vars
def GenerateVehicleParamater(WorkingDir, Filename, method="JustSweepL123"):
    
    if method == "JustSweepL123":
         
        # 示例调用: X_vars = GenerateVehicleParamater(WorkingDir, Filename="config_opt.xlsx", method="JustSweepL123")
        
        print(f"[INFO] 仅需扫略 L1、L2、L3 三个参数, 其他车辆参数保持与基准值相同") # 扫略 L1、L2、L3
        
        # 1) 调用函数读取 config_opt.xlsx，获取基准值 X_base
        Opt_Config = read_config_opt_excel(WorkingDir)
        X_base = Opt_Config["基准值"].to_list()
        # print("基准值 X_base:", X_base)

        # 2) 生成 X_vars (32 x N_combos)
        Lx1_sweep = np.arange(0, 0.64 + 0.001, 0.04)    # 从0到0.64，共17个元素
        Lx2_sweep = np.arange(0, 0.60 + 0.001, 0.04)    # 从0到0.60，共16个元素
        Lx3_sweep = np.arange(0, 0.001, 0.1)            # Lx3_sweep = np.arange(-0.6, 0.40 + 0.001, 0.1)

        Lx123_combinations = list(itertools.product(Lx1_sweep, Lx2_sweep, Lx3_sweep))
        X_vars_columns = []

        for (lx1, lx2, lx3) in Lx123_combinations:
            x_temp = X_base.copy()
            x_temp[29] = lx1 # 从 0 开始编号
            x_temp[30] = lx2
            x_temp[31] = lx3
            X_vars_columns.append(x_temp)

        X_vars = np.column_stack(X_vars_columns)
        print("用于扫略计算的 X_vars的形状: ", X_vars.shape)
        
    elif method == "FromExcel":
        
        # 示例调用: X_vars = GenerateVehicleParamater(WorkingDir, Filename="ParameterSweep_fromExcel.xlsx", method="FromExcel")
        
        print(f"[INFO] 从 EXCEL 表格中获取车辆参数, 该表格记录了前沿解对应的被优化参数")
        print("元文件名称:", Filename)
        # 读取 config_opt.xlsx 获取基准值 X_base 和是否优化的标志 is2opt
        Opt_Config = read_config_opt_excel(WorkingDir, excel_name="config_opt.xlsx")
        X_base = Opt_Config["基准值"].to_list()
        is2opt = Opt_Config["是否优化"].to_list()

        # 读取外部文件 ParameterSweep_fromExcel.xlsx
        Param_Sweep_Config = pd.read_excel(f"{WorkingDir}/{Filename}", sheet_name="Sheet1", header=None)
        N_Xvars = Param_Sweep_Config.shape[1] - 1 # 从Excel 表格中读取到的有多少组 X_vars 需要被计算
        ChangingVars = Param_Sweep_Config.iloc[..., 1: Param_Sweep_Config.shape[1]]

        N_2opt = len(is2opt) # X_vars 有多少维度, 应该是32
                
        # 根据外部excel行数初始化 X_vars
        X_vars = np.zeros((N_2opt, N_Xvars)) 

        optCount = 0
        for LineId in range(0, N_2opt):
            if is2opt[LineId] == 0:
                X_vars[LineId] = X_base[LineId] * np.ones((1, N_Xvars))
            elif is2opt[LineId] == 1:
                X_vars[LineId] = ChangingVars.iloc[optCount, ].to_numpy()
                optCount = optCount + 1 
        # 特别关心  $_Kpy	一系悬挂刚度-横向，与3耦合； $_Ksy	二系悬挂刚度-横向，与7耦合
        X_vars[3]  = X_vars[2]
        X_vars[7]  = X_vars[6]

        print("用于扫略计算的 X_vars的形状: ", X_vars.shape)  # 打印 X_vars 的形状
        
    elif method == "FromNPZ":
                
        # 示例调用 1: X_vars = GenerateVehicleParamater(WorkingDir, Filename="res_history.npz", method="FromNPZ")
        # 示例调用 2: X_vars = GenerateVehicleParamater(WorkingDir, Filename="generation_150_nondom.npz", method="FromNPZ")
        
        # print(f"[INFO] 从 res_history.npz 或者 generation_i_nondom.npz 中获取车辆参数, 该 .npz 文件记录了前沿解的 final_X")
        data = np.load(Filename, allow_pickle=True)
        
        try:
            # 尝试加载final_X
            final_X = data["final_X"].T
            print(f"[INFO] 从 res_history.npz 中获取车辆参数, 该 .npz 文件记录了最终前沿解的 final_X")
            print("元文件名称:", Filename)
        except KeyError:
            # 如果没有final_X，尝试加载X
            final_X = data["X"].T
            print(f"[INFO] 从 generation_i_nondom.npz 中获取车辆参数, 该 .npz 文件记录了某一代际前沿解的 final_X")
            print("元文件名称:", Filename)

        Opt_Config = read_config_opt_excel(WorkingDir, excel_name="config_opt.xlsx")
        X_base = Opt_Config["基准值"].to_list()
        is2opt = Opt_Config["是否优化"].to_list()

        N_Xvars = len(final_X[0])
        N_2opt = len(is2opt) # X_vars 有多少维度, 应该是32
                        
        # 初始化 X_vars
        X_vars = np.zeros((N_2opt, N_Xvars)) 

        optCount = 0
        for LineId in range(0, N_2opt):
            if is2opt[LineId] == 0:
                    X_vars[LineId] = X_base[LineId] * np.ones((1, N_Xvars))
            elif is2opt[LineId] == 1:
                    X_vars[LineId] = final_X[optCount, ]
                    optCount = optCount + 1 
            # 特别关心  $_Kpy	一系悬挂刚度-横向，与3耦合； $_Ksy	二系悬挂刚度-横向，与7耦合
            X_vars[3]  = X_vars[2]
            X_vars[7]  = X_vars[6]

        print("用于扫略计算的 X_vars的形状: ", X_vars.shape)  # 打印 X_vars 的形状

    return X_vars

# 主函数
def main():
    
    start_time = time.time()  # 获取当前时间戳(秒)
    
    # 工作目录路径定义
    WorkingDir = os.getcwd()
    # 修改点 0
    # 实验标识符
    tag="Sweep"  
    
    # 二分法速度上下限、二分深度
    StartVel = 100/3.6 
    EndVel =  900/3.6
    N_depth = 7 
    
    # 调用 X_vars 生成函数, 以进行 正向参数扫略 / 前沿解回顾验证

    # 修改点 1
    X_vars = GenerateVehicleParamater(WorkingDir, Filename="config_opt.xlsx", method="JustSweepL123")
    
    # X_vars = GenerateVehicleParamater(WorkingDir, Filename="Overall_FinalSolutions.npz", method="FromNPZ")
    # 示例调用 1: X_vars = GenerateVehicleParamater(WorkingDir, Filename="res_history.npz", method="FromNPZ")

    # 创建子文件夹 ChkPnt（checkpoint (检查点)）如果不存在
    checkpoint_dir = os.path.join(WorkingDir, "ChkPnt")
     # 如果子文件夹已经存在且不为空，先删除它以及它包含的所有内容
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # 再次新建一个空白的 ChkPnt(checkpoint) 文件夹，便于后续写入
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 保存 X_vars 
    np.save(os.path.join(checkpoint_dir, f"myXvars_{tag}.npy"), X_vars)
    
    # 3) 设置批次和并行
    # 修改点 2
    BatchSize_parallel = 10
    total_columns = X_vars.shape[1]
    num_batches = math.ceil(total_columns / BatchSize_parallel)
    print("总的参数组合数：", total_columns)
    print("并行任务数：", BatchSize_parallel)
    print("批次数量：", num_batches)

    # 假设每个组合计算和保存的维度数
    # 修改点 3
    result_dim = 7  
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
                col_idx_in_batch, cVel, RigidWN_CRV, RigidLatMax_CRV, IrwWN_CRV, IrwLatMax_CRV, SperlingY_AAR5, SperlingZ_AAR5 = future.result()
                # 修改点 4
                # 并行池 return (col_idx_in_batch, cVel, WearNumber_CRV, LatDispMax_CRV, SperlingY_AAR5, SperlingZ_AAR5)
                batch_result[0, col_idx_in_batch] = cVel 
                batch_result[1, col_idx_in_batch] = RigidWN_CRV
                batch_result[2, col_idx_in_batch] = RigidLatMax_CRV
                batch_result[3, col_idx_in_batch] = IrwWN_CRV
                batch_result[4, col_idx_in_batch] = IrwLatMax_CRV
                batch_result[5, col_idx_in_batch] = SperlingY_AAR5
                batch_result[6, col_idx_in_batch] = SperlingZ_AAR5
        
        all_batch_results.append(batch_result)
        # 这里 all_batch_results 是一个 list，其中每个元素都是当前批次的结果 (batch_result)

        batch_result_toSave = batch_result # 保存数据解耦，避免保存“列表”时，NumPy 内部会尝试把 all_batch_results 转成一个统一的 ndarray
        np.save(os.path.join(checkpoint_dir, f"batch_result_{tag}_batch{batch_idx}.npy"), batch_result_toSave)

    # 拼接所有批次的结果
    final_results = np.concatenate(all_batch_results, axis=1)
    print("final_results shape:", final_results.shape)
    print("计算完成，前10列临界速度 = ", final_results[0, :10])
    
    # 代码计时
    elapsed = time.time() - start_time
    print(f"该代码块耗时: {elapsed:.6f} 秒")
    
    # 最终结果保存
    np.save(f"Xvars_{tag}.npy", X_vars)
    np.save(f"Result_{tag}.npy", final_results)
    
if __name__ == "__main__":
    main()
    # ShowMeshgrid()
    
"""
命令行调用：

    # 需要启动 pypack 环境的命令行
    
    F:  # 切换盘符                                                                                                             
    cd F:\ResearchMainStream\0.ResearchBySection\C.动力学模型\参数优化\参数优化实现\ParallelSweepSimpack                           
    python -X utf8 SweepLx1Lx2xL3x.py # 执行本程序                                                                                                                                   

"""
