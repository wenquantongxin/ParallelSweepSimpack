# -*- coding: utf-8 -*-

# 曲线性能评估

import os
import time
import numpy as np
import pandas as pd
import subprocess
from FindCrticalVelocity import (Import_Subvars_To_File_idx, run_simpack_cmd)

# 读取指定 .dat 数据
# 返回从数据文件中获得的磨耗数与最大横移量
def ReadCRVDat(
    dat_path,
    line_sumwear=6,                 # 从 1 计数的行数
    lines_lateral=(46, 51, 56, 61)  # 从 1 计数的行数
):
    """
    读取给定 dat 文件，返回磨耗总数和最大横向位移。
    
    参数:
    ----
    dat_path: str
        .dat 文件路径
    line_sumwear: int
        .dat 文件中存放总磨耗数据的行号（从1开始计数）
    lines_lateral: tuple or list
        .dat 文件中存放横向位移数据的行号（从1开始计数），可传入多个

    返回:
    ----
    (SumWearNumber, maxLatDisp) : (float, float)
        - SumWearNumber: float，总磨耗数
        - maxLatDisp: float，横向位移中的最大值
    """
    with open(dat_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 确保行号不超过文件总行数
    total_lines = len(lines)
    if line_sumwear > total_lines:
        raise ValueError(f"line_sumwear = {line_sumwear} 超过文件总行数 {total_lines}。")
    for idx in lines_lateral:
        if idx > total_lines:
            raise ValueError(f"横向位移行号 {idx} 超过文件总行数 {total_lines}。")

    # 解析总磨耗(第 line_sumwear 行)
    # 因为 Python 列表是从 0 开始索引，所以对应行号 line_sumwear => lines[line_sumwear - 1]
    line_sum = lines[line_sumwear - 1]
    parts_sum = line_sum.split(';')
    SumWearNumber_CRV_fromDat = float(parts_sum[1].strip())

    # 依次读取指定的横向位移行
    LatDisp_array = []
    for idx in lines_lateral:
        line_disp = lines[idx - 1]
        parts_disp = line_disp.split(';')
        LatDisp_array.append(float(parts_disp[1].strip()))

    maxLatDisp_CRV_fromDat = max(LatDisp_array)
    
    return SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat

# 单线程内计算曲线通过的磨耗数与最大横移量
def CRVCal_idx(
    work_dir,
    filemidname, # 用于区分刚性轮对或者独立轮对，filemidname 可能为 IRWCRV300m 或者 NativeRigidCRV300m
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # 对应于 Result_IRWCRV300m_Opt_0125_0.spf 或者 Result_NativeRigidCRV300m_Opt_0125_0.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # 输出的 .dat 文件的名称
   
    # 拼出 SPF 文件的绝对路径
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # 脚本位置
    qs_script_path = os.path.join(work_dir, qs_script)

    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return (-99.0111, -99.0112) 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return (-99.0121, -99.0122)

    # 1) 调用 simpack-post 的脚本 .qs
    # BatchTmp 子文件夹内，以命令行执行: simpack-post -s SbrExport_SPCKResult.qs Result_NativeRigidCRV300m_Opt_0125_0.spf DatResult_NativeRigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    # 调用函数执行
    result = run_simpack_cmd(cmd, work_dir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (-99.0131, -99.0132)
    else:
        print(f"成功执行 qs 脚本调用")
        print("命令执行完成")
    
    time.sleep(2)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] 后处理结果文件未找到: {dat_path}")
        return (-99.0151, -99.0152) 
      
    # 3) 解析文件
    try:
        # 根据 filemidname 区分读取 .dat 的方式
        if filemidname == "IRWCRV300m":
            SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat = ReadCRVDat(dat_path) # 行号默认是第6行保存磨耗值，第46、51、56、61行保存横向位移
        elif filemidname == "NativeRigidCRV300m":
            SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat = ReadCRVDat(dat_path, line_sumwear=6, lines_lateral=(26, 31, 36, 41))
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        SumWearNumber_CRV = -99.025
        maxLatDisp_CRV = -99.026
    else:
        SumWearNumber_CRV = SumWearNumber_CRV_fromDat
        maxLatDisp_CRV = maxLatDisp_CRV_fromDat
        
    # 返回上部函数 CRVPerf_idx
    # 注意检查 CRVCal_idx 各个故障返回码的维度，应与 CRVCal_idx 函数的 return 相同
    return SumWearNumber_CRV, maxLatDisp_CRV

def CRVPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"对于模型 {idx} 进行曲线线路通过性能测试")
    
    # =========== 1. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = 60/3.6      # 曲线评估时，采用 60 km/h 速度通过 R300 曲线，使用 TargetVel 覆盖该速度取值
    
    sprCpz         = X_vars_col[1]
    Kpx            = X_vars_col[2]
    Kpy            = X_vars_col[3]
    Kpz            = X_vars_col[4]
    Cpz            = X_vars_col[5]
    Ksx            = X_vars_col[6]
    Ksy            = X_vars_col[7]
    Ksz            = X_vars_col[8]
    Csz            = X_vars_col[9]
    Kld            = X_vars_col[10]
    Cld            = X_vars_col[11]
    Kaar           = X_vars_col[12]
    Kstr           = X_vars_col[13]
    Chx            = X_vars_col[14]
    Mc             = X_vars_col[15]
    hc             = X_vars_col[16]
    Icx            = X_vars_col[17]
    Icy            = X_vars_col[18]
    Icz            = X_vars_col[19]
    Mt             = X_vars_col[20]
    ht             = X_vars_col[21]
    Itx            = X_vars_col[22]
    Ity            = X_vars_col[23]
    Itz            = X_vars_col[24]
    Mw             = X_vars_col[25]
    Iwx            = X_vars_col[26]
    Iwy            = X_vars_col[27]
    Iwz            = X_vars_col[28]
    Lx1            = X_vars_col[29]
    Lx2            = X_vars_col[30]
    Lx3            = X_vars_col[31]

    # =========== 2. 生成 .subvar 文件 ===========
    #   调用 Import_Subvars_To_File_idx(...)
    Import_Subvars_To_File_idx(
        WorkingDir=WorkingDir,
        tag=tag,
        idx=idx,
        TargetVelocity=TargetVelocity,
        sprCpz=sprCpz,
        Kpx=Kpx, Kpy=Kpy, Kpz=Kpz, Cpz=Cpz,
        Ksx=Ksx, Ksy=Ksy, Ksz=Ksz, Csz=Csz,
        Kld=Kld, Cld=Cld, Kaar=Kaar, Kstr=Kstr, Chx=Chx,
        Mc=Mc, hc=hc, Icx=Icx, Icy=Icy, Icz=Icz,
        Mt=Mt, ht=ht, Itx=Itx, Ity=Ity, Itz=Itz,
        Mw=Mw, Iwx=Iwx, Iwy=Iwy, Iwz=Iwz,
        Lx1=Lx1, Lx2=Lx2, Lx3=Lx3
    )

    # =========== 3.1 调用 SIMPACK 仿真  ===========
    # ===========      刚性轮对模型      ===========
    spck_name = f"Vehicle4WDB_NativeRigidCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    # 调用函数执行
    result = run_simpack_cmd(cmd, WorkingDir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (-99.11, -99.12, -99.13, -99.14) # 故障标记返回值
    else:
        print(f"成功执行 qs 脚本调用")
        # print("命令执行完成")
    time.sleep(1)  
    
    # =========== 3.2 调用 SIMPACK 仿真  ===========
    # ===========      独立轮对模型      ===========  
    
    # 注意仿真文件名称区别，核对仿真模型文件
    spck_name = f"Vehicle4WDB_IRWCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    # 调用函数执行
    result = run_simpack_cmd(cmd, WorkingDir, timeout_seconds = 10 * 60)
    # print(f"[INFO] 跳过 IRW 模型测试"); result = 0
 
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (-99.21, -99.22, -99.23, -99.24)
    else:
        print(f"成功执行 qs 脚本调用")
        # print("命令执行完成")

    time.sleep(1)    
    
    # =========== 4. 分析返回值 ===========    
    # 刚性轮对后处理结果导出与分析
    filemidname = r"NativeRigidCRV300m"
    SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV = CRVCal_idx(WorkingDir, filemidname, tag, idx)
    
    # 独立轮对后处理结果导出与分析
    filemidname = r"IRWCRV300m"
    SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = CRVCal_idx(WorkingDir, filemidname, tag, idx)
    
    # # print(f"[INFO] IRW 模型输出结果，使用123代替");
    # SumWearNumber_IRWCRV300m_CRV= 123 
    # maxLatDisp_IRWCRV300m_CRV = 123
    # WearNumber_CRV = 0.0
    # LatDispMax_CRV = 0.0
    
    return (SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV)
    # 注意检查 CRVPerf_idx 各个故障返回码的维度，应与 CRVPerf_idx 函数的 return 相同


