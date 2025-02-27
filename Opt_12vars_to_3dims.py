# -*- coding: utf-8 -*-

import numpy as np
import os
import math
import concurrent.futures
import time
import pandas as pd
import itertools
import shutil
import matplotlib
import matplotlib.pyplot as plt
import subprocess
matplotlib.use("TkAgg") 
import pickle

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter

#############################################################################################################
##############################               准备仿真文件函数组                    ##########################
#############################################################################################################

# 读取给定工作目录(working directory)下的config_opt.xlsx文件的Sheet1内容
def opt_ReadConfigExcel(working_dir, excel_name="config_opt.xlsx", sheet_name="Sheet1"):
    excel_path = os.path.join(working_dir, excel_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df

# 清空 BatchTmp 子文件夹
def opt_ClearBatchTmpFolder(WorkingDir):
    batch_tmp_path = os.path.join(WorkingDir, "BatchTmp")
    if os.path.exists(batch_tmp_path):
        shutil.rmtree(batch_tmp_path)
    os.makedirs(batch_tmp_path, exist_ok=True)
    
# 根据 Base 文件，复制并生成 .spck, .subvar, .spf 文件
def opt_PrepareSpckFilesForEachBatch(WorkingDir, tag, start_idx, end_idx):
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
    spck_spf_list = [
        (
            os.path.join(WorkingDir, "Vehicle4WDB_NativeRigidCRV300m_OptBase.spck"),
            os.path.join(WorkingDir, "Result_NativeRigidCRV300m.spf"),
            "Vehicle4WDB_NativeRigidCRV300m_Opt",
            "Result_NativeRigidCRV300m_Opt",
        ),
        (
            os.path.join(WorkingDir, "Vehicle4WDB_NativeRigidSTR80kmph_OptBase.spck"),
            os.path.join(WorkingDir, "Result_NativeRigidSTR80kmph.spf"),
            "Vehicle4WDB_NativeRigidSTR80kmph_Opt",
            "Result_NativeRigidSTR80kmph_Opt",
        ),
        (
            os.path.join(WorkingDir, "Vehicle4WDB_NativeRigidCriticalVel_OptBase.spck"),
            os.path.join(WorkingDir, "Result_NativeRigidCriticalVel.spf"),
            "Vehicle4WDB_NativeRigidCriticalVel_Opt",
            "Result_NativeRigidCriticalVel_Opt",
        ),
    ]
    
    # =========== 3. 创建输出文件夹 BatchTmp ===========
    batch_tmp_dir = os.path.join(WorkingDir, "BatchTmp")
    os.makedirs(batch_tmp_dir, exist_ok=True)
    
    # =========== 4. 对 spck_spf_list 中的每一对进行处理 ===========
    for base_spck_path, base_spf_path, spck_prefix, spf_prefix in spck_spf_list:
        # --- 4.1 读取源 spck 和 spf 文件内容(行) ---
        with open(base_spck_path, "r", encoding="utf-8") as f_spck:
            base_spck_lines = f_spck.readlines()
        with open(base_spf_path, "r", encoding="utf-8") as f_spf:
            base_spf_lines = f_spf.readlines()
        
        # --- 4.2 在 [start_idx, end_idx) 范围内, 逐个 i 生成目标文件 ---
        for i in range(start_idx, end_idx):
            
            # =========== 4.2.1 处理 spck 文件 ===========
            new_spck_name = f"{spck_prefix}_{tag}_{i}.spck"
            new_spck_path = os.path.join(batch_tmp_dir, new_spck_name)
            
            lines_spck_mod = base_spck_lines.copy()
            # 修改第 26(索引25), 61(索引60), 69(索引68) 行
            # 1) 第 26 行(索引 25)指定 subvar 文件名称
            lines_spck_mod[25] = (
                f"subvarset.file (          1                                       ) = "
                f"'./subvars_Opt_{tag}_{i}.subvar' ! subvarset filename\n"
            )
            # 2) 第 61 行(索引 60)
            lines_spck_mod[60] = (
                "substr.file (                       $S_NativeRigidWSBogie_Front             ) = "
                "'../ref_files/Bogie_RW_4WDBv31.spck' ! Filename\n"
            )
            # 3) 第 69 行(索引 68)
            lines_spck_mod[68] = (
                "substr.file (                       $S_NativeRigidWSBogie_Rear              ) = "
                "'../ref_files/Bogie_RW_4WDBv31.spck' ! Filename\n"
            )
            
            # 写出新的 .spck 文件
            with open(new_spck_path, "w", encoding="utf-8") as f_out_spck:
                f_out_spck.writelines(lines_spck_mod)
            
            # =========== 4.2.2 复制 subvar 文件 ===========
            # 文件名：subvars_Opt_test_0.subvar
            new_subvar_name = f"subvars_Opt_{tag}_{i}.subvar"
            new_subvar_path = os.path.join(batch_tmp_dir, new_subvar_name)
            shutil.copyfile(base_subvar_path, new_subvar_path)
            
            # =========== 4.2.3 处理 spf 文件 ===========
            new_spf_name = f"{spf_prefix}_{tag}_{i}.spf"
            new_spf_path = os.path.join(batch_tmp_dir, new_spf_name)
            
            lines_spf_mod = base_spf_lines.copy()
            # 修改第 9 行(索引8)
            lines_spf_mod[8] = (
                f'<ResFile filename="{spck_prefix}_{tag}_{i}.output/{spck_prefix}_{tag}_{i}.sbr" '
                'generatorVersion="20210000" id="resf1" type="sbr"/>\n'
            )
            
            # 写出新的 spf 文件
            with open(new_spf_path, "w", encoding="utf-8") as f_out_spf:
                f_out_spf.writelines(lines_spf_mod)
            
            # print(f"[INFO] 已生成文件: {new_spck_name}, {new_subvar_name}, {new_spf_name}")
    
    print("[INFO] 本大批次（包含多个并行小批次）所有文件已生成完毕！")
    time.sleep(1)

# 为每个 idx 的 subvar 文件导入参量
def opt_WriteSubvarsFile_idx(
    WorkingDir,
    tag,
    idx,
    TargetVelocity,
    sprCpz, Kpx, Kpy, Kpz, Cpz,
    Ksx, Ksy, Ksz, Csz, Kld, Cld,
    Kaar, Kstr, Chx,
    Mc, hc, Icx, Icy, Icz,
    Mt, ht, Itx, Ity, Itz,
    Mw, Iwx, Iwy, Iwz,
    Lx1, Lx2, Lx3
):
   
    subvar_filename = f"subvars_Opt_{tag}_{idx}.subvar"
    subvar_path = os.path.join(WorkingDir, "BatchTmp", subvar_filename)

    # 以写模式('w')打开文件，覆盖其原有内容
    with open(subvar_path, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write("!file.version=3.5! Removing this line will make the file unreadable\n\n")
        f.write("!**********************************************************************\n")
        f.write("! SubVars\n")
        f.write("!**********************************************************************\n")

        # 逐行写入 subvar(...) 语句
        f.write(f"subvar($_TargetVelocity, str= '{TargetVelocity:.6g}') ! $_TargetVelocity\n")
        f.write(f"subvar($_sprCpz, str= '{sprCpz:.6g}') ! $_sprCpz\n")
        f.write(f"subvar($_Kpx, str= '{Kpx:.6g}') ! $_Kpx\n")
        f.write(f"subvar($_Kpy, str= '{Kpy:.6g}') ! $_Kpy\n")
        f.write(f"subvar($_Kpz, str= '{Kpz:.6g}') ! $_Kpz\n")
        f.write(f"subvar($_Cpz, str= '{Cpz:.6g}') ! $_Cpz\n")
        f.write(f"subvar($_Ksx, str= '{Ksx:.6g}') ! $_Ksx\n")
        f.write(f"subvar($_Ksy, str= '{Ksy:.6g}') ! $_Ksy\n")
        f.write(f"subvar($_Ksz, str= '{Ksz:.6g}') ! $_Ksz\n")
        f.write(f"subvar($_Csz, str= '{Csz:.6g}') ! $_Csz\n")
        f.write(f"subvar($_Kld, str= '{Kld:.6g}') ! $_Kld\n")
        f.write(f"subvar($_Cld, str= '{Cld:.6g}') ! $_Cld\n")
        f.write(f"subvar($_Kaar, str= '{Kaar:.6g}') ! $_Kaar\n")
        f.write(f"subvar($_Kstr, str= '{Kstr:.6g}') ! $_Kstr\n")
        f.write(f"subvar($_Chx, str= '{Chx:.6g}') ! $_Chx\n")
        f.write(f"subvar($_Mc, str= '{Mc:.6g}') ! $_Mc\n")
        f.write(f"subvar($_hc, str= '{hc:.6g}') ! $_hc\n")
        f.write(f"subvar($_Icx, str= '{Icx:.6g}') ! $_Icx\n")
        f.write(f"subvar($_Icy, str= '{Icy:.6g}') ! $_Icy\n")
        f.write(f"subvar($_Icz, str= '{Icz:.6g}') ! $_Icz\n")
        f.write(f"subvar($_Mt, str= '{Mt:.6g}') ! $_Mt\n")
        f.write(f"subvar($_ht, str= '{ht:.6g}') ! $_ht\n")
        f.write(f"subvar($_Itx, str= '{Itx:.6g}') ! $_Itx\n")
        f.write(f"subvar($_Ity, str= '{Ity:.6g}') ! $_Ity\n")
        f.write(f"subvar($_Itz, str= '{Itz:.6g}') ! $_Itz\n")
        f.write(f"subvar($_Mw, str= '{Mw:.6g}') ! $_Mw\n")
        f.write(f"subvar($_Iwx, str= '{Iwx:.6g}') ! $_Iwx\n")
        f.write(f"subvar($_Iwy, str= '{Iwy:.6g}') ! $_Iwy\n")
        f.write(f"subvar($_Iwz, str= '{Iwz:.6g}') ! $_Iwz\n")
        f.write(f"subvar($_Lx1, str= '{Lx1:.6g}') ! $_Lx1\n")
        f.write(f"subvar($_Lx2, str= '{Lx2:.6g}') ! $_Lx2\n")
        f.write(f"subvar($_Lx3, str= '{Lx3:.6g}') ! $_Lx3\n")

    # 函数结束时，with 上下文会自动关闭文件
    print(f"[INFO] 已更新文件: {subvar_path}")
    
#############################################################################################################
##############################              读取.dat数据的函数组                    ##########################
#############################################################################################################

# 返回从数据文件中获得计算临界速度时，当前速度下的最大横移量
def opt_ReadCriticalVelDat(dat_path):
    val_array = [0.0]*4
    with open(dat_path, "r", encoding="utf-8") as f:
        # 跳过前5行
        for _ in range(5):
            f.readline()
        
        # 读取关键行
        line6 = f.readline()
        parts = line6.split(';')
        val_array[0] = float(parts[1].strip())

        for _ in range(4):
            f.readline()

        line11 = f.readline()
        parts = line11.split(';')
        val_array[1] = float(parts[1].strip())

        for _ in range(4):
            f.readline()

        line16 = f.readline()
        parts = line16.split(';')
        val_array[2] = float(parts[1].strip())

        for _ in range(4):
            f.readline()

        line21 = f.readline()
        parts = line21.split(';')
        val_array[3] = float(parts[1].strip())
    
    maxLatY_fromDat = max(val_array)
    
    return maxLatY_fromDat

# 返回从数据文件中获得的 Sperling 指标
def opt_ReadAAR5Dat(dat_path):
    with open(dat_path, "r", encoding="utf-8") as f:
        for _ in range(5):
            f.readline()
        line6 = f.readline()
        parts = line6.split(';')
        Sperling_Y_fromDat = float(parts[1].strip()) 
        for _ in range(4): 
            f.readline()
        line11 = f.readline()
        parts = line11.split(';')
        Sperling_Z_fromDat = float(parts[1].strip())
    
    return Sperling_Y_fromDat, Sperling_Z_fromDat

# 读取曲线仿真 .dat 文件，获得曲线磨耗数与最大横移
def opt_ReadCRVDat(
    dat_path,
    line_sumwear=6,
    lines_lateral=(26, 31, 36, 41)
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

#############################################################################################################
##############################        命令行调用Simpack前处理与后处理函数组          ##########################
#############################################################################################################

# 执行 simpack-cmd 调用 simpack-slv 或者 simpack-qs 脚本
def opt_RunSPCKCmd (cmd, work_dir, timeout_seconds):
    
    try:
        # 使用 Popen 启动进程
        process = subprocess.Popen(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

         # 等待进程完成，最多等待 timeout_seconds 时间
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        # 如果进程在时间限制内完成
        if process.returncode == 0:
            return 0
        else:
            # 打印标准错误输出
            print(f"[ERROR] simpack 执行失败，返回码={process.returncode}")
            return -99.5  # 返回错误码

    except subprocess.TimeoutExpired:
        # 如果超时，终止进程
        print("[ERROR] simpack 执行超时，终止进程！")
        process.terminate()
        process.wait()  # 确保进程被完全终止
        return -99.4  # 返回超时错误码

    except Exception as e:
        print(f"[ERROR] 执行命令失败: {e}")
        return -99.3  # 其他错误

#############################################################################################################
##########################                直曲线单线程计算组(命令启动+后处理)           #####################
#############################################################################################################

# 单线程内计算四组轮对的最大横移量
def opt_MaxLatY_idx(
    work_dir,
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs",
    wait_seconds = 3
):
    
    spf_filename = f"Result_NativeRigidCriticalVel_Opt_{tag}_{idx}.spf"
    out_result_prefix = f"DatResult_NativeRigidCriticalVel_{tag}_{idx}"
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    qs_script_path = os.path.join(work_dir, qs_script)
    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return +99.0 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return +99.1 
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    # 调用函数执行
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60) # 10 * 60
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return +99.2 
    else:
        print(f"成功执行 qs 脚本调用")
    time.sleep(wait_seconds)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"后处理结果文件未找到: {dat_path}")

    # 3) 解析文件
    try:
        maxLatY_fromDat = opt_ReadCriticalVelDat(dat_path)
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        maxLatY = -99.4
    else:
        maxLatY = maxLatY_fromDat

    return maxLatY

# 单线程内计算曲线通过的磨耗数与最大横移量
def opt_CRVCal_idx(
    work_dir,
    filemidname, # 用于区分刚性轮对或者独立轮对，filemidname 为 NativeRigidCRV300m
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # 对应于 Result_NativeRigidCRV300m_Opt_0125_0.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # 输出的 .dat 文件的名称
   
    # 拼出 SPF 文件的绝对路径
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # 脚本位置
    qs_script_path = os.path.join(work_dir, qs_script)

    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return (+999.0111, +999.0112) 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return (+999.0121, +999.0122)

    # 1) 调用 simpack-post 的脚本 .qs
    # BatchTmp 子文件夹内，以命令行执行: simpack-post -s SbrExport_SPCKResult.qs Result_NativeRigidCRV300m_Opt_0125_0.spf DatResult_NativeRigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    # 调用函数执行
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (+999.0131, +999.0132)
    else:
        print(f"成功执行 qs 脚本调用")
        print("命令执行完成")
    
    time.sleep(2)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] 后处理结果文件未找到: {dat_path}")
        return (+999.0151, +999.0152) 
      
    # 3) 解析文件
    try:
        SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat = opt_ReadCRVDat(dat_path)
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        SumWearNumber_CRV = +9.025
        maxLatDisp_CRV = +9.026
    else:
        SumWearNumber_CRV = SumWearNumber_CRV_fromDat
        maxLatDisp_CRV = maxLatDisp_CRV_fromDat
        
    # 返回上部函数 CRVPerf_idx
    # 注意检查 CRVCal_idx 各个故障返回码的维度，应与 CRVCal_idx 函数的 return 相同
    return SumWearNumber_CRV, maxLatDisp_CRV

# 单线程内计算直线 AAR5 线路的 Sperling 参量
def STRSperling_idx(
    work_dir,
    filemidname, # 用于区分刚性轮对或者独立轮对，filemidname 为 RigidSTR80kmph
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # 对应于 Result_NativeRigidSTR80kmph_Opt_0125_20.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # 输出的 .dat 文件的名称
   
    # 拼出 SPF 文件的绝对路径
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # 脚本位置
    qs_script_path = os.path.join(work_dir, qs_script)

    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return (+9.0211, +9.0212) 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return (+9.0221, +9.0222)

    # 1) 调用 simpack-post 的脚本 .qs
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    # 调用函数执行
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (+9.0231, +9.0232)
    else:
        print(f"成功执行 slv 或 qs 脚本调用")
        # print("命令执行完成")
    
    time.sleep(2)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] 后处理结果文件未找到: {dat_path}")
        return (+9.0251, +9.0252) 
      
    # 3) 解析文件
    try:
        Sperling_Y_fromDat, Sperling_Z_fromDat = opt_ReadAAR5Dat(dat_path)
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        Sperling_Y = +9.0261
        Sperling_Z = +9.0262
    else:
        Sperling_Y = Sperling_Y_fromDat
        Sperling_Z = Sperling_Z_fromDat

    return Sperling_Y, Sperling_Z


#############################################################################################################
##########################           直曲线单线程计算组(参数载入+上述计算函数调用)          #####################
#############################################################################################################

# 曲线优化目标 - 刚性轮对转向架曲线通过的磨耗数
# 注意与 CRVPerf_idx 函数区分
def opt_CRVPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"参数寻优B：对于模型 {idx} 进行曲线通过性能评估")
    
    # =========== 1. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = 60/3.6      # 曲线评估时，采用 60 km/h 速度通过 R300 曲线，使用 TargetVel 覆盖该速度取值
    
    sprCpz         = X_vars_col[0]
    Kpx            = X_vars_col[1]
    Kpy            = Kpx                # X_vars_col[3]
    Kpz            = X_vars_col[2]
    Cpz            = 5000000            # X_vars_col[3]
    Ksx            = X_vars_col[3]
    Ksy            = Ksx                # X_vars_col[7]
    Ksz            = X_vars_col[4]
    Csz            = X_vars_col[5]
    Kld            = X_vars_col[6]
    Cld            = X_vars_col[7]
    Kaar           = 2500000            # X_vars_col[12]
    Kstr           = 20000000           # X_vars_col[13]
    Chx            = X_vars_col[8]
    Mc             = 25000              # X_vars_col[15]
    hc             = -1.8               # X_vars_col[16]
    Icx            = 50000              # X_vars_col[17]
    Icy            = 1200000            # X_vars_col[18]
    Icz            = 1200000            # X_vars_col[19]
    Mt             = 5000 # X_vars_col[20]
    ht             = 0.0  # X_vars_col[21]
    Itx            = 2000 # X_vars_col[22]
    Ity            = 3000 # X_vars_col[23]
    Itz            = 5000 # X_vars_col[24]
    Mw             = 1000 # X_vars_col[25]
    Iwx            = 560 # X_vars_col[26]
    Iwy            = 100 # X_vars_col[27]
    Iwz            = 560 # X_vars_col[28]
    Lx1            = X_vars_col[9]
    Lx2            = X_vars_col[10]
    Lx3            = X_vars_col[11]

    # =========== 2. 生成 .subvar 文件 ===========

    opt_WriteSubvarsFile_idx(
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

    # ===========  调用 SIMPACK 仿真  ===========
    # ===========      独立轮对模型      ===========  
    spck_name = f"Vehicle4WDB_NativeRigidCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    # 调用函数执行
    result = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60)

    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (+999.21, +999.22)
    else:
        print(f"成功执行 qs 脚本调用")
        # print("命令执行完成")

    time.sleep(1)    
    
    # =========== 4. 分析返回值 ===========    
    # 独立轮对后处理结果导出与分析
    filemidname = r"NativeRigidCRV300m"
    SumWearNumber_NativeRigidCRV300m_CRV, maxLatDisp_NativeRigidCRV300m_CRV = opt_CRVCal_idx(WorkingDir, filemidname, tag, idx)
        
    return (SumWearNumber_NativeRigidCRV300m_CRV, maxLatDisp_NativeRigidCRV300m_CRV)

# 直线 AAR5 评估
def opt_STRPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"对于模型 {idx} 进行典型直线线路的 Sperling 指标测试")
    
    # =========== 1. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = 80/3.6      # 直线评估时，采用 80 km/h 速度通过 AAR5 直线线路，使用 TargetVel 覆盖该速度取值

    sprCpz         = X_vars_col[0]
    Kpx            = X_vars_col[1]
    Kpy            = Kpx                # X_vars_col[3]
    Kpz            = X_vars_col[2]
    Cpz            = 5000000            # X_vars_col[3]
    Ksx            = X_vars_col[3]
    Ksy            = Ksx                # X_vars_col[7]
    Ksz            = X_vars_col[4]
    Csz            = X_vars_col[5]
    Kld            = X_vars_col[6]
    Cld            = X_vars_col[7]
    Kaar           = 2500000            # X_vars_col[12]
    Kstr           = 20000000           # X_vars_col[13]
    Chx            = X_vars_col[8]
    Mc             = 25000              # X_vars_col[15]
    hc             = -1.8               # X_vars_col[16]
    Icx            = 50000              # X_vars_col[17]
    Icy            = 1200000            # X_vars_col[18]
    Icz            = 1200000            # X_vars_col[19]
    Mt             = 5000 # X_vars_col[20]
    ht             = 0.0  # X_vars_col[21]
    Itx            = 2000 # X_vars_col[22]
    Ity            = 3000 # X_vars_col[23]
    Itz            = 5000 # X_vars_col[24]
    Mw             = 1000 # X_vars_col[25]
    Iwx            = 560 # X_vars_col[26]
    Iwy            = 100 # X_vars_col[27]
    Iwz            = 560 # X_vars_col[28]
    Lx1            = X_vars_col[9]
    Lx2            = X_vars_col[10]
    Lx3            = X_vars_col[11]

    # =========== 2. 生成 .subvar 文件 ===========
    opt_WriteSubvarsFile_idx(
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
    spck_name = f"Vehicle4WDB_NativeRigidSTR80kmph_Opt_{tag}_{idx}.spck"   # 例如: Vehicle4WDB_NativeRigidSTR80kmph_Opt_0125_23
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    result = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return (+99.31, +99.32)
    else:
        print(f"成功执行 qs 脚本调用")
        # print("命令执行完成")
        
    time.sleep(1)

    # =========== 4. 分析返回值 ===========    
    # 刚性轮对后处理结果导出与分析
    filemidname = r"NativeRigidSTR80kmph"
    SperlingY_AAR5, SperlingZ_AAR5 = STRSperling_idx(WorkingDir, filemidname, tag, idx)
  
    return (SperlingY_AAR5, SperlingZ_AAR5)

# 临界速度计算子项目
# 判断编号为 idx 的SIMPACK模型是否稳定
def opt_CheckStable_Idx(WorkingDir, X_vars, tag, idx, TargetVel):
    
    # =========== 1. 失稳阈值设置 ===========
    UnstableThreshold = 3.0 / 1000.0  # 3 mm

    # =========== 2. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = TargetVel      # X_vars_col[0] # 使用 TargetVel 覆盖该速度取值
    sprCpz         = X_vars_col[0]
    Kpx            = X_vars_col[1]
    Kpy            = Kpx                # X_vars_col[3]
    Kpz            = X_vars_col[2]
    Cpz            = 5000000            # X_vars_col[3]
    Ksx            = X_vars_col[3]
    Ksy            = Ksx                # X_vars_col[7]
    Ksz            = X_vars_col[4]
    Csz            = X_vars_col[5]
    Kld            = X_vars_col[6]
    Cld            = X_vars_col[7]
    Kaar           = 2500000            # X_vars_col[12]
    Kstr           = 20000000           # X_vars_col[13]
    Chx            = X_vars_col[8]
    Mc             = 25000              # X_vars_col[15]
    hc             = -1.8               # X_vars_col[16]
    Icx            = 50000              # X_vars_col[17]
    Icy            = 1200000            # X_vars_col[18]
    Icz            = 1200000            # X_vars_col[19]
    Mt             = 5000 # X_vars_col[20]
    ht             = 0.0  # X_vars_col[21]
    Itx            = 2000 # X_vars_col[22]
    Ity            = 3000 # X_vars_col[23]
    Itz            = 5000 # X_vars_col[24]
    Mw             = 1000 # X_vars_col[25]
    Iwx            = 560 # X_vars_col[26]
    Iwy            = 100 # X_vars_col[27]
    Iwz            = 560 # X_vars_col[28]
    Lx1            = X_vars_col[9]
    Lx2            = X_vars_col[10]
    Lx3            = X_vars_col[11]

    # =========== 3. 生成 .subvar 文件 ===========
    #   调用 Import_Subvars_To_File_idx(...)
    opt_WriteSubvarsFile_idx(
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

    # =========== 4. 调用 SIMPACK 仿真 ===========
    spck_name = f"Vehicle4WDB_NativeRigidCriticalVel_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # 执行命令
    status = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60) # 10 * 60
    if status != 0:
        print(f"[ERROR] SIMPACK仿真失败，命令返回码: {status}")
        return 0.1
    else:
        # 仿真成功, 继续后处理 -> 读取最大横移量
        maxLatY = opt_MaxLatY_idx(WorkingDir, tag, idx)
        
        # 与阈值比较
        if abs(maxLatY) >= UnstableThreshold:
            return 0.2  # 表示失稳
        else:
            return 1.0  # 表示稳定

# 二分搜索临界速度函数
def opt_HalfSearchCrticalVelocity(
    WorkingDir,
    X_vars,
    tag,
    idx,
    StartVel,
    EndVel,
    N_depth
):
    
    print("开始二分搜索：")
    # 你可以在这里做更多打印，类似于MATLAB
    print(f"  - tag: {tag}, idx: {idx}")
    print(f"  - 起始速度：{StartVel:.2f} m/s ({StartVel*3.6:.2f} km/h)")
    print(f"  - 终止速度：{EndVel:.2f} m/s ({EndVel*3.6:.2f} km/h)")
    print(f"  - 二分次数：{N_depth}")
    print("-----------------------------------")

    low_vel = StartVel
    high_vel = EndVel

    for i_depth in range(1, N_depth + 1):
        mid_vel = 0.5 * (low_vel + high_vel)
        # 调用稳定性判断函数
        is_stable = opt_CheckStable_Idx(
            WorkingDir=WorkingDir,
            X_vars=X_vars,
            tag=tag,
            idx=idx,
            TargetVel=mid_vel # 传入 mid_vel 进行测试
        )

        if is_stable == 1.0:
            # 车辆在 mid_vel 下稳定 => 临界速度可能更高
            low_vel = mid_vel
            print(f"深度 {i_depth}: {mid_vel:.2f} m/s 稳定, 收缩区间到 [{low_vel:.2f}, {high_vel:.2f}]")
        else:
            # 返回 0.1 或 0.2，均视为不稳定 => 临界速度在 mid_vel 以下
            high_vel = mid_vel
            print(f"模型{idx}: 二分深度 {i_depth} 运行速度{mid_vel:.2f} m/s时不稳定, 收缩区间到 [{low_vel:.2f}, {high_vel:.2f}]")

    # 取收缩区间的中值作为近似临界速度
    critical_vel = 0.5 * (low_vel + high_vel)

    print("-----------------------------------")
    print(f"搜索结束，得到临界速度 ≈ {critical_vel:.2f} m/s ({critical_vel*3.6:.2f} km/h)\n")

    return critical_vel


################################################################
# 1) 并行 worker：只返回 4 个值，最后一列是我们关心的第三个目标
################################################################

# 并行任务函数
def opt_parallel_worker(args):
    
    (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth) = args
    
    # 实际上的全局列索引
    actual_idx = start_idx + col_idx_in_batch

    # 并行任务组
    # 并行任务 1：调用半搜索函数，返回临界速度
    CrticalVelocity = opt_HalfSearchCrticalVelocity(WorkingDir, X_vars, tag, actual_idx, StartVel, EndVel, N_depth)
    time.sleep(1)

    # 并行任务 2：调用曲线计算模型，返回曲线磨耗数、横移量
    SumWearNumber_NativeRigidCRV300m_CRV, maxLatDisp_NativeRigidCRV300m_CRV = opt_CRVPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # 并行任务 3：调用典型 AAR5 直线计算模型 性能评估，返回 Sperling 指标 RMS 值
    SperlingY_AAR5, SperlingZ_AAR5 = opt_STRPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    try:
        SperlingYZ_RMS = math.sqrt( (SperlingY_AAR5 ** 2 + SperlingZ_AAR5 ** 2) / 2 )
    except Exception as e:
        print(f"[ERROR] SperlingYZ指标计算异常，异常信息：{e}")
        return +9.77
        
    time.sleep(1)
    # 返回并行计算该 idx 的结果组向量
    # 三个目标
    return (col_idx_in_batch, CrticalVelocity,  SumWearNumber_NativeRigidCRV300m_CRV, SperlingYZ_RMS)

################################################################
# 2) 封装“分批并行”逻辑的函数
################################################################
def opt_XEvalPrl(X, WorkingDir, tag, StartVel, EndVel, N_depth, BatchSize_parallel=5):
    """
    X: shape=(N, 12)  # N个解, 每个解12个变量
    返回: shape=(N, 3) 的目标矩阵F
          其中F[:, 0] = -cVel  (若要最大化cVel就取负)
               F[:, 1] = NativeRigidWN_CRV
               F[:, 2] = SperlingYZ_RMS
    """
    # 转置 pymoo 传过来的参数，以与后续代码所需匹配
    X = X.T

    print(f"==== Debug: 转置后的 X.shape = {X.shape}")
    print("==== Debug: 这批候选解(按12个参数变量分组)：\n", X[:12]) 
    
    # N个解
    N_opt = X.shape[1]
    
    # 每个解要计算 3 个目标(这里的 3 是您想保留的指标数量)
    result_dim = 3 
    
    # 用于存放所有解的目标值(先用 shape=(3, N) 存，最后转置)
    batch_result_full = np.zeros((result_dim, N_opt))

    num_batches = math.ceil(N_opt / BatchSize_parallel)
    print("总的参数组合数 = 种群个体数：", N_opt)
    print("并行任务数：", BatchSize_parallel)
    print("批次数量：", num_batches)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BatchSize_parallel
        end_idx   = min((batch_idx + 1) * BatchSize_parallel, N_opt)
        
        print(f"第 {batch_idx+1} / {num_batches} 批：行索引范围 [{start_idx}:{end_idx})")
        
        # ============== 清理 & 生成文件 ==============
        opt_ClearBatchTmpFolder(WorkingDir)
        print("已清理上个小批次的文件")
        
        opt_PrepareSpckFilesForEachBatch(WorkingDir, tag, start_idx, end_idx)
        print("已准备好本小批次的文件")     
        
        # 取本批次的解(按行切分)
        X_batch = X[:, start_idx:end_idx]  # 而不是 X[start_idx:end_idx, :]

        # 结果暂存 (3行 x batch_size列)
        batch_result = np.zeros((result_dim, X_batch.shape[1]))

        # ============== (b) 并行处理 ==============
        with concurrent.futures.ProcessPoolExecutor(max_workers=BatchSize_parallel) as executor:
            future_list = []
            for col_idx_in_batch in range(X_batch.shape[1]):
                args = (col_idx_in_batch, start_idx, WorkingDir, X, tag, StartVel, EndVel, N_depth)
                future = executor.submit(opt_parallel_worker, args)
                future_list.append(future)

            # 收集结果
            for future in concurrent.futures.as_completed(future_list):
                col_idx_in_batch, cVel, NativeRigidWN_CRV, SperlingYZ_RMS = future.result()
                # 并行池 return 
                batch_result[0, col_idx_in_batch] = cVel 
                batch_result[1, col_idx_in_batch] = NativeRigidWN_CRV
                batch_result[2, col_idx_in_batch] = SperlingYZ_RMS
                
        # 将本批次结果放到全局 big array
        batch_size_actual = end_idx - start_idx
        batch_result_full[:, start_idx:end_idx] = batch_result[:, :batch_size_actual]
        
    # 现在 batch_result_full.shape=(3, N), 转置成 (N,3)
    # 例如 cVel 越大越好 => -cVel
    cVel_all = batch_result_full[0, :]
    NativeRigidWN_CRV_all = batch_result_full[1, :]
    SperlingYZ_all = batch_result_full[2, :]

    # 构造最终返回F: (N,3)
    f1 = -cVel_all  # 因为要最大化cVel, pymoo默认最小化
    f2 = NativeRigidWN_CRV_all
    # 注意！对于 Sperling 指标，此处乘以放缩系数 100
    f3 = SperlingYZ_all * 100 

    F = np.vstack([f1, f2, f3]).T  # (3,N).T => (N,3)
    print("==== Debug: 本轮 pymoo 所有候选解（包含多个小批次）优化目标汇总 batch_result_full:\n ", batch_result_full) 
    print("Sperling 指标，此处为原始值，实际以放缩系数 100 进入优化迭代\n")
    
    return F

# 保存最终结果，包括所有历史迭代优化的记录
def SaveItersResult(res, filename):
    """
    将 pymoo 'Result' 对象的部分信息写入 npz 文件:
      - X, F, G, CV, history

    注意:
      - 不包含 'feasible' 等其他属性, 以避免某些场景下报错.
      - 'history' 中包含每一代的快照, 若问题规模很大, 文件也可能较大.
    """
    data_dict = {
        "X"      : res.X,         # 非支配解(多目标)或最优解(单目标)
        "F"      : res.F,         # 对应的目标值
        "G"      : res.G,         # 不等式约束(若无则可能是 None)
        "CV"     : res.CV,        # 约束违反度 (Constraint Violation), 同样可能 None
        "history": res.history    # 如果 save_history=True, 则这里有每代历史
    }
    # 保存为 npz
    np.savez(filename, **data_dict)
    print(f"[SaveItersResult] 已将 X, F, G, CV, history 保存到 {filename}.")

# callback 函数回调，保存迭代过程中的 X, F, G
def my_callback(algorithm, working_dir=None, **kwargs):
    """
    在每一代结束后被调用:
      - algorithm: 当前算法对象(例如 NSGA2 instance)
      - n_gen: 当前代数 (从1开始)
      - **kwargs: 其他参数 (pymoo 内部保留参数)

    这里将当前种群的 X, F, G 等保存到 working_dir/ChkPnt 下，
    并且提取非支配解 (nd_X, nd_F, nd_G) 另外保存.
    """
    
    global history_F # 全局列表，用于存储历代种群目标值

    n_gen = algorithm.n_gen
    pop = algorithm.pop
    X = pop.get("X")  # shape=(pop_size, n_var)
    F = pop.get("F")  # shape=(pop_size, n_obj)
    G = pop.get("G")  # shape=(pop_size, n_ieq_constr) 或 None
    history_F.append(F.copy())
    print(f"[Callback] 第 {n_gen} 代, F_gen.shape = {F.shape}")
    
    # 准备 res 代际结果的目录
    if working_dir is None:
        working_dir = os.getcwd() 
    chkpt_dir = os.path.join(working_dir, "ChkPnt")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir, exist_ok=True)

    # 以 generation_{n_gen}.npz 形式保存所有解
    filename_all = os.path.join(chkpt_dir, f"generation_{n_gen}.npz")
    print(f"[Callback] 正在保存第 {n_gen} 代的所有种群到 {filename_all}")
    np.savez(filename_all, X=X, F=F, G=G)

    # ============== 提取当代非支配解并单独保存 ==============
    nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_X = X[nd_front]
    nd_F = F[nd_front]
    nd_G = G[nd_front] if G is not None else None

    filename_nd = os.path.join(chkpt_dir, f"generation_{n_gen}_nondom.npz")
    print(f"[Callback] 正在保存第 {n_gen} 代的非支配解到 {filename_nd} ...")
    np.savez(filename_nd, X=nd_X, F=nd_F, G=nd_G)

    print(f"[Callback] 第 {n_gen} 代数据保存完毕, 种群规模 = {X.shape[0]}, 非支配解数量 = {len(nd_front)}")

class MyBatchProblem(Problem):
    def __init__(self,
                 batch_size=5,
                 WorkingDir=os.getcwd(),
                 tag="demo",                # 修改点 0
                 StartVel = 100/3.6,
                 EndVel = 900/3.6,
                 N_depth = 7,
                 **kwargs):
        """
        # 修改点 1
        使用 (N,12) 决策变量, 3 个目标 
        xl=np.array([ 2000,  160000,  120000,  24000,  30000,   4000,  1600000,  10000,     100,    0,    0, -0.6]),
        xu=np.array([50000, 4000000, 3000000, 600000, 750000, 100000, 40000000, 250000, 3000000, 0.64, 0.64,  0.4]),
        """
        super().__init__(
            n_var=12,
            n_obj=3,
            # 3个约束条件：n_ieq_constr=3
            n_ieq_constr = 3,  
            xl=np.array([ 2000,  50000,  120000,  15000,  15000,   2000,  1600000,  10000,     100,    0,    0, -0.6]),
            xu=np.array([50000, 4000000, 3000000, 600000, 750000, 100000, 60000000, 250000, 3000000, 0.64, 0.64,  0.4]),
            elementwise_evaluation=False,
            **kwargs
        )

        self.batch_size = batch_size
        self.WorkingDir = WorkingDir
        self.tag = tag
        self.StartVel = StartVel
        self.EndVel = EndVel
        self.N_depth = N_depth

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X.shape = (N, 12)
        """
        # 1) 计算目标值 F (N,3)
        F = opt_XEvalPrl(
            X,
            WorkingDir=self.WorkingDir,
            tag=self.tag,
            StartVel=self.StartVel,
            EndVel=self.EndVel,
            N_depth=self.N_depth,
            BatchSize_parallel=self.batch_size
        )
        
        # 2) 计算不等式约束 G (N,2)
        # 约束1: cVel >= 250/3.6 => -cVel <= -250/3.6 => f1 <= -250/3.6
        # f1 = -cVel =>  g1 = f1 - (-250/3.6) = f1 + 250/3.6 <= 0
        # 约束2: f2 <= 1000 =>  g2 = f2 - 1000 <= 0
        # 约束3: f3 <= 300  =>  g3 = f3 - 300 <= 0
        G = np.zeros((F.shape[0], 3))
        
        # 使 f1 <= -250/3.6，对应于临界速度大于 250 km/h
        G[:, 0] = F[:, 0] + 250.0/3.6 
        # 使 f2 <= 1000，对应于磨耗数小于 1000
        G[:, 1] = F[:, 1] - 1000.00 
        # 使 f3 <= 300，对应于 Sperling 指标小于 3
        G[:, 2] = F[:, 2] - math.sqrt( (3 * 3 + 3 * 3) / 2 ) * 100
          
        out["F"] = F
        out["G"] = G

################################################################
# 4) 测试入口
################################################################
def main():
    
    # 历史适应度函数 F
    global history_F
    history_F = []
    
    # 定义问题
    # 修改点 2：并行计算池
    problem = MyBatchProblem( batch_size = 10 ) 
    
    #--------------------------------------------------------------------------------------------#
    # 多目标 NSGA2 算法
    algorithm_NSGA2 = NSGA2( pop_size = 66 )
    # RNSGA2 算法
    ref_points = np.load('RefPnt_fromNSGA2.npy') # 导入 Analysis_GenRefPnt.ipynb 生成的参考点
    algorithm_RNSGA2 = RNSGA2(ref_points=ref_points, pop_size=66, epsilon=15, normalization='no') 
    # NSGA3 算法
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=17) 
    # 组合结果C(17+2,2)=171<180=pop_size, pop_size is equal or larger than the number of reference directions. 
    algorithm_NSGA3 = NSGA3(pop_size=180,ref_dirs=ref_dirs)
    
    # 修改点 3：选择优化算法
    selected_algorithm = algorithm_NSGA2  # 可以选择 algorithm_NSGA2 或 algorithm_NSGA3
    #--------------------------------------------------------------------------------------------#
    
    # tag 在1033行定义 tag="demo"
    
    # 修改点 4：终止条件
    termination = get_termination("n_gen", 88)  # 遗传迭代数
    
    # 运行优化
    res = minimize(
        problem, 
        selected_algorithm,
        termination, 
        seed=1,
        verbose = True, 
        save_history = True, 
        callback = my_callback)  # 传入回调函数
    
    # 查看结果
    print("\n==== 优化完成 ====")
    print("非支配解数量:", len(res.X))
    print("示例解X[0] :", res.X[0])
    print("对应目标F[0]:", res.F[0])
    print("约束G[0]:", res.G[0], "(注: G[i] <= 0表示可行)")
    
    # 结果保存(2种方法，csv 和 npz)
    nd_X = res.X 
    nd_F = res.F

    np.savetxt("final_solutions.csv",
            np.hstack([nd_X, nd_F]),
            delimiter=",",
            comments="",  # 去掉注释符号
            fmt="%.6f")   # 控制输出精度
    np.savez("final_solutions.npz", X=nd_X, F=nd_F)
    
    # res 的结果保存
    #  `res.X, res.F` => 最终解；history_F => (list of arrays)
    np.savez(
        "res_history.npz",
        final_X=res.X,
        final_F=res.F,
        history_F=np.array(history_F, dtype=object) # history_F 是一个列表, 其中每代都是一个 (pop_size, n_obj) 的数组
    )
    print("[Main] 已保存 final_X, final_F, 以及各代 F_gen 到 res_history.npz.")

if __name__ == "__main__":
    main()
    
    """
    命令行调用：
    
    F:  # 切换盘符                                                                                                             
    cd F:\ResearchMainStream\0.ResearchBySection\C.动力学模型\参数优化\参数优化实现\ParallelSweepSimpack                        
    python -X utf8  Opt_12vars_to_3dims.py # 执行本程序                                  
    

    附录A: MATLAB GA 函数对应设置
    MaxGenerations (Generations) 功能：算法的最大迭代数。默认 ga 是 100×变量数，gamultiobj 是 200×变量数。
    PopulationSize 功能：设置种群大小（即每一代的个体数量）。如果变量数 ≤ 5，默认 50；否则默认 200。
    
    附录B: 后处理
    (1) 查看 Pareto Front 分布，详见 Analysis_OptsResults.ipynb
    (2) 根据{X,F}计算结果，分析参数变化对于各动力学指标的相关性，详见 Analysis_Corr.ipynb
    (3) 根据NSGA2最终前沿解，拟合分布曲面上的散点作为参考点，详见 Analysis_GenRefPnt.ipynb

    """
