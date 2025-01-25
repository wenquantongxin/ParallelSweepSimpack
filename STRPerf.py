# -*- coding: gbk -*-

# 典型直线 AAR5 线路 性能评估

import os
import time
import numpy as np
import pandas as pd
import subprocess
from FindCrticalVelocity import (Import_Subvars_To_File_idx)

# 读取指定 .dat 数据
# 返回从数据文件中获得的 Sperling 指标
def ReadAAR5Dat(dat_path):
    with open(dat_path, "r", encoding="utf-8") as f:
        
        # 跳过前5行
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
    
# 单线程内计算直线 AAR5 线路的 Sperling 参量
def STRSperling_idx(
    work_dir,
    filemidname, # 用于区分刚性轮对或者独立轮对，filemidname 可能为 RigidSTR80kmph 或 IRWSTR80kmph（暂无）
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # 对应于 Result_RigidSTR80kmph_Opt_0125_20.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # 输出的 .dat 文件的名称
   
    # 拼出 SPF 文件的绝对路径
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # 脚本位置
    qs_script_path = os.path.join(work_dir, qs_script)

    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return (-99.0211, -99.0212) 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return (-99.0221, -99.0222)

    # 1) 调用 simpack-post 的脚本 .qs
    # BatchTmp 子文件夹内，以命令行执行: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCRV300m_Opt_0125_0.spf DatResult_RigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    try:
        ret = subprocess.run(cmd, cwd=work_dir, check=True)
    except subprocess.CalledProcessError as e:
        # 外部命令返回非 0
        print(f"[ERROR] simpack-post命令出错，返回码={e.returncode}")
        return (-99.0231, -99.0232) 
    except Exception as e:
        # 其他异常，如找不到可执行文件、工作目录不存在等
        print(f"[ERROR] 无法执行simpack-post命令，异常信息：{e}")
        return (-99.0241, -99.0242) 
    time.sleep(2)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] 后处理结果文件未找到: {dat_path}")
        return (-99.0251, -99.0252) 
      
    # 3) 解析文件
    try:
        Sperling_Y_fromDat, Sperling_Z_fromDat = ReadAAR5Dat(dat_path)
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        Sperling_Y = -99.0261
        Sperling_Z = -99.0262
    else:
        Sperling_Y = Sperling_Y_fromDat
        Sperling_Z = Sperling_Z_fromDat

    return Sperling_Y, Sperling_Z


def STRPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"对于模型 {idx} 进行典型直线线路的 Sperling 指标测试")
    
    # =========== 1. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = 80/3.6      # 直线评估时，采用 80 km/h 速度通过 AAR5 直线线路，使用 TargetVel 覆盖该速度取值
    
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
    spck_name = f"Vehicle4WDB_RigidSTR80kmph_Opt_{tag}_{idx}.spck"   # 例如: Vehicle4WDB_RigidSTR80kmph_Opt_0125_23
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # 执行命令
    try:
        ret = subprocess.run(cmd, cwd=WorkingDir)
        status = ret.returncode
        # 如果需要查看输出： ret.stdout, ret.stderr
    except Exception as e:
        # 如果出现异常，比如命令行执行错误
        print(f"[ERROR] SIMPACK仿真调用出现异常: {e}")
        return (-99.31, -99.32) # 故障标记返回值
    time.sleep(1)
    
    # =========== 4. 分析返回值 ===========    
    # 刚性轮对后处理结果导出与分析
    filemidname = r"RigidSTR80kmph"
    SperlingY_AAR5, SperlingZ_AAR5 = STRSperling_idx(WorkingDir, filemidname, tag, idx)
  
    return (SperlingY_AAR5, SperlingZ_AAR5)

