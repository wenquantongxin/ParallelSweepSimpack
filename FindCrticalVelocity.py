# -*- coding: gbk -*-

import os
import time
import numpy as np
import pandas as pd
import subprocess

# 为每个 idx 的 subvar 文件导入参量
def Import_Subvars_To_File_idx(
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
    """
    对应于MATLAB的 Fun_GenerateSubvarsFile 函数。
    在子文件夹 BatchTmp 下，根据传入的 tag, idx 确定要写入的目标 .subvar 文件名称，
    然后覆盖写入Simpack外部参数定义文件内容。

    参数:
    ----------
    WorkingDir : str
        主工作目录，包含 BatchTmp 子文件夹
    tag : str
        区分批次或试验环境的标识符
    idx : int
        组合或模型编号
    其余参数:
        与原MATLAB函数中的子变量一一对应 (TargetVelocity, sprCpz, Kpx, ... , Lx1, Lx2, Lx3)。
"""
    # 构造需要写入的子变量文件路径
    # 例如:  .../BatchTmp/subvars_Opt_{tag}_{idx}.subvar
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

# 读取指定 .dat 数据
# 返回从数据文件中获得计算临界速度时，当前速度下的最大横移量
def ReadCriticalVelDat(dat_path):
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

# 执行 simpack-cmd 调用 simpack-slv 或者 simpack-qs 脚本
def run_simpack_cmd (cmd, work_dir, timeout_seconds):
    """
    运行 simpack-post 命令，并在指定时间内监控其执行，超时则终止进程。

    参数:
    cmd : list
        需要执行的命令及其参数。
    work_dir : str
        工作目录路径，指向 simpack-post 需要运行的目录。
    timeout_seconds : int
        最大执行时间（秒）。如果命令运行超时，将会终止它。

    返回:
    result : int
        返回 0 表示成功执行，其他值表示错误。
    """
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


# 单线程内计算四组轮对的最大横移量
def MaxLatY_idx(
    work_dir,
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs",
    wait_seconds = 3
):
    """
    Python版函数，对应 MATLAB 的 Fun_maxLatY_fromSPCKpost。
    
    参数:
    ----------
    work_dir : str
        主工作目录，包含脚本 SbrExport_SPCKResult.qs。
    spf_filename : str
        SPF文件名（例如 "OptTargetResult_Opt_test_66.spf"），实际位于 work_dir/BatchTmp 下。
    out_result_prefix : str
        生成结果的前缀（"OptResult_test_66"），实际 dat 文件会写到 work_dir/BatchTmp/下。
    qs_script : str
        默认 "SbrExport_SPCKResult.qs"，在 work_dir 下。
    wait_seconds : int or float
        命令执行后等待秒数，保证文件写入。
    """
    spf_filename = f"Result_RigidCriticalVel_Opt_{tag}_{idx}.spf"
    out_result_prefix = f"DatResult_RigidCriticalVel_{tag}_{idx}"
    # spf_filename = f"OptTargetResult_Opt_{tag}_{idx}.spf"
    # out_result_prefix = f"OptResult_{tag}_{idx}"
    
    # 拼出 SPF 文件的绝对路径
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    # 拼出导出的结果前缀 (dat 最终会成为 "BatchTmp/out_result_prefix.dat")
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # 脚本位置
    qs_script_path = os.path.join(work_dir, qs_script)

    # 若有需要可检查文件存在
    if not os.path.isfile(qs_script_path):
        print(f"后处理脚本不存在: {qs_script_path}")
        return -99.0 
    if not os.path.isfile(spf_path):
        print(f".spf文件不存在: {spf_path}")
        return -99.1 

    # 1) 调用 simpack-post 的脚本 .qs
    # BatchTmp 子文件夹内，以命令行执行: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCriticalVel_Opt_AAA_9.spf DatResult_RigidCriticalVel_AAA_9     
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF 文件路径
        out_result_full_prefix  # 输出前缀
    ]
    
    # 调用函数执行
    result = run_simpack_cmd(cmd, work_dir, timeout_seconds = 10 * 60) # 10 * 60
    if result != 0:
        print(f"运行失败，错误码：{result}")
        return -99.2 
    else:
        print(f"成功执行 qs 脚本调用")

    
    # try:
    #     ret = subprocess.run(cmd, cwd=work_dir, check=True)
    # except subprocess.CalledProcessError as e:
    #     # 外部命令返回非 0
    #     print(f"[ERROR] simpack-post命令出错，返回码={e.returncode}")
    #     
    # except Exception as e:
    #     # 其他异常，如找不到可执行文件、工作目录不存在等
    #     print(f"[ERROR] 无法执行simpack-post命令，异常信息：{e}")
    #     return -99.3 
       
    time.sleep(wait_seconds)
    
    # 2) 拼出最终 .dat 文件所在路径
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"后处理结果文件未找到: {dat_path}")

    # 3) 解析文件
    try:
        maxLatY_fromDat = ReadCriticalVelDat(dat_path)
    except Exception as e:
        print(f"[ERROR] 解析 {dat_path} 时出现异常: {e}")
        maxLatY = -99.4
    else:
        maxLatY = maxLatY_fromDat

    return maxLatY

# 判断编号为 idx 的SIMPACK模型是否稳定
def Check_SPCK_IsStable_Idx(WorkingDir, X_vars, tag, idx, TargetVel):
    """
    判断在当前参数列 (X_vars[:, idx]) 对应的速度、悬挂参数下，车辆是否会失稳。
    
    输入：
    ----------
    WorkingDir : str
        主工作目录，包含 "BatchTmp" 子文件夹，以及生成的 .spck/.subvar/.spf 文件等。
    tag : str
        标识符，用于拼装文件名，如 '0121A'。    
    X_vars : np.ndarray
        形状 (32, N) 的数组，每一列代表一组待仿真的参数；
        对应顺序: [ TargetVelocity, sprCpz, Kpx, Kpy, Kpz, Cpz, ... Lx1, Lx2, Lx3 ] 共 32 个。
    idx : int
        指定当前要仿真的列索引。
    TargetVel : float
        指定运行速度，以评价当前运行条件下的稳定性。
    返回：
    ----------
    is_stable : float
        稳定性判断结果：
            1.0  表示稳定（蛇行）运行
            0.2  表示失稳（横移超过阈值）
            0.1  表示SIMPACK仿真失败，未成功获得结果；仿真崩溃意味着不稳定
    """

    # =========== 1. 失稳阈值设置 ===========
    UnstableThreshold = 3.0 / 1000.0  # 3 mm

    # =========== 2. 解包 X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # 依照既定顺序解包
    TargetVelocity = TargetVel      # X_vars_col[0] # 使用 TargetVel 覆盖该速度取值
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

    # =========== 3. 生成 .subvar 文件 ===========
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

    # =========== 4. 调用 SIMPACK 仿真 ===========
    #   需要根据你在 prepare_SpckFiles_eachBatch 中生成的 .spck 文件名确定下列名称
    #   例如: Vehicle_Opt_{tag}_{idx}.spck
    # spck_name = f"Vehicle_Opt_{tag}_{idx}.spck"
    spck_name = f"Vehicle4WDB_RigidCriticalVel_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # 构建运行命令
    # 例如 "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # 执行命令
    status = run_simpack_cmd(cmd, WorkingDir, timeout_seconds = 10 * 60) # 10 * 60
    if status != 0:
        print(f"[ERROR] SIMPACK仿真失败，命令返回码: {status}")
        return 0.1
    else:
        # 仿真成功, 继续后处理 -> 读取最大横移量
        maxLatY = MaxLatY_idx(WorkingDir, tag, idx)
        
        # 与阈值比较
        if abs(maxLatY) >= UnstableThreshold:
            return 0.2  # 表示失稳
        else:
            return 1.0  # 表示稳定

# Python版本的二分搜索临界速度函数，对应于MATLAB的 Fun_HalfSearchCrticalVelocity
def HalfSearch_CrticalVelocity(
    WorkingDir,
    X_vars,
    tag,
    idx,
    StartVel,
    EndVel,
    N_depth
):
    """
    参数：
    ----------
    WorkingDir : str
        主工作目录路径。
    X_vars : np.ndarray
        形状 (32, N) 的数组，每一列代表一组待仿真的参数。
    tag : str
        用于拼装文件名，如 '0121A'。
    idx : int
        当前参数列索引，对应 X_vars[:, idx]。
    StartVel : float
        二分搜索初始左边界 (m/s)，如 50/3.6 。
    EndVel : float
        二分搜索初始右边界 (m/s)，如 612/3.6 。
    N_depth : int
        二分次数，如 6 或 7。

    返回：
    ----------
    critical_vel : float
        在 [StartVel, EndVel] 区间内，通过 N_depth 步二分搜索得到的近似临界速度 (单位：m/s)。
    """

    # 这里可根据需要打印一些提示信息
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
        is_stable = Check_SPCK_IsStable_Idx(
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