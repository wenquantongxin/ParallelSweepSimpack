# -*- coding: gbk -*-

import os
import pandas as pd
import numpy as np
import shutil
import time

# 读取给定工作目录(working directory)下的config_opt.xlsx文件的Sheet1内容
def read_config_opt_excel(working_dir, excel_name="config_opt.xlsx", sheet_name="Sheet1"):
    """
    参数：
    working_dir (str): 指定的工作目录路径
    excel_name (str): excel文件名称，默认为 config_opt.xlsx
    sheet_name (str): 需要读取的表名，默认为 Sheet1
    返回：
    DataFrame: pandas的DataFrame类型，包含读取的表格内容
    """
    
    # 1. 设置工作目录(working directory)
    # 2. 读取指定的Excel表格（只读Sheet1）
    excel_path = os.path.join(working_dir, excel_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    return df

# 清空 BatchTmp 子文件夹
def ClearBatchTmpFolder(WorkingDir):
    batch_tmp_path = os.path.join(WorkingDir, "BatchTmp")
    
    # 如果子文件夹已经存在且不为空，先删除它以及它包含的所有内容
    if os.path.exists(batch_tmp_path):
        shutil.rmtree(batch_tmp_path)
    
    # 再次新建一个空白的 BatchTmp 文件夹，便于后续写入
    os.makedirs(batch_tmp_path, exist_ok=True)
    
# 根据 Base 文件，复制并生成 .spck, .subvar, .spf 文件
def prepare_SpckFiles_eachBatch(WorkingDir, tag, start_idx, end_idx):
    """
    在 [start_idx, end_idx) 范围内，为每个 i，
    针对多个 base_spck / base_spf 文件，生成对应的目标文件到 BatchTmp 文件夹。
    
    每对 (base_spck, base_spf) 文件要生成 3 个文件：
      1. 对应 spck 文件
         - 基于 base_spck 复制，修改第 26 (索引25)、61 (索引60)、69 (索引68) 行
      2. subvar 文件
         - 基于 subvars_OptBase.subvar 复制（无任何内容修改）
      3. 对应 spf 文件
         - 基于 base_spf 复制，修改第 9 行 (索引 8)
    
    参数:
    ----------
    WorkingDir : str
        工作目录路径, 包含 .spck, .subvar, .spf 原始文件
    tag : str
        标记字符串, 用于识别不同实验环境, 如 "test", "runA" 等
    start_idx, end_idx : int
        索引范围 [start_idx, end_idx), 对每个 i 生成对应的 3 个文件
    """
    
    # =========== 1. 定义固定的 subvar 原始文件路径 ===========
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
    
    # base_spck1_path = os.path.join(WorkingDir, "Vehicle4WDB_IRWCRV300m_OptBase.spck")
    # base_spf1_path = os.path.join(WorkingDir, "Result_IRWCRV300m.spf")
    # base_spck2_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidCRV300m_OptBase.spck")
    # base_spf2_path = os.path.join(WorkingDir, "Result_RigidCRV300m.spf")
    # base_spck3_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidSTR80kmph_OptBase.spck")
    # base_spf3_path = os.path.join(WorkingDir, "Result_RigidSTR80kmph.spf")    
    # base_spck4_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidCriticalVel_OptBase.spck")
    # base_spf4_path = os.path.join(WorkingDir, "Result_RigidCriticalVel.spf")
    
    # =========== 2. 将所有需要处理的 (spck, spf) 文件放在列表中 ===========
    #    为方便维护，可以在这里新增或删减需要处理的 Base 文件。
    #    列表中每个元素是一个元组：
    #    ( base_spck_路径, base_spf_路径, 给输出文件用的前缀名 或 其它自定义信息... )
    spck_spf_list = [
        (
            os.path.join(WorkingDir, "Vehicle4WDB_IRWCRV300m_OptBase.spck"),
            os.path.join(WorkingDir, "Result_IRWCRV300m.spf"),
            "Vehicle4WDB_IRWCRV300m_Opt",    # 用于命名输出 spck 文件前缀
            "Result_IRWCRV300m_Opt",        # 用于命名输出 spf 文件前缀
        ),
        (
            os.path.join(WorkingDir, "Vehicle4WDB_RigidCRV300m_OptBase.spck"),
            os.path.join(WorkingDir, "Result_RigidCRV300m.spf"),
            "Vehicle4WDB_RigidCRV300m_Opt",
            "Result_RigidCRV300m_Opt",
        ),
        (
            os.path.join(WorkingDir, "Vehicle4WDB_RigidSTR80kmph_OptBase.spck"),
            os.path.join(WorkingDir, "Result_RigidSTR80kmph.spf"),
            "Vehicle4WDB_RigidSTR80kmph_Opt",
            "Result_RigidSTR80kmph_Opt",
        ),
        (
            os.path.join(WorkingDir, "Vehicle4WDB_RigidCriticalVel_OptBase.spck"),
            os.path.join(WorkingDir, "Result_RigidCriticalVel.spf"),
            "Vehicle4WDB_RigidCriticalVel_Opt",
            "Result_RigidCriticalVel_Opt",
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
            # 输出文件名示例: Vehicle4WDB_IRWCRV300m_Opt_test_0.spck
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
                "substr.file (                       $S_IRWBogie_Front             ) = "
                "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
            )
            # 3) 第 69 行(索引 68)
            lines_spck_mod[68] = (
                "substr.file (                       $S_IRWBogie_Rear              ) = "
                "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
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
            # 输出文件名示例: Result_IRWCRV300m_Opt_test_0.spf
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
    
    print("[INFO] 本批次所有文件已生成完毕！")
    time.sleep(1)

