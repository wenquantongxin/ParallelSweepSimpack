# -*- coding: gbk -*-

import os
import pandas as pd
import numpy as np
import shutil

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
    在 [start_idx, end_idx) 范围内，为每个 i 生成如下文件到 BatchTmp 文件夹：
      1. Vehicle_Opt_{tag}_{i}.spck
         - 基于 Vehicle_Genera_OptBase.spck 复制，修改第 26、61、69 行
      2. subvars_Opt_{tag}_{i}.subvar
         - 基于 subvars_OptBase.subvar 复制，内容无需修改
      3. OptTargetResult_Opt_{tag}_{i}.spf
         - 基于 OptTargetResult_OptBasel.spf 复制，修改第 9 行 (行索引 8)
    
    参数:
    ----------
    WorkingDir : str
        工作目录路径, 包含 .spck, .subvar, .spf 原始文件
    tag : str
        标记字符串, 用于识别不同实验环境, 如 "test", "runA" 等
    start_idx, end_idx : int
        索引范围 [start_idx, end_idx), 对每个 i 生成对应的 3 个文件
    """

    # =========== 1. 定义原始文件路径 ===========
    base_spck_path = os.path.join(WorkingDir, "Vehicle_Genera_OptBase.spck")
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
    base_spf_path = os.path.join(WorkingDir, "OptTargetResult_OptBasel.spf")
    
    # =========== 2. 创建输出文件夹 BatchTmp ===========
    batch_tmp_dir = os.path.join(WorkingDir, "BatchTmp")
    os.makedirs(batch_tmp_dir, exist_ok=True)

    # =========== 3. 读取源 spck 和 spf 文件行 ===========
    #    (subvar 文件只需要复制, 无需修改)
    with open(base_spck_path, "r", encoding="utf-8") as f_spck:
        base_spck_lines = f_spck.readlines()

    with open(base_spf_path, "r", encoding="utf-8") as f_spf:
        base_spf_lines = f_spf.readlines()
    
    # =========== 4. 逐个索引 i, 生成 3 个目标文件 ===========
    for i in range(start_idx, end_idx):
        # -------- 4.1 处理 spck 文件 -----------
        new_spck_name = f"Vehicle_Opt_{tag}_{i}.spck"
        new_spck_path = os.path.join(batch_tmp_dir, new_spck_name)

        lines_spck_mod = base_spck_lines.copy()
        # 修改第 26(索引25), 61(索引60), 69(索引68) 行
        lines_spck_mod[25] = (
            f"subvarset.file (          1                                       ) = "
            f"'./subvars_Opt_{tag}_{i}.subvar' ! subvarset filename\n"
        )
        lines_spck_mod[60] = (
            "substr.file (                       $S_IRWBogie_Front             ) = "
            "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
        )
        lines_spck_mod[68] = (
            "substr.file (                       $S_IRWBogie_Rear              ) = "
            "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
        )

        # 写出新的 .spck 文件
        with open(new_spck_path, "w", encoding="utf-8") as f_out_spck:
            f_out_spck.writelines(lines_spck_mod)
        # print(f"[INFO] 已生成文件: {new_spck_path}")

        # -------- 4.2 复制 subvar 文件 -----------
        new_subvar_name = f"subvars_Opt_{tag}_{i}.subvar"
        new_subvar_path = os.path.join(batch_tmp_dir, new_subvar_name)

        shutil.copyfile(base_subvar_path, new_subvar_path)
        # print(f"[INFO] 已生成文件: {new_subvar_path}")

        # -------- 4.3 处理 spf 文件 -----------
        new_spf_name = f"OptTargetResult_Opt_{tag}_{i}.spf"
        new_spf_path = os.path.join(batch_tmp_dir, new_spf_name)

        lines_spf_mod = base_spf_lines.copy()
        # 修改第 9 行(索引8):
        #   <ResFile filename="Vehicle_4WDB_General.output/Vehicle_4WDB_General.sbr" ... >
        # 替换为
        #   <ResFile filename="Vehicle_Opt_{tag}_{i}.output/Vehicle_Opt_{tag}_{i}.sbr" ... >
        lines_spf_mod[8] = (
            f'<ResFile filename="Vehicle_Opt_{tag}_{i}.output/Vehicle_Opt_{tag}_{i}.sbr" '
            'generatorVersion="20210000" id="resf1" type="sbr"/>\n'
        )

        # 写出新的 spf 文件
        with open(new_spf_path, "w", encoding="utf-8") as f_out_spf:
            f_out_spf.writelines(lines_spf_mod)
        # print(f"[INFO] 已生成文件: {new_spf_path}")
        
