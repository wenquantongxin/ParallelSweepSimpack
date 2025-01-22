# -*- coding: gbk -*-

import os
import pandas as pd
import numpy as np
import shutil

# ��ȡ��������Ŀ¼(working directory)�µ�config_opt.xlsx�ļ���Sheet1����
def read_config_opt_excel(working_dir, excel_name="config_opt.xlsx", sheet_name="Sheet1"):
    """
    ������
    working_dir (str): ָ���Ĺ���Ŀ¼·��
    excel_name (str): excel�ļ����ƣ�Ĭ��Ϊ config_opt.xlsx
    sheet_name (str): ��Ҫ��ȡ�ı�����Ĭ��Ϊ Sheet1
    ���أ�
    DataFrame: pandas��DataFrame���ͣ�������ȡ�ı������
    """
    
    # 1. ���ù���Ŀ¼(working directory)
    # 2. ��ȡָ����Excel���ֻ��Sheet1��
    excel_path = os.path.join(working_dir, excel_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    return df

# ��� BatchTmp ���ļ���
def ClearBatchTmpFolder(WorkingDir):
    batch_tmp_path = os.path.join(WorkingDir, "BatchTmp")
    
    # ������ļ����Ѿ������Ҳ�Ϊ�գ���ɾ�����Լ�����������������
    if os.path.exists(batch_tmp_path):
        shutil.rmtree(batch_tmp_path)
    
    # �ٴ��½�һ���հ׵� BatchTmp �ļ��У����ں���д��
    os.makedirs(batch_tmp_path, exist_ok=True)
    
# ���� Base �ļ������Ʋ����� .spck, .subvar, .spf �ļ�
def prepare_SpckFiles_eachBatch(WorkingDir, tag, start_idx, end_idx):
    """
    �� [start_idx, end_idx) ��Χ�ڣ�Ϊÿ�� i ���������ļ��� BatchTmp �ļ��У�
      1. Vehicle_Opt_{tag}_{i}.spck
         - ���� Vehicle_Genera_OptBase.spck ���ƣ��޸ĵ� 26��61��69 ��
      2. subvars_Opt_{tag}_{i}.subvar
         - ���� subvars_OptBase.subvar ���ƣ����������޸�
      3. OptTargetResult_Opt_{tag}_{i}.spf
         - ���� OptTargetResult_OptBasel.spf ���ƣ��޸ĵ� 9 �� (������ 8)
    
    ����:
    ----------
    WorkingDir : str
        ����Ŀ¼·��, ���� .spck, .subvar, .spf ԭʼ�ļ�
    tag : str
        ����ַ���, ����ʶ��ͬʵ�黷��, �� "test", "runA" ��
    start_idx, end_idx : int
        ������Χ [start_idx, end_idx), ��ÿ�� i ���ɶ�Ӧ�� 3 ���ļ�
    """

    # =========== 1. ����ԭʼ�ļ�·�� ===========
    base_spck_path = os.path.join(WorkingDir, "Vehicle_Genera_OptBase.spck")
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
    base_spf_path = os.path.join(WorkingDir, "OptTargetResult_OptBasel.spf")
    
    # =========== 2. ��������ļ��� BatchTmp ===========
    batch_tmp_dir = os.path.join(WorkingDir, "BatchTmp")
    os.makedirs(batch_tmp_dir, exist_ok=True)

    # =========== 3. ��ȡԴ spck �� spf �ļ��� ===========
    #    (subvar �ļ�ֻ��Ҫ����, �����޸�)
    with open(base_spck_path, "r", encoding="utf-8") as f_spck:
        base_spck_lines = f_spck.readlines()

    with open(base_spf_path, "r", encoding="utf-8") as f_spf:
        base_spf_lines = f_spf.readlines()
    
    # =========== 4. ������� i, ���� 3 ��Ŀ���ļ� ===========
    for i in range(start_idx, end_idx):
        # -------- 4.1 ���� spck �ļ� -----------
        new_spck_name = f"Vehicle_Opt_{tag}_{i}.spck"
        new_spck_path = os.path.join(batch_tmp_dir, new_spck_name)

        lines_spck_mod = base_spck_lines.copy()
        # �޸ĵ� 26(����25), 61(����60), 69(����68) ��
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

        # д���µ� .spck �ļ�
        with open(new_spck_path, "w", encoding="utf-8") as f_out_spck:
            f_out_spck.writelines(lines_spck_mod)
        # print(f"[INFO] �������ļ�: {new_spck_path}")

        # -------- 4.2 ���� subvar �ļ� -----------
        new_subvar_name = f"subvars_Opt_{tag}_{i}.subvar"
        new_subvar_path = os.path.join(batch_tmp_dir, new_subvar_name)

        shutil.copyfile(base_subvar_path, new_subvar_path)
        # print(f"[INFO] �������ļ�: {new_subvar_path}")

        # -------- 4.3 ���� spf �ļ� -----------
        new_spf_name = f"OptTargetResult_Opt_{tag}_{i}.spf"
        new_spf_path = os.path.join(batch_tmp_dir, new_spf_name)

        lines_spf_mod = base_spf_lines.copy()
        # �޸ĵ� 9 ��(����8):
        #   <ResFile filename="Vehicle_4WDB_General.output/Vehicle_4WDB_General.sbr" ... >
        # �滻Ϊ
        #   <ResFile filename="Vehicle_Opt_{tag}_{i}.output/Vehicle_Opt_{tag}_{i}.sbr" ... >
        lines_spf_mod[8] = (
            f'<ResFile filename="Vehicle_Opt_{tag}_{i}.output/Vehicle_Opt_{tag}_{i}.sbr" '
            'generatorVersion="20210000" id="resf1" type="sbr"/>\n'
        )

        # д���µ� spf �ļ�
        with open(new_spf_path, "w", encoding="utf-8") as f_out_spf:
            f_out_spf.writelines(lines_spf_mod)
        # print(f"[INFO] �������ļ�: {new_spf_path}")
        
