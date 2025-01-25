# -*- coding: gbk -*-

import os
import pandas as pd
import numpy as np
import shutil
import time

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
    �� [start_idx, end_idx) ��Χ�ڣ�Ϊÿ�� i��
    ��Զ�� base_spck / base_spf �ļ������ɶ�Ӧ��Ŀ���ļ��� BatchTmp �ļ��С�
    
    ÿ�� (base_spck, base_spf) �ļ�Ҫ���� 3 ���ļ���
      1. ��Ӧ spck �ļ�
         - ���� base_spck ���ƣ��޸ĵ� 26 (����25)��61 (����60)��69 (����68) ��
      2. subvar �ļ�
         - ���� subvars_OptBase.subvar ���ƣ����κ������޸ģ�
      3. ��Ӧ spf �ļ�
         - ���� base_spf ���ƣ��޸ĵ� 9 �� (���� 8)
    
    ����:
    ----------
    WorkingDir : str
        ����Ŀ¼·��, ���� .spck, .subvar, .spf ԭʼ�ļ�
    tag : str
        ����ַ���, ����ʶ��ͬʵ�黷��, �� "test", "runA" ��
    start_idx, end_idx : int
        ������Χ [start_idx, end_idx), ��ÿ�� i ���ɶ�Ӧ�� 3 ���ļ�
    """
    
    # =========== 1. ����̶��� subvar ԭʼ�ļ�·�� ===========
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
    
    # base_spck1_path = os.path.join(WorkingDir, "Vehicle4WDB_IRWCRV300m_OptBase.spck")
    # base_spf1_path = os.path.join(WorkingDir, "Result_IRWCRV300m.spf")
    # base_spck2_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidCRV300m_OptBase.spck")
    # base_spf2_path = os.path.join(WorkingDir, "Result_RigidCRV300m.spf")
    # base_spck3_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidSTR80kmph_OptBase.spck")
    # base_spf3_path = os.path.join(WorkingDir, "Result_RigidSTR80kmph.spf")    
    # base_spck4_path = os.path.join(WorkingDir, "Vehicle4WDB_RigidCriticalVel_OptBase.spck")
    # base_spf4_path = os.path.join(WorkingDir, "Result_RigidCriticalVel.spf")
    
    # =========== 2. ��������Ҫ����� (spck, spf) �ļ������б��� ===========
    #    Ϊ����ά��������������������ɾ����Ҫ����� Base �ļ���
    #    �б���ÿ��Ԫ����һ��Ԫ�飺
    #    ( base_spck_·��, base_spf_·��, ������ļ��õ�ǰ׺�� �� �����Զ�����Ϣ... )
    spck_spf_list = [
        (
            os.path.join(WorkingDir, "Vehicle4WDB_IRWCRV300m_OptBase.spck"),
            os.path.join(WorkingDir, "Result_IRWCRV300m.spf"),
            "Vehicle4WDB_IRWCRV300m_Opt",    # ����������� spck �ļ�ǰ׺
            "Result_IRWCRV300m_Opt",        # ����������� spf �ļ�ǰ׺
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
    
    # =========== 3. ��������ļ��� BatchTmp ===========
    batch_tmp_dir = os.path.join(WorkingDir, "BatchTmp")
    os.makedirs(batch_tmp_dir, exist_ok=True)
    
    # =========== 4. �� spck_spf_list �е�ÿһ�Խ��д��� ===========
    for base_spck_path, base_spf_path, spck_prefix, spf_prefix in spck_spf_list:
        # --- 4.1 ��ȡԴ spck �� spf �ļ�����(��) ---
        with open(base_spck_path, "r", encoding="utf-8") as f_spck:
            base_spck_lines = f_spck.readlines()
        with open(base_spf_path, "r", encoding="utf-8") as f_spf:
            base_spf_lines = f_spf.readlines()
        
        # --- 4.2 �� [start_idx, end_idx) ��Χ��, ��� i ����Ŀ���ļ� ---
        for i in range(start_idx, end_idx):
            
            # =========== 4.2.1 ���� spck �ļ� ===========
            # ����ļ���ʾ��: Vehicle4WDB_IRWCRV300m_Opt_test_0.spck
            new_spck_name = f"{spck_prefix}_{tag}_{i}.spck"
            new_spck_path = os.path.join(batch_tmp_dir, new_spck_name)
            
            lines_spck_mod = base_spck_lines.copy()
            # �޸ĵ� 26(����25), 61(����60), 69(����68) ��
            # 1) �� 26 ��(���� 25)ָ�� subvar �ļ�����
            lines_spck_mod[25] = (
                f"subvarset.file (          1                                       ) = "
                f"'./subvars_Opt_{tag}_{i}.subvar' ! subvarset filename\n"
            )
            # 2) �� 61 ��(���� 60)
            lines_spck_mod[60] = (
                "substr.file (                       $S_IRWBogie_Front             ) = "
                "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
            )
            # 3) �� 69 ��(���� 68)
            lines_spck_mod[68] = (
                "substr.file (                       $S_IRWBogie_Rear              ) = "
                "'../ref_files/Bogie_IRWs_4WDBv3.spck' ! Filename\n"
            )
            
            # д���µ� .spck �ļ�
            with open(new_spck_path, "w", encoding="utf-8") as f_out_spck:
                f_out_spck.writelines(lines_spck_mod)
            
            # =========== 4.2.2 ���� subvar �ļ� ===========
            # �ļ�����subvars_Opt_test_0.subvar
            new_subvar_name = f"subvars_Opt_{tag}_{i}.subvar"
            new_subvar_path = os.path.join(batch_tmp_dir, new_subvar_name)
            shutil.copyfile(base_subvar_path, new_subvar_path)
            
            # =========== 4.2.3 ���� spf �ļ� ===========
            # ����ļ���ʾ��: Result_IRWCRV300m_Opt_test_0.spf
            new_spf_name = f"{spf_prefix}_{tag}_{i}.spf"
            new_spf_path = os.path.join(batch_tmp_dir, new_spf_name)
            
            lines_spf_mod = base_spf_lines.copy()
            # �޸ĵ� 9 ��(����8)
            lines_spf_mod[8] = (
                f'<ResFile filename="{spck_prefix}_{tag}_{i}.output/{spck_prefix}_{tag}_{i}.sbr" '
                'generatorVersion="20210000" id="resf1" type="sbr"/>\n'
            )
            
            # д���µ� spf �ļ�
            with open(new_spf_path, "w", encoding="utf-8") as f_out_spf:
                f_out_spf.writelines(lines_spf_mod)
            
            # print(f"[INFO] �������ļ�: {new_spck_name}, {new_subvar_name}, {new_spf_name}")
    
    print("[INFO] �����������ļ���������ϣ�")
    time.sleep(1)

