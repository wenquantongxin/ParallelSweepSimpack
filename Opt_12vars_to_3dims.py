# -*- coding: gbk -*-

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
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

#############################################################################################################
##############################               ׼�������ļ�������                    ##########################
#############################################################################################################

# ��ȡ��������Ŀ¼(working directory)�µ�config_opt.xlsx�ļ���Sheet1����
def opt_ReadConfigExcel(working_dir, excel_name="config_opt.xlsx", sheet_name="Sheet1"):
    excel_path = os.path.join(working_dir, excel_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df

# ��� BatchTmp ���ļ���
def opt_ClearBatchTmpFolder(WorkingDir):
    batch_tmp_path = os.path.join(WorkingDir, "BatchTmp")
    if os.path.exists(batch_tmp_path):
        shutil.rmtree(batch_tmp_path)
    os.makedirs(batch_tmp_path, exist_ok=True)
    
# ���� Base �ļ������Ʋ����� .spck, .subvar, .spf �ļ�
def opt_PrepareSpckFilesForEachBatch(WorkingDir, tag, start_idx, end_idx):
    base_subvar_path = os.path.join(WorkingDir, "subvars_OptBase.subvar")
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
    
    print("[INFO] �������Σ������������С���Σ������ļ���������ϣ�")
    time.sleep(1)

# Ϊÿ�� idx �� subvar �ļ��������
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

    # ��дģʽ('w')���ļ���������ԭ������
    with open(subvar_path, 'w', encoding='utf-8') as f:
        # д��ͷ����Ϣ
        f.write("!file.version=3.5! Removing this line will make the file unreadable\n\n")
        f.write("!**********************************************************************\n")
        f.write("! SubVars\n")
        f.write("!**********************************************************************\n")

        # ����д�� subvar(...) ���
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

    # ��������ʱ��with �����Ļ��Զ��ر��ļ�
    print(f"[INFO] �Ѹ����ļ�: {subvar_path}")
    
#############################################################################################################
##############################              ��ȡ.dat���ݵĺ�����                    ##########################
#############################################################################################################

# ���ش������ļ��л�ü����ٽ��ٶ�ʱ����ǰ�ٶ��µ���������
def opt_ReadCriticalVelDat(dat_path):
    val_array = [0.0]*4
    with open(dat_path, "r", encoding="utf-8") as f:
        # ����ǰ5��
        for _ in range(5):
            f.readline()
        
        # ��ȡ�ؼ���
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

# ���ش������ļ��л�õ� Sperling ָ��
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

# ��ȡ���߷��� .dat �ļ����������ĥ������������
def opt_ReadCRVDat(dat_path):
    
    LatDisp_array = [0.0]*4
    
    with open(dat_path, "r", encoding="utf-8") as f:
        # ����ǰ5��
        for _ in range(5):
            f.readline()
        # ��ȡ�ؼ���
        line6 = f.readline()
        parts = line6.split(';')
        # ��¼��ĥ��������λ�� .dat �ļ��ĵ� 6 �е�һ���ֺ�֮��
        SumWearNumber_CRV_fromDat = float(parts[1].strip()) 
        # ��ʱΪ 5 + 1 = 6 ��
        
        for _ in range(39): 
            f.readline()
        # ��ʱΪ 6 + 39 = 45 ��
        
        line46 = f.readline()
        # ��ʱΪ 45 + 1 = 46 ��
        
        parts = line46.split(';')
        LatDisp_array[0] = float(parts[1].strip())
        
        for _ in range(4):
            f.readline()
        # ��ʱΪ 46 + 4 = 50 ��
        
        line51 = f.readline()
        # ��ʱΪ 50 + 1 = 51 ��
        
        parts = line51.split(';')
        LatDisp_array[1] = float(parts[1].strip())
        
        for _ in range(4):
            f.readline()
        # ��ʱΪ 51 + 4 = 55 ��
        
        line56 = f.readline()
        # ��ʱΪ 55 + 1 = 56 ��
        
        parts = line56.split(';')
        LatDisp_array[2] = float(parts[1].strip())

        for _ in range(4):
            f.readline()
        # ��ʱΪ 56 + 4 = 60 ��
        
        line61 = f.readline()
        # ��ʱΪ 60 + 1 = 61 ��
        
        parts = line61.split(';')
        LatDisp_array[3] = float(parts[1].strip())
            
    maxLatDisp_CRV_fromDat = max(LatDisp_array)
    
    return SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat

#############################################################################################################
##############################        �����е���Simpackǰ�������������          ##########################
#############################################################################################################

# ִ�� simpack-cmd ���� simpack-slv ���� simpack-qs �ű�
def opt_RunSPCKCmd (cmd, work_dir, timeout_seconds):
    
    try:
        # ʹ�� Popen ��������
        process = subprocess.Popen(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

         # �ȴ�������ɣ����ȴ� timeout_seconds ʱ��
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        # ���������ʱ�����������
        if process.returncode == 0:
            return 0
        else:
            # ��ӡ��׼�������
            print(f"[ERROR] simpack ִ��ʧ�ܣ�������={process.returncode}")
            return -99.5  # ���ش�����

    except subprocess.TimeoutExpired:
        # �����ʱ����ֹ����
        print("[ERROR] simpack ִ�г�ʱ����ֹ���̣�")
        process.terminate()
        process.wait()  # ȷ�����̱���ȫ��ֹ
        return -99.4  # ���س�ʱ������

    except Exception as e:
        print(f"[ERROR] ִ������ʧ��: {e}")
        return -99.3  # ��������

#############################################################################################################
##########################                ֱ���ߵ��̼߳�����(��������+����)           #####################
#############################################################################################################

# ���߳��ڼ��������ֶԵ���������
def opt_MaxLatY_idx(
    work_dir,
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs",
    wait_seconds = 3
):
    
    spf_filename = f"Result_RigidCriticalVel_Opt_{tag}_{idx}.spf"
    out_result_prefix = f"DatResult_RigidCriticalVel_{tag}_{idx}"
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    qs_script_path = os.path.join(work_dir, qs_script)
    # ������Ҫ�ɼ���ļ�����
    if not os.path.isfile(qs_script_path):
        print(f"����ű�������: {qs_script_path}")
        return +99.0 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return +99.1 
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF �ļ�·��
        out_result_full_prefix  # ���ǰ׺
    ]
    
    # ���ú���ִ��
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60) # 10 * 60
    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return +99.2 
    else:
        print(f"�ɹ�ִ�� qs �ű�����")
    time.sleep(wait_seconds)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"�������ļ�δ�ҵ�: {dat_path}")

    # 3) �����ļ�
    try:
        maxLatY_fromDat = opt_ReadCriticalVelDat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        maxLatY = -99.4
    else:
        maxLatY = maxLatY_fromDat

    return maxLatY

# ���߳��ڼ�������ͨ����ĥ��������������
def opt_CRVCal_idx(
    work_dir,
    filemidname, # �������ָ����ֶԻ��߶����ֶԣ�filemidname ����Ϊ IRWCRV300m ���� RigidCRV300m
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # ��Ӧ�� Result_IRWCRV300m_Opt_0125_0.spf ���� Result_RigidCRV300m_Opt_0125_0.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # ����� .dat �ļ�������
   
    # ƴ�� SPF �ļ��ľ���·��
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # �ű�λ��
    qs_script_path = os.path.join(work_dir, qs_script)

    # ������Ҫ�ɼ���ļ�����
    if not os.path.isfile(qs_script_path):
        print(f"����ű�������: {qs_script_path}")
        return (+999.0111, +999.0112) 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return (+999.0121, +999.0122)

    # 1) ���� simpack-post �Ľű� .qs
    # BatchTmp ���ļ����ڣ���������ִ��: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCRV300m_Opt_0125_0.spf DatResult_RigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF �ļ�·��
        out_result_full_prefix  # ���ǰ׺
    ]
    
    # ���ú���ִ��
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return (+999.0131, +999.0132)
    else:
        print(f"�ɹ�ִ�� qs �ű�����")
        print("����ִ�����")
    
    time.sleep(2)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] �������ļ�δ�ҵ�: {dat_path}")
        return (+999.0151, +999.0152) 
      
    # 3) �����ļ�
    try:
        SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat = opt_ReadCRVDat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        SumWearNumber_CRV = +9.025
        maxLatDisp_CRV = +9.026
    else:
        SumWearNumber_CRV = SumWearNumber_CRV_fromDat
        maxLatDisp_CRV = maxLatDisp_CRV_fromDat
        
    # �����ϲ����� CRVPerf_idx
    # ע���� CRVCal_idx �������Ϸ������ά�ȣ�Ӧ�� CRVCal_idx ������ return ��ͬ
    return SumWearNumber_CRV, maxLatDisp_CRV

# ���߳��ڼ���ֱ�� AAR5 ��·�� Sperling ����
def STRSperling_idx(
    work_dir,
    filemidname, # �������ָ����ֶԻ��߶����ֶԣ�filemidname ����Ϊ RigidSTR80kmph �� IRWSTR80kmph�����ޣ�
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs"
):
    
    spf_filename = f"Result_{filemidname}_Opt_{tag}_{idx}.spf" # ��Ӧ�� Result_RigidSTR80kmph_Opt_0125_20.spf
    out_result_prefix = f"DatResult_{filemidname}_{tag}_{idx}" # ����� .dat �ļ�������
   
    # ƴ�� SPF �ļ��ľ���·��
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # �ű�λ��
    qs_script_path = os.path.join(work_dir, qs_script)

    # ������Ҫ�ɼ���ļ�����
    if not os.path.isfile(qs_script_path):
        print(f"����ű�������: {qs_script_path}")
        return (+9.0211, +9.0212) 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return (+9.0221, +9.0222)

    # 1) ���� simpack-post �Ľű� .qs
    # BatchTmp ���ļ����ڣ���������ִ��: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCRV300m_Opt_0125_0.spf DatResult_RigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF �ļ�·��
        out_result_full_prefix  # ���ǰ׺
    ]
    
    # ���ú���ִ��
    result = opt_RunSPCKCmd(cmd, work_dir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return (+9.0231, +9.0232)
    else:
        print(f"�ɹ�ִ�� slv �� qs �ű�����")
        # print("����ִ�����")
    
    time.sleep(2)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] �������ļ�δ�ҵ�: {dat_path}")
        return (+9.0251, +9.0252) 
      
    # 3) �����ļ�
    try:
        Sperling_Y_fromDat, Sperling_Z_fromDat = opt_ReadAAR5Dat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        Sperling_Y = +9.0261
        Sperling_Z = +9.0262
    else:
        Sperling_Y = Sperling_Y_fromDat
        Sperling_Z = Sperling_Z_fromDat

    return Sperling_Y, Sperling_Z


#############################################################################################################
##########################           ֱ���ߵ��̼߳�����(��������+�������㺯������)          #####################
#############################################################################################################

# �����Ż�Ŀ�� - IRW ת�ٿ���ģ�͵�ĥ����
# ע���� CRVPerf_idx ��������
def opt_CRVPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"����Ѱ��B������ģ�� {idx} ��������ͨ����������")
    
    # =========== 1. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = 60/3.6      # ��������ʱ������ 60 km/h �ٶ�ͨ�� R300 ���ߣ�ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ
    
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

    # =========== 2. ���� .subvar �ļ� ===========

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

    # ===========  ���� SIMPACK ����  ===========
    # ===========      �����ֶ�ģ��      ===========  
    
    spck_name = f"Vehicle4WDB_IRWCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    # ���ú���ִ��
    result = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60)

    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return (+999.21, +999.22)
    else:
        print(f"�ɹ�ִ�� qs �ű�����")
        # print("����ִ�����")

    time.sleep(1)    
    
    # =========== 4. ��������ֵ ===========    

    # �����ֶԺ��������������
    filemidname = r"IRWCRV300m"
    SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = opt_CRVCal_idx(WorkingDir, filemidname, tag, idx)
        
    return (SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV)

# ֱ�� AAR5 ����
def opt_STRPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"����ģ�� {idx} ���е���ֱ����·�� Sperling ָ�����")
    
    # =========== 1. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = 80/3.6      # ֱ������ʱ������ 80 km/h �ٶ�ͨ�� AAR5 ֱ����·��ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ

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

    # =========== 2. ���� .subvar �ļ� ===========
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

    # =========== 3.1 ���� SIMPACK ����  ===========
    # ===========      �����ֶ�ģ��      ===========
    spck_name = f"Vehicle4WDB_RigidSTR80kmph_Opt_{tag}_{idx}.spck"   # ����: Vehicle4WDB_RigidSTR80kmph_Opt_0125_23
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]
    
    result = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60)
    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return (+99.31, +99.32)
    else:
        print(f"�ɹ�ִ�� qs �ű�����")
        # print("����ִ�����")
        
    time.sleep(1)

    # =========== 4. ��������ֵ ===========    
    # �����ֶԺ��������������
    filemidname = r"RigidSTR80kmph"
    SperlingY_AAR5, SperlingZ_AAR5 = STRSperling_idx(WorkingDir, filemidname, tag, idx)
  
    return (SperlingY_AAR5, SperlingZ_AAR5)

# �ٽ��ٶȼ�������Ŀ
# �жϱ��Ϊ idx ��SIMPACKģ���Ƿ��ȶ�
def opt_CheckStable_Idx(WorkingDir, X_vars, tag, idx, TargetVel):
    
    # =========== 1. ʧ����ֵ���� ===========
    UnstableThreshold = 3.0 / 1000.0  # 3 mm

    # =========== 2. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = TargetVel      # X_vars_col[0] # ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ
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

    # =========== 3. ���� .subvar �ļ� ===========
    #   ���� Import_Subvars_To_File_idx(...)
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

    # =========== 4. ���� SIMPACK ���� ===========
    spck_name = f"Vehicle4WDB_RigidCriticalVel_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # ִ������
    status = opt_RunSPCKCmd(cmd, WorkingDir, timeout_seconds = 10 * 60) # 10 * 60
    if status != 0:
        print(f"[ERROR] SIMPACK����ʧ�ܣ��������: {status}")
        return 0.1
    else:
        # ����ɹ�, �������� -> ��ȡ��������
        maxLatY = opt_MaxLatY_idx(WorkingDir, tag, idx)
        
        # ����ֵ�Ƚ�
        if abs(maxLatY) >= UnstableThreshold:
            return 0.2  # ��ʾʧ��
        else:
            return 1.0  # ��ʾ�ȶ�

# ���������ٽ��ٶȺ���
def opt_HalfSearchCrticalVelocity(
    WorkingDir,
    X_vars,
    tag,
    idx,
    StartVel,
    EndVel,
    N_depth
):
    
    print("��ʼ����������")
    # ������������������ӡ��������MATLAB
    print(f"  - tag: {tag}, idx: {idx}")
    print(f"  - ��ʼ�ٶȣ�{StartVel:.2f} m/s ({StartVel*3.6:.2f} km/h)")
    print(f"  - ��ֹ�ٶȣ�{EndVel:.2f} m/s ({EndVel*3.6:.2f} km/h)")
    print(f"  - ���ִ�����{N_depth}")
    print("-----------------------------------")

    low_vel = StartVel
    high_vel = EndVel

    for i_depth in range(1, N_depth + 1):
        mid_vel = 0.5 * (low_vel + high_vel)
        # �����ȶ����жϺ���
        is_stable = opt_CheckStable_Idx(
            WorkingDir=WorkingDir,
            X_vars=X_vars,
            tag=tag,
            idx=idx,
            TargetVel=mid_vel # ���� mid_vel ���в���
        )

        if is_stable == 1.0:
            # ������ mid_vel ���ȶ� => �ٽ��ٶȿ��ܸ���
            low_vel = mid_vel
            print(f"��� {i_depth}: {mid_vel:.2f} m/s �ȶ�, �������䵽 [{low_vel:.2f}, {high_vel:.2f}]")
        else:
            # ���� 0.1 �� 0.2������Ϊ���ȶ� => �ٽ��ٶ��� mid_vel ����
            high_vel = mid_vel
            print(f"ģ��{idx}: ������� {i_depth} �����ٶ�{mid_vel:.2f} m/sʱ���ȶ�, �������䵽 [{low_vel:.2f}, {high_vel:.2f}]")

    # ȡ�����������ֵ��Ϊ�����ٽ��ٶ�
    critical_vel = 0.5 * (low_vel + high_vel)

    print("-----------------------------------")
    print(f"�����������õ��ٽ��ٶ� �� {critical_vel:.2f} m/s ({critical_vel*3.6:.2f} km/h)\n")

    return critical_vel


################################################################
# 1) ���� worker��ֻ���� 4 ��ֵ�����һ�������ǹ��ĵĵ�����Ŀ��
################################################################

# ����������
def opt_parallel_worker(args):
    
    (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth) = args
    
    # ʵ���ϵ�ȫ��������
    actual_idx = start_idx + col_idx_in_batch

    # ����������
    # �������� 1�����ð����������������ٽ��ٶ�
    CrticalVelocity = opt_HalfSearchCrticalVelocity(WorkingDir, X_vars, tag, actual_idx, StartVel, EndVel, N_depth)
    time.sleep(1)

    # �������� 2���������߼���ģ�ͣ���������ĥ������������
    SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = opt_CRVPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # �������� 3�����õ��� AAR5 ֱ�߼���ģ�� ��������������Sperlingָ��
    SperlingY_AAR5, SperlingZ_AAR5 = opt_STRPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    try:
        SperlingYZ = math.sqrt(SperlingY_AAR5 ** 2 + SperlingZ_AAR5 ** 2)
    except Exception as e:
        print(f"[ERROR] SperlingYZָ������쳣���쳣��Ϣ��{e}")
        return +9.77
        
    
    time.sleep(1)
    # ���ز��м���� idx �Ľ��������
    # ����Ŀ��
    return (col_idx_in_batch, CrticalVelocity,  SumWearNumber_IRWCRV300m_CRV, SperlingYZ)



################################################################
# 2) ��װ���������С��߼��ĺ���
################################################################
def opt_XEvalPrl(X, WorkingDir, tag, StartVel, EndVel, N_depth, BatchSize_parallel=5):
    """
    X: shape=(N, 12)  # N����, ÿ����12������
    ����: shape=(N, 3) ��Ŀ�����F
          ����F[:, 0] = -cVel  (��Ҫ���cVel��ȡ��)
               F[:, 1] = IrwWN_CRV
               F[:, 2] = SperlingYZ
    """
    # ת�� pymoo �������Ĳ��������������������ƥ��
    X = X.T

    print(f"==== Debug: ת�ú�� X.shape = {X.shape}")
    print("==== Debug: ������ѡ��(��12��������������)��\n", X[:12]) 
    
    # N����
    N_opt = X.shape[1]
    
    # ÿ����Ҫ���� 3 ��Ŀ��(����� 3 �����뱣����ָ������)
    result_dim = 3 
    
    # ���ڴ�����н��Ŀ��ֵ(���� shape=(3, N) �棬���ת��)
    batch_result_full = np.zeros((result_dim, N_opt))

    num_batches = math.ceil(N_opt / BatchSize_parallel)
    print("�ܵĲ�������� = ��Ⱥ��������", N_opt)
    print("������������", BatchSize_parallel)
    print("����������", num_batches)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BatchSize_parallel
        end_idx   = min((batch_idx + 1) * BatchSize_parallel, N_opt)
        
        print(f"�� {batch_idx+1} / {num_batches} ������������Χ [{start_idx}:{end_idx})")
        
        # ============== ���� & �����ļ� ==============
        opt_ClearBatchTmpFolder(WorkingDir)
        print("�������ϸ�С���ε��ļ�")
        
        opt_PrepareSpckFilesForEachBatch(WorkingDir, tag, start_idx, end_idx)
        print("��׼���ñ�С���ε��ļ�")     
        
        # ȡ�����εĽ�(�����з�)
        X_batch = X[:, start_idx:end_idx]  # ������ X[start_idx:end_idx, :]

        # ����ݴ� (3�� x batch_size��)
        batch_result = np.zeros((result_dim, X_batch.shape[1]))

        # ============== (b) ���д��� ==============
        with concurrent.futures.ProcessPoolExecutor(max_workers=BatchSize_parallel) as executor:
            future_list = []
            for col_idx_in_batch in range(X_batch.shape[1]):
                args = (col_idx_in_batch, start_idx, WorkingDir, X, tag, StartVel, EndVel, N_depth)
                future = executor.submit(opt_parallel_worker, args)
                future_list.append(future)

            # �ռ����
            for future in concurrent.futures.as_completed(future_list):
                col_idx_in_batch, cVel, IrwWN_CRV, SperlingYZ = future.result()
                # ���г� return 
                batch_result[0, col_idx_in_batch] = cVel 
                batch_result[1, col_idx_in_batch] = IrwWN_CRV
                batch_result[2, col_idx_in_batch] = SperlingYZ
                
        # �������ν���ŵ�ȫ�� big array
        batch_size_actual = end_idx - start_idx
        batch_result_full[:, start_idx:end_idx] = batch_result[:, :batch_size_actual]
        
    # ���� batch_result_full.shape=(3, N), ת�ó� (N,3)
    # ���� cVel Խ��Խ�� => -cVel
    cVel_all = batch_result_full[0, :]
    IrwWN_CRV_all = batch_result_full[1, :]
    SperlingYZ_all = batch_result_full[2, :]

    # �������շ���F: (N,3)
    f1 = -cVel_all  # ��ΪҪ���cVel, pymooĬ����С��
    f2 = IrwWN_CRV_all
    # ע�⣡���� Sperling ָ�꣬�˴����Է���ϵ�� 100
    f3 = SperlingYZ_all * 100 

    F = np.vstack([f1, f2, f3]).T  # (3,N).T => (N,3)
    print("==== Debug: ���� pymoo ���к�ѡ�⣨�������С���Σ��Ż�Ŀ����� batch_result_full:\n ", batch_result_full) 
    print("Sperling ָ�꣬�˴�Ϊԭʼֵ��ʵ���Է���ϵ�� 100 �����Ż�����\n")
    
    return F

# �������ս��������������ʷ�����Ż��ļ�¼
def SaveItersResult(res, filename):
    """
    �� pymoo 'Result' ����Ĳ�����Ϣд�� npz �ļ�:
      - X, F, G, CV, history

    ע��:
      - ������ 'feasible' ����������, �Ա���ĳЩ�����±���.
      - 'history' �а���ÿһ���Ŀ���, �������ģ�ܴ�, �ļ�Ҳ���ܽϴ�.
    """
    data_dict = {
        "X"      : res.X,         # ��֧���(��Ŀ��)�����Ž�(��Ŀ��)
        "F"      : res.F,         # ��Ӧ��Ŀ��ֵ
        "G"      : res.G,         # ����ʽԼ��(����������� None)
        "CV"     : res.CV,        # Լ��Υ���� (Constraint Violation), ͬ������ None
        "history": res.history    # ��� save_history=True, ��������ÿ����ʷ
    }
    # ����Ϊ npz
    np.savez(filename, **data_dict)
    print(f"[SaveItersResult] �ѽ� X, F, G, CV, history ���浽 {filename}.")

# callback �����ص���������������е� X, F, G
def my_callback(algorithm, working_dir=None, **kwargs):
    """
    ��ÿһ�������󱻵���:
      - algorithm: ��ǰ�㷨����(���� NSGA2 instance)
      - n_gen: ��ǰ���� (��1��ʼ)
      - **kwargs: �������� (pymoo �ڲ���������)

    ���ｫ��ǰ��Ⱥ�� X, F, G �ȱ��浽 working_dir/ChkPnt �£�
    ������ȡ��֧��� (nd_X, nd_F, nd_G) ���Ᵽ��.
    """
    
    global history_F # ȫ���б����ڴ洢������ȺĿ��ֵ

    n_gen = algorithm.n_gen
    pop = algorithm.pop
    X = pop.get("X")  # shape=(pop_size, n_var)
    F = pop.get("F")  # shape=(pop_size, n_obj)
    G = pop.get("G")  # shape=(pop_size, n_ieq_constr) �� None
    history_F.append(F.copy())
    print(f"[Callback] �� {n_gen} ��, F_gen.shape = {F.shape}")
    
    # ׼�� res ���ʽ����Ŀ¼
    if working_dir is None:
        working_dir = os.getcwd() 
    chkpt_dir = os.path.join(working_dir, "ChkPnt")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir, exist_ok=True)

    # �� generation_{n_gen}.npz ��ʽ�������н�
    filename_all = os.path.join(chkpt_dir, f"generation_{n_gen}.npz")
    print(f"[Callback] ���ڱ���� {n_gen} ����������Ⱥ�� {filename_all}")
    np.savez(filename_all, X=X, F=F, G=G)

    # ============== ��ȡ������֧��Ⲣ�������� ==============
    nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_X = X[nd_front]
    nd_F = F[nd_front]
    nd_G = G[nd_front] if G is not None else None

    filename_nd = os.path.join(chkpt_dir, f"generation_{n_gen}_nondom.npz")
    print(f"[Callback] ���ڱ���� {n_gen} ���ķ�֧��⵽ {filename_nd} ...")
    np.savez(filename_nd, X=nd_X, F=nd_F, G=nd_G)

    print(f"[Callback] �� {n_gen} �����ݱ������, ��Ⱥ��ģ = {X.shape[0]}, ��֧������� = {len(nd_front)}")

class MyBatchProblem(Problem):
    def __init__(self,
                 batch_size=5,
                 WorkingDir=os.getcwd(),
                 tag="demo",
                 StartVel = 100/3.6,
                 EndVel = 900/3.6,
                 N_depth = 7,
                 **kwargs):
        """
        ʹ�� (N,12) ���߱���, 3 ��Ŀ��
        """
        super().__init__(
            n_var=12,
            n_obj=3,
            # 3��Լ��������n_ieq_constr=3
            n_ieq_constr = 3,  
            xl=np.array([2000, 160000, 120000, 24000, 30000, 4000, 1600000, 10000, 100,   0,    0,   -0.6]),
            xu=np.array([50000,4000000,3000000,600000,750000,100000,40000000,250000,3000000,0.64,0.64, 0.4]),
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
        # 1) ����Ŀ��ֵ F (N,3)
        F = opt_XEvalPrl(
            X,
            WorkingDir=self.WorkingDir,
            tag=self.tag,
            StartVel=self.StartVel,
            EndVel=self.EndVel,
            N_depth=self.N_depth,
            BatchSize_parallel=self.batch_size
        )
        
        # 2) ���㲻��ʽԼ�� G (N,2)
        # Լ��1: cVel >= 250/3.6 => -cVel <= -250/3.6 => f1 <= -250/3.6
        # f1 = -cVel =>  g1 = f1 - (-250/3.6) = f1 + 250/3.6 <= 0
        # Լ��2: f2 <= 1000 =>  g2 = f2 - 1000 <= 0
        # Լ��3: f3 <= 300  =>  g3 = f3 - 300 <= 0
        G = np.zeros((F.shape[0], 3))
        
        # ʹ f1 <= -250/3.6����Ӧ���ٽ��ٶȴ��� 250 km/h
        G[:, 0] = F[:, 0] + 250.0/3.6 
        # ʹ f2 <= 1000����Ӧ��ĥ����С�� 1000
        G[:, 1] = F[:, 1] - 1000.00 
        # ʹ f3 <= 300����Ӧ�� Sperling ָ��С�� 3
        G[:, 2] = F[:, 2] - math.sqrt(3 * 3 + 3 * 3) * 100
          
        out["F"] = F
        out["G"] = G

################################################################
# 4) �������
################################################################
def main():
    
    # ��ʷ��Ӧ�Ⱥ��� F
    global history_F
    history_F = []
    
    # ��������
    problem = MyBatchProblem( batch_size = 3)
    # �����㷨 (��Ŀ�� NSGA2)
    algorithm = NSGA2( pop_size = 5 )
    # ��ֹ����
    termination = get_termination("n_gen", 3)  # �Ŵ�����������������ʵ��
    # �����Ż�
    res = minimize(
        problem, 
        algorithm, 
        termination, 
        seed=1, 
        verbose=True, 
        save_history=True, 
        callback=my_callback)  # ����ص�����
    
    # �鿴���
    print("\n==== �Ż���� ====")
    print("��֧�������:", len(res.X))
    print("ʾ����X[0] :", res.X[0])
    print("��ӦĿ��F[0]:", res.F[0])
    print("Լ��G[0]:", res.G[0], "(ע: G[i] <= 0��ʾ����)")
    
    # �������(2�ַ�����csv �� npz)
    nd_X = res.X 
    nd_F = res.F

    np.savetxt("final_solutions.csv",
            np.hstack([nd_X, nd_F]),
            delimiter=",",
            comments="",  # ȥ��ע�ͷ���
            fmt="%.6f")   # �����������
    np.savez("final_solutions.npz", X=nd_X, F=nd_F)
    
    # res �Ľ������
    #  `res.X, res.F` => ���ս⣻history_F => (list of arrays)
    np.savez(
        "res_history.npz",
        final_X=res.X,
        final_F=res.F,
        history_F=np.array(history_F, dtype=object) # history_F ��һ���б�, ����ÿ������һ�� (pop_size, n_obj) ������
    )
    print("[Main] �ѱ��� final_X, final_F, �Լ����� F_gen �� res_history.npz.")

if __name__ == "__main__":
    main()
    
    """
    �����е��ã�
    
    F:  # �л��̷�                                                                                                             
    cd F:\ResearchMainStream\0.ResearchBySection\C.����ѧģ��\�����Ż�\�����Ż�ʵ��\ParallelSweepSimpack                        
    python Opt_12vars_to_3dims.py # ִ�б�����                    
    
    """    
    
    """
    ��¼A: MATLAB GA ������Ӧ����
    MaxGenerations (Generations) ���ܣ��㷨������������Ĭ�� ga �� 100����������gamultiobj �� 200����������
    PopulationSize ���ܣ�������Ⱥ��С����ÿһ���ĸ�������������������� �� 5��Ĭ�� 50������Ĭ�� 200��
    
    ��¼B: �ں���ʱ���鿴 Pareto Front �ֲ������ Analysis_FindOpts.ipynb

    """
    
    
    