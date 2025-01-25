# -*- coding: gbk -*-

# ������������

import os
import time
import numpy as np
import pandas as pd
import subprocess
from FindCrticalVelocity import (Import_Subvars_To_File_idx)

# ��ȡָ�� .dat ����
# ���ش������ļ��л�õ�ĥ��������������
def ReadCRVDat(dat_path):
    
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

# ���߳��ڼ�������ͨ����ĥ��������������
def CRVCal_idx(
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
        return (-99.0111, -99.0112) 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return (-99.0121, -99.0122)

    # 1) ���� simpack-post �Ľű� .qs
    # BatchTmp ���ļ����ڣ���������ִ��: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCRV300m_Opt_0125_0.spf DatResult_RigidCRV300m_0125_0    
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF �ļ�·��
        out_result_full_prefix  # ���ǰ׺
    ]
    
    try:
        ret = subprocess.run(cmd, cwd=work_dir, check=True)
    except subprocess.CalledProcessError as e:
        # �ⲿ����ط� 0
        print(f"[ERROR] simpack-post�������������={e.returncode}")
        return (-99.0131, -99.0132)
    except Exception as e:
        # �����쳣�����Ҳ�����ִ���ļ�������Ŀ¼�����ڵ�
        print(f"[ERROR] �޷�ִ��simpack-post����쳣��Ϣ��{e}")
        return (-99.0141, -99.0142)
    time.sleep(2)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] �������ļ�δ�ҵ�: {dat_path}")
        return (-99.0151, -99.0152) 
      
    # 3) �����ļ�
    try:
        SumWearNumber_CRV_fromDat, maxLatDisp_CRV_fromDat = ReadCRVDat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        SumWearNumber_CRV = -99.025
        maxLatDisp_CRV = -99.026
    else:
        SumWearNumber_CRV = SumWearNumber_CRV_fromDat
        maxLatDisp_CRV = maxLatDisp_CRV_fromDat
        
    # �����ϲ����� CRVPerf_idx
    # ע���� CRVCal_idx �������Ϸ������ά�ȣ�Ӧ�� CRVCal_idx ������ return ��ͬ
    return SumWearNumber_CRV, maxLatDisp_CRV

def CRVPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"����ģ�� {idx} ����������·ͨ�����ܲ���")
    
    # =========== 1. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = 60/3.6      # ��������ʱ������ 60 km/h �ٶ�ͨ�� R300 ���ߣ�ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ
    
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

    # =========== 2. ���� .subvar �ļ� ===========
    #   ���� Import_Subvars_To_File_idx(...)
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

    # =========== 3.1 ���� SIMPACK ����  ===========
    # ===========      �����ֶ�ģ��      ===========
    spck_name = f"Vehicle4WDB_RigidCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # ִ������
    try:
        ret = subprocess.run(cmd, cwd=WorkingDir)
        status = ret.returncode
        # �����Ҫ�鿴����� ret.stdout, ret.stderr
    except Exception as e:
        # ��������쳣������������ִ�д���
        print(f"[ERROR] SIMPACK������ó����쳣: {e}")
        return (-99.11, -99.12, -99.13, -99.14) # ���ϱ�Ƿ���ֵ
    time.sleep(1)
    
    # =========== 3.2 ���� SIMPACK ����  ===========
    # ===========      �����ֶ�ģ��      ===========  
    
    # ע������ļ��������𣬺˶Է���ģ���ļ�
    spck_name = f"Vehicle4WDB_IRWCRV300m_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # ִ������
    try:
        ret = subprocess.run(cmd, cwd=WorkingDir)
        status = ret.returncode
    except Exception as e:
        # ��������쳣������������ִ�д���
        print(f"[ERROR] SIMPACK������ó����쳣: {e}")
        return (-99.21, -99.22, -99.23, -99.24) # ���ϱ�Ƿ���ֵ
    time.sleep(1)    
    
    # =========== 4. ��������ֵ ===========    
    # �����ֶԺ��������������
    filemidname = r"RigidCRV300m"
    SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV = CRVCal_idx(WorkingDir, filemidname, tag, idx)
    
    # �����ֶԺ��������������
    filemidname = r"IRWCRV300m"
    SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = CRVCal_idx(WorkingDir, filemidname, tag, idx)

    # WearNumber_CRV = 0.0
    # LatDispMax_CRV = 0.0
    
    return (SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV)
    # ע���� CRVPerf_idx �������Ϸ������ά�ȣ�Ӧ�� CRVPerf_idx ������ return ��ͬ


