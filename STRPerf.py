# -*- coding: gbk -*-

# ����ֱ�� AAR5 ��· ��������

import os
import time
import numpy as np
import pandas as pd
import subprocess
from FindCrticalVelocity import (Import_Subvars_To_File_idx)

# ��ȡָ�� .dat ����
# ���ش������ļ��л�õ� Sperling ָ��
def ReadAAR5Dat(dat_path):
    with open(dat_path, "r", encoding="utf-8") as f:
        
        # ����ǰ5��
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
        return (-99.0211, -99.0212) 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return (-99.0221, -99.0222)

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
        return (-99.0231, -99.0232) 
    except Exception as e:
        # �����쳣�����Ҳ�����ִ���ļ�������Ŀ¼�����ڵ�
        print(f"[ERROR] �޷�ִ��simpack-post����쳣��Ϣ��{e}")
        return (-99.0241, -99.0242) 
    time.sleep(2)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        print(f"[ERROR] �������ļ�δ�ҵ�: {dat_path}")
        return (-99.0251, -99.0252) 
      
    # 3) �����ļ�
    try:
        Sperling_Y_fromDat, Sperling_Z_fromDat = ReadAAR5Dat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        Sperling_Y = -99.0261
        Sperling_Z = -99.0262
    else:
        Sperling_Y = Sperling_Y_fromDat
        Sperling_Z = Sperling_Z_fromDat

    return Sperling_Y, Sperling_Z


def STRPerf_idx(WorkingDir, X_vars, tag, idx):

    print(f"����ģ�� {idx} ���е���ֱ����·�� Sperling ָ�����")
    
    # =========== 1. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = 80/3.6      # ֱ������ʱ������ 80 km/h �ٶ�ͨ�� AAR5 ֱ����·��ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ
    
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
    spck_name = f"Vehicle4WDB_RigidSTR80kmph_Opt_{tag}_{idx}.spck"   # ����: Vehicle4WDB_RigidSTR80kmph_Opt_0125_23
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
        return (-99.31, -99.32) # ���ϱ�Ƿ���ֵ
    time.sleep(1)
    
    # =========== 4. ��������ֵ ===========    
    # �����ֶԺ��������������
    filemidname = r"RigidSTR80kmph"
    SperlingY_AAR5, SperlingZ_AAR5 = STRSperling_idx(WorkingDir, filemidname, tag, idx)
  
    return (SperlingY_AAR5, SperlingZ_AAR5)

