# -*- coding: gbk -*-

import os
import time
import numpy as np
import pandas as pd
import subprocess

# Ϊÿ�� idx �� subvar �ļ��������
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
    ��Ӧ��MATLAB�� Fun_GenerateSubvarsFile ������
    �����ļ��� BatchTmp �£����ݴ���� tag, idx ȷ��Ҫд���Ŀ�� .subvar �ļ����ƣ�
    Ȼ�󸲸�д��Simpack�ⲿ���������ļ����ݡ�

    ����:
    ----------
    WorkingDir : str
        ������Ŀ¼������ BatchTmp ���ļ���
    tag : str
        �������λ����黷���ı�ʶ��
    idx : int
        ��ϻ�ģ�ͱ��
    �������:
        ��ԭMATLAB�����е��ӱ���һһ��Ӧ (TargetVelocity, sprCpz, Kpx, ... , Lx1, Lx2, Lx3)��
"""
    # ������Ҫд����ӱ����ļ�·��
    # ����:  .../BatchTmp/subvars_Opt_{tag}_{idx}.subvar
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

# ��ȡָ�� .dat ����
# ���ش������ļ��л�ü����ٽ��ٶ�ʱ����ǰ�ٶ��µ���������
def ReadCriticalVelDat(dat_path):
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

# ִ�� simpack-cmd ���� simpack-slv ���� simpack-qs �ű�
def run_simpack_cmd (cmd, work_dir, timeout_seconds):
    """
    ���� simpack-post �������ָ��ʱ���ڼ����ִ�У���ʱ����ֹ���̡�

    ����:
    cmd : list
        ��Ҫִ�е�����������
    work_dir : str
        ����Ŀ¼·����ָ�� simpack-post ��Ҫ���е�Ŀ¼��
    timeout_seconds : int
        ���ִ��ʱ�䣨�룩������������г�ʱ��������ֹ����

    ����:
    result : int
        ���� 0 ��ʾ�ɹ�ִ�У�����ֵ��ʾ����
    """
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


# ���߳��ڼ��������ֶԵ���������
def MaxLatY_idx(
    work_dir,
    tag,
    idx,
    qs_script="SbrExport_SPCKResult.qs",
    wait_seconds = 3
):
    """
    Python�溯������Ӧ MATLAB �� Fun_maxLatY_fromSPCKpost��
    
    ����:
    ----------
    work_dir : str
        ������Ŀ¼�������ű� SbrExport_SPCKResult.qs��
    spf_filename : str
        SPF�ļ��������� "OptTargetResult_Opt_test_66.spf"����ʵ��λ�� work_dir/BatchTmp �¡�
    out_result_prefix : str
        ���ɽ����ǰ׺��"OptResult_test_66"����ʵ�� dat �ļ���д�� work_dir/BatchTmp/�¡�
    qs_script : str
        Ĭ�� "SbrExport_SPCKResult.qs"���� work_dir �¡�
    wait_seconds : int or float
        ����ִ�к�ȴ���������֤�ļ�д�롣
    """
    spf_filename = f"Result_RigidCriticalVel_Opt_{tag}_{idx}.spf"
    out_result_prefix = f"DatResult_RigidCriticalVel_{tag}_{idx}"
    # spf_filename = f"OptTargetResult_Opt_{tag}_{idx}.spf"
    # out_result_prefix = f"OptResult_{tag}_{idx}"
    
    # ƴ�� SPF �ļ��ľ���·��
    spf_path = os.path.join(work_dir, "BatchTmp", spf_filename)
    # ƴ�������Ľ��ǰ׺ (dat ���ջ��Ϊ "BatchTmp/out_result_prefix.dat")
    out_result_full_prefix = os.path.join(work_dir, "BatchTmp", out_result_prefix)
    # �ű�λ��
    qs_script_path = os.path.join(work_dir, qs_script)

    # ������Ҫ�ɼ���ļ�����
    if not os.path.isfile(qs_script_path):
        print(f"����ű�������: {qs_script_path}")
        return -99.0 
    if not os.path.isfile(spf_path):
        print(f".spf�ļ�������: {spf_path}")
        return -99.1 

    # 1) ���� simpack-post �Ľű� .qs
    # BatchTmp ���ļ����ڣ���������ִ��: simpack-post -s SbrExport_SPCKResult.qs Result_RigidCriticalVel_Opt_AAA_9.spf DatResult_RigidCriticalVel_AAA_9     
    cmd = [
        "simpack-post",
        "-s", qs_script_path,
        spf_path,               # SPF �ļ�·��
        out_result_full_prefix  # ���ǰ׺
    ]
    
    # ���ú���ִ��
    result = run_simpack_cmd(cmd, work_dir, timeout_seconds = 10 * 60) # 10 * 60
    if result != 0:
        print(f"����ʧ�ܣ������룺{result}")
        return -99.2 
    else:
        print(f"�ɹ�ִ�� qs �ű�����")

    
    # try:
    #     ret = subprocess.run(cmd, cwd=work_dir, check=True)
    # except subprocess.CalledProcessError as e:
    #     # �ⲿ����ط� 0
    #     print(f"[ERROR] simpack-post�������������={e.returncode}")
    #     
    # except Exception as e:
    #     # �����쳣�����Ҳ�����ִ���ļ�������Ŀ¼�����ڵ�
    #     print(f"[ERROR] �޷�ִ��simpack-post����쳣��Ϣ��{e}")
    #     return -99.3 
       
    time.sleep(wait_seconds)
    
    # 2) ƴ������ .dat �ļ�����·��
    dat_path = out_result_full_prefix + ".dat"
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"�������ļ�δ�ҵ�: {dat_path}")

    # 3) �����ļ�
    try:
        maxLatY_fromDat = ReadCriticalVelDat(dat_path)
    except Exception as e:
        print(f"[ERROR] ���� {dat_path} ʱ�����쳣: {e}")
        maxLatY = -99.4
    else:
        maxLatY = maxLatY_fromDat

    return maxLatY

# �жϱ��Ϊ idx ��SIMPACKģ���Ƿ��ȶ�
def Check_SPCK_IsStable_Idx(WorkingDir, X_vars, tag, idx, TargetVel):
    """
    �ж��ڵ�ǰ������ (X_vars[:, idx]) ��Ӧ���ٶȡ����Ҳ����£������Ƿ��ʧ�ȡ�
    
    ���룺
    ----------
    WorkingDir : str
        ������Ŀ¼������ "BatchTmp" ���ļ��У��Լ����ɵ� .spck/.subvar/.spf �ļ��ȡ�
    tag : str
        ��ʶ��������ƴװ�ļ������� '0121A'��    
    X_vars : np.ndarray
        ��״ (32, N) �����飬ÿһ�д���һ�������Ĳ�����
        ��Ӧ˳��: [ TargetVelocity, sprCpz, Kpx, Kpy, Kpz, Cpz, ... Lx1, Lx2, Lx3 ] �� 32 ����
    idx : int
        ָ����ǰҪ�������������
    TargetVel : float
        ָ�������ٶȣ������۵�ǰ���������µ��ȶ��ԡ�
    ���أ�
    ----------
    is_stable : float
        �ȶ����жϽ����
            1.0  ��ʾ�ȶ������У�����
            0.2  ��ʾʧ�ȣ����Ƴ�����ֵ��
            0.1  ��ʾSIMPACK����ʧ�ܣ�δ�ɹ���ý�������������ζ�Ų��ȶ�
    """

    # =========== 1. ʧ����ֵ���� ===========
    UnstableThreshold = 3.0 / 1000.0  # 3 mm

    # =========== 2. ��� X_vars[:, idx] ===========
    X_vars_col = X_vars[:, idx]
    # ���ռȶ�˳����
    TargetVelocity = TargetVel      # X_vars_col[0] # ʹ�� TargetVel ���Ǹ��ٶ�ȡֵ
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

    # =========== 3. ���� .subvar �ļ� ===========
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

    # =========== 4. ���� SIMPACK ���� ===========
    #   ��Ҫ�������� prepare_SpckFiles_eachBatch �����ɵ� .spck �ļ���ȷ����������
    #   ����: Vehicle_Opt_{tag}_{idx}.spck
    # spck_name = f"Vehicle_Opt_{tag}_{idx}.spck"
    spck_name = f"Vehicle4WDB_RigidCriticalVel_Opt_{tag}_{idx}.spck"
    spck_path = os.path.join(WorkingDir, "BatchTmp", spck_name)

    # ������������
    # ���� "simpack-slv.exe" + spck_path
    cmd = ["simpack-slv.exe", "--silent", spck_path]

    # ִ������
    status = run_simpack_cmd(cmd, WorkingDir, timeout_seconds = 10 * 60) # 10 * 60
    if status != 0:
        print(f"[ERROR] SIMPACK����ʧ�ܣ��������: {status}")
        return 0.1
    else:
        # ����ɹ�, �������� -> ��ȡ��������
        maxLatY = MaxLatY_idx(WorkingDir, tag, idx)
        
        # ����ֵ�Ƚ�
        if abs(maxLatY) >= UnstableThreshold:
            return 0.2  # ��ʾʧ��
        else:
            return 1.0  # ��ʾ�ȶ�

# Python�汾�Ķ��������ٽ��ٶȺ�������Ӧ��MATLAB�� Fun_HalfSearchCrticalVelocity
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
    ������
    ----------
    WorkingDir : str
        ������Ŀ¼·����
    X_vars : np.ndarray
        ��״ (32, N) �����飬ÿһ�д���һ�������Ĳ�����
    tag : str
        ����ƴװ�ļ������� '0121A'��
    idx : int
        ��ǰ��������������Ӧ X_vars[:, idx]��
    StartVel : float
        ����������ʼ��߽� (m/s)���� 50/3.6 ��
    EndVel : float
        ����������ʼ�ұ߽� (m/s)���� 612/3.6 ��
    N_depth : int
        ���ִ������� 6 �� 7��

    ���أ�
    ----------
    critical_vel : float
        �� [StartVel, EndVel] �����ڣ�ͨ�� N_depth �����������õ��Ľ����ٽ��ٶ� (��λ��m/s)��
    """

    # ����ɸ�����Ҫ��ӡһЩ��ʾ��Ϣ
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
        is_stable = Check_SPCK_IsStable_Idx(
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