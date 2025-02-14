# -*- coding: gbk -*-

# SweepLx1Lx2xL3x.py
import os
import pandas as pd
import numpy as np
import itertools
import math
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # ���� "TkAgg"
import shutil
import time

from PrepareBatchFiles import (
    read_config_opt_excel,
    ClearBatchTmpFolder,
    prepare_SpckFiles_eachBatch
)

from FindCrticalVelocity import (HalfSearch_CrticalVelocity)
from STRPerf import (STRPerf_idx)
from CRVPerf import (CRVPerf_idx)

# ����άͼ����ʾ �ٽ��ٶ��� L1��L2 �Ĺ�ϵ
def ShowMeshgrid():
    CriticalVel = np.load("myCriticalVel.npy") 
    Xvars = np.load("myXvars.npy") 
    Z = CriticalVel
    lx1_all = Xvars[29,:] 
    lx2_all = Xvars[30,:] 

    # 1) �ҵ�Ψһֵ������
    lx1_unique = np.unique(lx1_all)
    lx2_unique = np.unique(lx2_all)
    m = len(lx1_unique)
    n = len(lx2_unique)
    print("lx1_unique:", lx1_unique)
    print("lx2_unique:", lx2_unique)
    print("m,n =", m,n)
    # 2) �� meshgrid
    # ע�� meshgrid ��Ĭ�� (x, y) => X.shape = (n,m), Y.shape = (n,m)
    X, Y = np.meshgrid(lx1_unique, lx2_unique)
    print("X.shape = ", X.shape)  # (n,m)
    # 3) reshape Z
    Z_mat = Z.reshape(n, m)  # ��Ϊ n ��, m �� => shape=(len(lx2_unique), len(lx1_unique))
    # 4) �� surf
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z_mat, cmap='viridis')
    ax.set_xlabel("Lx1")
    ax.set_ylabel("Lx2")
    ax.invert_yaxis()  # ʹ�� Y �ᷴ�����
    ax.set_zlabel("Critical Velocity (m/s)")
    fig.colorbar(surf, shrink=0.5)
    plt.show()
    
# ����������
def parallel_worker(args):
    """
    ������������Ĳ���������������pickle����
    
    args ��һ��Ԫ�飬������
        (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth)
    �����ں����н����ִ�� HalfSearch_CrticalVelocity��
    ���� (col_idx_in_batch, cVel)���ⲿ���ɴ��˽���еĽ��
    """
    (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth) = args
    
    # ʵ���ϵ�ȫ��������
    actual_idx = start_idx + col_idx_in_batch

    # ����������
    # �������� 1�����ð����������������ٽ��ٶ�
    CrticalVelocity = HalfSearch_CrticalVelocity(WorkingDir, X_vars, tag, actual_idx, StartVel, EndVel, N_depth)
    time.sleep(1)
    # print("[INFO] ���ԣ���ʱ�����ٽ��ٶȼ���")  
    # CrticalVelocity = 666.66
    
    # �������� 2���������߼���ģ�ͣ���������ĥ������������
    SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV = CRVPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # �������� 3�����õ��� AAR5 ֱ�߼���ģ�� ��������������Sperlingָ��
    SperlingY_AAR5, SperlingZ_AAR5 = STRPerf_idx(WorkingDir, X_vars, tag, actual_idx)
    time.sleep(1)
    
    # ���ز��м���� idx �Ľ��������
    return (col_idx_in_batch, CrticalVelocity, SumWearNumber_RigidCRV300m_CRV, maxLatDisp_RigidCRV300m_CRV, SumWearNumber_IRWCRV300m_CRV, maxLatDisp_IRWCRV300m_CRV, SperlingY_AAR5, SperlingZ_AAR5)

# ���ɼ����������������������� X_vars
def GenerateVehicleParamater(WorkingDir, Filename, method="JustSweepL123"):
    
    if method == "JustSweepL123":
         
        # ʾ������: X_vars = GenerateVehicleParamater(WorkingDir, Filename="config_opt.xlsx", method="JustSweepL123")
        
        print(f"[INFO] ����ɨ�� L1��L2��L3 ��������, �������������������׼ֵ��ͬ") # ɨ�� L1��L2��L3
        
        # 1) ���ú�����ȡ config_opt.xlsx����ȡ��׼ֵ X_base
        Opt_Config = read_config_opt_excel(WorkingDir)
        X_base = Opt_Config["��׼ֵ"].to_list()
        # print("��׼ֵ X_base:", X_base)

        # 2) ���� X_vars (32 x N_combos)

        Lx1_sweep = np.arange(0, 0.64 + 0.001, 0.04)  
        Lx2_sweep = np.arange(0, 0.60 + 0.001, 0.04)  
        Lx3_sweep = np.arange(-0.6, 0.40 + 0.001, 0.1) 

        Lx123_combinations = list(itertools.product(Lx1_sweep, Lx2_sweep, Lx3_sweep))
        X_vars_columns = []

        for (lx1, lx2, lx3) in Lx123_combinations:
            x_temp = X_base.copy()
            x_temp[29] = lx1 # �� 0 ��ʼ���
            x_temp[30] = lx2
            x_temp[31] = lx3
            X_vars_columns.append(x_temp)

        X_vars = np.column_stack(X_vars_columns)
        print("����ɨ�Լ���� X_vars����״: ", X_vars.shape)
        
    elif method == "FromExcel":
        
        # ʾ������: X_vars = GenerateVehicleParamater(WorkingDir, Filename="ParameterSweep_fromExcel.xlsx", method="FromExcel")
        
        print(f"[INFO] �� EXCEL ����л�ȡ��������, �ñ���¼��ǰ�ؽ��Ӧ�ı��Ż�����")
        print("Ԫ�ļ�����:", Filename)
        # ��ȡ config_opt.xlsx ��ȡ��׼ֵ X_base ���Ƿ��Ż��ı�־ is2opt
        Opt_Config = read_config_opt_excel(WorkingDir, excel_name="config_opt.xlsx")
        X_base = Opt_Config["��׼ֵ"].to_list()
        is2opt = Opt_Config["�Ƿ��Ż�"].to_list()

        # ��ȡ�ⲿ�ļ� ParameterSweep_fromExcel.xlsx
        Param_Sweep_Config = pd.read_excel(f"{WorkingDir}/{Filename}", sheet_name="Sheet1", header=None)
        N_Xvars = Param_Sweep_Config.shape[1] - 1 # ��Excel ����ж�ȡ�����ж����� X_vars ��Ҫ������
        ChangingVars = Param_Sweep_Config.iloc[..., 1: Param_Sweep_Config.shape[1]]

        N_2opt = len(is2opt) # X_vars �ж���ά��, Ӧ����32
                
        # �����ⲿexcel������ʼ�� X_vars
        X_vars = np.zeros((N_2opt, N_Xvars)) 

        optCount = 0
        for LineId in range(0, N_2opt):
            if is2opt[LineId] == 0:
                X_vars[LineId] = X_base[LineId] * np.ones((1, N_Xvars))
            elif is2opt[LineId] == 1:
                X_vars[LineId] = ChangingVars.iloc[optCount, ].to_numpy()
                optCount = optCount + 1 
        # �ر����  $_Kpy	һϵ���Ҹն�-������3��ϣ� $_Ksy	��ϵ���Ҹն�-������7���
        X_vars[3]  = X_vars[2]
        X_vars[7]  = X_vars[6]

        print("����ɨ�Լ���� X_vars����״: ", X_vars.shape)  # ��ӡ X_vars ����״
        
    elif method == "FromNPZ":
                
        # ʾ������ 1: X_vars = GenerateVehicleParamater(WorkingDir, Filename="res_history.npz", method="FromNPZ")
        # ʾ������ 2: X_vars = GenerateVehicleParamater(WorkingDir, Filename="generation_150_nondom.npz", method="FromNPZ")
        
        # print(f"[INFO] �� res_history.npz ���� generation_i_nondom.npz �л�ȡ��������, �� .npz �ļ���¼��ǰ�ؽ�� final_X")
        data = np.load(Filename, allow_pickle=True)
        
        try:
            # ���Լ���final_X
            final_X = data["final_X"].T
            print(f"[INFO] �� res_history.npz �л�ȡ��������, �� .npz �ļ���¼������ǰ�ؽ�� final_X")
            print("Ԫ�ļ�����:", Filename)
        except KeyError:
            # ���û��final_X�����Լ���X
            final_X = data["X"].T
            print(f"[INFO] �� generation_i_nondom.npz �л�ȡ��������, �� .npz �ļ���¼��ĳһ����ǰ�ؽ�� final_X")
            print("Ԫ�ļ�����:", Filename)

        Opt_Config = read_config_opt_excel(WorkingDir, excel_name="config_opt.xlsx")
        X_base = Opt_Config["��׼ֵ"].to_list()
        is2opt = Opt_Config["�Ƿ��Ż�"].to_list()

        N_Xvars = len(final_X[0])
        N_2opt = len(is2opt) # X_vars �ж���ά��, Ӧ����32
                        
        # ��ʼ�� X_vars
        X_vars = np.zeros((N_2opt, N_Xvars)) 

        optCount = 0
        for LineId in range(0, N_2opt):
            if is2opt[LineId] == 0:
                    X_vars[LineId] = X_base[LineId] * np.ones((1, N_Xvars))
            elif is2opt[LineId] == 1:
                    X_vars[LineId] = final_X[optCount, ]
                    optCount = optCount + 1 
            # �ر����  $_Kpy	һϵ���Ҹն�-������3��ϣ� $_Ksy	��ϵ���Ҹն�-������7���
            X_vars[3]  = X_vars[2]
            X_vars[7]  = X_vars[6]

        print("����ɨ�Լ���� X_vars����״: ", X_vars.shape)  # ��ӡ X_vars ����״

    return X_vars

# ������
def main():
    
    start_time = time.time()  # ��ȡ��ǰʱ���(��)
    
    # ����Ŀ¼·������
    WorkingDir = os.getcwd()
    # ʵ���ʶ��
    tag="Sweep"  
    
    # ���ַ��ٶ������ޡ��������
    StartVel = 100/3.6 
    EndVel =  900/3.6
    N_depth = 7 
    
    # ���� X_vars ���ɺ���, �Խ��� ����ɨ�� / ǰ�ؽ�ع���֤
    # ���÷�ʽ�������ڲ�˵��
    X_vars = GenerateVehicleParamater(WorkingDir, Filename="config_opt.xlsx", method="JustSweepL123")

    # �������ļ��� ChkPnt��checkpoint (����)�����������
    checkpoint_dir = os.path.join(WorkingDir, "ChkPnt")
     # ������ļ����Ѿ������Ҳ�Ϊ�գ���ɾ�����Լ�����������������
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # �ٴ��½�һ���հ׵� ChkPnt(checkpoint) �ļ��У����ں���д��
    os.makedirs(checkpoint_dir, exist_ok=True)
    # ���� X_vars 
    np.save(os.path.join(checkpoint_dir, f"myXvars_{tag}.npy"), X_vars)
    
    # 3) �������κͲ���
    # �޸ĵ� 1
    BatchSize_parallel = 10
    total_columns = X_vars.shape[1]
    num_batches = math.ceil(total_columns / BatchSize_parallel)
    print("�ܵĲ����������", total_columns)
    print("������������", BatchSize_parallel)
    print("����������", num_batches)

    # ����ÿ����ϼ���ͱ����ά����
    # �޸ĵ� 2
    result_dim = 7  
    all_batch_results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BatchSize_parallel
        end_idx   = min((batch_idx + 1) * BatchSize_parallel, total_columns)

        # ============== (a) ���� & �����ļ� ==============
        ClearBatchTmpFolder(WorkingDir)
        prepare_SpckFiles_eachBatch(WorkingDir, tag, start_idx, end_idx)

        X_vars_batch = X_vars[:, start_idx:end_idx]
        print(f"�� {batch_idx+1} ������������Χ [{start_idx}:{end_idx}), ������С = {X_vars_batch.shape[1]}")

        # �����洢�����εĽ�� (shape=(1, batch_size))
        batch_result = np.zeros((result_dim, X_vars_batch.shape[1]))

        # ============== (b) ���д��� ==============
        with concurrent.futures.ProcessPoolExecutor(max_workers=BatchSize_parallel) as executor:
            future_list = []
            for col_idx_in_batch in range(X_vars_batch.shape[1]):
                args = (col_idx_in_batch, start_idx, WorkingDir, X_vars, tag, StartVel, EndVel, N_depth)
                future = executor.submit(parallel_worker, args)
                future_list.append(future)

            # �ռ����
            for future in concurrent.futures.as_completed(future_list):
                col_idx_in_batch, cVel, RigidWN_CRV, RigidLatMax_CRV, IrwWN_CRV, IrwLatMax_CRV, SperlingY_AAR5, SperlingZ_AAR5 = future.result()
                # �޸ĵ� 3
                # ���г� return (col_idx_in_batch, cVel, WearNumber_CRV, LatDispMax_CRV, SperlingY_AAR5, SperlingZ_AAR5)
                batch_result[0, col_idx_in_batch] = cVel 
                batch_result[1, col_idx_in_batch] = RigidWN_CRV
                batch_result[2, col_idx_in_batch] = RigidLatMax_CRV
                batch_result[3, col_idx_in_batch] = IrwWN_CRV
                batch_result[4, col_idx_in_batch] = IrwLatMax_CRV
                batch_result[5, col_idx_in_batch] = SperlingY_AAR5
                batch_result[6, col_idx_in_batch] = SperlingZ_AAR5
        
        all_batch_results.append(batch_result)
        # ���� all_batch_results ��һ�� list������ÿ��Ԫ�ض��ǵ�ǰ���εĽ�� (batch_result)

        batch_result_toSave = batch_result # �������ݽ�����Ᵽ�桰�б�ʱ��NumPy �ڲ��᳢�԰� all_batch_results ת��һ��ͳһ�� ndarray
        np.save(os.path.join(checkpoint_dir, f"batch_result_{tag}_batch{batch_idx}.npy"), batch_result_toSave)

    # ƴ���������εĽ��
    final_results = np.concatenate(all_batch_results, axis=1)
    print("final_results shape:", final_results.shape)
    print("������ɣ�ǰ10���ٽ��ٶ� = ", final_results[0, :10])
    
    # �����ʱ
    elapsed = time.time() - start_time
    print(f"�ô�����ʱ: {elapsed:.6f} ��")
    
    # ���ս������
    np.save(f"Xvars_{tag}.npy", X_vars)
    np.save(f"Result_{tag}.npy", final_results)
    
if __name__ == "__main__":
    main()
    # ShowMeshgrid()
    
"""
�����е��ã�

    ���� pypack ������������
    F:  # �л��̷�                                                                                                             
    cd F:\ResearchMainStream\0.ResearchBySection\C.����ѧģ��\�����Ż�\�����Ż�ʵ��\ParallelSweepSimpack                            
    python SweepLx1Lx2xL3x.py # ִ�б�����                                                                                                                                   

"""
