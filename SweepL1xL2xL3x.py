# SweepL1xL2xL3x.py
# -*- coding: gbk -*-

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

from PrepareBatchFiles import (
    read_config_opt_excel,
    ClearBatchTmpFolder,
    prepare_SpckFiles_eachBatch
)
from HalfSearch_CrticalVelocity import (HalfSearch_CrticalVelocity, CurvePerf_idx)


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

    # �޸ĵ� 0
    
    # ��������1�����ð����������������ٽ��ٶ�
    cVel = HalfSearch_CrticalVelocity(
        WorkingDir,
        X_vars,
        tag,
        actual_idx,  # ����ж�Ӧ [start_idx, end_idx) ��Χ�ڵ�һ��
        StartVel,
        EndVel,
        N_depth
    )
    
    # ��������2���������߼�����򣬷�������ĥ������������
    WearNumber, LatDisp = CurvePerf_idx(WorkingDir, X_vars, tag, actual_idx)
    
    # ���ز��м���� idx �Ľ��������
    return (col_idx_in_batch, cVel, WearNumber, LatDisp)

# ����άͼ����ʾ �ٽ��ٶ��� L1��L2 �Ĺ�ϵ
def ShowMeshgrid():
    CriticalVel = np.load("myCriticalVel.npy") 
    Xvars = np.load("myXvars.npy") 

    Z = CriticalVel  # Z.size = 169
    lx1_all = Xvars[29,:]  # �� 169 ��ֵ�����������ظ�/����
    lx2_all = Xvars[30,:]  # ͬ��

    # 1) �ҵ�Ψһֵ������
    lx1_unique = np.unique(lx1_all)
    lx2_unique = np.unique(lx2_all)
    m = len(lx1_unique)
    n = len(lx2_unique)
    print("lx1_unique:", lx1_unique)
    print("lx2_unique:", lx2_unique)
    print("m,n =", m,n)
    # ���� m*n Ӧ�� = 169

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

# ������
def main():
    
    # �޸ĵ� 1    
    # ����Ŀ¼·������
    WorkingDir = r"F:\ResearchMainStream\0.ResearchBySection\C.����ѧģ��\�����Ż�\�����Ż�ʵ��\���л�����ɨ�Լ���SIMPACKģ���ٽ��ٶ�"
    # ʵ���ʶ��
    tag="Test"  
    
    # �޸ĵ� 2
    # ���ַ��ٶ������ޡ��������
    StartVel = 50/3.6 
    EndVel =  800/3.6
    N_depth = 7 
    
    # 1) ���ú�����ȡ config_opt.xlsx����ȡ��׼ֵ X_base
    Opt_Config = read_config_opt_excel(WorkingDir)
    X_base = Opt_Config["��׼ֵ"].to_list()
    print("��׼ֵ X_base:", X_base)

    # 2) ���� X_vars (32 x N_combos)
    # �޸ĵ� 3
    Lx1_sweep = np.arange(0, 0.6 + 0.001, 0.05)  
    Lx2_sweep = np.arange(0, 0.6 + 0.001, 0.05)  
    Lx3_sweep = np.arange(0, 0.05, 0.05)  # ʾ��

    Lx123_combinations = list(itertools.product(Lx1_sweep, Lx2_sweep, Lx3_sweep))
    X_vars_columns = []

    for (lx1, lx2, lx3) in Lx123_combinations:
        x_temp = X_base.copy()
        x_temp[29] = lx1
        x_temp[30] = lx2
        x_temp[31] = lx3
        X_vars_columns.append(x_temp)

    X_vars = np.column_stack(X_vars_columns)
    print("X_vars����״: ", X_vars.shape)
    
    # �������� X_vars 
    # �������ļ��� ChkPnt��checkpoint (����)�����������
    checkpoint_dir = os.path.join(WorkingDir, "ChkPnt")
     # ������ļ����Ѿ������Ҳ�Ϊ�գ���ɾ�����Լ�����������������
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # �ٴ��½�һ���հ׵� ChkPnt(checkpoint) �ļ��У����ں���д��
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    np.save(os.path.join(checkpoint_dir, f"myXvars_{tag}.npy"), X_vars)
    
    # 3) �������κͲ���
    # �޸ĵ� 4
    BatchSize_parallel = 10
    total_columns = X_vars.shape[1]
    num_batches = math.ceil(total_columns / BatchSize_parallel)
    print("�ܵĲ����������", total_columns)
    print("������������", BatchSize_parallel)
    print("����������", num_batches)

    # ����ÿ����ϼ���ͱ����ά����
    # �޸ĵ� 5
    result_dim = 3  
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
                col_idx_in_batch, cVel, WearNumber, LatDisp = future.result()
                # �޸ĵ� 6
                batch_result[0, col_idx_in_batch] = cVel
                batch_result[1, col_idx_in_batch] = WearNumber
                batch_result[2, col_idx_in_batch] = LatDisp
        
        all_batch_results.append(batch_result)
        # ���� all_batch_results ��һ�� list������ÿ��Ԫ�ض��ǵ�ǰ���εĽ�� (batch_result)

        batch_result_toSave = batch_result # �������ݽ�����Ᵽ�桰�б�ʱ��NumPy �ڲ��᳢�԰� all_batch_results ת��һ��ͳһ�� ndarray
        np.save(os.path.join(checkpoint_dir, f"batch_result_{tag}_batch{batch_idx}.npy"), batch_result_toSave)

    # ƴ���������εĽ��
    final_results = np.concatenate(all_batch_results, axis=1)
    print("final_results shape:", final_results.shape)
    print("ʾ����ǰ10���ٽ��ٶ� = ", final_results[0, :10])
    
    # ���ս������
    np.save("myXvars.npy", X_vars)
    np.save("myCriticalVel.npy", final_results)
    
if __name__ == "__main__":
    main()
    # ShowMeshgrid()
    
"""
�����е��ã�

    ���� pypack ������������
    F:  �л��̷�
    cd F:\ResearchMainStream\0.ResearchBySection\C.����ѧģ��\�����Ż�\�����Ż�ʵ��\���л�����ɨ�Լ���SIMPACKģ���ٽ��ٶ�
    python SweepL1xL2xL3x.py ִ�б�����
    
������װ��
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install  pymoo pandas numpy ipywidgets matplotlib openpyxl
    pip install -U pymoo
    
"""
