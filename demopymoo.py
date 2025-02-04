# -*- coding: gbk -*-
# ʾ�����룬ʹ�� pymoo ���� NSGA-2 ��Ŀ��Ѱ�š����У�����ѧָ�������������档

import numpy as np
import math
import concurrent.futures
import time

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


def external_evaluate(x):
    """
    ģ���ⲿ��������:
    x �ǳ���Ϊ12��һά����(������).
    ����򵥷���3�������, ����3��Ŀ��(������ʾ).
    ʵ��Ӧ����, ��Ӧ�����ⲿ����ѧ��������ʵĿ��ֵ.

    ע��: ���ĳ��Ŀ����Ҫ���, �ں��� _evaluate ��ǵ�ȡ��ֵ.
    """
    time.sleep(0.01)  # ģ���ʱ
    # ���� (f1, f2, f3) ������Ŀ��ֵ��
    return np.random.random(), np.random.random(), np.random.random()


class MyBatchProblem(Problem):
    def __init__(self,
                 batch_size=10,
                 **kwargs):
        """
        - n_var=12: ���߱���ά��
        - n_obj=3 : Ŀ�꺯������(�˴���ʾ��3��Ŀ��)
        - xl, xu : ���߱�����������
        - elementwise_evaluation=False: ��������, һ���Խ������� X

        batch_size: �����Լ�����Ĳ���, ��ʾ�� _evaluate �ж� X �������д���.
        """
        super().__init__(
            n_var=12,
            n_obj=3,
            xl=np.zeros(12),       # ����12���������½綼Ϊ0
            xu=np.ones(12)*10.0,   # ����12���������Ͻ綼Ϊ10
            elementwise_evaluation=False,
            **kwargs
        )

        self.batch_size = batch_size

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X.shape = (N, 12), ���� N = ��Ⱥ��ģ(pop_size) + ������һЩoffspring.
        ���ｫ�� N ����������߳�����.

        ������õ� (N, 3) ��Ŀ��ֵ���� F, ����ֵ�� out["F"].
        """
        N = X.shape[0]
        F = np.zeros((N, 3))

        # 1) ����������
        num_batches = math.ceil(N / self.batch_size)

        # 2) ��������
        start_index = 0
        for b in range(num_batches):
            end_index = min(start_index + self.batch_size, N)

            X_batch = X[start_index:end_index, :]  # ��ǰ����

            # �����ʾ����, ������ ThreadPoolExecutor �����߳�.
            # Ҳ���Ի��� ProcessPoolExecutor / ���в����߼���.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = []
                for i in range(X_batch.shape[0]):
                    x_i = X_batch[i, :]
                    # �ύ��������ⲿ��������
                    future = executor.submit(external_evaluate, x_i)
                    futures.append(future)

                # �ռ����
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    f1, f2, f3 = future.result()
                    # ����������Ŀ�궼����С������; 
                    # ���ĳ��Ŀ����Ҫ���, ���Ըĳ� f = -g ֮��.
                    F[start_index + i, :] = [f1, f2, f3]

            start_index = end_index

        # 3) ���ս�Ŀ����󸳸� out["F"]
        out["F"] = F


def main():
    # 1) ��ʼ������
    problem = MyBatchProblem(
        batch_size=10   # ÿ�����10���Ⲣ��
    )

    # 2) ѡ���㷨 (��Ŀ��NSGA2), ��Ⱥ��ģΪ20������ʾ
    algorithm = NSGA2(pop_size=20)

    # 3) ��ֹ����: ���� 5 ��(����ʾ, ���ɸĴ�һЩ)
    termination = get_termination("n_gen", 5)

    # 4) �Ż�
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True  # ���ÿһ���Ľ���
    )

    # 5) �鿴���
    # res.X.shape = (N_non_dominated, 12)
    # res.F.shape = (N_non_dominated, 3)
    print("\n==================== �Ż����� ====================")
    print("��֧�������:", len(res.X))
    print("���ֽ�ʾ��X[0]:", res.X[0])
    print("��ӦĿ��F[0]:", res.F[0])
    
    nd_X = res.X 
    nd_F = res.F
    
    # # д�� CSV �ļ�
    # np.savetxt("final_solutions.csv",
    #         np.hstack([nd_X, nd_F]),
    #         delimiter=",",
    #         comments="",  # ȥ��ע�ͷ���
    #         fmt="%.6f")   # �����������


if __name__ == '__main__':
    main()
