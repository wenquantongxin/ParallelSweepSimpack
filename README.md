# ParallelSweepSimpack

使用并行化编程进行转向架设计参数的扫略，并行调用 SIMPACK 车辆模型计算临界速度。

## 工程简介

本工程主要包含以下内容：

- **参数配置**：如 `config_opt.xlsx` 中保存的优化相关信息
- **关键 Python 脚本**：
  - `FindCrticalVelocity`：二分搜索算法求解临界速度
  - `PrepareBatchFiles.py`：生成批量仿真文件
  - `SweepL1xL2xL3x.py`：针对三个维度的扫频脚本
  - `CRVPerf.py`：曲线通过性能计算
  - `STRPerf.py`：直线（主要是平稳性）计算
- **SIMPACK 文件**：如 `Vehicle4WDB_IRWCRV300m_OptBase.spck`、`subvars_OptBase.subvar` 等
- **输出文件**：仿真结果 `.sbr` / `.sir` / `.spf` / `.dat` 等

## 工程目录结构

```plaintext
F:\ResearchMainStream\......\并行化直曲线运行综合评价
│  config_opt.xlsx
│  CRVPerf.py
│  FindCrticalVelocity.py
│  PrepareBatchFiles.py
│  Result_IRWCRV300m.spf
│  Result_RigidCriticalVel.spf
│  Result_RigidCRV300m.spf
│  Result_RigidSTR80kmph.spf
│  SbrExport_SPCKResult.qs
│  STRPerf.py
│  subvars_OptBase.subvar
│  SweepL1xL2xL3x.py
│  tree.txt
│  Untitled-1.ipynb
│  Vehicle4WDB_IRWCRV300m_OptBase.spck
│  Vehicle4WDB_RigidCriticalVel_OptBase.spck
│  Vehicle4WDB_RigidCRV300m_OptBase.spck
│  Vehicle4WDB_RigidSTR80kmph_OptBase.spck
├─BatchTmp
│  │  DatResult_RigidCriticalVel_ALL_120.dat
│  │  Result_IRWCRV300m_Opt_ALL_120.spf
│  │  Result_RigidCriticalVel_Opt_ALL_120.spf
│  │  Result_RigidCRV300m_Opt_ALL_120.spf
│  │  Result_RigidSTR80kmph_Opt_ALL_120.spf
│  │  subvars_Opt_ALL_120.subvar
│  │  Vehicle4WDB_IRWCRV300m_Opt_ALL_120.spck
│  │  Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.spck
│  │  Vehicle4WDB_RigidCRV300m_Opt_ALL_120.spck
│  │  Vehicle4WDB_RigidSTR80kmph_Opt_ALL_120.spck
│  ├─Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.output
│  │      Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.intinfo
│  │      Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.licreq.log
│  │      Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.sbr
│  │      Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.sir
│  │      Vehicle4WDB_RigidCriticalVel_Opt_ALL_120.spckst
│  ├─Vehicle4WDB_RigidCRV300m_Opt_ALL_120.output
│  │      Vehicle4WDB_RigidCRV300m_Opt_ALL_120.intinfo
│  │      Vehicle4WDB_RigidCRV300m_Opt_ALL_120.licreq.log
│  │      Vehicle4WDB_RigidCRV300m_Opt_ALL_120.sir     
├─ChkPnt
│      batch_result_ALL_batch0.npy
│      batch_result_ALL_batch1.npy
│      myXvars_ALL.npy
├─ref_files
│  │  Bogie_IRWs_4WDBv3.spck
│  │  IRW_4WDBv31.spck
│  │  subvars_FWDBv31-备份.subvar
│  │  几何模型_STL版本_构架.STL
│  │  几何模型_STL版本_轴桥.STL
│  └─Bogie_IRWs_4WDBv3.output
│          Bogie_IRWs_4WDBv3.licreq.log
├─__pycache__   
├─备份
│  └─checkpoint
└─结果分析组
            
