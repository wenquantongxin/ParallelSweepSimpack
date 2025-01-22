
# ParallelSweepSimpack

使用并行化编程进行转向架设计参数的扫略，并行调用SIMPACK车辆模型计算临界速度。

工程文件夹所含内容如下：

F:\...\参数优化\参数优化实现\基于PYMOO框架优化
|   config_opt.xlsx
|   HalfSearch_CrticalVelocity.py
|   myCriticalVel.npy
|   myXvars.npy
|   OptTargetResult_OptBasel.spf
|   PrepareBatchFiles.py
|   SbrExport_Opt.qs
|   subvars_OptBase.subvar
|   SweepL1xL2xL3x.py
|   SweepLx123.ipynb
|   Vehicle_Genera_OptBase.spck
|
+---BatchTmp
|   |   OptResult_Test_160.dat
|   |   OptResult_Test_XXX.dat
|   |   OptResult_Test_YYY.dat
|   |   OptTargetResult_Opt_Test_160.spf
|   |   OptTargetResult_Opt_Test_XXX.spf
|   |   OptTargetResult_Opt_Test_YYY.spf
|   |   subvars_Opt_Test_160.subvar
|   |   subvars_Opt_Test_XXX.subvar
|   |   subvars_Opt_Test_YYY.subvar
|   |   Vehicle_Opt_Test_160.spck
|   |   Vehicle_Opt_Test_XXX.spck
|   |   Vehicle_Opt_Test_YYY.spck
|   |
|   +---Vehicle_Opt_Test_160.output
|   |       Vehicle_Opt_Test_160.intinfo
|   |       Vehicle_Opt_Test_160.licreq.log
|   |       Vehicle_Opt_Test_160.sbr
|   |       Vehicle_Opt_Test_160.sir
|   |       Vehicle_Opt_Test_160.spckst
|   |
|   +---Vehicle_Opt_Test_XXX.output
|   |       Vehicle_Opt_Test_XXX.intinfo
|   |       Vehicle_Opt_Test_XXX.licreq.log
|   |       Vehicle_Opt_Test_XXX.sbr
|   |       Vehicle_Opt_Test_XXX.sir
|   |       Vehicle_Opt_Test_XXX.spckst
|   |
|   \---Vehicle_Opt_Test_YYY.output
|           Vehicle_Opt_Test_168.intinfo
|           Vehicle_Opt_Test_168.licreq.log
|           Vehicle_Opt_Test_168.sbr
|           Vehicle_Opt_Test_168.sir
|           Vehicle_Opt_Test_168.spckst
|
+---ref_files
|   |   Bogie_IRWs_4WDBv3.spck
|   |   IRW_4WDBv31.spck
|   |   subvars_FWDBv31-备份.subvar
|   |   几何模型_STL版本_构架.STL
|   |   几何模型_STL版本_轴桥.STL
|   |
|   \---Bogie_IRWs_4WDBv3.output
|           Bogie_IRWs_4WDBv3.licreq.log
|
+---Vehicle_Genera_OptBase.output
|       Vehicle_Genera_OptBase.licreq.log
|
\---__pycache__
        check_SPCK_stable.cpython-310.pyc
        HalfSearch_CrticalVelocity.cpython-310.pyc
        PrepareBatchFiles.cpython-310.pyc
        SweepL1xL2xL3x.cpython-310.pyc
