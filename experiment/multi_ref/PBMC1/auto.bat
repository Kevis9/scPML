@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=PBMC1

@REM Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[0]!
Rscript ..\..\..\utils\pre_process.R
python ..\..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[0]! --subpath=data





