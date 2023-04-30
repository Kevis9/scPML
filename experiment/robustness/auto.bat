@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=0
set list[1]=0.05
set list[2]=0.1
set list[3]=0.15
set list[4]=0.2

set basepath=E:\YuAnHuang\kevislin\Cell_Classification\experiment\robustness\85241_5061\dropout
for /l %%n in (0,1,4) do (
    Rscript ..\..\utils\get_sm.R %basepath%\!list[%%n]!
    Rscript ..\..\utils\pre_process.R %basepath%\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=%basepath%\!list[%%n]! --subpath=raw_data
    python ..\..\utils\data_csv2h5.py --path=%basepath%\!list[%%n]! --subpath=data
)


set basepath=E:\YuAnHuang\kevislin\Cell_Classification\experiment\robustness\85241_5061\gaussian
for /l %%n in (0,1,4) do (
    Rscript ..\..\utils\get_sm.R %basepath%\!list[%%n]!
    Rscript ..\..\utils\pre_process.R %basepath%\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=%basepath%\!list[%%n]! --subpath=raw_data
    python ..\..\utils\data_csv2h5.py --path=%basepath%\!list[%%n]! --subpath=data
)

