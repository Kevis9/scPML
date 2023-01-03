@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=alpha
set list[1]=beta
set list[2]=gamma
set list[3]=delta



for /l %%n in (2,1,2) do (
@REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
    Rscript ..\..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[%%n]!
    python ..\..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[%%n]!
)



