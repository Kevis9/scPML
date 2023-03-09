@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=alpha
set list[1]=beta
set list[2]=gamma
set list[3]=delta
set list[4]=even

Rscript ..\..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[4]!
Rscript ..\..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[4]!
python ..\..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[4]! --subpath=raw_data
python ..\..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[4]! --subpath=data


@REM for /l %%n in (2,1,2) do (
@REM @REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM @REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM     Rscript ..\..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[%%n]!
@REM     python ..\..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\gse84133human_emtab5061\!list[%%n]!
@REM )
@REM


