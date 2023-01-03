@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=GSE84133
Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[0]!
python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[0]!

@REM for /l %%n in (20,1,23) do (
@REM @REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM @REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM @REM     Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM )



