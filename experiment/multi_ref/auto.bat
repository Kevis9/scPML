@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=PBMC1
set list[1]=PBMC2
set list[2]=MCA_liver
set list[3]=Haber
@REM Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[3]!
@REM Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[3]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[3]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[3]! --subpath=data

for /l %%n in (3,1,3) do (
@REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
    Rscript ..\..\utils\old_get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\multi_ref\!list[%%n]! --subpath=data
)



