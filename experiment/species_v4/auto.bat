@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=gsemouse_gsehuman
set list[1]=gsehuman_gsemouse
set list[2]=mouse_combine
set list[3]=combine_mouse

@REM Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[30]!
@REM Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[29]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[29]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[30]!

for /l %%n in (0,1,2) do (
@REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
         Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v4\!list[%%n]!
         Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v4\!list[%%n]!
         python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v4\!list[%%n]! --subpath=raw_data
         python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v4\!list[%%n]! --subpath=data
@REM     Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!

)



