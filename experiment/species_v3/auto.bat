@REM 预处理
@echo off
setlocal enabledelayedexpansion

set list[0]=gse\mouse_human
set list[1]=gse\human_mouse
set list[2]=mouse_combine
set list[3]=combine_mouse
@REM set list[4]=gsemouse_gse85241
@REM set list[5]=gsemouse_emtab_leftjoin

@REM Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[30]!
@REM Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[29]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[29]!
@REM python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[30]!

for /l %%n in (1,1,3) do (
@REM          Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[%%n]!
    Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[%%n]! --subpath=raw_data
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\!list[%%n]! --subpath=data
)



