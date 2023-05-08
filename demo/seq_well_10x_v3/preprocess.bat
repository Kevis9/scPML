@echo off
setlocal enabledelayedexpansion
set list[0]=seq_well_10x_v3
for /l %%n in (0,1,0) do (
    Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
    Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]! --subpath=raw_data
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]! --subpath=data
)



