@REM python data.py
@REM Rscript ..\..\utils\get_sm2.R


@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=10x_v3
set list[1]=cel_seq
set list[2]=dropseq
set list[3]=emtab5061
set list[4]=gse81608
set list[5]=gse84133_human
set list[6]=gse84133_mouse
set list[7]=gse85241
set list[8]=indrop
set list[9]=seq_well
set list[10]=smart_seq

for /l %%n in (0,1,10) do (
@REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset\!list[%%n]!
)

python main.py