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

set list[11]=Cao_2020_stomach
set list[12]=GSE72056
set list[13]=GSE98638
set list[14]=GSE99254
set list[15]=GSE108989
set list[16]=MacParland

set list[18]=Guo
set list[19]=He_Calvarial_Bone
set list[20]=Enge
set list[21]=Hu
set list[22]=Wu_human
@REM 下面是withindatast的mouse
set list[23]=GSE115746
set list[24]=GSM3271044
set list[25]=Guo_2021
set list[26]=Loo_E14.5
@REM 以上

@REM python split_and_process.py

for /l %%n in (14,1,14) do (
@REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
    Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset\!list[%%n]!
    Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset\!list[%%n]!
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset\!list[%%n]! --subpath=raw_data
    python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset\!list[%%n]! --subpath=data
)