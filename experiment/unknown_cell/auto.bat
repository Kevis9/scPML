@REM 预处理
@echo off
setlocal enabledelayedexpansion

set list[0]=GSE72056_GSE103322_B_cell
set list[1]=GSE72056_GSE103322_Endothelial
set list[2]=GSE72056_GSE103322_Macrophage
set list[3]=GSE72056_GSE103322_malignant
set list[4]=GSE72056_GSE103322_T_cell

set list[5]=GSE84133_EMTAB5061_alpha
set list[6]=GSE84133_EMTAB5061_beta
set list[7]=GSE84133_EMTAB5061_delta
set list[8]=GSE84133_EMTAB5061_gamma

set list[9]=GSE103322_GSE72056_malignant2
set list[10]=GSE72056_GSE103322
set list[11]=GSE118056_GSE117988


@REM set list[1]=GSE118056_GSE117988
for /l %%n in (3,1,3) do (
         Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\unknown_cell\!list[%%n]!
@REM          Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\unknown_cell\!list[%%n]!
         python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\unknown_cell\!list[%%n]! --subpath=raw_data
         python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\unknown_cell\!list[%%n]! --subpath=data

)



