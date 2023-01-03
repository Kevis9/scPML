@REM mkdir cel_seq_smart_seq
@REM mkdir cel_seq_drop_seq
@REM mkdir cel_seq_10x
@REM mkdir seq_well_smart_seq
@REM mkdir seq_well_drop_seq
@REM mkdir seq_well_10x
@REM mkdir indrop_drop_seq
@REM mkdir indrop_10x
@REM mkdir drop_seq_10x
@REM mkdir drop_seq_smart_seq
@REM mkdir 84133_5061
@REM mkdir 84133_85241
@REM mkdir 84133_81608
@REM
@REM @REM 从platoform目录中先把数据复制过来
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_smart_seq\raw_data cel_seq_smart_seq\raw_data
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_drop_seq\raw_data cel_seq_drop_seq\raw_data
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_10x\data cel_seq_10x\raw_data
@REM cp -r ..\platform\new_version\seq_well\seq_well_smart_seq\data seq_well_smart_seq\raw_data
@REM cp -r ..\platform\new_version\seq_well\seq_well_dropseq\data seq_well_drop_seq\raw_data
@REM cp -r ..\platform\new_version\seq_well\seq_well_10x\data seq_well_10x\raw_data
@REM cp -r ..\platform\new_version\indrop\indrop_dropseq\data indrop_drop_seq\raw_data
@REM cp -r ..\platform\new_version\indrop\indrop_10x\data indrop_10x\raw_data
@REM cp -r ..\platform\new_version\dropseq\dropseq_10x\data drop_seq_10x\raw_data
@REM cp -r ..\platform\new_version\dropseq\dropseq_smart_seq\data drop_seq_smart_seq\raw_data
@REM cp -r ..\platform\new_version\84133_5061\raw_data 84133_5061\raw_data
@REM cp -r ..\platform\new_version\84133_85241\data 84133_85241\raw_data
@REM cp -r ..\platform\new_version\84133_81608\data 84133_81608\raw_data
@REM
@REM @REM 再次复制到data目录
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_smart_seq\raw_data cel_seq_smart_seq\data
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_drop_seq\raw_data cel_seq_drop_seq\data
@REM cp -r ..\platform\new_version\cel_seq\cel_seq_10x\data cel_seq_10x\data
@REM cp -r ..\platform\new_version\seq_well\seq_well_smart_seq\data seq_well_smart_seq\data
@REM cp -r ..\platform\new_version\seq_well\seq_well_dropseq\data seq_well_drop_seq\data
@REM cp -r ..\platform\new_version\seq_well\seq_well_10x\data seq_well_10x\data
@REM cp -r ..\platform\new_version\indrop\indrop_dropseq\data indrop_drop_seq\data
@REM cp -r ..\platform\new_version\indrop\indrop_10x\data indrop_10x\data
@REM cp -r ..\platform\new_version\dropseq\dropseq_10x\data drop_seq_10x\data
@REM cp -r ..\platform\new_version\dropseq\dropseq_smart_seq\data drop_seq_smart_seq\data
@REM cp -r ..\platform\new_version\84133_5061\raw_data 84133_5061\data
@REM cp -r ..\platform\new_version\84133_85241\data 84133_85241\data
@REM cp -r ..\platform\new_version\84133_81608\data 84133_81608\data
@REM
@REM @REM 删除data目录下的data_1.csv文件
@REM rm -f cel_seq_smart_seq\data\ref\data_1.csv
@REM rm -f cel_seq_smart_seq\data\query\data_1.csv
@REM rm -f cel_seq_drop_seq\data\ref\data_1.csv
@REM rm -f cel_seq_drop_seq\data\query\data_1.csv
@REM rm -f cel_seq_10x\data\ref\data_1.csv
@REM rm -f cel_seq_10x\data\query\data_1.csv
@REM rm -f seq_well_smart_seq\data\ref\data_1.csv
@REM rm -f seq_well_smart_seq\data\query\data_1.csv
@REM rm -f seq_well_dropseq\data\ref\data_1.csv
@REM rm -f seq_well_dropseq\data\query\data_1.csv
@REM rm -f seq_well_10x\data\ref\data_1.csv
@REM rm -f seq_well_10x\data\query\data_1.csv
@REM rm -f indrop_dropseq\data\ref\data_1.csv
@REM rm -f indrop_dropseq\data\query\data_1.csv
@REM rm -f indrop_10x\data\ref\data_1.csv
@REM rm -f indrop_10x\data\query\data_1.csv
@REM rm -f dropseq_10x\data\ref\data_1.csv
@REM rm -f dropseq_10x\data\query\data_1.csv
@REM rm -f dropseq_smart_seq\data\ref\data_1.csv
@REM rm -f dropseq_smart_seq\data\query\data_1.csv
@REM rm -f 84133_5061\data\ref\data_1.csv
@REM rm -f 84133_5061\data\query\data_1.csv
@REM rm -f 84133_85241\data\ref\data_1.csv
@REM rm -f 84133_85241\data\query\data_1.csv
@REM rm -f 84133_81608\data\ref\data_1.csv
@REM rm -f 84133_81608\data\query\data_1.csv




@REM 预处理
@echo off
setlocal enabledelayedexpansion
set list[0]=cel_seq_smart_seq
set list[1]=cel_seq_drop_seq
set list[2]=cel_seq_10x
set list[3]=seq_well_smart_seq
set list[4]=seq_well_drop_seq
set list[5]=seq_well_10x
set list[6]=indrop_drop_seq
set list[7]=indrop_10x
set list[8]=drop_seq_10x
set list[9]=drop_seq_smart_seq
set list[10]=84133_5061
set list[11]=84133_85241
set list[12]=84133_81608
set list[13]=cel_seq_10x_v3
set list[14]=seq_well_10x_v3
set list[15]=indrop_10x_v3
set list[16]=indrop_smart_seq
set list[17]=drop_seq_10x_v3
set list[18]=drop_seq_indrop
set list[19]=drop_seq_seq_well
set list[20]=10x_v3_cel_seq
set list[21]=10x_v3_seq_well
set list[22]=10x_v3_drop_seq
set list[23]=10x_v3_indrop
set list[23]=smart_seq_seq_well
set list[24]=smart_seq_indrop
set list[25]=smart_seq_drop_seq
set list[26]=smart_seq_10x_v3
set list[27]=5061_84133
set list[28]=84133_combine
set list[29]=combine_84133
@REM Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[18]!
@REM Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[28]!
python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[29]!

@REM for /l %%n in (20,1,23) do (
@REM @REM     rm -rf E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     cp -r E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\raw_data E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!\data
@REM @REM     Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM @REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM @REM     Rscript ..\..\utils\get_sm.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM     python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2\!list[%%n]!
@REM )



