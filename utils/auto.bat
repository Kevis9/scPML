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

for /l %%n in (0,1,12) do (
    python data_csv2h5.py --path=..\experiment\platform_v2\!list[%%n]!
)
@REM python data_csv2h5.py --path=..\experiment\platform_v2\cel_seq_10x
@REM python data_csv2h5.py --path=..\experiment\platform_v2\cel_seq_10x