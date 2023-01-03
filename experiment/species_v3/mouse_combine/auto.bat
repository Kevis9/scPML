cd ..\..\..\
@REM python common_type.py
cd utils
@REM Rscript get_sm.R
@REM Rscript pre_process.R
Rscript ../../utils/get_sm.R
python data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3\mouse_combine



