
Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\omic_v2\lung
Rscript ..\..\utils\pre_process.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\omic_v2\kidney

python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\omic_v2\lung
python ..\..\utils\data_csv2h5.py --path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\omic_v2\kidney
