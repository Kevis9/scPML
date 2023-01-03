import pandas as pd


def check_data(path):
    label = pd.read_csv(path)
    print(label.iloc[:, 0].value_counts())


check_data("emtab_label.csv")
check_data("gsehuman_label.csv")
check_data("gse85241_label.csv")
check_data("gse81608_label.csv")