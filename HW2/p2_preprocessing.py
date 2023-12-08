import numpy as np
import pandas as pd
from datetime import datetime


def preprocessing_data():
    FIGRX_2004 = pd.read_csv("./data/FIGRX_2004.csv")[["Date", "Close"]].values.tolist()
    FIGRX_2005 = pd.read_csv("./data/FIGRX_2005.csv")[["Date", "Close"]].values.tolist()
    EURUSD_2004 = pd.read_csv("./data/EURUSD_2004.csv", header=None)[
        [1, 3]
    ].values.tolist()
    EURUSD_2005 = pd.read_csv("./data/EURUSD_2005.csv", header=None)[
        [1, 3]
    ].values.tolist()
    GSPC_2004 = (
        pd.read_csv("./data/GSPC_2004.csv")[["Date", "Close"]]
        .iloc[::-1]
        .values.tolist()
    )
    GSPC_2005 = (
        pd.read_csv("./data/GSPC_2005.csv")[["Date", "Close"]]
        .iloc[::-1]
        .values.tolist()
    )

    # change EURUSD_2004, EURUSD_2005 time format 1/6/2004, to 2004-01-06
    for i in range(len(EURUSD_2004)):
        EURUSD_2004[i][0] = datetime.strptime(EURUSD_2004[i][0], "%m/%d/%Y").strftime(
            "%Y-%m-%d"
        )
    for i in range(len(EURUSD_2005)):
        EURUSD_2005[i][0] = datetime.strptime(EURUSD_2005[i][0], "%m/%d/%Y").strftime(
            "%Y-%m-%d"
        )

    def change_date(data, year):
        for i in range(len(data)):
            data[i][0] = (
                datetime.strptime(data[i][0], "%Y-%m-%d") - datetime(year, 1, 1)
            ).days
        return data

    # change all data date tyup to the differ from 2004-01-01
    FIGRX_2004 = change_date(FIGRX_2004, 2004)
    FIGRX_2005 = change_date(FIGRX_2005, 2005)
    EURUSD_2004 = change_date(EURUSD_2004, 2004)
    EURUSD_2005 = change_date(EURUSD_2005, 2005)
    GSPC_2004 = change_date(GSPC_2004, 2004)
    GSPC_2005 = change_date(GSPC_2005, 2005)

    # fill the missing date with NaN
    def fill_missing_date(data):
        for i in range(366):
            if i >= len(data) or data[i][0] != i:
                data.insert(i, [i, np.nan])
        new_data = []
        for d in data:
            new_data.append(d[1])
        return new_data

    FIGRX_2004 = fill_missing_date(FIGRX_2004)
    FIGRX_2005 = fill_missing_date(FIGRX_2005)
    EURUSD_2004 = fill_missing_date(EURUSD_2004)
    EURUSD_2005 = fill_missing_date(EURUSD_2005)
    GSPC_2004 = fill_missing_date(GSPC_2004)
    GSPC_2005 = fill_missing_date(GSPC_2005)

    def remove_missing_data(FIGRX, EURUSD, GSPC):
        new_FIGRX = []
        new_EURUSD = []
        new_GSPC = []
        for i in range(366):
            if np.isnan(FIGRX[i]) or np.isnan(EURUSD[i]) or np.isnan(GSPC[i]):
                continue
            new_FIGRX.append(FIGRX[i])
            new_EURUSD.append(EURUSD[i])
            new_GSPC.append(GSPC[i])
        return new_FIGRX, new_EURUSD, new_GSPC

    FIGRX_2004, EURUSD_2004, GSPC_2004 = remove_missing_data(
        FIGRX_2004, EURUSD_2004, GSPC_2004
    )
    FIGRX_2005, EURUSD_2005, GSPC_2005 = remove_missing_data(
        FIGRX_2005, EURUSD_2005, GSPC_2005
    )

    def daily_change(data):
        new_data = []
        for i in range(1, len(data)):
            new_data.append((data[i] - data[i - 1]) / data[i - 1])
        return new_data

    FIGRX_2004 = daily_change(FIGRX_2004)
    FIGRX_2005 = daily_change(FIGRX_2005)
    EURUSD_2004 = daily_change(EURUSD_2004)
    EURUSD_2005 = daily_change(EURUSD_2005)
    GSPC_2004 = daily_change(GSPC_2004)
    GSPC_2005 = daily_change(GSPC_2005)

    def get_data(FIGRX, EURUSD, GSPC):
        data = {}
        data["X"] = []
        data["Y"] = []
        data["y"] = []
        for i in range(len(FIGRX) - 1):
            data["X"].append([GSPC[i], EURUSD[i]])
            data["Y"].append(FIGRX[i + 1])
            data["y"].append(np.sign(FIGRX[i + 1]))
        return data

    train_data = get_data(FIGRX_2004, EURUSD_2004, GSPC_2004)
    test_data = get_data(FIGRX_2005, EURUSD_2005, GSPC_2005)

    return train_data, test_data
