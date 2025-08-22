import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

def read_file(dsname):
    dic = {}
    classmember = 0

    try:
        if not os.path.exists("./data/uci_download/"):
            os.makedirs("./data/uci_download/")
        if dsname == "shuttle": # UCI: DOI:10.24432/C5WS31, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/shuttle_label.npy"):
                statlog_shuttle = fetch_ucirepo(id=148)
                X = statlog_shuttle.data.features.to_numpy()
                y = statlog_shuttle.data.targets.to_numpy()
                np.save("./data/uci_download/shuttle_data.npy", X)
                np.save("./data/uci_download/shuttle_label.npy", y)
            X = np.load("./data/uci_download/shuttle_data.npy")
            y = np.load("./data/uci_download/shuttle_label.npy")
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "adult": # UCI: DOI:10.24432/C5XW20, Barry Becker, Ronny Kohavi, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/adult_label.npy"):
                adult = fetch_ucirepo(id=2)
                X = adult.data.features.to_numpy()
                y = adult.data.targets.to_numpy()
                np.save("./data/uci_download/adult_data.npy", X)
                np.save("./data/uci_download/adult_label.npy", y)
            X = np.load("./data/uci_download/adult_data.npy")
            y = np.load("./data/uci_download/adult_label.npy")
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "sensorless": # UCI: DOI:10.24432/C5VP5F, Martyna Bator, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.,  added commas for proper import
            file = open("./data/uci/Sensorless_drive_diagnosis.txt", "r")
        else:
            # data from https://github.com/milaan9/Clustering-Datasets/tree/master
            file = open("./data/synthetic_milaan9" + "/" + dsname + ".arff", "r")
    except:
        try:
            # data from https://github.com/milaan9/Clustering-Datasets/tree/master
            file = open("./data/uci_milaan9" + "/" + dsname + ".arff", "r")
        except:
            raise FileNotFoundError

    x = []
    label = []
    for line in file:
        if (line.startswith("@") or line.startswith("%") or "class" in line or "duration" in line or len(line.strip()) == 0):
            pass
        else:
            j = line.split(",")
            if ("?" in j):
                continue
            k = []

            if dsname == "kddcup":
                k.append(float(j[0]))
                k.append(j[1])
                k.append(j[2])
                k.append(j[3])
                for i in range(4, len(j) - 1):
                    k.append(float(j[i]))
            else:
                for i in range(len(j) - 1):
                    k.append(float(j[i]))
            if (not j[len(j) - 1].startswith("noise")):
                clsname = j[len(j) - 1].rstrip()
                if (clsname in dic.keys()):
                    label.append(dic[clsname])
                else:
                    dic[clsname] = classmember
                    label.append(dic[clsname])
                    classmember += 1
            else:
                label.append(-1)
            x.append(k)

    return np.array(x), np.array(label).reshape(1, len(label))[0]

def load_data(dsname):
    scaler = MinMaxScaler()
    data, labels = read_file(dsname)
    scaler.fit(data)
    data = scaler.transform(data)
    return data, labels

if __name__ == "__main__":
    X, y =load_data("sensorless")
    print(X.shape, y.shape)