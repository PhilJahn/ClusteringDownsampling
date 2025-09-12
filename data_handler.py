import os
from doctest import UnexpectedException

import numpy as np
from densired import datagen
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
            X = np.load("./data/uci_download/shuttle_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/shuttle_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "adult": # UCI: DOI:10.24432/C5XW20, Barry Becker, Ronny Kohavi, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/adult_label.npy"):
                adult = fetch_ucirepo(id=2)
                X = adult.data.features.to_numpy()
                y = adult.data.targets.to_numpy()
                np.save("./data/uci_download/adult_data.npy", X)
                np.save("./data/uci_download/adult_label.npy", y)
            X = np.load("./data/uci_download/adult_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/adult_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "pendigits": # UCI: DOI:10.24432/C5MG6K, E. Alpaydin, Fevzi. Alimoglu, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/pendigits_label.npy"):
                pendigits = fetch_ucirepo(id=81)
                X = pendigits.data.features.to_numpy()
                y = pendigits.data.targets.to_numpy()
                np.save("./data/uci_download/pendigits_data.npy", X)
                np.save("./data/uci_download/pendigits_label.npy", y)
            X = np.load("./data/uci_download/pendigits_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/pendigits_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "magic_gamma": # UCI: DOI:10.24432/C52C8B, R. Bock, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/magic_gamma_label.npy"):
                magic_gamma = fetch_ucirepo(id=159)
                X = magic_gamma.data.features.to_numpy()
                y_name = magic_gamma.data.targets.to_numpy()
                y_integer = []
                for i in range(len(y_name)):
                    if y_name[i] == "g":
                        y_integer.append(1)
                    elif y_name[i] == "h":
                        y_integer.append(0)
                    else:
                        print(f"found {y_name[i]} label")
                        raise UnexpectedException
                y = np.array(y_integer)
                np.save("./data/uci_download/magic_gamma_data.npy", X)
                np.save("./data/uci_download/magic_gamma_label.npy", y)
            X = np.load("./data/uci_download/magic_gamma_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/magic_gamma_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "wine_quality": # UCI: DOI:10.24432/C56S3T, Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/wine_quality_label.npy"):
                wine_quality = fetch_ucirepo(id=186)
                X = wine_quality.data.features.to_numpy()
                y = wine_quality.data.targets.to_numpy()
                np.save("./data/uci_download/wine_quality_data.npy", X)
                np.save("./data/uci_download/wine_quality_label.npy", y)
            X = np.load("./data/uci_download/wine_quality_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/wine_quality_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "isolet": # UCI: DOI:10.24432/C51G69, Ron Cole, Mark Fanty, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/isolet_label.npy"):
                isolet = fetch_ucirepo(id=54)
                X = isolet.data.features.to_numpy()
                y = isolet.data.targets.to_numpy()
                np.save("./data/uci_download/isolet_data.npy", X)
                np.save("./data/uci_download/isolet_label.npy", y)
            X = np.load("./data/uci_download/isolet_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/isolet_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "sensorless": # UCI: DOI:10.24432/C5VP5F, Martyna Bator, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.,  added commas for proper import
            file = open("./data/uci/Sensorless_drive_diagnosis.txt", "r")
        elif dsname == "har": # UCI: DOI:10.24432/C54S4K, Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            X = []
            file = open("./data/uci/HAR/X_test.txt")
            for line in file:
                linesplit = line.split(" ")
                x = []
                for split in linesplit:
                    if len(split) > 0:
                        x.append(float(split))
                X.append(x)
            file = open("./data/uci/HAR/X_train.txt")
            for line in file:
                linesplit = line.split(" ")
                x = []
                for split in linesplit:
                    if len(split) > 0:
                        x.append(float(split))
                X.append(x)
            X = np.array(X)
            #X_train = np.loadtxt("./data/uci/HAR/X_train.txt", delimiter=" ", skiprows=0, dtype=str)
            y_test = np.loadtxt("./data/uci/HAR/y_test.txt", delimiter=" ", skiprows=0, dtype=np.int32)
            y_train = np.loadtxt("./data/uci/HAR/y_train.txt", delimiter=" ", skiprows=0, dtype=np.int32)
            y = y_test.tolist()
            y.extend(y_train.tolist())
            y = np.array(y).reshape(1, len(y))[0]
            return X,y
        elif dsname == "densired":
            file = open("./data/synth/densired_no_noise.txt", "r")
        elif dsname == "densired_noise":
            file = open("./data/synth/densired_noise.txt", "r")
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
            if (not j[len(j) - 1].startswith("noise") and not j[len(j) - 1].startswith("-1")):
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

    file.close()

    return np.array(x), np.array(label).reshape(1, len(label))[0]

def make_densired_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=50, ratio_noise = 0.1, max_retry=5, dens_factors=[1,1,0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True,
                   clunum= 10, seed = 6, core_num= 200, momentum=[0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
                   branch=[0,0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
                   con_min_dist=0.8, verbose=True, safety=True, domain_size = 20)
    data = densired_gen.generate_data(20000)
    print(data.shape)
    with open("./data/synth/densired_noise.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)
    with open("./data/synth/densired_no_noise.txt", 'w') as f:
        for x in data:
            if x[-1] >= 0:
                strx = ""
                for xi in x:
                    strx += str(xi) + ","
                strx = strx[:-1] + "\n"
                f.write(strx)

def load_data(dsname):
    scaler = MinMaxScaler()
    data, labels = read_file(dsname)
    scaler.fit(data)
    data = scaler.transform(data)
    label_max = max(np.unique(labels)) + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = label_max
            label_max += 1
    return data, labels

if __name__ == "__main__":
    # make_densired_ds()
    # #
    ds = "densired_noise"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)