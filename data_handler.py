import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_file(dsname):
    dic = {}
    classmember = 0

    try:
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