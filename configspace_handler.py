import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition


class_num = {}
class_num["complex9"] = 9
class_num["diamond9"] = 9
class_num["letter"] = 26
class_num["EEG Eye State"] = 2

def get_configspace(method, ds, data_points):
    config_space = ConfigurationSpace()
    if ds in class_num.keys():
        num_classes = class_num[ds]
    else:
        raise NotImplementedError(ds)
    max_nn = max(((math.floor((math.log2(len(data_points))))**2) + 5), 100)
    d = len(data_points[0])

    if method == "dbscan":
        dbscan_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        dbscan_2 = Integer("min_samples", (1, 100), default=5)
        config_space.add([dbscan_1, dbscan_2])
    else:
        raise NotImplementedError(method)
    return config_space