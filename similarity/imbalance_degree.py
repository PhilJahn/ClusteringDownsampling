import numpy as np

# histogram to use for comparison with balanced distribution
def imbalance_degree_histogram(hist):
    labels = []
    for i in range(len(hist)):
        labels.extend([i]*hist[i])
    return imbalance_degree(labels)

# reimplementation from Measuring the class-imbalance extent of multi-class problems
def imbalance_degree(labels):

    uniq, counts = np.unique(labels, return_counts=True)

    class_ratio = []
    for i in range(len(counts)):
        class_ratio.append(counts[i] / len(labels))

    expected = 1 / len(np.unique(labels))
    num_major = 0
    for i in range(len(class_ratio)):
        if class_ratio[i] >= expected:
            num_major += 1

    if num_major != len(class_ratio):
        e = [1 / len(counts)] * len(counts)
        upper_diff = np.sum(np.abs(np.subtract(class_ratio, e))) / 2  # distance.euclidean(class_ratio, e)

        num_minor = len(counts) - num_major

        most_distant_dist = [1 - ((num_major - 1) / len(counts))] * 1 + [1 / len(counts)] * (num_major - 1) + [
            0] * num_minor
        # print(class_ratio)
        # print(most_distant_dist)
        lower_diff = np.sum(np.abs(np.subtract(most_distant_dist, e))) / 2
        # print(most_distant_dist, lower_diff)

        imb_degree = (upper_diff / lower_diff) + num_minor - 1
    else:
        imb_degree = 0

    return counts, imb_degree