from data_handler import load_data
from subset_handler import load_subset


class OptimizationRunHandler:
    def __init__(self, args):
        ds = args.ds
        size = args.size
        sampling = args.sampling
        data_seed = args.data_seed

        if size == 1.0:
            self.data_points, self.labels = load_data(ds)
        else:
            self.data_points, self.labels = load_subset(ds, size, sampling, data_seed)

    def get_data(self):
        return self.data_points, self.labels
