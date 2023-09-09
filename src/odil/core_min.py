import numpy as np


class Domain():

    def __init__(self, domain):
        self.ndim = np.copy(domain.ndim)
        self.lower = np.copy(domain.lower)
        self.upper = np.copy(domain.upper)
        self.varnames = np.copy(domain.varnames)
        self.fieldnames = np.copy(domain.fieldnames)
        self.dtype = domain.dtype
        self.shape = domain.shape

    def step_by_dim(self, i):
        return (self.upper[i] - self.lower[i]) / self.shape[i]

    def cell_center_1d(self, i):
        s = self.shape[i]
        x = np.asarray(self.lower[i] + (np.arange(s) + 0.5) / s *
                       (self.upper[i] - self.lower[i]),
                       dtype=self.dtype)
        return x

    def cell_center_all(self):
        xx = [self.cell_center_1d(i) for i in range(self.ndim)]
        res = np.meshgrid(*xx, indexing='ij')
        return res
