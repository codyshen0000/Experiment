import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data
from data import benchmark


class Benchmark(benchmark.Benchmark):
    def __init__(self, args, name='', train=True, benchmark=True):
        self.q_factor = int(name.split('Q')[-1])
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_LU')
        self.ext = ('.png', '.jpeg')
