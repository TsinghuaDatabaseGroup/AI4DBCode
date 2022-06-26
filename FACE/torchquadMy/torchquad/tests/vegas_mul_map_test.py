import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.vegas_mul_map import VEGASMultiMap

from helper_functions import setup_test_for_backend
import torch

# n, N_intervals, dim, backend, dtype, alpha=0.5

# VEGASMultiMap(n=200, N_intervals=1000, dim=6, backend="torch", dtype=torch.cuda.FloatTensor)
vmap = VEGASMultiMap(n=200, N_intervals=1000, dim=6, backend="torch", dtype=torch.float)


# rng_y = torch.rand((200, 6, 2000))
rng_y = torch.rand((6, 200, 2000))

# print(rng_y.min(), rng_y.max())
rng_x = vmap.get_X(rng_y)
# print(rng_x)

jf2 = torch.rand_like(rng_y)

vmap.accumulate_weight(rng_y, jf2)


VEGASMultiMap._smooth_map(vmap.weights, vmap.counts, 0.5)

vmap.update_map()




