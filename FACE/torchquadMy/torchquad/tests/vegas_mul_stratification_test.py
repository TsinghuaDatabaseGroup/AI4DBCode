import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.vegas_mul_stratification import VEGASMultiStratification

from helper_functions import setup_test_for_backend
import torch
from integration.rng import RNG

n = 200
dim = 6
rng = RNG(backend="torch", seed=123)

vstrat = VEGASMultiStratification(n=n, N_increment=20000, dim=dim, rng=rng, backend="torch", dtype=torch.float, beta=0.75)

# Test if get_NH works correctly for a fresh VEGASStratification
neval = vstrat.get_NH(40000)
# assert neval.dtype == dtype_int
assert neval.shape == (n, vstrat.N_cubes,)
# assert (
#     anp.max(anp.abs(neval - neval[0])) == 0
# ), "Varying number of evaluations per hypercube for a fresh VEGASStratification"


print(vstrat.dh.shape)
print("neval shape", neval.shape)

# Test if sample point calculation works correctly for a
# fresh VEGASStratification
y = vstrat.get_Y(neval)
# print("y shape", y.shape)
# assert y.dtype == dtype_float
assert y.shape == (n, anp.sum(neval[0,:]), dim)
# assert anp.all(y >= 0.0) and anp.all(y <= 1.0), "Sample points are out of bounds"


# Test accumulate_weight
# Use exp to get a peak in a corner
f_eval = anp.prod(anp.exp(y), axis=2)
jf, jf2 = vstrat.accumulate_weight(neval, f_eval)
# assert jf.dtype == jf2.dtype == dtype_float
# assert jf.shape == jf2.shape == (vstrat.N_cubes,)
assert anp.min(jf2) >= 0.0, "Sums of squared values should be non-negative"
assert (
        anp.min(jf ** 2 - jf2) >= 0.0
), "Squared sums should be bigger than summed squares"



# Test the dampened sample counts update
vstrat.update_DH()


# Test if get_NH still works correctly
neval = vstrat.get_NH(40000)
assert neval[0,-1] > neval[0,0], "The hypercube at the peak should have more points"



# Test if sample point calculation still works correctly
y = vstrat.get_Y(neval)
assert anp.all(y >= 0.0) and anp.all(y <= 1.0), "Sample points are out of bounds"


