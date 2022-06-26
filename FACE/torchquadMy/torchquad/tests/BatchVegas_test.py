import sys

sys.path.append("../")
import time
from autoray import numpy as anp
from autoray import to_backend_dtype, astype
import timeit
import cProfile
import pstats
import torch
from unittest.mock import patch

from integration.vegas import VEGAS
from integration.rng import RNG
from integration.BatchVegas import BatchVEGAS

import integration.utils as utils

DEVICE= "cuda"

from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
    get_test_functions

)


def f_batch(inp):
    print(inp.shape)
    inp[:, 2] = inp[:, 2] * 2
    print(torch.sum(inp, axis=1).shape)
    return torch.sum(inp, axis=1)
    # print(torch.prod(inp, axis=1).shape)
    # return torch.prod(inp, axis=1)

def _run_simple_funcs_of_BatchVegas():
    z = BatchVEGAS()

    dim = 10
    n = 200
    N = 200000
    # legal_tensors = torch.Tensor(n * [dim * [[0,1]] ]).to('cuda')
    legal_tensors = torch.Tensor(n * [dim * [[0,1]] ]).to(DEVICE)
    legal_tensors[0, :] = torch.Tensor([dim * [[0, 0.5]]]).to(DEVICE)
    legal_tensors[1, :] = torch.Tensor([dim * [[0.2, 0.5]]]).to(DEVICE)
    print(legal_tensors.shape)

    # f_batch = get_test_functions(dim, "torch")[0]

    full_integration_domain = torch.Tensor(dim * [[0,1]])


    vegas = VEGAS()
    bigN = 100000 * 40

    domain_starts = full_integration_domain[:, 0]
    domain_sizes = full_integration_domain[:, 1] - domain_starts
    domain_volume = torch.prod(domain_sizes)
    result = vegas.integrate(f_batch, dim=dim,
                             N=bigN,
                             integration_domain=full_integration_domain,
                             #                              use_warmup=True,
                             use_warmup=True,
                             use_grid_improve=True,
                             max_iterations=20
                             #                              backend='torch'
                             )
    print('result is ', result)


    z.setValues(f_batch,
                dim=dim,
                N=N,
                n=n,
                integration_domains=legal_tensors,
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                target_map=vegas.map,
                target_domain_starts = domain_starts,
                target_domain_sizes = domain_sizes
                )


    # z.setValues(f_batch,
    #             dim=dim,
    #             N=N,
    #             n=n,
    #             integration_domains=legal_tensors,
    #             rng=None,
    #             seed=1234
    #             )
    # z.integrate()
    ret = z.integrate1()

    print("see ret")
    print(ret)
    # BVegas.

def _run_example_integrations(backend, dtype_name):
    """Test the integrate method in VEGAS for the given backend and example test functions using compute_integration_test_errors"""
    print(f"Testing VEGAS+ with example functions with {backend}, {dtype_name}")
    vegas = VEGAS()

    # 1D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 1, "seed": 0},
        dim=1,
        use_complex=False,
        backend=backend,
    )
    print("1D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:3]:
        assert error < 5e-3

    for error in errors:
        assert error < 9.0

    for error in errors[6:]:
        assert error < 6e-3

    # 3D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 3, "seed": 0},
        dim=3,
        use_complex=False,
        backend=backend,
    )
    print("3D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 0.61

    # 10D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 10, "seed": 0},
        dim=10,
        use_complex=False,
        backend=backend,
    )
    print("10D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 12.5


def _run_BatchVegas_tests(backend, dtype_name):
    utils.add_time = 0
    st = time.time()
    """Test if VEGAS+ works with example functions and is accurate as expected"""
    # _run_vegas_accuracy_checks(backend, dtype_name)
    _run_simple_funcs_of_BatchVegas()
    print("Total add_time is ",utils.add_time)
    en = time.time()
    print("Total time is ", en-st)
    # _run_example_integrations(backend, dtype_name)

test_integrate_torch = setup_test_for_backend(_run_BatchVegas_tests, "torch", "float64")


if __name__ == "__main__":
    # used to run this test individually
    # test_integrate_numpy()

    profile_torch = False

    if profile_torch:
        profiler = cProfile.Profile()
        profiler.enable()
        start = timeit.default_timer()
        test_integrate_torch()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats()
        stop = timeit.default_timer()
        print("Test ran for ", stop - start, " seconds.")
    else:
        test_integrate_torch()
