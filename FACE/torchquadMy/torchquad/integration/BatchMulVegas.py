
from autoray import numpy as anp
from autoray import infer_backend, astype
from loguru import logger


from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain
from .vegas import VEGAS
import torch
from .rng import RNG
from .vegas_mul_map import VEGASMultiMap
from .vegas_mul_stratification import VEGASMultiStratification


class BatchMulVEGAS(BaseIntegrator):
    # Number of different vegas integrators
    _n = None

    # Number of sampling points used for each integrator
    _N = None

    # Integration domain list  (backend tensor)
    _integration_domains = None

    def __init__(self):
        super().__init__()


    def setValues(self,
                  fn,
                  dim,
                  N,
                  n,
                  integration_domains,
                  iterations=20,
                  backend="torch",
                  dtype="float32",
                  eps_rel=0,
                  alpha=0.5,
                  beta=0.75,
                  target_map=None,
                  target_domain_starts=None,
                  target_domain_sizes=None,
                  reuse_sample_points=False,
                  rng=None,
                  seed=None,
                  ):
        """ Initialize required attributes"""
        self._fn = fn
        self._dim = dim
        self._N = N
        self._n = n
        # (n, dim, 2)
        self._integration_domains = integration_domains

        self._iterations = iterations
        self.it = 0

        self._backend = backend
        self._dtype = integration_domains[0].dtype
        if rng is None:
            rng = RNG(backend=self._backend, seed=seed)
        elif seed is not None:
            raise ValueError("seed and rng cannot both be passed")
        self.rng = rng


        self._N_increment = N // (self._iterations )
        self._starting_N_val = N // (self._iterations )
        self._starting_N = self._starting_N_val
        self.N_intervals = max(2, self._N_increment // 10)  # for small N intervals set 2

        self.use_grid_improve = True
        self._eps_rel = eps_rel
        self._alpha = alpha
        self._beta = beta


        self.results = torch.zeros((self._iterations, self._n))
        self.sigma2  = torch.zeros((self._iterations, self._n))

        """ variables related to transfer map"""
        self.target_map = target_map
        self.reuse_sample_points = reuse_sample_points
        # (dim)
        self.target_domain_starts = target_domain_starts
        # (dim)
        self.target_domain_sizes = target_domain_sizes




    def integrate(self):
        self._check_inputs(dim=self._dim,
                           N=self._N,
                           integration_domain=self._integration_domains[0,:])


        # self._domain_starts = torch.empty((self._dim, self._n))
        self._domain_starts = torch.empty((self._n, self._dim))
        self._domain_sizes = torch.empty_like(self._domain_starts)
        self._domain_volume = torch.empty((self._n))


        for i, integration_domain in enumerate(self._integration_domains):
            # Transform the integrand into the [0,1]^dim domain
            domain_starts = integration_domain[:, 0]
            domain_sizes = integration_domain[:, 1] - domain_starts
            domain_volume = anp.prod(domain_sizes)
            # (n, dim)
            self._domain_starts[i,:] = domain_starts
            self._domain_sizes[i,:]  = domain_sizes
            # (n)
            self._domain_volume[i]   = domain_volume

        # (dim,n)
        self._domain_starts = self._domain_starts.permute(1,0)
        self._domain_sizes = self._domain_sizes.permute(1,0)
        # (dim, n, 1)
        self._domain_starts = self._domain_starts.unsqueeze(2)
        self._domain_sizes = self._domain_sizes.unsqueeze(2)
        # (n, 1)
        self._domain_volume = self._domain_volume.unsqueeze(1)


        self.map = VEGASMultiMap(
            n=self._n,
            alpha=self._alpha,
            N_intervals=self.N_intervals,
            dim=self._dim,
            backend=self._backend,
            dtype=self._dtype)

        self.strat = VEGASMultiStratification(
            n=self._n,
            N_increment=self._N_increment,
            dim=self._dim,
            rng=self.rng,
            backend=self._backend,
            dtype=self._dtype,
            beta=self._beta)


        if self.reuse_sample_points:
            # self.transfer_map(self.target_map)
            self.transfer_map_vec(self.target_map)

        # Main loop
        while True:
            print("#"*40)
            print("    Current iteration is 【{}】".format(self.it))
            print("#"*40)
            # Compute current iteration for every integrator
            self._run_iteration()
            self.it = self.it + 1
            if self.it >= self._iterations:
                break

        return self._get_result()

    def _run_iteration(self):

        self._generate_sampling_points()
        self._evaluate_sampling_points()
        self._update_each_integrator()


        return

    def _generate_sampling_points(self):
        # (n, N_cubes)
        self.nevals = self.strat.get_NH(self._starting_N)

        # (n, bla, dim)
        self.y = self.strat.get_Y(self.nevals)
        # (dim, n, bla)
        self.y = self.y.permute(2,0,1)
        # (dim, n, bla)  [0,1]的
        self.x = self.map.get_X(self.y)

        # (dim, n, bla)  变回真正范围的
        self.x = (self.x * self._domain_sizes + self._domain_starts)


    def _evaluate_sampling_points(self):
        # (n, bla ,dim)
        inp = self.x.permute(1,2,0)

        # (total_n, dim)
        inp = inp.view(-1, inp.shape[-1])

        # (total_n, 1)
        self.f_eval = self._eval(inp)

        # (n, bla)
        self.f_eval = self.f_eval.view(self._n, -1)
        print("Classic f_eval ", self.f_eval.max(), self.f_eval.median())

    def _update_each_integrator(self):
        # self.f_eval shape:  (n, bla)
        self.f_eval = self.f_eval * self._domain_volume
        # (n, bla)
        jac = self.map.get_Jac(self.y)
        jf_vec = self.f_eval * jac
        jf_vec2 = jf_vec ** 2
        jf_vec2 = jf_vec2.detach()

        if self.use_grid_improve:
            self.map.accumulate_weight(self.y, jf_vec2)
        jf, jf2 = self.strat.accumulate_weight(self.nevals, jf_vec)  # update strat
        neval_inverse = 1.0 / astype(self.nevals, self.y.dtype)

        ih = jf * (neval_inverse * self.strat.V_cubes)  # Compute integral per cube

        # Collect results
        sig2 = jf2 * neval_inverse * (self.strat.V_cubes ** 2) - pow(ih, 2)
        sig2 = sig2.detach()

        # Sometimes rounding errors produce negative values very close to 0
        sig2 = anp.abs(sig2)
        self.results[self.it, :] = ih.sum(axis=1)  # store results
        self.sigma2[self.it, :] = (sig2 * neval_inverse).sum(axis=1)

        if self.use_grid_improve:  # if on, update adaptive map
            logger.debug("Running grid improvement")
            self.map.update_map()

        self.strat.update_DH()  # update stratification

        # Estimate an accuracy for the logging
        # acc = anp.sqrt(self.sigma2[self.it, :])
        # if torch.count_nonzero(self.results[self.it,:]) == self._n:
        #     acc = acc/ anp.abs(self.results[self.it,:])


    def _get_error(self):
        """Estimates error from variance , EQ 31.

        Returns:
            backend-specific float: Estimated error.

        """
        # Skip variances which are zero and return a backend-specific float

        res = torch.sum(1./self.sigma2, axis=1)
        res = 1./anp.sqrt(res)
        return res


    def _get_chisq(self):
        """Computes chi square from estimated integral and variance, EQ 32.

        Returns:
            backend-specific float: Chi squared.
        """

        use_results = self.results.clone()
        use_results[0,:] = 0

        use_results = use_results.permute(1,0)
        use_sigma2 = self.sigma2.clone()
        use_sigma2 = use_sigma2.permute(1,0)

        res = torch.sum(
            (use_results - self.res_abs) ** 2 / use_sigma2, axis=1
        )
        return res


    def _enhance_iterations(self):
        """ Every fifth iteration reset the sample integral results
                and adjust the number of evaluations per iteration

                Returns:
                    Bool: True iff VEGAS should abort
                """
        # Abort only every fifth iteration
        if self.it % 5 > 0 or self.it >= self._iterations or self.it <=0:
            return False


        self._starting_N += self._N_increment
        return False

    def _get_result(self):

        results = self.results.permute(1,0)
        sigma2 = self.sigma2.permute(1,0)



        res_num = torch.sum(results/sigma2, dim=1)
        res_den = torch.sum(1./sigma2, dim=1)
        res = res_num / res_den
        return res

    """ helper functions"""
    def get_dx_x_edges(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (self._dim, 2)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self._dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self._dim
        """
        ret_x_edges  = torch.empty((n_edges, self._dim))
        # ret_x_edges = torch.empty((self._dim, n_edges))


        for i in range(self._dim):
            if y_lims[i,1] > 0.999999:
                y_lims[i,1] = 0.999999
            ret_x_edges[:,i] = torch.linspace(y_lims[i, 0], y_lims[i, 1], n_edges)

        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_dx_edges = ret_x_edges[1:, :] - ret_x_edges[:-1, :]


        _tmp_max, _ = ret_x_edges.max(dim = 0)
        _tmp_min, _ = ret_x_edges.min(dim = 0)
        siz = _tmp_max - _tmp_min

        # siz = siz / (self.target_domain_sizes)
        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges


    def get_dx_x_edges_vec(self, target_map, n_edges, y_lims):
        """
        Args:
            target_map (VEGASMap)   : target map that this range(y_lims) should be transformed from
            n_interval (backend int): number of edges
            y_lims (backend tensor):  range limits (lower bound & upper bound) for each dimension   size: (n, 2, self._dim)

        Returns:
            dx_edges (backend tensor) : dx of edges for each dimension   size: (n_edges - 1) * self._dim
            x_edges  (backend tensor) : edges for each dimension         size: n_edges * self._dim
        """
        ret_x_edges  = torch.empty((self._n, n_edges, self._dim))
        # ret_x_edges = torch.empty((self._dim, n_edges))

        # (n, 2, dim)
        y_lims[:, 1, :] = torch.clamp(y_lims[:, 1, :], max=0.999999)


        for i in range(self._n):
            for j in range(self._dim):
                ret_x_edges[i, :, j] = torch.linspace(y_lims[i, 0, j], y_lims[i, 1, j], n_edges)

        ret_x_edges = ret_x_edges.view(-1, self._dim)

        ret_x_edges = target_map.get_X(ret_x_edges)

        ret_x_edges = ret_x_edges.view(self._n, n_edges, self._dim)
        ret_dx_edges = ret_x_edges[:,1:,:] -  ret_x_edges[:, :-1, :]
        # ret_dx_edges = ret_x_edges[1:, :] - ret_x_edges[:-1, :]
        assert ret_dx_edges.shape == (self._n, n_edges-1, self._dim)


        _tmp_max, _ = ret_x_edges.max(dim = 1)
        _tmp_min, _ = ret_x_edges.min(dim = 1)
        # print("see tmp_max shape", _tmp_max.shape)
        _tmp_max = _tmp_max.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])
        _tmp_min = _tmp_min.view(ret_x_edges.shape[0], 1, ret_x_edges.shape[2])

        siz = _tmp_max - _tmp_min

        print("shapes ", ret_x_edges.shape, _tmp_max.shape, _tmp_min.shape, siz.shape)
        # siz = siz / (self.target_domain_sizes)
        ret_x_edges = (ret_x_edges - _tmp_min) / siz
        ret_dx_edges = ret_dx_edges / siz

        return ret_dx_edges, ret_x_edges

    """ new helper function 3: re-initialize maps in map_list using full_map  """
    def transfer_map(self, full_map):
        # TODO: vectorization
        for i in range(self._n):
            # 20,10,2    n, dim, min|max
            # if i==0:
            #     continue


            this_x = self._integration_domains[i, :].T   # (2,dim)
            this_x = (this_x - self.target_domain_starts) / self.target_domain_sizes # (2,dim)


            ys = full_map.get_Y(this_x).T
            # print("see ys")
            # print("ys.shape ", ys.shape)
            # print(ys)

            _dx_edges, _x_edges = self.get_dx_x_edges(full_map, self.N_intervals + 1, ys)

            self.map.set_map_edges_i(i, dx_edges=_dx_edges, x_edges=_x_edges)

        return



    def transfer_map_vec(self, full_map):
        """ vectorize version of transfer_map"""


        x_lims = self._integration_domains.permute(0, 2, 1)

        x_lims = (x_lims - self.target_domain_starts) / self.target_domain_sizes # (n, 2, dim)


        x_lims = x_lims.reshape(-1, x_lims.shape[-1])


        y_lims = full_map.get_Y(x_lims)
        _dx_edges, _x_edges = self.get_dx_x_edges_vec(full_map, self.N_intervals + 1, y_lims.view(self._n, 2, -1))


        self.map.set_map_edges(dx_edges=_dx_edges, x_edges=_x_edges)

        print(x_lims)



