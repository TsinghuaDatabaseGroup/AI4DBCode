from autoray import numpy as anp
from autoray import astype

from .utils import _add_at_indices
import torch
import time
class VEGASMultiStratification:

    def __init__(self, n, N_increment, dim, rng, backend, dtype, beta=0.75):
        """Initialize the VEGAS stratification.

        Args:
            n (int): Number of integrators
            N_increment (int): Number of evaluations per iteration.
            dim (int): Dimensionality
            rng (RNG): Random number generator
            backend (string): Numerical backend
            dtype (backend dtype): dtype used for the calculations
            beta (float, optional): Beta parameter from VEGAS Enhanced. Defaults to 0.75.
        """
        self.n = n
        self.rng = rng
        self.dim = dim
        # stratification steps per dim, EQ 41
        self.N_strat = int((N_increment / 4.0) ** (1.0 / dim))
        self.N_strat = 1000 if self.N_strat > 1000 else self.N_strat
        self.beta = beta  # variable controlling adaptiveness in stratification 0 to 1
        self.N_cubes = self.N_strat**self.dim  # total number of subdomains
        self.V_cubes = (1.0 / self.N_strat) ** self.dim  # volume of hypercubes

        self.dtype = dtype
        self.backend = backend


        # jacobian times f eval and jacobian^2 times f
        # (self.n, self.N_cubes)
        self.JF = anp.zeros([self.n, self.N_cubes], dtype=self.dtype, like=backend)
        self.JF2 = anp.zeros([self.n, self.N_cubes], dtype=self.dtype, like=backend)

        # dampened counts
        # (self.n, self.N_cubes)
        self.dh = (
            anp.ones([self.n, self.N_cubes], dtype=self.dtype, like=backend)
            * 1.0
            / self.N_cubes
        )

        # current index counts as floating point numbers
        # (self.n, self.N_cubes)
        self.strat_counts = anp.zeros([self.n, self.N_cubes], dtype=self.dtype, like=backend)


        nevals_arange = torch.arange(self.N_cubes, dtype=self.dtype)
        # (N_cubes, dim)
        self.positions = self._get_indices(nevals_arange)


    def accumulate_weight(self, nevals, weight_all_cubes):
        """Accumulate weights for the cubes.

        Args:
            nevals (backend tensor): Number of evals belonging to each cube (sorted).  （n, N_cube)
            weight_all_cubes (backend tensor): Function values. (n, total bla)
            【must be sorted! according to indices】


        Returns:
            backend tensor, backend tensor: Computed JF and JF2  (n, N_cube)
        """
        # indices maps each index of weight_all_cubes to the corresponding
        # hypercube index.

        N_cubes_arange = anp.arange(self.N_cubes, dtype=nevals.dtype, like=self.backend)


        N_cubes_arange_n = anp.repeat(N_cubes_arange.view(1, -1), self.n, axis=0).view(-1)
        indices = anp.repeat(N_cubes_arange_n, nevals.view(-1))

        indices = indices.reshape(self.n, -1)



        # Reset JF and JF2, and accumulate the weights and squared weights
        # into them.
        self.JF = anp.zeros([self.n, self.N_cubes], dtype=self.dtype, like=self.backend)
        self.JF2 = anp.zeros([self.n, self.N_cubes], dtype=self.dtype, like=self.backend)


        integrator_id = torch.arange(self.n, dtype=int)
        integrator_id = integrator_id.reshape(-1, 1)

        # (n, total_bla)
        integrator_id = anp.repeat(integrator_id, indices.shape[1], axis=1)


        JF_shape = self.JF.shape
        weight_all_cubes = weight_all_cubes.view(-1)
        # (0,0,0……, n-1, n-1)
        integrator_id = integrator_id.view(-1)
        # (0,0,0, ……, N_cubes-1, N_cubes-1, ……,  0, 0, 0, N_cubes-1)
        indices = indices.view(-1)
        idx = integrator_id * self.N_cubes + indices

        self.JF = self.JF.view(-1)
        self.JF2 = self.JF2.view(-1)

        # self.JF.index_add_(dim=0, index=idx, source=weight_all_cubes)
        # self.JF2.index_add_(dim=0, index=idx, source=weight_all_cubes**2)
        self.JF.scatter_add_(dim=0, index=idx, src=weight_all_cubes)
        self.JF2.scatter_add_(dim=0, index=idx, src=weight_all_cubes ** 2)

        self.JF = self.JF.view(JF_shape)
        self.JF2 = self.JF2.view(JF_shape)


        # Store counts
        self.strat_counts = astype(nevals, self.dtype)

        return self.JF, self.JF2

    def update_DH(self):
        """Update the dampened sample counts."""

        # EQ 42
        V2 = self.V_cubes * self.V_cubes
        d_tmp = (
            V2 * self.JF2 / self.strat_counts
            - (self.V_cubes * self.JF / self.strat_counts) ** 2
        )
        # Sometimes rounding errors produce negative values very close to 0
        d_tmp[d_tmp < 0.0] = 0.0

        # (n, N_cubes)
        self.dh = d_tmp**self.beta

        # Normalize dampening
        # (n, 1)
        d_sum = anp.sum(self.dh,axis=1).view(-1,1)


        assert torch.count_nonzero(d_sum) ==d_sum.shape[0]
        self.dh = self.dh / d_sum




    def get_NH(self, nevals_exp):
        """Recalculate sample points per hypercube, EQ 44.

        Args:
            nevals_exp (int): Expected number of evaluations.

        Returns:
            backend tensor: Stratified sample counts per cube.
        """
        # (n, N_cubes)

        st_time = time.time()
        nh = anp.floor(self.dh * nevals_exp)
        nh = anp.clip(nh, 2, None)
        en_time = time.time()
        print("get nh originally took ", en_time - st_time)




        st_time = time.time()

        cur_nevals = torch.sum(nh, dim=1)
        max_nevals = cur_nevals.max()

        delta_nevals = max_nevals - cur_nevals


        assert delta_nevals.min() >=0
        delta_nevals = delta_nevals.int()

        nh = astype(nh, "int64")


        pst = time.time()
        for i in range(self.n):
            if delta_nevals[i] > 0:

                # ids = ids_[:delta_nevals[i]]
                weights = self.dh[i,:]
                ids = torch.multinomial(weights,  num_samples=delta_nevals[i], replacement=True)

                # nh[i,:].index_add_(0, ids, torch.ones_like(ids, dtype=nh.dtype))
                nh[i, :].scatter_add_(0, ids, torch.ones_like(ids, dtype=nh.dtype))
        print("For took ",time.time() - pst)

        cur_nevals = torch.sum(nh, dim=1)
        delta_nevals = max_nevals - cur_nevals
        assert delta_nevals.count_nonzero() ==0
        en_time = time.time()
        print("get nh additionally took ",en_time-st_time)
        return astype(nh, "int64")

    def _get_indices(self, idx):
        """Maps point to stratified point.

        Args:
            idx (int backend tensor): Target points indices.

        Returns:
            int backend tensor: Mapped points.
        """
        # A commented-out alternative way for mapped points calculation if
        # idx is anp.arange(len(nevals), like=nevals).
        # torch.meshgrid's indexing argument was added in version 1.10.1,
        # so don't use it yet.
        """
        grid_1d = anp.arange(self.N_strat, like=self.backend)
        points = anp.meshgrid(*([grid_1d] * self.dim), indexing="xy", like=self.backend)
        points = anp.stack(
            [mg.ravel() for mg in points], axis=1, like=self.backend
        )
        return points
        """
        # Repeat idx via broadcasting and divide it by self.N_strat ** d
        # for all dimensions d
        points = anp.reshape(idx, [idx.shape[0], 1])
        strides = self.N_strat ** anp.arange(self.dim, like=points)
        if self.backend == "torch":
            # Torch shows a compatibility warning with //, so use torch.div
            # instead
            points = anp.div(points, strides, rounding_mode="floor")
        else:
            points = points // strides
        # Calculate the component-wise remainder: points mod self.N_strat
        points[:, :-1] = points[:, :-1] - self.N_strat * points[:, 1:]

        return points

    def get_Y(self, nevals):
        """Compute randomly sampled points.

        Args:
            nevals (int backend tensor): Number of samples to draw per stratification cube.  (n, N_cubes)

        Returns:
            backend tensor: Sampled points.
        """
        # Get integer positions for each hypercube

        assert self.N_cubes == nevals.shape[1], print("Not every cube is sampled! ", self.N_cubes, nevals.shape[1])
        nevals_arange = torch.arange(self.N_cubes, dtype=int)

        # (n, self.N_cubes)
        print("nevals.shape", nevals.shape)

        N_cubes_arange_n = anp.repeat(nevals_arange.view(1, -1), self.n, axis=0).view(-1)
        indices = anp.repeat(N_cubes_arange_n, nevals.view(-1))
        positions = self.positions[indices]
        # (n, bla, dim)
        positions = positions.reshape(self.n, -1, self.dim)
        # (n, bla, dim)
        random_uni = self.rng.uniform(
            size=[positions.shape[0],positions.shape[1], positions.shape[2]], dtype=self.dtype
        )
        positions = (positions + random_uni) / self.N_strat
        positions[positions >= 1.0] = 0.999999
        positions_list = positions

        return positions_list
