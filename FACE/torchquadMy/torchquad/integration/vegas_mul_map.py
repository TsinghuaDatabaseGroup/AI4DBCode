
from autoray import numpy as anp
from autoray import astype, to_backend_dtype
from loguru import logger
import torch
from .utils import _add_at_indices


class VEGASMultiMap:

    def __init__(self, n, N_intervals, dim, backend, dtype, alpha=0.5) -> None:
        self.n = n
        self.dim = dim
        self.N_intervals = N_intervals  # # of subdivisions
        self.N_edges = self.N_intervals + 1  # # of subdivsion boundaries
        self.alpha = alpha  # Weight smoothing
        self.backend = backend
        self.dtype = dtype

        # Boundary locations x_edges and subdomain stepsizes dx_edges
        # Subdivide the domain [0,1]^dim equally spaced in N-d, EQ 8




        # (dim,n, N_intervals)
        self.dx_edges = (
                # anp.ones((self.n, self.dim, self.N_intervals), dtype=self.dtype, like=self.backend)
                anp.ones((self.dim, self.n, self.N_intervals), dtype=self.dtype, like=self.backend)
                / self.N_intervals
        )

        # N_edges
        x_edges_per_dim = anp.linspace(
            0.0, 1.0, self.N_edges, dtype=self.dtype, like=self.backend
        )
        # (1, 1, N_edges)
        tmp = anp.reshape(x_edges_per_dim, [1, 1, self.N_edges])
        # (1, n, N_edges)
        tmp = anp.repeat(tmp, self.n, axis=1)
        # (dim, n, N_edges)
        self.x_edges = anp.repeat(tmp, self.dim, axis=0)

        # # Initialize self.weights and self.counts
        self._reset_weight()

    def get_X(self, y):
        """Get mapped sampling points, EQ 9.

        Args:
            y (backend tensor): Randomly sampled location(s) 【shape】 (DIM, n, blabla)

        Returns:
            backend tensor: Mapped points.
        """


        # (DIM, n, bla)
        ID, offset = self._get_interval_ID(y), self._get_interval_offset(y)

        dim_id = torch.arange(self.dim)
        dim_id = anp.repeat(dim_id.reshape(-1, 1, 1), ID.shape[2],axis=2)
        dim_id = anp.repeat(dim_id, ID.shape[1], axis=1)

        integrator_id = torch.arange(ID.shape[1])
        integrator_id = anp.repeat(integrator_id.reshape(1, -1, 1), ID.shape[2], axis=2)
        integrator_id = anp.repeat(integrator_id, ID.shape[0], axis=0)


        res = self.x_edges[dim_id, integrator_id, ID] + self.dx_edges[dim_id, integrator_id, ID] * offset

        return res

    def get_Jac(self, y):
        """Computes the jacobian of the mapping transformation, EQ 12.

        Args:
            y : (DIM, n, bla)

        Returns:
            backend tensor: Jacobian  (n,bla)
        """
        # (DIM,n,bla)
        ID = self._get_interval_ID(y)

        dim_id = torch.arange(ID.shape[0])
        dim_id = anp.repeat(dim_id.reshape(-1, 1, 1), ID.shape[2], axis=2)
        dim_id = anp.repeat(dim_id, ID.shape[1], axis=1)

        integrator_id = torch.arange(ID.shape[1])
        integrator_id = anp.repeat(integrator_id.reshape(1, -1, 1), ID.shape[2], axis=2)
        integrator_id = anp.repeat(integrator_id, ID.shape[0], axis=0)

        # (DIM, n, bla)
        jac_tmp = self.N_intervals * self.dx_edges[dim_id, integrator_id, ID]

        # (n, bla, dim)
        jac_tmp= jac_tmp.permute(1, 2, 0)
        # (n, bla)
        jac = torch.prod(jac_tmp, dim=2)
        return jac


    def _get_interval_ID(self, y):
        """Get the integer part of the desired mapping , EQ 10.

        Args:
            y (backend tensor): Sampled points

        Returns:
            backend tensor: Integer part of mapped points.
        """
        return astype(anp.floor(y * float(self.N_intervals)), "int64")

    def _get_interval_offset(self, y):
        """Get the fractional part of the desired mapping , EQ 11.

        Args:
            y (backend tensor): Sampled points.

        Returns:
            backend tensor: Fractional part of mapped points.
        """
        y = y * float(self.N_intervals)
        return y - anp.floor(y)


    def _reset_weight(self):
        """Reset or initialize weights and counts."""
        # weights in each intervall

        # (dim, n, N_intervals)
        self.weights = anp.zeros(
            # (self.n, self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
            (self.dim, self.n, self.N_intervals), dtype=self.dtype, like=self.backend
        )
        # numbers of random samples in specific interval
        # (dim, n, N_intervals)
        self.counts = anp.zeros(
            (self.dim, self.n, self.N_intervals),
            dtype=to_backend_dtype("int64", like=self.backend),
            like=self.backend,
        )


    def accumulate_weight(self, y, jf_vec2):
        """Accumulate weights and counts of the map.

        Args:
            y (backend tensor): Sampled points.  (dim,n,bla)
            jf_vec2 (backend tensor): Square of the product of function values and jacobians  (n,bla)
        """

        ID = self._get_interval_ID(y)
        # (dim, n, bla)
        dim_id = torch.arange(self.dim)
        dim_id = anp.repeat(dim_id.reshape(-1, 1, 1), ID.shape[2], axis=2)
        dim_id = anp.repeat(dim_id, ID.shape[1], axis=1)

        integrator_id = torch.arange(ID.shape[1])
        integrator_id = anp.repeat(integrator_id.reshape(1, -1, 1), ID.shape[2], axis=2)
        integrator_id = anp.repeat(integrator_id, ID.shape[0], axis=0)

        ones = torch.ones_like(jf_vec2, dtype=self.counts.dtype)


        n = ID.shape[1]
        dim_id = dim_id.view(-1)
        integrator_id = integrator_id.view(-1)
        ID = ID.reshape(-1)


        # (1, n, bla)
        jf_vec2 = jf_vec2.unsqueeze(0)
        # (dim, n, bla)
        jf_vec2 = anp.repeat(jf_vec2, self.dim, axis=0)
        jf_vec2 = jf_vec2.view(-1)

        ones = ones.view(-1)
        ones = anp.repeat(ones, self.dim, axis=0)

        weights_shape = self.weights.shape
        counts_shape  = self.counts.shape

        self.weights = self.weights.view(-1)
        self.counts = self.counts.view(-1)
        idx = dim_id * n * self.N_intervals + integrator_id * self.N_intervals + ID

        # self.weights.index_add_(dim=0, index=idx, source=jf_vec2)
        # self.counts.index_add_(dim=0, index=idx, source=ones)

        self.weights.scatter_add_(dim=0, index=idx, src=jf_vec2)
        self.counts.scatter_add_(dim=0, index=idx, src=ones)

        self.weights = self.weights.view(weights_shape)
        self.counts  = self.counts.view(counts_shape)
        print("Total counts sum is ", self.counts.sum())


    @staticmethod
    def _smooth_map(weights, counts, alpha):
        z_idx = counts == 0  # zero count indices

        if anp.any(z_idx):
            print("******** ANY")
            nnz_idx = anp.logical_not(z_idx)

            weights[nnz_idx] /= counts[nnz_idx]
            logger.opt(lazy=True).debug(
                "The integrand was not evaluated in {z_idx_sum} of {num_weights} VEGASMap intervals. "
                "Filling the weights for some of them with neighbouring values.",
                z_idx_sum=lambda: anp.sum(z_idx),
                num_weights=lambda: counts.shape[0] * counts.shape[1],
            )
            # Set the weights of the intervals with zero count to weights from
            # their nearest neighbouring intervals
            # (up to a distance of 10 indices).
            for _ in range(10):
                weights[:,:, :-1] = anp.where(
                    z_idx[:,:, :-1], weights[:,:, 1:], weights[:,:, :-1]
                )
                # The asterisk corresponds to a logical And here
                z_idx[:,:, :-1] = z_idx[:,:, :-1] * z_idx[:,:, 1:]
                weights[:,:, 1:] = anp.where(
                    z_idx[:,:, 1:], weights[:,:, :-1], weights[:,:, 1:]
                )
                z_idx[:,:, 1:] = z_idx[:,:, 1:] * z_idx[:,:, :-1]
                logger.opt(lazy=True).debug(
                    "  remaining intervals: {z_idx_sum}",
                    z_idx_sum=lambda: anp.sum(z_idx, dim=2),
                )
                if not anp.any(z_idx):
                    break


                print("itr = {}  zero counts {}".format(_, anp.sum(anp.any(z_idx))))
        else:
            weights /= counts

        # Convolve with [1/8, 6/8, 1/8] in each dimension to smooth the
        # weights; boundary behaviour: repeat border values.
        # Divide by d_sum to normalize (divide by the sum before smoothing)
        # EQ 18
        dim, n, N_intervals = weights.shape
        # (dim, n, 1)
        weights_sums = anp.reshape(anp.sum(weights, axis=2, dtype=torch.float), [dim, n, 1])
        print("see weights_sums")
        print(weights_sums.min())
        if anp.any(weights_sums == 0.0):

            # The VEGASMap cannot be updated in dimensions where all weights
            # are zero.
            return None

        i_tmp = N_intervals - 2
        # (dim, n, N_intervals)
        d_tmp = anp.concatenate(
            [
                7.0 * weights[:, :, 0:1] + weights[:, :, 1:2],
                weights[:, :, :-2] + 6.0 * weights[:, :, 1:-1] + weights[:, :, 2:],
                weights[:, :, i_tmp: i_tmp + 1] + 7.0 * weights[:, :, i_tmp + 1: i_tmp + 2],
            ],
            axis=2,
            like=weights,
        )

        d_tmp = d_tmp / (8.0 * weights_sums)

        # Range compression
        # EQ 19
        d_tmp[d_tmp != 0] = (
                        (d_tmp[d_tmp != 0] - 1.0) / anp.log(d_tmp[d_tmp != 0])
                ) ** alpha

        return d_tmp


    def update_map(self):
        # (dim, n, N_intervals)
        smoothed_weights = self._smooth_map(self.weights, self.counts, self.alpha)
        if smoothed_weights is None:
            logger.warning(
                "Cannot update the VEGASMap. This can happen with an integrand "
                "which evaluates to zero everywhere."
            )
            self._reset_weight()
            return
        # The amount of the sum of smoothed_weights for each interval of
        # the new 1D grid, for each dimension
        # EQ 20
        # (dim, n)
        delta_weights = anp.sum(smoothed_weights, axis=2) / self.N_intervals

        # (dim, n, 1)
        delta_weights = delta_weights.reshape(self.dim, self.n, 1)


        # (dim, n, N_intervals-1)
        delta_d_multiples = torch.cumsum(smoothed_weights[:, :, :-1].double(), dim=2)



        delta_d_multiples = (delta_d_multiples / delta_weights).long()


        # For each number of delta_d multiples in {0, 1, …, N_intervals},
        # determine how many intervals belong to it (num_sw_per_dw)
        # and the sum of smoothed weights in these intervals (val_sw_per_dw)
        dtype_int = delta_d_multiples.dtype
        num_sw_per_dw = torch.zeros(
            [self.dim, self.n, self.N_intervals + 1], dtype=dtype_int
        )

        dim_id = torch.arange(self.dim)
        dim_id = anp.repeat(dim_id.reshape(-1, 1, 1), delta_d_multiples.shape[2], axis=2)
        dim_id = anp.repeat(dim_id, delta_d_multiples.shape[1], axis=1)

        integrator_id = torch.arange(delta_d_multiples.shape[1])
        integrator_id = anp.repeat(integrator_id.reshape(1, -1, 1), delta_d_multiples.shape[2], axis=2)
        integrator_id = anp.repeat(integrator_id, delta_d_multiples.shape[0], axis=0)


        val_sw_per_dw = torch.zeros(
            [self.dim, self.n, self.N_intervals + 1], dtype=self.dtype
        )

        dim_shape = dim_id.shape
        integrator_shape = integrator_id.shape

        dim_id = dim_id.view(-1)
        integrator_id = integrator_id.view(-1)
        delta_d_multiples = delta_d_multiples.view(-1)
        # (dim, n, N_intervals-1)  -> (dim * n * (N_intervals-1) )
        tmp_smoothed_weights = smoothed_weights[:,:,:-1].reshape(-1)

        ones = torch.ones_like(delta_d_multiples)
        ones = ones.view(-1)

        num_shape = num_sw_per_dw.shape
        val_shape = val_sw_per_dw.shape

        num_sw_per_dw = num_sw_per_dw.view(-1)
        val_sw_per_dw = val_sw_per_dw.view(-1)

        idx = dim_id * self.n * (self.N_intervals +1) + integrator_id * (self.N_intervals+1) + delta_d_multiples
        # idx = dim_id * self.n * self.N_intervals  + integrator_id * self.N_intervals + delta_d_multiples
        # val_sw_per_dw.index_add_(dim=0, index=idx, source=tmp_smoothed_weights)
        # num_sw_per_dw.index_add_(dim=0, index=idx, source=ones)
        val_sw_per_dw.scatter_add_(dim=0, index=idx, src=tmp_smoothed_weights)
        num_sw_per_dw.scatter_add_(dim=0, index=idx, src=ones)

        num_sw_per_dw = num_sw_per_dw.view(num_shape)
        val_sw_per_dw = val_sw_per_dw.view(val_shape)

        dim_id = dim_id.view(dim_shape)
        integrator_id = integrator_id.view(integrator_shape)


        # (dim, n, N_intervals-1)
        indices= torch.cumsum(num_sw_per_dw[:,:,:-2], dim=2)
        d_accu_i = torch.cumsum(delta_weights - val_sw_per_dw[:,:,:-2], dim=2)

        # (DIM, n, N_edges)
        self.x_edges[:, :, 1:-1] = (
                self.x_edges[dim_id, integrator_id, indices]
                + d_accu_i / smoothed_weights[dim_id, integrator_id, indices] * self.dx_edges[dim_id, integrator_id, indices]
        )
        finite_edges = torch.isfinite(self.x_edges)
        if not anp.all(finite_edges):
            # With float64 precision the delta_d_multiples calculation
            # usually doesn't have rounding errors.
            # If it is nonetheless too inaccurate, few values in
            # smoothed_weights[i][indices] can be zero, which leads to
            # invalid edges.
            num_edges = self.x_edges.shape[2]
            logger.warning(
                f"{num_edges - anp.sum(finite_edges)} out of {num_edges} calculated VEGASMap edges were infinite"
            )
            # Replace inf edges with the average of their two neighbours
            middle_edges = 0.5 * (self.x_edges[:,:,:-2] + self.x_edges[:,:,2:])
            self.x_edges[:,:,1:-1] = anp.where(
                finite_edges[:,:,1:-1], self.x_edges[:,:,1:-1], middle_edges
            )
            if not anp.all(anp.isfinite(self.x_edges)):
                raise RuntimeError("Could not replace all infinite edges")

        self.dx_edges = self.x_edges[:,:,1:] - self.x_edges[:,:,:-1]

        self._reset_weight()

    def set_map_edges_i(self, i, dx_edges, x_edges):

        # x_edges (N_edges, dim)   -> (dim, N_edges)
        # dx_edges (N_intervals, dim)   -> (dim, N_intervals)

        x_edges = x_edges.permute(1, 0)
        dx_edges = dx_edges.permute(1, 0)
        # (dim, n, N_edges)
        self.x_edges[:, i, :] = x_edges
        # self.x_edges[i,:,:] = x_edges
        # (dim, n, N_intervals)
        # self.dx_edges[i,:,:] = dx_edges
        self.dx_edges[:, i, :] = dx_edges

    def set_map_edges(self, dx_edges, x_edges):

        # (n, n_edges, dim)   ->   (dim, n, n_edges)
        x_edges = x_edges.permute(2, 0, 1)
        dx_edges = dx_edges.permute(2, 0, 1)

        self.x_edges[:, :, :] = x_edges
        self.dx_edges[:, :, :] = dx_edges
