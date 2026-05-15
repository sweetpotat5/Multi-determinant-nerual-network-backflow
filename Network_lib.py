
import jax
import jax.numpy as jnp
import flax.linen as nn
from netket.utils.types import NNInitFunc

from typing import Any, Callable, Sequence

DType = Any


import jax.numpy as jnp
from flax import linen as nn

class EmbeddingModule(nn.Module):
    embedding_dim: int
    patch_dim: int
    lattice_row: int
    lattice_col: int
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, configs: jnp.ndarray) -> jnp.ndarray:
        B, _ = configs.shape
        nh, nw = self.lattice_row // self.patch_dim, self.lattice_col // self.patch_dim
        
        # 1. Fold into 5D: (Batch, Spin, Row, Col) -> (Batch, Row, Col, Spin)
        x = configs.reshape(B, 2, self.lattice_row, self.lattice_col).transpose(0, 2, 3, 1)

        # 2. Patchify: (B, Row, Col, S) -> (B, n_h, p_h, n_w, p_w, S)
        # Then move patch indices together: (B, n_h, n_w, p_h, p_w, S)
        x = x.reshape(B, nh, self.patch_dim, nw, self.patch_dim, 2).transpose(0, 1, 3, 2, 4, 5)
        
        # 3. Flatten patches into tokens and project
        tokens = x.reshape(B, nh * nw, -1)
        
        return nn.Dense(self.embedding_dim, param_dtype=self.param_dtype)(tokens)




import jax.numpy as jnp
from flax import linen as nn
from typing import Callable

class StaticKernel(nn.Module):
    embedding_dim: int
    num_kernels: int
    n_patches: int
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        d_head = self.embedding_dim // self.num_kernels
        
        # Combined projection and split
        v = nn.Dense(self.embedding_dim, use_bias=False, param_dtype=self.param_dtype)(x)
        v = v.reshape((-1, self.n_patches, self.num_kernels, d_head))
        
        # Learnable attention weights
        alpha = self.param("alpha", nn.initializers.xavier_uniform(), 
                           (self.num_kernels, self.n_patches, self.n_patches), self.param_dtype)

        # Single einsum replaces manual transposes and matmul
        # 'h p k' (alpha) x 'b k h d' (v) -> 'b p h d'
        x = jnp.einsum('hpk, bkhd -> bphd', alpha, v)
        
        # Re-combine heads and final projection
        x = x.reshape((-1, self.n_patches, self.embedding_dim))
        return nn.Dense(self.embedding_dim, use_bias=False, param_dtype=self.param_dtype)(x)

class Encoder(nn.Module):
    num_layers: int
    embedding_dim: int
    num_kernels: int
    n_patches: int
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            # --- Attention Block ---
            res = nn.LayerNorm(param_dtype=self.param_dtype, name=f"norm1_{i}")(x)
            x = x + StaticKernel(
                self.embedding_dim, 
                self.num_kernels, 
                self.n_patches, 
                self.param_dtype, 
                name=f"kernel_{i}"
            )(res)
            
            # --- Feed-Forward Block ---
            res = nn.LayerNorm(param_dtype=self.param_dtype, name=f"norm2_{i}")(x)
            res = nn.Dense(4 * self.embedding_dim, param_dtype=self.param_dtype, name=f"ff1_{i}")(res)
            res = nn.gelu(res)
            res = nn.Dense(self.embedding_dim, param_dtype=self.param_dtype, name=f"ff2_{i}")(res)
            x = x + res
            
        return x




# uniform initializer
def uniform_initializer(minval=-1.0, maxval=1.0):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype=dtype, minval=minval, maxval=maxval)
    return init

def lecun_normal_initializer():
    def init(key, shape, dtype=jnp.float32):
        fan_in = jnp.prod(jnp.array(shape)[:-1])  # assumes shape is (fan_in, fan_out)

        variance = jnp.array(1.0 / fan_in, dtype=dtype)
        stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
        
        return jax.random.truncated_normal(key, -2, 2, shape, dtype) * jnp.array(stddev, dtype)
    return init



# get occupied orbital indices
def find_nonzero(n_row,size):
    return jnp.nonzero(n_row,size=size)[0]  # This returns the indices of non-zero elements in the row
find_nonzero = jax.vmap(find_nonzero, in_axes=(0,None))

# log determinant
@jax.vmap
def _logdet_cmplx(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


from functools import partial
import jax
import jax.numpy as jnp
from jax._src.lax import lax
from jax._src.numpy import reductions, tensor_contractions, ufuncs
from jax._src.lax import linalg as lax_linalg
import numpy as np
from jax._src import api
from jax._src.lax import control_flow

def slogdet_LU_input(a, lu_, pivot):
    dtype = lax.dtype(a)
    diag = jnp.diagonal(lu_, axis1=-2, axis2=-1)
    is_zero = reductions.any(diag == jnp.array(0, dtype=dtype), axis=-1)
    iota = lax.expand_dims(jnp.arange(a.shape[-1], dtype=pivot.dtype),
                         range(pivot.ndim - 1))
    parity = reductions.count_nonzero(pivot != iota, axis=-1)
    if jnp.iscomplexobj(a):
        sign = reductions.prod(diag / ufuncs.abs(diag).astype(diag.dtype), axis=-1)
    else:
        sign = jnp.array(1, dtype=dtype)
        parity = parity + reductions.count_nonzero(diag < 0, axis=-1)
    sign = jnp.where(is_zero,
                     jnp.array(0, dtype=dtype),
                     sign * jnp.array(-2 * (parity % 2) + 1, dtype=dtype))
    logdet = jnp.where(is_zero, jnp.array(-np.inf, dtype=dtype),
                       reductions.sum(ufuncs.log(ufuncs.abs(diag)).astype(dtype), axis=-1))
    return sign, ufuncs.real(logdet)

def linear_solve_LU_input(a, b, lu_, permutation):
    lax_linalg._check_solve_shapes(a, b)
    # Broadcast leading dimensions of b to the shape of a, as is required by custom_linear_solve.
    out_shape = tuple(d_a if d_b == 1 else d_b 
                      for d_a, d_b in zip(a.shape[:-1] + (1,), b.shape))
    b = lax.broadcast_in_dim(b, out_shape, range(b.ndim))
    # With custom_linear_solve, we can reuse the same factorization when computing sensitivities. This is considerably faster.
    # lu_, _, permutation = lax_linalg.lu(lax.stop_gradient(a))
    custom_solve = partial(control_flow.custom_linear_solve,
                           lambda x: lax_linalg._broadcasted_matvec(a, x),
                           solve=lambda _, x: lax_linalg.lu_solve(lu_, permutation, x, trans=0),
                           transpose_solve=lambda _, x: lax_linalg.lu_solve(lu_, permutation, x, trans=1))
    if a.ndim == b.ndim + 1: # b.shape == [..., m]
        x = custom_solve(b)
    else: # b.shape == [..., m, k]
        x = api.vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)
    return x

def batched_solve_all(linear_solve_fn, A, lu_batch, permutation_batch, b_batch):

    # Vectorize over A first, then b
    result = jax.vmap(lambda a, lu_, perm: 
                      jax.vmap(lambda b: linear_solve_fn(a, b, lu_, perm))(b_batch),
                      in_axes=(0, 0, 0)
                     )(A, lu_batch, permutation_batch)
    return result




class BackflowNet(nn.Module):
    num_sites : int 
    num_electrons: int
    n_patches: int
    embed_dim : int
    def setup(self):
        
        self.F_modifier_left = nn.Dense(features=self.num_sites * 2,
                           kernel_init=nn.initializers.lecun_normal(),
                           name='F_dense_modifer_up')
        
        self.F_modifier_right = nn.Dense(features=self.num_electrons,
                           kernel_init=nn.initializers.lecun_normal(),
                           name='F_dense_modifer_dn')
        
        
        self.project_num_electrons = nn.Dense(features=self.num_electrons,
                        kernel_init=nn.initializers.lecun_normal(),
                        name='F_num_electron_projection')
        self.hidden_embed_2 = nn.Dense(features=self.num_electrons,
                        kernel_init=nn.initializers.lecun_normal(),
                        name='F_hidden_2')
        
        self.project_num_sites = nn.Dense(features=self.num_sites,
                        kernel_init=nn.initializers.lecun_normal(),
                        name='F_num_sites_projection')
        self.hidden_embed_4 = nn.Dense(features=self.num_sites,
                        kernel_init=nn.initializers.lecun_normal(),
                        name='F_hidden_4')

        self.U = self.param("U", uniform_initializer(-0.05, 0.05), (2*self.num_sites, self.n_patches))
        self.V = self.param("V", uniform_initializer(-0.05, 0.05), (self.embed_dim, self.num_electrons))

        self.U2 = self.param("U2", uniform_initializer(-0.05, 0.05), (2*self.num_sites, self.n_patches))
        self.V2 = self.param("V2", uniform_initializer(-0.05, 0.05), (self.num_electrons, self.num_electrons))

        
    def __call__(self, x):
        @jax.vmap
        def backflow(x):   
            # --- Electron-space projection ---
            x_proj = self.project_num_electrons(x)  # (num_patches, num_electrons)
            x1 = nn.gelu(x_proj)
            x1 = self.hidden_embed_2(x1)
            x1 = nn.gelu(x1)
            F_right = self.F_modifier_right(x1)  # (num_patches, num_electrons)
    
            # --- Site-space projection ---
            x_proj_site = self.project_num_sites(F_right.T)  # (2*num_sites, num_electrons)
            x2 = nn.gelu(x_proj_site)
            x2 = self.hidden_embed_4(x2)
            x2 = nn.gelu(x2)
            F = self.F_modifier_left(x2).T  # (2*num_sites, num_electrons)
            
            return F*(1+self.U@x@self.V + self.U2@F_right@self.V2)
            
        return backflow(x)



class MultiDeterminant(nn.Module):
    num_sites: int
    num_electrons: int
    num_det_M: int
    num_det_F: int
    rank: int
    kernel_init: nn.initializers.Initializer = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, F_modifier, spins):
        # 0. Shared Orbital Parameters
        M = self.param('M', self.kernel_init, (self.num_sites * 2, self.num_electrons))
        F = self.param('F', self.kernel_init, (self.num_sites * 2, self.num_electrons))
        
        # 1. Identify occupied sites
        nonzero_indices = find_nonzero(spins, self.num_electrons)
        
        # 2. Parameters for low-rank updates (U and V matrices)
        # delta_M = U_M @ W_M
        U_M = self.param('U_left_M', self.kernel_init, (self.num_det_M, self.num_sites*2, self.rank))
        W_M = self.param('U_right_M', self.kernel_init, (self.num_det_M, self.rank, self.num_electrons))
        
        # delta_F = U_F @ W_F
        U_F = self.param('U_left_F', self.kernel_init, (self.num_det_F, self.num_sites*2, self.rank))
        W_F = self.param('U_right_F', self.kernel_init, (self.num_det_F, self.rank, self.num_electrons))

        # Build delta_F for the batched computation
        delta_F_full = jnp.einsum('kir, krj -> kij', U_F, W_F)

        def compute_logdets_single(f_mod, occ):
            # Base matrix: M + F * f_mod
            # Extra matrices: M + (F + delta_F_k) * f_mod
            A_extra = (M[None, :, :] + (F[None, :, :] + delta_F_full) * f_mod)[:, occ, :]
            A_base = (M + F * f_mod)[occ, :][None, :, :]
            
            A = jnp.concatenate((A_base, A_extra), axis=0) # (K+1, Ne, Ne)
            
            # LU Solve + Matrix Determinant Lemma
            Q_M = U_M[:, occ, :]
            lu_, pivot, permutation = lax_linalg.lu(A)
            sign_A, logamp_A = slogdet_LU_input(A, lu_, pivot)
            
            A_inv_Q_M = batched_solve_all(linear_solve_LU_input, A, lu_, permutation, Q_M)
            B = jnp.einsum("imk,bikn->bimn", W_M, A_inv_Q_M) + jnp.eye(self.rank)
            
            sign_B, logamp_B = jnp.linalg.slogdet(B)
            
            # Combine results
            logpsi_A = logamp_A.astype(complex) + jnp.log(sign_A.astype(complex))
            logpsi_extra = (logamp_A[:, None] + logamp_B).astype(complex) + jnp.log((sign_A[:, None] * sign_B).astype(complex))
            
            # Flatten to a single vector of determinants
            return jnp.concatenate((logpsi_A[0:1], logpsi_extra.flatten()))

        # Batch over the input configurations
        logdets = jax.vmap(compute_logdets_single)(F_modifier, nonzero_indices)
        return logdets.reshape(spins.shape[0], -1)


class Net(nn.Module):
    num_layers: int
    embedding_dim: int
    num_kernels: int
    patch_dim: int
    num_sites: int
    num_electrons: int
    num_det_M: int
    num_det_F: int
    rank: int
    lattice_row: int
    lattice_col: int

    kernel_init: nn.initializers.Initializer = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, spins):

        # 1. Transformer Backflow Path
        # Calculates F_modifier (f_mod) via Encoder and BackflowNet
        n_patches = (self.lattice_row // self.patch_dim) * (self.lattice_col // self.patch_dim)
        
        x_embedded = EmbeddingModule(self.embedding_dim, self.patch_dim, self.lattice_row, self.lattice_col)(spins)
        x_encoded = Encoder(self.num_layers, self.embedding_dim, self.num_kernels, n_patches)(x_embedded)
        F_modifier = BackflowNet(self.num_sites, self.num_electrons, n_patches, self.embedding_dim)(x_encoded)

        # 2. Multi-Determinant Computation
        # This calls the new module created above
        logdets = MultiDeterminant(
            num_sites=self.num_sites, 
            num_electrons=self.num_electrons,
            num_det_M=self.num_det_M,
            num_det_F=self.num_det_F,
            rank = self.rank
        )(F_modifier, spins)

        # 3. Final Pooling and Weighting
        x_encoded_pool = jnp.mean(x_encoded, axis=1)
        logits = nn.Dense(logdets.shape[-1], kernel_init=self.kernel_init)(x_encoded_pool)
        coeffs = nn.softmax(logits, axis=-1)

        # Log-Sum-Exp for numerical stability
        max_real = jnp.max(jnp.real(logdets), axis=1)
        dets_s = jnp.exp(logdets - max_real[:, None])
        
        return max_real + jnp.log(jnp.sum(dets_s * coeffs, axis=1) + 1e-40)
