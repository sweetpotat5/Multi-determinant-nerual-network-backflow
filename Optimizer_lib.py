import sys
import pickle
import os

from netket._src.driver.vmc_sr import *
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp

@jax.jit
def _flatten_samples(x):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(x, 0, x.ndim - 1)


@jax.jit
def gradient_norm_clip(pytree_grad, C, eta):
    sq_norm = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x ** 2), pytree_grad, initializer=0.0)
    norm_clip = jnp.sqrt(C/sq_norm)
    lr = jnp.minimum(eta, norm_clip)
    normalized_grad = jax.tree_util.tree_map(lambda x: x * lr, pytree_grad)
    return normalized_grad, norm_clip


class VMC_SR_norm_clip(AbstractVariationalDriver):

    # Settings set by user
    diag_shift: ScalarOrSchedule = struct.field(serialize=False)
    proj_reg: ScalarOrSchedule = struct.field(serialize=False)

    momentum: bool = struct.field(serialize=False, default=False)

    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    _mode: str = struct.field(serialize=False)
    _chunk_size_bwd: int | None = struct.field(serialize=False)
    _use_ntk: bool = struct.field(serialize=False)
    _on_the_fly: bool = struct.field(serialize=False)
    _linear_solver_fn: Any = struct.field(serialize=False)

    # Internal things cached
    _unravel_params_fn: Any = struct.field(serialize=False)

    # Serialized state
    _old_updates: PyTree = None
    _dp: PyTree = struct.field(serialize=False)
    info: Any | None = None

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        proj_reg: ScalarOrSchedule | None = None,
        momentum: ScalarOrSchedule | None = None,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        variational_state: MCState = None,
        chunk_size_bwd: int | None = None,
        mode: JacobianMode | None = None,
        use_ntk: bool | None = None,
        on_the_fly: bool | None = None,
        
        eta: ScalarOrSchedule,
        C: ScalarOrSchedule,
    ):
        
        if isinstance(variational_state, FullSumState):
            raise TypeError(
                "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
            )
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        if use_ntk is None:
            use_ntk = variational_state.n_parameters > variational_state.n_samples
            print("Automatic SR implementation choice: ", "NTK" if use_ntk else "QGT")

        if on_the_fly is None:
            if use_ntk:
                on_the_fly = True
            else:
                on_the_fly = False
        elif on_the_fly and not use_ntk:
            raise ValueError(
                """
                `onthefly` is only supported when `use_ntk=True`.

                We plan to support this mode for the standard NGD in the future.
                In the meantime, use a standard VMC+SR QGTOnTheFly.
                """
            )

        self.eta = eta
        self.C = C
        
        self._ham = hamiltonian

        self.diag_shift = diag_shift
        self.proj_reg = proj_reg
        self.momentum = momentum

        self.chunk_size_bwd = chunk_size_bwd
        self._use_ntk = use_ntk
        self.mode = mode
        self._on_the_fly = on_the_fly

        self._linear_solver_fn = linear_solver_fn

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

        self._old_updates: PyTree = None
        self._dp: PyTree = None

        # PyTree to pass on information from the solver, e.g, the quadratic model
        self.info = None

        params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex parameters. "
                "Hybrid structures are not yet supported (but we would welcome contributions. Get in touch with us!)"
            )
        self.previous_E = 1000 # ccc: delete after check

    @timing.timed
    def _forward_and_backward(self):
        # import time
        # t0 = time.perf_counter()
        self.state.reset()
        # t1 = time.perf_counter()
        # Compute the local energy estimator and average Energy
        local_energies = self.state.local_estimators(self._ham)
                
        self._loss_stats = nkstats.statistics(local_energies)
        # t2 = time.perf_counter()
        # Extract the hyperparameters which might be iteration dependent
        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)

        if self.use_ntk:
            if self.on_the_fly:
                compute_sr_update_fun = srt_onthefly
            else:
                compute_sr_update_fun = srt
        else:
            if self.on_the_fly:
                raise NotImplementedError
            else:
                compute_sr_update_fun = sr


        samples = _flatten_samples(self.state.samples)
        self._dp, self._old_updates, self.info = compute_sr_update_fun(
            self.state._apply_fun,
            local_energies,
            self.state.parameters,
            self.state.model_state,
            samples,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver_fn,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd,
        )
        # t3 = time.perf_counter()
        # #norm clip
        # eta = 0.01
        # C = 0.01
        # flat_phi, unravel_fn = ravel_pytree(self._dp)
        # norm_clip = jnp.sqrt(self.C) / jnp.linalg.norm(flat_phi)
        # flat_updates = flat_phi * jnp.minimum(self.eta, norm_clip)
        # self._dp = unravel_fn(flat_updates)
        
        eta = self.eta
        C = self.C
        if callable(eta):
            eta = eta(self.step_count)
        if callable(C):
            C = C(self.step_count)

        # t31 = time.perf_counter()
        # flat_phi, unravel_fn = ravel_pytree(self._dp)
        # t32 = time.perf_counter()
        # norm_clip = jnp.sqrt(C) / jnp.linalg.norm(flat_phi)
        # t33 = time.perf_counter()
        # flat_updates = flat_phi * jnp.minimum(eta, norm_clip)
        # t34 = time.perf_counter()
        # self._dp = unravel_fn(flat_updates)
        # t35 = time.perf_counter()
        

        
        self._dp, norm_clip = gradient_norm_clip(self._dp, C, eta)
        
        # print('\n')
        # print(f'eta={eta}, C={C}, norm_clip={norm_clip}')
        # print('eta smaller *#*') if eta<norm_clip else print('norm_clip smaller $@$')
        # t4 = time.perf_counter()
        # print(f'sampling time:{t1-t0}, energy calculation time:{t2-t1}, SR time:{t3-t2}, normclip time:{t4-t3}')
        # print(f'ravel:{t32-t31}, normclip:{t33-t32}, flatupdates:{t34-t33}, unravel:{t35-t34}, print:{t4-t35}')
        return self._dp

    


    @timing.timed
    def _log_additional_data(self, log_dict: dict, step: int):
        """
        Method to be implemented in sub-classes of AbstractVariationalDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.
        """
        # Always log the acceptance.
        if hasattr(self.state, "sampler_state"):
            acceptance = getattr(self.state.sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict["acceptance"] = acceptance

        # Log the quadratic model if requested.
        if self.info is not None:
            log_dict["info"] = self.info

    @property
    def mode(self) -> JacobianMode:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`, `'complex'`, or 'onthefly'.

        - `'real'` mode truncates imaginary part of the wavefunction, useful for real-valued wf with a sign.
        - `'complex'` is the general implementation that always works.
        - `onthefly` uses a lazy implementation of the neural tangent kernel and does not compute the jacobian.

        This internally uses :func:`netket.jax.jacobian`. See that function for a more complete documentation.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str | JacobianMode | None):
        # TODO: Add support for 'onthefly' mode
        # At the moment, onthefly is only supported for use_ntk=True.
        # We raise a warning if the user tries to use it with use_ntk=False.
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.hilbert.random_state(jax.random.key(1), 3),
                warn=False,
            )

        # TODO: Add support for 'holomorphic' mode
        # At the moment we only support 'real' and 'complex' modes for jacobian.
        # We raise an error if the user tries to use 'holomorphic' mode.
        if mode not in ["complex", "real"]:
            raise ValueError(
                "`mode` only supports 'jacobian_real' for real-valued wavefunctions, and 'jacobian_complex' for complex valued wave functions."
                "`holomorphic` is not yet supported, but could be contributed in the future. \n"
                f"You gave {mode}"
            )

        self._mode = mode

    @property
    def on_the_fly(self) -> bool:
        """
        Whether
        """
        return self._on_the_fly

    @property
    def use_ntk(self) -> bool:
        r"""
        Whether to use the Neural Tangent Kernel (NTK) instead of the Quantum Geometric Tensor (QGT) to compute the update.
        """
        return self._use_ntk

    @property
    def chunk_size_bwd(self) -> int:
        """
        Chunk size for backward-mode differentiation. This reduces memory pressure at a potential cost of higher computation time.

        If computing the jacobian, the jacobian is computed in blocks of `chunk_size_bwd` rows.
        If computing the NTK lazily, this is the number of rows of NTK evaluated in a single sweep.
        The chunk size does not affect the result, up to numerical precision.
        """
        return self._chunk_size_bwd

    @chunk_size_bwd.setter
    def chunk_size_bwd(self, value: int | None):
        if not isinstance(value, int | None):
            raise TypeError("chunk_size must be an integer or None")
        self._chunk_size_bwd = value

