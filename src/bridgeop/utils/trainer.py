from functools import partial
import time
import logging

import jax
import jax.numpy as jnp
from typing import Callable
from flax import struct
from flax.training import train_state, checkpoints
import optax

from ..neuralop.uno import CTUNO
from ..diffusion.bridge import DiffusionBridge

def flatten_batch(x):
    b_size, t_size, *_ = x.shape
    return x.reshape(b_size*t_size, *x.shape[2:])

def unflatten_batch(x, b_size):
    bt_size, *_ = x.shape
    t_size = bt_size // b_size
    return x.reshape(b_size, t_size, *x.shape[1:])

def flatten_dim(x):
    return x.reshape(*x.shape[:-2], -1)

def unflatten_dim(x, dim=2):
    return x.reshape(*x.shape[:-1], x.shape[-1]//dim, dim)


# @struct.dataclass
# class TrainState(train_state.TrainState):
#     batch_stats: dict
#     apply_fn: Callable
#     tx: optax.GradientTransformation
#     opt_state: optax.OptState

@struct.dataclass
class TrainState(train_state.TrainState):
    batch_stats: dict = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            **kwargs,
        )

class TrainerModule:

    def __init__(self,
                 config,
                ):
        # Diffusion bridge 
        self.db = DiffusionBridge(**config["diffusion_bridge"])
        self.neural_op = CTUNO(**config["neural_op"])

        # Training
        training_config = config["training"]
        self.dir = training_config["dir"]
        self.rng_key = jax.random.PRNGKey(training_config["seed"])

        self.n_train_pts = training_config["n_train_pts"]
        self.lr = training_config["lr"]
        self.batch_sz = training_config["batch_sz"]
        self.opt_name = training_config["opt_name"]
        self.clip_grads = training_config["clip_grads"] 
        self.warmup_steps = training_config["warmup_steps"] if "warmup_steps" in training_config else 0
        self.train_n_iters = training_config["train_n_iters"]
        self.checkpoint_freq = training_config["checkpoint_freq"]

        self._create_train_function()
        self._init_model()

    def _create_train_function(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xs, ts, bs = batch
            xs_flatten = flatten_batch(xs) 
            bs_flatten = flatten_batch(bs)
            ts_expand = jnp.tile(jnp.expand_dims(ts, axis=0), (self.batch_sz, 1))
            ts_flatten = flatten_batch(ts_expand)                                              

            outs = self.neural_op.apply(
                {"params": params, "batch_stats": batch_stats},
                x=xs_flatten,
                t=ts_flatten,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds_flatten, new_model_state = outs if train else (outs, None)
            assert preds_flatten.shape == bs_flatten.shape
            preds = unflatten_batch(preds_flatten, self.batch_sz)
            loss = self.db.dsm_loss(preds, bs)
            return loss, new_model_state
    
        def train_step(state, batch):
            loss_fn = lambda params: compute_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state = ret
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        self.train_step = jax.jit(train_step)

    def _init_model(self):
        dummy_xs = jnp.zeros((1, 1, self.n_train_pts, self.neural_op.in_co_dim))
        dummy_ts = jnp.zeros((1, 1, ))
        dummy_xs_flatten = flatten_batch(dummy_xs)
        dummy_ts_flatten = flatten_batch(dummy_ts)
        variables = self.neural_op.init(
            self.rng_key, 
            x=dummy_xs_flatten, 
            t=dummy_ts_flatten, 
            train=True
        )
        
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None
        del dummy_xs, dummy_ts, dummy_xs_flatten, dummy_ts_flatten
    
    def _init_optimizer(self):

        if self.opt_name.lower() == 'sgd':
            opt_class = optax.sgd
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=self.lr,
                boundaries_and_scales={
                    int(self.train_n_iters*0.6): 0.1,
                    int(self.train_n_iters*0.85): 0.1
                }
            )
        elif self.opt_name.lower() == 'adam':
            opt_class = optax.adam
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.lr,
                warmup_steps=self.warmup_steps,
                decay_steps=int(self.train_n_iters*0.85),
                end_value=0.01*self.lr
            )
        else:
            raise NotImplementedError(f"{self.opt_name} has not implemented!")
        
        optimizer = optax.chain(
            optax.clip(1.0) if self.clip_grads else optax.identity(),
            opt_class(lr_schedule)
        )
        self.state = TrainState.create(
            apply_fn=self.neural_op.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )
    

    def train_model(self, x0, mode='train', step=None):
        assert x0.shape == (self.n_train_pts, self.neural_op.in_co_dim)

        if mode == 'train':
            self._init_optimizer()
            start_iter = 0
        elif mode == 'resume':
            if step is None:
                raise ValueError("Step must be provided when resuming training.")
            self.load_model(prefix="checkpoint", step=step)
            start_iter = step
        elif mode == 'pretrained':
            if step is None:
                step = self.train_n_iters
            self.load_model(prefix="pretrained", step=step)
            return  # No training needed for pretrained mode
        else:
            raise ValueError("Invalid mode. Choose 'train', 'resume', or 'pretrained'.")

        all_train_losses = []
        tmp_train_losses = []
        all_train_times = []

        stage_start_time = time.time()
        for i in range(start_iter, self.train_n_iters):
            iter_start_time = time.time()
            self.rng_key, step_key = jax.random.split(self.rng_key)
            batch = self.db.solve_forward_sde(
                rng_key=step_key,
                x0=x0,
                n_batches=self.batch_sz
            )
            self.state, loss = self.train_step(self.state, batch)
            iter_running_time = time.time() - iter_start_time
            all_train_times.append(iter_running_time)
            all_train_losses.append(loss)
            tmp_train_losses.append(loss)
            if len(tmp_train_losses) == self.checkpoint_freq:
                avg_train_loss = jnp.stack(tmp_train_losses).mean()
                stage_running_time = time.time() - stage_start_time
                print(f"Iter {i+1:<6} / {self.train_n_iters}, avg train loss: {avg_train_loss:.4f}, stage running time: {stage_running_time:.4f}s")
                stage_start_time = time.time()
                tmp_train_losses = []
            
            if (i + 1) % self.checkpoint_freq == 0:
                self.save_model(step=i+1)

        with open(self.dir + "/records.txt", "w") as f:
            f.write("Loss, Time\n")  # Add header for the two columns
            for loss, time_spent in zip(all_train_losses, all_train_times):
                f.write(f"{loss}, {time_spent}\n")

        self.save_model(step=self.train_n_iters)
        print(f"Model saved to {self.dir + '/pretrained'}")
        print(f"Training loss saved to {self.dir + '/records.txt'}")
        print(f"Training finished in {sum(all_train_times)/60:.4f}m {sum(all_train_times):.4f}s")
            
    def infer_model(self, batch):
        xs, ts = batch
        outs = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            xs,
            ts,
            train=False,
            mutable=False
        )
        return outs

    def save_model(self, step):
        try:
            ckpt_path = checkpoints.save_checkpoint(
                ckpt_dir=self.dir,
                target={
                    "params": self.state.params,
                    "batch_stats": self.state.batch_stats,
                    "optimizer_state": self.state.opt_state,
                    "step": step,
                    "rng_key": self.rng_key,
                },
                step=step,
                prefix="checkpoint_",
                keep=1000000,  # Keep all checkpoints
                overwrite=True
            )
            print(f"Model checkpoint saved successfully to {ckpt_path}")
            return ckpt_path
        except Exception as e:
            print(f"Failed to save model checkpoint: {str(e)}")
            return None
    
    def load_model(self, prefix, step):
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=self.dir, 
                                              target=None,
                                              step=step,
                                              prefix=prefix)
        if self.state is None:
            self._init_optimizer()
        
        self.state = TrainState(
            apply_fn=self.neural_op.apply,
            params=ckpt["params"],
            tx=self.state.tx,
            batch_stats=ckpt["batch_stats"],
            opt_state=ckpt["optimizer_state"]
        )
        self.rng_key = ckpt["rng_key"]
        print(f"Model loaded from {self.dir}/{prefix}_step_{step}")
        
    def update_db(self, db_config):
        self.db = DiffusionBridge(**db_config)
        
class ScoreModel:
    """ This model serves as the wrapper for the trained nn to fit in the reverse bridge solver 
    """
    def __init__(self, trainer: TrainerModule):
        self.infer = jax.jit(trainer.infer_model)
        # self.infer = trainer.infer_model

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Model inference

        Args:
            t (float): time step
            x (jnp.ndarray): flatten x, shape (n*co_dim, )

        Returns:
            jnp.ndarray: model output, shape (n*do_dim, )
        """
        x_expanded = jnp.expand_dims(x, axis=0)
        t_expanded = jnp.full((1,), t)
        
        out = self.infer((x_expanded, t_expanded))
        return out.squeeze().ravel()

    # @partial(jax.jit, static_argnums=(0,))
    def batch_call(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """ Batched model inference

        Args:
            t (jnp.ndarray): time steps, shape (batch_size,)
            x (jnp.ndarray): flatten x, shape (batch_size, n*co_dim)

        Returns:
            jnp.ndarray: model output, shape (batch_size, n*do_dim)
        """
        x_reshaped = x.reshape(x.shape[0], -1, self.co_dim)
        return jax.vmap(self.infer)((x_reshaped, t[:, None])).reshape(x.shape[0], -1)