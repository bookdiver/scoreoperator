from absl import logging
logging.set_verbosity(logging.ERROR)
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from flax.training import train_state, checkpoints
import optax

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

    def __init__(self, db, model, training_config):
        
        # Diffusion bridge 
        self.db = db
        self.model = model
        
        # Training
        self.dir = training_config.dir
        self.rng_key = jax.random.PRNGKey(training_config.seed)
        self.lr = training_config.lr
        self.batch_size = training_config.batch_size
        self.n_iters = training_config.n_iters
        self.log_freq = training_config.log_freq

        self._create_train_function()
        self._init_model()

    def _create_train_function(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xs, ts, bs = batch
            xs_flatten = flatten_batch(xs) 
            ts_flatten = flatten_batch(ts)                                              

            outs = self.model.apply(
                variables={
                    "params": params, 
                    "batch_stats": batch_stats
                },
                x=xs_flatten,
                t=ts_flatten,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds_flatten, new_model_state = outs if train else (outs, None)
            preds = unflatten_batch(preds_flatten, self.batch_size)
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
        dummy_xs = jnp.zeros((1, 1, *self.db.sde.x_shape))
        dummy_ts = jnp.zeros((1, 1, ))
        dummy_xs_flatten = flatten_batch(dummy_xs)
        dummy_ts_flatten = flatten_batch(dummy_ts)
        variables = self.model.init(
            self.rng_key, 
            x=dummy_xs_flatten, 
            t=dummy_ts_flatten, 
            train=True
        )
        
        # Print the number of trainable parameters
        num_params = sum(param.size for param in jax.tree_util.tree_leaves(variables["params"]))
        print(f"Number of trainable parameters: {num_params:,}")
        
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None
        del dummy_xs, dummy_ts, dummy_xs_flatten, dummy_ts_flatten
    
    def _init_optimizer(self):
        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=0.0,
        #     peak_value=self.lr,
        #     warmup_steps=int(0.1*self.n_iters),
        #     decay_steps=int(0.85*self.n_iters),
        #     end_value=0.01*self.lr
        # )
        
        # optimizer = optax.chain(
        #     optax.adam(lr_schedule),
        #     optax.ema(0.995)
        # )
        optimizer = optax.adam(self.lr)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )
    
    def train_model(self, x0, mode='train'):
        if mode == 'train':
            self._init_optimizer()
        elif mode == 'pretrained':
            self.load_model(prefix="checkpoint_", step=self.n_iters)
            return None
        else:
            raise ValueError("Invalid mode. Choose 'train', or 'pretrained'.")

        all_train_losses = []
        all_train_times = []
        
        tmp_train_loss = 0.0

        stage_start_time = time.time()
        print(f"Training started, total iterations: {self.n_iters}")
        train_rng_key, _ = jax.random.split(self.rng_key)
        for i in range(1, self.n_iters+1):
            iter_start_time = time.time()
            
            current_rng_key = jr.fold_in(train_rng_key, i)
            sub_keys = jr.split(current_rng_key, self.batch_size)
            xss, tss, bss = jax.vmap(
                self.db.solve_forward_sde,
                in_axes=(0, None)
            )(sub_keys, x0)
            batch = (xss, tss, bss)
            self.state, loss = self.train_step(self.state, batch)
            iter_running_time = time.time() - iter_start_time
            
            all_train_times.append(iter_running_time)
            all_train_losses.append(loss)
            
            tmp_train_loss += loss
            
            if i % self.log_freq == 0 or i == self.n_iters:
                avg_train_loss = tmp_train_loss / self.log_freq
                tmp_train_loss = 0.0
                stage_running_time = time.time() - stage_start_time
                print(f"Iter [{i:<5} / {self.n_iters}]:")
                print("Stage statistics:")
                print(f"avg train loss: {avg_train_loss:.4f}, stage running time: {stage_running_time:.4f}s")
                stage_start_time = time.time()
                # self.save_model(step=i)

        with open(self.dir + "/records.txt", "w") as f:
            f.write("Loss, Time\n")  # Add header for the two columns
            for loss, time_spent in zip(all_train_losses, all_train_times):
                f.write(f"{loss}, {time_spent}\n")
        self.save_model(step=self.n_iters)

        print(f"Model saved to {self.dir + '/pretrained'}")
        print(f"Training loss saved to {self.dir + '/records.txt'}")
        print(f"Training finished in {sum(all_train_times)/60:.4f}m ({sum(all_train_times):.4f}s)")
            
    def infer_model(self, batch):
        xss_flat, tss_flat = batch
        outss_flat = self.state.apply_fn(
            variables={
                "params": self.state.params, 
                "batch_stats": self.state.batch_stats
            },
            x=xss_flat,
            t=tss_flat,
            train=False,
            mutable=False
        )
        return outss_flat

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
                keep=1000,  # Keep all checkpoints
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
            apply_fn=self.model.apply,
            params=ckpt["params"],
            tx=self.state.tx,
            batch_stats=ckpt["batch_stats"],
            opt_state=ckpt["optimizer_state"],
            step=step
        )
        self.rng_key = ckpt["rng_key"]
        print(f"Model loaded from {self.dir}/{prefix}_step_{step}")
        
class ScoreModel:
    """ This model serves as the wrapper for the trained nn to fit in the reverse bridge solver 
    """
    def __init__(self, trainer):
        self.infer = jax.jit(trainer.infer_model)

    def __call__(self, t, x):
        x_expanded = jnp.expand_dims(x, axis=0)
        t_expanded = jnp.full((1,), t)
        out = self.infer((x_expanded, t_expanded))
        return out.squeeze(axis=0)
