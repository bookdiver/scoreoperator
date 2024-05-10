import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm
import absl

absl.logging.set_verbosity(absl.logging.ERROR)

from ..models.neuralop.uno import UNO1D
from ..models.diffusion.diffuser import Diffuser
from ..models.diffusion.sde import SDE

def flatten_batch(x):
    b_size, t_size, *_ = x.shape
    return x.reshape(b_size*t_size, *x.shape[2:])

def unflatten_batch(x, b_size):
    bt_size, *_ = x.shape
    t_size = bt_size // b_size
    return x.reshape(b_size, t_size, *x.shape[1:])

def flatten_dim(x):
    return x.reshape(*x.shape[:-2], -1)

def unflatten_dim(x):
    return x.reshape(*x.shape[:-1], x.shape[-1]//2, 2)
    

class TrainState(train_state.TrainState):
    batch_stats: dict

class TrainerModule:

    def __init__(self,
                 config,
                ):
        # SDE
        self.sde = SDE(**config.sde)

        # Diffusion
        self.diffusion_dt = config.diffusion.dt
        self.matching_obj = config.diffusion.matching_obj

        # Model
        self.model = UNO1D(**config.model)

        # Training
        self.dir = config.training.dir
        self.seed = config.training.seed
        self.n_pts = config.training.n_pts
        self.learning_rate = config.training.learning_rate
        self.batch_size = config.training.batch_size
        self.optimizer_name = config.training.optimizer_name
        self.warmup_steps = config.training.warmup_steps if hasattr(config.training, "warmup_steps") else 0
        self.train_num_epochs = config.training.train_num_epochs
        self.train_num_steps_per_epoch = config.training.train_num_steps_per_epoch
        self.eval_num_steps_per_epoch = config.training.eval_num_steps_per_epoch if hasattr(config.training, "eval_num_steps_per_epoch") else int(0.1*self.train_num_steps_per_epoch)

        self.create_diffuser_loader()
        self.create_functions()
        self.init_model()

    def create_diffuser_loader(self):
        self.diffuser = Diffuser(
            self.seed, self.sde, self.diffusion_dt
        )

        if self.matching_obj== "score":
            self.dataloader = self.diffuser.get_trajectory_generator(
                x0=jnp.zeros(self.n_pts*2),
                batch_size=self.batch_size,
                noise_scaling="inv_g2",
                weighting_output="g2s"
            )
        elif self.matching_obj == "gscore":
            self.dataloader = self.diffuser.get_trajectory_generator(
                x0=jnp.zeros(self.n_pts*2),
                batch_size=self.batch_size,
                noise_scaling="inv_g",
                weighting_output="id"
            )
        elif self.matching_obj == "g2score":
            self.dataloader = self.diffuser.get_trajectory_generator(
                x0=jnp.zeros(self.n_pts*2),
                batch_size=self.batch_size,
                noise_scaling="none",
                weighting_output="id"
            )
        else:
            raise NotImplementedError(f"{self.matching_obj} has not available!")

    def create_functions(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xss, tss, gradss, weightss = batch      # (b_size, t_size, d_size)
            xs = flatten_batch(xss)                 # (b_size*t_size, d_size)
            xs = unflatten_dim(xs)                  # (b_size*t_size, d_size//2, 2)
            ts = flatten_batch(tss)                 # (b_size*t_size, )

            outs = self.model.apply(
                {"params": params, "batch_stats": batch_stats},
                xs,
                ts,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds, new_model_state = outs if train else (outs, None)
            predss = preds.reshape(*xss.shape)
            loss = self.diffuser.dsm_loss(predss, gradss, weightss)
            return loss, new_model_state
    
        def train_step(state, batch):
            loss_fn = lambda params: compute_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state = ret
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        def eval_step(state, batch):
            loss, _ = compute_loss(state.params, state.batch_stats, batch, train=False)
            return loss
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        xss, tss, *_ = next(self.dataloader)
        xs = flatten_batch(xss)
        xs = unflatten_dim(xs)
        ts = flatten_batch(tss)

        init_rng_key = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng_key, xs, ts, train=True)
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None
    
    def init_optimizer(self):
        total_steps = self.train_num_epochs * self.train_num_steps_per_epoch

        if self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=self.learning_rate,
                boundaries_and_scales={
                    int(total_steps*0.6): 0.1,
                    int(total_steps*0.85): 0.1
                }
            )
        elif self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=self.warmup_steps,
                decay_steps=int(total_steps*0.85),
                end_value=0.01*self.learning_rate
            )
        else:
            raise NotImplementedError(f"{self.optimizer_name} has not implemented!")
        
        optimizer = optax.chain(
            # optax.clip(1.0),
            opt_class(lr_schedule)
        )
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )

    def train_epoch(self):
        train_losses = []
        for _ in range(self.train_num_steps_per_epoch):
            batch = next(self.dataloader)
            self.state, loss = self.train_step(self.state, batch)
            train_losses.append(loss)
        train_losses = jax.device_get(train_losses)
        avg_train_loss = jnp.stack(train_losses).mean()
        return train_losses, avg_train_loss
    
    def eval_epoch(self):
        eval_losses = []
        for _ in range(self.eval_num_steps_per_epoch):
            batch = next(self.dataloader)
            eval_loss = self.eval_step(self.state, batch)
            eval_losses.append(eval_loss)
        eval_losses = jax.device_get(eval_losses)
        avg_eval_loss = jnp.stack(eval_losses).mean()
        return eval_losses, avg_eval_loss

    def train_model(self, pretrained=False, step=None):
        if pretrained:
            if step is None:
                self.load_model(prefix="pretrained", step=self.train_num_epochs)
            else:
                self.load_model(prefix="pretrained", step=step)
        else:
            self.init_optimizer()

            all_train_losses = []
            all_eval_losses = []

            with tqdm(range(self.train_num_epochs), desc="Training", unit="epoch") as pbar:
                for epoch in pbar:
                    epoch_train_losses, epoch_avg_train_loss = self.train_epoch()
                    epoch_eval_losses, epoch_avg_eval_loss = self.eval_epoch()
                    
                    all_train_losses += epoch_train_losses
                    all_eval_losses += epoch_eval_losses

                    pbar.set_postfix(Epoch=epoch, train_loss=f"{epoch_avg_train_loss:.4f}", eval_loss=f"{epoch_avg_eval_loss:.4f}")

            with open(self.dir + "/train_losses.txt", "w") as f:
                for loss in all_train_losses:
                    f.write(f"{loss}\n")
            with open(self.dir + "/eval_losses.txt", "w") as f:
                for loss in all_eval_losses:
                    f.write(f"{loss}\n")
            self.save_model("pretrained", step=self.train_num_epochs)
            print(f"Model saved to {self.dir + '/pretrained'}")
            print(f"Training loss saved to {self.dir + '/train_losses.txt'}")
            print(f"Evaluation loss saved to {self.dir + '/eval_losses.txt'}")
            print(f"Training finished!")

    def infer_model(self, batch):
        xs, ts = batch
        score_outs = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            xs,
            ts,
            train=False,
            mutable=False
        )
        return score_outs

    def save_model(self, prefix, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.dir,
                                    target={
                                        "params": self.state.params,
                                        "batch_stats": self.state.batch_stats
                                    },
                                    step=step,
                                    prefix=prefix,
                                    overwrite=True)
    
    def load_model(self, prefix, step):
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=self.dir, 
                                              target=None,
                                              step=step,
                                              prefix=prefix)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=ckpt["params"],
                                       batch_stats=ckpt["batch_stats"],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
        print(f"Model loaded from {self.dir}/{prefix}")


class Model:

    def __init__(self, trainer: TrainerModule):
        self.infer = trainer.infer_model
        self.matching_obj = trainer.matching_obj

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        x_ = jnp.expand_dims(x.reshape(-1, 2), axis=0)
        t_ = jnp.expand_dims(jnp.asarray(t), axis=0)
        out = self.infer((x_, t_))
        out = out.squeeze().flatten()
        return out
    
    def forward(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(t, x)