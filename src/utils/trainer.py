import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm
import absl
# import wandb

absl.logging.set_verbosity(absl.logging.ERROR)

from ..models.neuralop.uno import UNO1D
from ..models.diffusion.loss import dsm_loss
from ..models.diffusion.diffusion import Diffuser
from ..models.diffusion.sde import BrownianSDE, EulerianSDE

class TrainState(train_state.TrainState):
    batch_stats: dict


class TrainerModule:

    def __init__(self,
                 config,
                ):
        self.config = config

        # SDE
        self.sde_name = config.sde_name

        # Diffusion
        self.diffusion_dt = config.diffusion.dt

        # Model
        self.model = UNO1D(**config.model)

        # Training
        self.seed = config.training.seed
        self.n_pts = config.training.n_pts
        self.learning_rate = config.training.learning_rate
        self.batch_size = config.training.batch_size
        self.optimizer_name = config.training.optimizer_name
        self.warmup_steps = config.training.warmup_steps if hasattr(config.training, "warmup_steps") else 0
        self.train_num_epochs = config.training.train_num_epochs
        self.train_num_steps_per_epoch = config.training.train_num_steps_per_epoch

        self.create_sde()
        self.create_diffuser_loader()
        self.create_functions()
        self.init_model()

    def create_sde(self):
        if self.sde_name == "brownian":
            self.sde = BrownianSDE(**self.config.sde)
        elif self.sde_name == "eulerian":
            self.sde = EulerianSDE(**self.config.sde)
        else:
            raise NotImplementedError(f"{self.sde_name} has not implemented!")

    def create_diffuser_loader(self):
        self.diffuser = Diffuser(
            self.seed, self.sde, self.diffusion_dt
        )
        self.dataloader = self.diffuser.get_trajectory_generator(
            x0=jnp.zeros(self.n_pts*2),
            batch_size=self.batch_size
        )

    def create_functions(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xss, tss, gradss = batch # (b_size, t_size, d_size)
            b_size, t_size, d_size = xss.shape
            xs = xss.reshape(b_size*t_size, d_size//2, 2)
            ts = tss.reshape(b_size*t_size, )

            outs = self.model.apply(
                {"params": params, "batch_stats": batch_stats},
                xs,
                ts,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds, new_model_state = outs if train else (outs, None)
            predss = preds.reshape(b_size, t_size, d_size)
            loss = dsm_loss(predss, gradss, dt=self.diffusion_dt)
            return loss, new_model_state
    
        def train_step(state, batch):
            loss_fn = lambda params: compute_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state = ret
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        def eval_step(state, batch):
            xss, tss, gradss = batch # (b_size, t_size, d_size)
            b_size, t_size, d_size = xss.shape
            xs = xss.reshape(b_size*t_size, d_size//2, 2)
            ts = tss.reshape(b_size*t_size, )
            preds = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                xs,
                ts,
                train=False,
                mutable=False
            )
            predss = preds.reshape(b_size, t_size, d_size)
            return dsm_loss(predss, gradss, dt=self.diffusion_dt)
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        xss, tss, _ = next(self.dataloader)
        b_size, t_size, d_size = xss.shape
        xs = xss.reshape(b_size*t_size, d_size//2, 2)
        ts = tss.reshape(b_size*t_size, )
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
        avg_train_loss = jnp.stack(jax.device_get(train_losses)).mean()
        return avg_train_loss

    def train_model(self, pretrained=False, load_dir=None, prefix=None):
        if pretrained and load_dir is not None and prefix is not None:
            self.load_model(load_dir, prefix)
        else:
            self.init_optimizer()

            with tqdm(range(self.train_num_epochs), desc="Training", unit="epoch") as pbar:
                for epoch in pbar:
                    epoch_avg_loss = self.train_epoch()
                
                    eval_batch = next(self.dataloader)
                    eval_loss = self.eval_step(self.state, eval_batch)
                    
                    # if self.enable_wandb:
                    #     wandb.log({
                    #         "train_avg_loss": epoch_avg_loss,
                    #         "eval_loss": eval_loss 
                    #     }, step=epoch)
                    pbar.set_postfix(Epoch=epoch, train_loss=f"{epoch_avg_loss:.4f}", eval_loss=f"{eval_loss:.4f}")

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

    def save_model(self, save_dir, prefix, step=0):
        checkpoints.save_checkpoint(ckpt_dir=save_dir,
                                    target={
                                        "params": self.state.params,
                                        "batch_stats": self.state.batch_stats
                                    },
                                    step=step,
                                    prefix=prefix,
                                    overwrite=True)
    
    def load_model(self, load_dir, prefix):
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=load_dir, 
                                              target=None,
                                              prefix=prefix)
        model = UNO1D(**self.config.model)
        self.state = TrainState.create(apply_fn=model.apply,
                                       params=ckpt["params"],
                                       batch_stats=ckpt["batch_stats"],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
        print(f"Model loaded from {load_dir}")