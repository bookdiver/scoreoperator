import os
import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm
import wandb

class TrainState(train_state.TrainState):
    batch_stats: dict


class TrainerModule:

    def __init__(self,
                 config,
                 model,
                 dataloader,
                 enable_wandb,
                ):
        
        self.seed = config.seed
        self.learning_rate = config.learning_rate
        self.optimizer_name = config.optimizer_name
        self.warmup_steps = config.warmup_steps if hasattr(config, "warmup_steps") else 0
        self.train_num_epochs = config.train_num_epochs
        self.train_num_steps_per_epoch = config.train_num_steps_per_epoch

        self.model = model
        self.dataloader = dataloader

        self.enable_wandb = enable_wandb

        self.create_functions()
        self.init_model()

    @staticmethod
    def dsm_loss(preds, zs, sigmas):
        # zs = jax.vmap(lambda x, y: x * y)(zs, 1.0 / sigmas**2)
        # preds = jax.vmap(lambda x, y: x * y)(preds, sigmas**2)
        loss = jnp.mean(
            jnp.mean(
                jnp.sum((preds + zs)**2, axis=-1),
                axis=-1
            ),
            axis=0
        )
        return loss
    
    def create_functions(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xs, ts, zs, sigmas = batch

            outs = self.model.apply(
                {"params": params, "batch_stats": batch_stats},
                xs,
                ts,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds, new_model_state = outs if train else (outs, None)
            loss = self.dsm_loss(preds, zs, sigmas)
            return loss, new_model_state
    
        def train_step(state, batch):
            loss_fn = lambda params: compute_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state = ret
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        def eval_step(state, batch):
            xs, ts, zs, sigmas = batch
            preds = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                xs,
                ts,
                train=False,
                mutable=False
            )
            return self.dsm_loss(preds, zs, sigmas)
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        xs, ts, _, _ = next(self.dataloader)
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
            optax.clip(1.0),
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
                    
                    if self.enable_wandb:
                        wandb.log({
                            "train_avg_loss": epoch_avg_loss,
                            "eval_loss": eval_loss 
                        }, step=epoch)
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
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=load_dir, 
                                                    target=None,
                                                    prefix=prefix)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict["params"],
                                       batch_stats=state_dict["batch_stats"],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
        print(f"Model loaded from {load_dir}")