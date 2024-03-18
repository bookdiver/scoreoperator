import os

import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
from tqdm import tqdm
from ml_collections import ConfigDict
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
        
        self._seed = config.seed
        self._learning_rate = config.learning_rate
        self._optimizer_name = config.optimizer_name
        self._warmup_steps = config.warmup_steps if hasattr(config, "warmup_steps") else 0
        self._train_num_epochs = config.train_num_epochs
        self._train_num_steps_per_epoch = config.train_num_steps_per_epoch

        self._model = model
        self._dataloader = dataloader

        self._enable_wandb = enable_wandb

        self._create_functions()
        self._init_model()

    @staticmethod
    def _dsm_loss(preds, zs):
        loss = jnp.mean(
            jnp.mean(
                jnp.sum((preds + zs)**2, axis=-1),
                axis=-2
            ),
            axis=0
        )
        return loss
    
    def _create_functions(self):
        
        def compute_loss(params, batch_stats, batch, train):
            xs, ts, zs = batch

            outs = self._model.apply(
                {"params": params, "batch_stats": batch_stats},
                xs,
                ts,
                train=train,
                mutable=["batch_stats"] if train else False
            )

            preds, new_model_state = outs if train else (outs, None)
            loss = self.dsm_loss(preds, zs)
            return loss, new_model_state
    
        def train_step(state, batch):
            loss_fn = lambda params: compute_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state = ret
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        def eval_step(state, batch):
            xs, ts, zs = batch
            preds = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                xs,
                ts,
                train=False,
                mutable=False
            )
            return self.dsm_loss(preds, zs)
        
        self._train_step = jax.jit(train_step)
        self._eval_step = jax.jit(eval_step)


    def _init_model(self):
        xs, ts, _ = next(self._dataloader)
        init_rng_key = jax.random.PRNGKey(self._seed)
        variables = self._model.init(init_rng_key, xs, ts, train=True)
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None
    
    def _init_optimizer(self):
        total_steps = self._train_num_epochs * self._train_num_steps_per_epoch

        if self._optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=self._learning_rate,
                boundaries_and_scales={
                    int(total_steps*0.6): 0.1,
                    int(total_steps*0.85): 0.1
                }
            )
        elif self._optimizer_config.name.lower() == 'adam':
            opt_class = optax.adam
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self._learning_rate,
                warmup_steps=self._warmup_steps,
                decay_steps=int(total_steps*0.85),
                end_value=0.01*self._learning_rate
            )
        else:
            raise NotImplementedError(f"{self._optimizer_name} has not implemented!")
        
        optimizer = optax.chain(
            optax.clip(1.0),
            opt_class(lr_schedule)
        )
        self.state = TrainState.create(
            apply_fn=self._model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )

    def _train_epoch(self):
        train_losses = []
        for _ in range(self._train_num_steps_per_epoch):
            batch = next(self._dataloader)
            self.state, loss = self.train_step(self.state, batch)
            train_losses.append(loss)
        avg_train_loss = np.stack(jax.device_get(train_losses)).mean()
        return avg_train_loss


    def train_model(self):
        self._init_optimizer()

        with tqdm(range(self._training_config.num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                epoch_avg_loss = self.train_epoch()
            
                eval_batch = next(self._diffusion_eval_loader)
                eval_loss = self.eval_step(self.state, eval_batch)
                
                if self._enable_wandb:
                    wandb.log({
                        "train_avg_loss": epoch_avg_loss,
                        "eval_loss": eval_loss
                    }, step=epoch)
                pbar.set_postfix(Epoch=epoch, train_loss=f"{epoch_avg_loss:.4f}", eval_loss=f"{eval_loss:.4f}")

    def save_model(self, save_dir, step=0):
        ckpt_dir = save_dir if save_dir else self._record_config.save_dir
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                                    target={
                                        "params": self.state.params,
                                        "batch_stats": self.state.batch_stats
                                    },
                                    step=step,
                                    overwrite=True)
    
    def load_model(self, load_dir, pretrained=False):
        load_dir = load_dir if load_dir else self._record_config.save_dir
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self._record_config.save_dir, target=None)
        else:
            assert self.checkpoint_exists(load_dir), "There is no pretrained model available!"
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(load_dir, f"{self.model_name}.ckpt"), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict["params"],
                                       batch_stats=state_dict["batch_stats"],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
    
    def checkpoint_exists(self, load_dir):
        return os.path.isfile(os.path.join(load_dir, f"{self.model_name}.ckpt"))