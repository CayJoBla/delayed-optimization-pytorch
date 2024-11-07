import torch
import os
from tqdm import tqdm

from delay_optimizer import DelayedOptimizer
from .config import RunConfig


class Runner:
    def __init__(self, task, device=None, **task_kwargs):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._task = task       # This should be the task class, not an instance
        self._task_kwargs = task_kwargs
        self.reset()

    def reset(self):
        """Resets the task, training state, and runner configuration.
        Call between training runs when using the same Runner instance.
        """
        self.task = self._task(**self._task_kwargs)
        self._global_step = 0
        self._epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.config = None

    def run(self, config):
        # Initialize config
        if isinstance(config, dict):
            config = RunConfig(**config)
        if not isinstance(config, RunConfig):
            raise ValueError("Invalid config type: must be a dict or RunConfig object.")
        self.config = config

        # Prep data loaders
        self.task.data.initialize_dataloaders(self.config.batch_size)
            
        if self.config.do_train:
            # Initialize optimizer and lr scheduler
            self.optimizer = DelayedOptimizer(
                params=self.task.model.parameters(),
                optimizer_class=self.config.optimizer_class,
                **self.config.optimizer_kwargs          # Includes delay params
            )
            if self.config.lr_scheduler is not None:    # None = constant lr
                self.lr_scheduler = self.config.lr_scheduler(
                                        **self.config.lr_scheduler_kwargs)
            else:
                self.lr_scheduler = None

            # Set up logging
            self._init_wandb_logging()

            # Training loop
            for _ in range(self.config.num_epochs):
                # Run validation before each epoch
                if self.config.do_validate:
                    val_loss, val_acc = self._validate()

                self._epoch += 1
                self._train()

                # Update lr scheduler after each epoch
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # Final validation
        if self.config.do_validate:
            do_logging = self.config.do_wandb_logging and self.config.do_train
            val_loss, val_acc = self._validate(do_wandb_logging=do_logging)

        # Run on the test set
        if self.config.do_test:
            test_loss, test_acc = self._test()

        # End logging
        if self.config.do_wandb_logging:
            self._run.finish()

        # Save the model
        if self.config.save_dir is not None:
            os.makedirs(self.config.save_dir, exist_ok=True)
            path = os.path.join(slef.config.save_dir, f"{self._task.__name__}.pt")
            state = {
                "model": self.task.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "config": self.config,  # TODO: Is this the best way to do this?
            }
            torch.save(state, path)

        return {
            "final_val_loss": val_loss if self.config.do_validate else None,
            "final_val_acc": val_acc if self.config.do_validate else None,
            "final_test_loss": test_loss if self.config.do_test else None,
            "final_test_acc": test_acc if self.config.do_test else None,
        }

    def _init_wandb_logging(self):
        if not self.config.do_wandb_logging:
            self._run = None
            return

        import wandb
        wandb.login()

        defaults = {k: v for k, v in self.optimizer.defaults.items() 
                    if k != 'init_history'}
        wandb_config = {
            "name": self.config.run_name,
            "model": self.task.model.__class__.__name__,
            "optimizer": self.config.optimizer_class.__name__,
            "defaults": defaults,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "lr_scheduler": self.lr_scheduler.__class__.__name__ if \
                            self.lr_scheduler is not None else "None",
        }
        self._run = wandb.init(
            project=self.config.wandb_project, 
            name=self.config.run_name, 
            config=wandb_config,
        )

    def _train(self):
        """Run a training epoch."""
        train_loader = self.task.data.train_loader
        if self.config.do_progress_bar:
            train_loader = tqdm(
                train_loader, desc=f"Train (Epoch: {self._epoch})"
            )

        self.task.model.train()
        for i, (batch, labels) in enumerate(train_loader):
            self._global_step += 1
            batch = batch.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            self.optimizer.apply_delays()
            output = self.task.model(batch)
            loss = self.task.loss_func(output, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log training metrics
            if self._global_step % self.config.logging_steps == 0:
                train_preds = torch.argmax(output, dim=-1)
                train_correct = torch.sum(train_preds == labels).cpu().item()
                train_acc = train_correct / len(labels)
                if self.config.do_wandb_logging:
                    self._run.log({
                        "train/global_step": self._global_step,
                        "train/epoch": self._global_step / len(train_loader),
                        "train/loss": loss.cpu().item(),
                        "train/accuracy": train_acc,
                    })
                if self.config.do_progress_bar:
                    train_loader.set_postfix_str({
                        f"Train Loss: {loss:.4f}, "
                        f"Train Acc.: {train_acc:.4f}"
                    })

    def _validate(self, do_wandb_logging=None):
        """Run evaluation using the validation set."""
        do_wandb_logging = do_wandb_logging or self.config.do_wandb_logging

        val_loader = self.task.data.val_loader
        if self.config.do_progress_bar:
            val_loader = tqdm(val_loader)

        val_total_loss = 0
        val_correct = 0
        val_num_items = 0

        self.task.model.eval()
        with torch.no_grad():
            for i, (batch, labels) in enumerate(val_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.task.model(batch)
                val_total_loss += self.task.loss_func(output, labels).cpu().item()

                val_preds = torch.argmax(output, dim=-1)
                val_correct += torch.sum(val_preds == labels).cpu().item()
                val_num_items += len(labels)

                if self.config.do_progress_bar:
                    val_loader.set_postfix_str({
                        f"Validation Loss: {val_total_loss / (i+1):.4f}, "
                        f"Validation Acc.: {val_correct / val_num_items:.4f}"
                    })
        loss = val_total_loss / len(val_loader)
        acc = val_correct / val_num_items

        # Log validation metrics
        if do_wandb_logging is not None:
            self._run.log({
                "train/global_step": self._global_step,
                "train/epoch": self._epoch,
                "eval/loss": loss,
                "eval/accuracy": acc,
            })
            
        return loss, acc

    def _test(self):
        """Run evaluation using the test set."""
        test_loader = self.task.data.test_loader
        if self.config.do_progress_bar:
            test_loader = tqdm(test_loader)

        test_total_loss = 0
        test_correct = 0
        test_num_items = 0

        self.task.model.eval()
        with torch.no_grad():
            for i, (batch, labels) in enumerate(test_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                outputs = self.task.model(batch)
                test_total_loss += self.task.loss_func(outputs, labels).cpu().item()

                test_preds = torch.argmax(outputs, dim=-1)
                test_correct += torch.sum(test_preds == labels).cpu().item()
                test_num_items += len(labels)

                if self.config.do_progress_bar:
                    test_loader.set_postfix_str({
                        f"Test Loss: {test_total_loss / (i+1):.4f}, "
                        f"Test Acc.: {test_correct / val_num_items:.4f}"
                    })
        loss = test_total_loss / len(test_loader)
        acc = test_correct / test_num_items

        return loss, acc