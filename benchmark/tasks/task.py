import os
from tqdm import tqdm
import torch
import wandb

from .utils import parse_configs
from delay_optimizer import DelayedOptimizer

class OptimizationTask:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.loss_func = None
        self.model = None

    def run_train(self, config=dict(), **kwargs):
        # Get configs
        optimizer_config, train_config, logging_config = parse_configs(**config, **kwargs)

        # Prep data loaders
        self.data.initialize_dataloaders(train_config.batch_size)

        # Skip training if not requested
        if not train_config.do_train:
            if train_config.do_validate:
                return self.run_validation()
            else:
                return

        # Initialize optimizer
        optimizer = DelayedOptimizer(
            params=self.model.parameters(),
            optimizer_class=optimizer_config.optimizer_class,
            **optimizer_config.optimizer_kwargs
        )
        
        # Initialize lr scheduler
        lr_scheduler = train_config.lr_scheduler
        if lr_scheduler is not None:        # None = constant lr
            lr_scheduler = lr_scheduler(**train_config.lr_scheduler_kwargs)

        # Set up logging
        run = None
        if logging_config.wandb_project is not None:
            wandb.login()
            defaults = {k: v for k, v in optimizer.defaults.items() if k != 'init_history'}
            wandb_config = {
                "name": logging_config.run_name,
                "model": self.model.__class__.__name__,
                "optimizer": optimizer_config.optimizer_class.__name__,
                "defaults": defaults,
                "batch_size": train_config.batch_size,
                "num_epochs": train_config.num_epochs,
                "lr_scheduler": lr_scheduler.__class__.__name__ if lr_scheduler is not None else "None",
            }
            run = wandb.init(
                project=logging_config.wandb_project, 
                name=logging_config.run_name, 
                config=wandb_config
            )

        def train_log_callback(loss, output, labels, pbar=None):
            global_step += 1
            if global_step % logging_config.logging_steps == 0:
                train_preds = torch.argmax(output, dim=-1)
                train_correct = torch.sum(train_preds == labels).cpu().item()
                train_acc = train_correct / len(labels)
                if run is not None:
                    run.log({
                        "train/global_step": global_step,
                        "train/epoch": global_step / len(self.data.train_loader),
                        "train/loss": loss.cpu().item(),
                        "train/accuracy": train_acc,
                    })
                if logging_config.do_progress_bar and pbar is not None:
                    pbar.set_postfix_str({
                        f"Train Loss: {loss:.4f}, "
                        f"Train Acc.: {train_acc:.4f}"
                    })

        # Training loop
        global_step = 0
        do_progress_bar = logging_config.do_progress_bar
        for epoch in range(train_config.num_epochs):
            train_loader = self.data.train_loader
            if do_progress_bar:         # Set up progress bar
                train_loader = tqdm(train_loader, desc=f"Train (Epoch: {epoch})")

            self._train_epoch(train_loader, logging_callback=train_log_callback)

            # Update lr scheduler after each epoch
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Run validation on the model at the end of each epoch
            if train_config.do_validate:
                val_loss, val_acc = self.run_validation(do_progress_bar)

                # Log vaidation metrics
                if run is not None:
                    run.log({
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                        "eval/loss": eval_loss / len(self.data.val_loader),
                        "eval/accuracy": eval_correct / eval_num_items,
                    })

        # Save the model
        if logging_config.save_dir is not None:
            os.makedirs(logging_config.save_dir, exist_ok=True)
            task_name = self.__class__.__name__
            path = os.path.join(logging_config.save_dir, f"{task_name}.pth")
            train_state = {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "train_config": train_config,
            }
            torch.save(train_state, path)

    def _train_epoch(self, train_loader, logging_callback=None):
        self.model.train()
        for i, (batch, labels) in enumerate(train_loader):
            batch = batch.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            optimizer.apply_delays()
            output = self.model(batch)
            loss = self.loss_func(output, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log training metrics
            if logging_callback is not None:
                logging_callback(
                    loss.cpu().item(), output, labels, pbar=train_loader
                )

    def run_validation(self, do_progress_bar=True):
        val_loader = self.data.val_loader
        if do_progress_bar:
            val_loader = tqdm(val_loader)

        val_total_loss = 0
        val_correct = 0
        val_num_items = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch, labels) in enumerate(val_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                output = self.model(batch)
                val_total_loss += self.loss_func(output, labels).cpu().item()
                val_preds = torch.argmax(output, dim=-1)
                val_correct += torch.sum(val_preds == labels).cpu().item()
                val_num_items += len(labels)
                if do_progress_bar:
                    val_loader.set_postfix_str({
                        f"Validation Loss: {val_total_loss / (i+1):.4f}, "
                        f"Validation Acc.: {val_correct / val_num_items:.4f}"
                    })

        loss = val_total_loss / len(val_loader)
        acc = val_correct / val_num_items
        return loss, acc

    def run_test(self, do_progress_bar=True):
        if do_progress_bar:
            pbar = tqdm(self.data.test_loader) 
        else:
            pbar = test_loader

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for batch, labels in pbar:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(batch)
                preds = torch.argmax(outputs, dim=-1)
                total += len(labels)
                correct += torch.sum(preds == labels).cpu().item()

        return correct / total
