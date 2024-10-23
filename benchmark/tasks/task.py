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
            if train_config.do_test:
                return self.run_test(logging_config.do_progress_bar)
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

        # Setup logging context manager
        run = wandb.init(
            project=logging_config.wandb_project, 
            name=logging_config.run_name, 
            config=wandb_config
        )
        with run:
            # Training loop
            train_correct = 0
            train_num_items = 0
            global_step = 0
            for epoch in range(train_config.num_epochs):
                self.model.train()
                if logging_config.do_progress_bar:     # Initialize progress bar
                    pbar = tqdm(self.data.train_loader, desc=f"Epoch: {epoch}")
                    eval_pbar = tqdm(self.data.val_loader)
                else:
                    pbar = self.data.train_loader
                    eval_pbar = self.data.val_loader

                # Do training epoch
                for batch, labels in pbar:
                    global_step += 1
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    optimizer.apply_delays()
                    output = self.model(batch)
                    loss = self.loss_func(output, labels)

                    # Backward pass
                    loss.backward()
                    optimizer.step()                
                    optimizer.zero_grad()

                    # Compute training metrics
                    train_preds = torch.argmax(output, dim=-1)
                    train_correct += torch.sum(train_preds == labels).cpu().item()
                    train_num_items += len(labels)

                    # Log
                    if global_step % logging_config.logging_steps == 0:
                        train_acc = train_correct / train_num_items
                        run.log({
                            "train/global_step": global_step,
                            "train/epoch": global_step / len(self.data.train_loader),
                            "train/loss": loss.cpu().item(),
                            "train/accuracy": train_acc,
                        })
                        train_correct, train_num_items = 0, 0
                        if logging_config.do_progress_bar:
                            pbar.set_description(
                                f"Epoch: {epoch}, Loss: {loss.cpu().item():.4f}, Accuracy: {train_acc:.4f}"
                            )

                # Update lr scheduler after each epoch
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Run validation on the model at the end of each epoch
                if train_config.do_validate:
                    eval_loss = 0
                    eval_correct = 0
                    eval_num_items = 0

                    self.model.eval()
                    with torch.no_grad():
                        for i, (batch, labels) in enumerate(eval_pbar):
                            batch = batch.to(self.device)
                            labels = labels.to(self.device)
                            output = self.model(batch)
                            eval_loss += self.loss_func(output, labels).cpu().item()
                            eval_preds = torch.argmax(output, dim=-1)
                            eval_correct += torch.sum(eval_preds == labels).cpu().item()
                            eval_num_items += len(labels)
                            if logging_config.do_progress_bar:
                                eval_pbar.set_description(
                                    f"Eval Loss: {eval_loss / (i+1):.4f}, Eval Accuracy: {eval_correct / eval_num_items:.4f}"
                                )

                    # Log vaidation metrics
                    wandb.log({
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

        # Return test accuracy
        if train_config.do_test:
            return self.run_test(logging_config.do_progress_bar)

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
