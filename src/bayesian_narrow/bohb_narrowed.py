"""
This file performs hyperparameter optimization using Bayesian Optimization with the HyperBand scheduler from Ray Tune.
It integrates a Vision Transformer model, optimizing over a manually defined configuration space using ConfigSpace.

Read more about Bayesian Optimisation here:
https://wandb.ai/wandb_fc/articles/reports/Bayesian-Hyperparameter-Optimization-A-Primer--Vmlldzo1NDQyNzcw

"""
import sys
sys.path.append('/home/sur06423/wacv_paper/wacv_paper')
# OS Specific Imports
import os
import time
from datetime import datetime, timedelta
import logging

# Ray Specific Imports
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, with_resources
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS

# Configuration and ther mapping specific imports
import hydra
from omegaconf import DictConfig
from config_narrow import register_configs  # Importing custom config registration

# PyTorch specific and Local Imports
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from src.utils.metrics import Metrics
from src.utils.utils import setup_library_paths
from src.model.model import ModelFactory, DataLoaderFactory
from src.config.config_space import create_fine_tuning_config_space

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"hyperopt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ),
        logging.StreamHandler(),
    ],
)

class TrainViT(tune.Trainable):
    """A trainable class for Ray Tune that handles the training and validation of a Vision Transformer model."""

    def setup(self, config):
        """Prepares the model, data loaders, optimizer, and scheduler for training based on the configuration provided.

        Args:
            config (dict): Configuration dictionary containing hyperparameters and model settings.
        """
        #self.last_auth_time = datetime.now()  # Store the last authentication time
        #self.auth_interval = timedelta(hours=8)  # Set re-authentication interval
        #setup_ccname()  # Initial authentication

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu"
        )
        # Extract parameters from the unified config
        self.dataset_name = config["dataset_name"]
        self.num_classes = config["num_classes"]
        self.train_dir = config["train_dir"]
        self.val_dir = config["val_dir"]
        self.model_type = config["model_type"]
        self.batch_size = config["batch_size"]

        # Create the model and transforms using ModelFactory
        self.model, self.train_transform, self.eval_transform = (
            ModelFactory.get_model_and_transforms(
                model_type=self.model_type, num_classes=self.num_classes
            )
        )
        self.model.to(self.device)

        # Create the data loaders using DataLoaderFactory
        self.train_loader, self.val_loader = DataLoaderFactory.get_data_loaders(
            dataset_name=self.dataset_name,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
            batch_size=self.batch_size,
        )

        self.optimizer = self._initialize_optimizer(config)
        self.scheduler = self._initialize_scheduler(config)

    def _initialize_optimizer(self, config):
        """Initializes the optimizer based on the configuration.

        Args:
            config (dict): Configuration dictionary specifying the optimizer type and parameters.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        if config["optimizer"] == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=config["learning_rate"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        elif config["optimizer"] == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        elif config["optimizer"] == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        else:
            raise ValueError(f"Optimizer type '{config['optimizer']}' is not supported. Supported types are: SGD, Adam, AdamW.")
        

    def _initialize_scheduler(self, config):
        """Initializes the learning rate scheduler based on the configuration.

        Args:
            config (dict): Configuration dictionary specifying the scheduler type and parameters.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: Initialized scheduler, or None if not applicable.
        """
        scheduler_type = config["scheduler"]

        if scheduler_type == "StepLR":
            # Check for required parameters for StepLR
            if "step_size" not in config or "gamma" not in config:
                raise ValueError("For StepLR, both 'step_size' and 'gamma' parameters must be provided.")
            
            step_size = config["step_size"]
            gamma = config["gamma"]

            if not isinstance(step_size, int) or step_size <= 0:
                raise ValueError("'step_size' should be a positive integer.")
            if not (0 < gamma <= 1):
                raise ValueError("'gamma' should be a float between 0 and 1 (exclusive).")
            
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        elif scheduler_type == "CosineAnnealing":
            # Check for required parameters for CosineAnnealingLR
            if "T_max" not in config or "eta_min" not in config:
                raise ValueError("For CosineAnnealingLR, both 'T_max' and 'eta_min' parameters must be provided.")
            
            T_max = config["T_max"]
            eta_min = config["eta_min"]

            if not isinstance(T_max, int) or T_max <= 0:
                raise ValueError("'T_max' should be a positive integer.")
            if not (eta_min >= 0):
                raise ValueError("'eta_min' should be a non-negative float.")
            
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)

        elif scheduler_type == "ExponentialLR":
            # Check for required parameters for ExponentialLR
            if "gamma" not in config:
                raise ValueError("For ExponentialLR, the 'gamma' parameter must be provided.")
            
            gamma = config["gamma"]

            if not (0 < gamma <= 1):
                raise ValueError("'gamma' should be a float between 0 and 1 (exclusive).")
            
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        elif scheduler_type == "LinearLR":
            # Check for required parameters for LinearLR
            start_factor = config["start_factor"]
            end_factor = config["end_factor"]
            total_iters = config["total_iters"]

            # Validate factors for LinearLR
            if not (0 < start_factor <= 1.0):
                raise ValueError("start_factor must be a float between 0 and 1 (exclusive of 0).")
            if not (0 <= end_factor <= 1.0):
                raise ValueError("end_factor must be a float between 0 and 1 (inclusive).")
            if not isinstance(total_iters, int) or total_iters <= 0:
                raise ValueError("'total_iters' should be a positive integer.")

            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters
            )
        
        elif scheduler_type == "SequentialLR":
            """Initialize SequentialLR with Linear Warm-Up + CosineAnnealingWarmRestarts."""
            # Warm-up phase
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=config["start_factor"],
                end_factor=config["end_factor"],
                total_iters=config["total_iters"]
            )
            # Cosine annealing phase
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config["T_zero"],
                T_mult=config["T_mult"],
                eta_min=config["eta_min"]
            )
            # Combine schedulers
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[config["total_iters"]]
            )

        else:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported. Supported types are: StepLR, CosineAnnealingLR, ExponentialLR, LinearLR, SequentialLR.")


    def step(self):
        """Executes a single step of training and validation.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.
        
        current_time = datetime.now()
        if current_time - self.last_auth_time > self.auth_interval:
            setup_ccname()  # Re-authenticate
            self.last_auth_time = current_time  # Reset the authentication time
        """

        train_loss, train_acc, train_f1_score = self._train_one_epoch()
        if self.scheduler:
            self.scheduler.step()

        # Get the current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        val_loss, val_acc, val_f1_score = self._validate_one_epoch()
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1_score": train_f1_score,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_score": val_f1_score,
            "learning_rate": current_lr,
        }

    def _train_one_epoch(self):
        """Conducts a single epoch of training on the entire training dataset.

        Returns:
            Tuple[float, float]: Training loss and Balanced accuracy.
        """
        self.model.train()
        running_loss = 0.0
        all_predictions, all_labels = [], []
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            all_predictions.append(torch.argmax(torch.softmax(outputs, dim=1), dim=1))
            all_labels.append(labels)
        avg_loss = running_loss / len(self.train_loader.dataset)
        balanced_acc = Metrics.calculate_balanced_accuracy_torchmetrics(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        f1_score = Metrics.calculate_f1_score(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        return avg_loss, balanced_acc, f1_score

    def _validate_one_epoch(self):
        """Conducts validation on the entire validation dataset and computes loss and accuracy.

        Returns:
            Tuple[float, float]: Validation loss and Balanced accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                all_predictions.append(
                    torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                )
                all_labels.append(labels)
        avg_loss = running_loss / len(self.val_loader.dataset)
        balanced_acc = Metrics.calculate_balanced_accuracy_torchmetrics(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        f1_score = Metrics.calculate_f1_score(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        return avg_loss, balanced_acc, f1_score

    def save_checkpoint(self, checkpoint_dir):
        """Saves the current model and optimizer state to a checkpoint.

        Args:
            checkpoint_dir (str): Directory path to save the checkpoint.

        Returns:
            str: Path to the checkpoint directory.
        """
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        """Loads the model and optimizer state from a checkpoint.

        Args:
            checkpoint_dir (str): Directory path from which to load the checkpoint.
        """
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



# Register the configurations
register_configs()


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_narrow"
)  # Hydra Decorator
def main(cfg: DictConfig):
    """Main function to set up and execute the hyperparameter tuning."""
    print("Loaded Configuration:")
    print(cfg)
    setup_library_paths()

    # Create the ConfigSpace
    config_space = create_fine_tuning_config_space(cfg)

    # Set the max_t to the maximum possible number of epochs from the
    # hyperparameter space
    max_epochs = max(config_space["epochs"].choices)

    # Create the HyperBandForBOHB scheduler with the max_t parameter set to
    # max_epochs
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_epochs,
        reduction_factor=2, # Moderate early stopping aggressiveness.
        stop_last_trials=False,
    )

    # Initial hyperparameter configurations to try first,it will seed the search with configurations based on prior knowledge or results.
    points_to_evaluate = cfg.hyperparameters.points_to_evaluate

    # Create the TuneBOHB search algorithm
    bohb_search = TuneBOHB(
        space=config_space,
        metric="val_acc", 
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )

    bohb_search = tune.search.ConcurrencyLimiter(
        bohb_search, max_concurrent=cfg.tuner.max_concurrent
    )

    # Update the RunConfig to stop based on the dynamically adjusted epochs
    run_config = ray.train.RunConfig(
        name=cfg.run_config.name,
        storage_path=cfg.run_config.storage_path,
        stop={"training_iteration": max_epochs},
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_frequency=cfg.run_config.checkpoint_frequency,
            checkpoint_at_end=True,
        ),
    )

    # Run the Tuner with the updated configuration
    tuner = Tuner(
        trainable=with_resources(
            TrainViT,
            resources=lambda config: (
                {
                    "gpu": cfg.tuner.resources_per_trial.gpu,
                    "cpu": cfg.tuner.resources_per_trial.cpu,
                }
                if config.get("use_gpu", False)
                else {"cpu": cfg.tuner.resources_per_trial.cpu}
            ),
        ),
        param_space={},  # Leave param_space empty as TuneBOHB uses config_space
        tune_config=TuneConfig(
            metric="val_acc",
            mode="max",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=cfg.tuner.num_samples,
            # reuse_actors=True,
        ),
        run_config=run_config,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_acc", mode="max")
    logging.info("Best trial config: {}".format(best_result.config))
    logging.info(
        "Best trial final validation accuracy: {}".format(
            best_result.metrics["val_acc"]
        )
    )


if __name__ == "__main__":
    main()
