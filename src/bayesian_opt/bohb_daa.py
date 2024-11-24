"""
This file performs hyperparameter optimization using Bayesian Optimization with the HyperBand scheduler from Ray Tune.
It integrates a Vision Transformer model, optimizing over a manually defined configuration space using ConfigSpace.
"""
import sys
sys.path.append('/home/sur06423/wacv_paper/wacv_paper')
# OS Specific Imports
import os
import time
from datetime import datetime, timedelta
import logging
import getpass

# Ray Specific Imports
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, with_resources
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
from ConfigSpace.conditions import EqualsCondition, OrConjunction

# Configuration and ther mapping specific imports
import hydra
from omegaconf import DictConfig
from config import register_configs  # Importing custom config registration

# PyTorch specific and Local Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from src.utils.utils import ImageFolderCustom, LinearClassifier
from src.utils.utils import (
    make_classification_eval_transform,
    make_classification_train_transform,
    setup_ccname,
)

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


def create_fine_tuning_config_space(cfg: DictConfig):
    """Creates a configuration space for fine-tuning using ConfigSpace and includes dataset and model selection.

    Args:
        cfg (DictConfig): Configuration dictionary provided by Hydra.

    Returns:
        CS.ConfigurationSpace: ConfigSpace object for BOHB.
    """
    config_space = CS.ConfigurationSpace()
    # Access the chosen dataset and model keys
    chosen_dataset_key = cfg.experiment.dataset
    chosen_model_key = cfg.experiment.model

    # Retrieve the chosen dataset and model configurations
    dataset_config = cfg.datasets[chosen_dataset_key]
    model_config = cfg.models[chosen_model_key]

    # Add dataset and model type as constant hyperparameters
    # CS.Constant is used to define a hyperparameter with a fixed value
    # that does not change during the hyperparameter optimization process.
    dataset_name = CS.Constant("dataset_name", dataset_config.name)
    num_classes = CS.Constant("num_classes", dataset_config.num_classes)
    train_dir = CS.Constant("train_dir", dataset_config.train_dir)
    val_dir = CS.Constant("val_dir", dataset_config.val_dir)
    model_type = CS.Constant("model_type", model_config.type)

    # Define main hyperparameters
    # ConfigSpace converts the list into a set for internal use because
    # A set automatically ensures that there are no duplicate values in the
    # choices.
    batch_size = CS.Constant("batch_size", cfg.hyperparameters.batch_size)
    optimizer = CS.CategoricalHyperparameter(
        "optimizer", cfg.hyperparameters.optimizer.choices
    )
    learning_rate = CS.UniformFloatHyperparameter(
        "learning_rate",
        lower=cfg.hyperparameters.learning_rate.lower,
        upper=cfg.hyperparameters.learning_rate.upper,
        log=cfg.hyperparameters.learning_rate.log,
    )
    weight_decay = CS.UniformFloatHyperparameter(
        "weight_decay",
        lower=cfg.hyperparameters.weight_decay.lower,
        upper=cfg.hyperparameters.weight_decay.upper,
        log=cfg.hyperparameters.weight_decay.log,
    )
    momentum = CS.UniformFloatHyperparameter(
        "momentum",
        lower=cfg.hyperparameters.momentum.lower,
        upper=cfg.hyperparameters.momentum.upper,
    )
    scheduler = CS.CategoricalHyperparameter(
        "scheduler", cfg.hyperparameters.scheduler.choices
    )
    step_size = CS.CategoricalHyperparameter(
        "step_size", cfg.hyperparameters.step_size.choices
    )
    gamma = CS.UniformFloatHyperparameter(
        "gamma",
        lower=cfg.hyperparameters.gamma.lower,
        upper=cfg.hyperparameters.gamma.upper,
    )
    T_max = CS.CategoricalHyperparameter("T_max", cfg.hyperparameters.T_max.choices)
    eta_min = CS.Constant("eta_min", cfg.hyperparameters.eta_min)
    start_lr = CS.Constant(
        "start_lr", cfg.hyperparameters.start_lr
    )  # Can be set relative to learning_rate
    end_lr = CS.Constant("end_lr", cfg.hyperparameters.end_lr)
    epochs = CS.CategoricalHyperparameter("epochs", cfg.hyperparameters.epochs.choices)
    # CS.CategoricalHyperparameter, expects an iterable,
    # so converting the boolean value into a list with one element
    use_gpu = CS.CategoricalHyperparameter(
        "use_gpu", [cfg.hyperparameters.use_gpu.enabled]
    )  # Explicitly set use_gpu in a list

    # Add all hyperparameters to the configuration space
    config_space.add(
        [
            dataset_name,
            num_classes,
            train_dir,
            val_dir,
            model_type,
            batch_size,
            optimizer,
            learning_rate,
            weight_decay,
            momentum,
            scheduler,
            step_size,
            gamma,
            T_max,
            eta_min,
            start_lr,
            end_lr,
            epochs,
            use_gpu,
        ]
    )

    # Add conditions for conditional hyperparameters
    momentum_condition = EqualsCondition(momentum, optimizer, "SGD")
    step_size_condition = EqualsCondition(step_size, scheduler, "StepLR")

    # Combine conditions for gamma using OrConjunction
    gamma_condition = OrConjunction(
        EqualsCondition(gamma, scheduler, "StepLR"),
        EqualsCondition(gamma, scheduler, "ExponentialLR"),
    )

    T_max_condition = EqualsCondition(T_max, scheduler, "CosineAnnealing")
    eta_min_condition = EqualsCondition(eta_min, scheduler, "CosineAnnealing")
    end_lr_condition = EqualsCondition(end_lr, scheduler, "LinearLR")

    # Add all conditions to the configuration space
    config_space.add(
        [
            momentum_condition,
            step_size_condition,
            gamma_condition,
            T_max_condition,
            eta_min_condition,
            end_lr_condition,
        ]
    )

    return config_space


class ModelFactory:
    """Factory class for creating different models and their associated transforms."""

    @staticmethod
    def get_model_and_transforms(model_type, num_classes):
        """Returns the model and transforms based on the specified model type.

        Args:
            model_type (str): The type of model to use ('vit', 'dinov2', etc.).
            num_classes (int): Number of classes for the output layer.

        Returns:
            Tuple[torch.nn.Module, callable, callable]: The model, train transform, and eval transform.
        """
        if model_type == "vit_b_16":
            # Vision Transformer Model
            pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            model = torchvision.models.vit_b_16(weights=pretrained_weights)
            for param in model.parameters():
                param.requires_grad = False
            model.heads = nn.Linear(in_features=768, out_features=num_classes)
            train_transform = pretrained_weights.transforms()
            eval_transform = pretrained_weights.transforms()
            return model, train_transform, eval_transform

        elif model_type == "dinov2_vitb14":
            # DINOv2 ViT-B/14 Model
            dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            model = LinearClassifier(
                backbone=dinov2_vitb14, in_features=768, num_classes=num_classes
            )
            for param in model.backbone.parameters():
                param.requires_grad = False
            train_transform = make_classification_train_transform()
            eval_transform = make_classification_eval_transform()
            return model, train_transform, eval_transform

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class DataLoaderFactory:
    """Factory class for creating dataset loaders based on the dataset name."""

    @staticmethod
    def get_data_loaders(
        dataset_name,
        train_dir,
        val_dir,
        train_transform=None,
        eval_transform=None,
        batch_size=None,
    ):
        """Creates training and validation data loaders based on the dataset name.

        Args:
            dataset_name (str): The name of the dataset ('Statefarm', 'Pizza_dataset', 'DAA').
            train_transform (callable): Transform for the training data.
            eval_transform (callable): Transform for the validation data.
            batch_size (int): Number of images in each batch of data.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Data loaders for training and validation.
        """
        # Use custom dataset class for DAA, otherwise use ImageFolder for
        # standard datasets like Cifar-10
        if dataset_name == "DAA":
            train_dataset = ImageFolderCustom(
                train_dir, 
                transform=train_transform,
            )
            val_dataset = ImageFolderCustom(
                val_dir, 
                transform=eval_transform,
            )
        else:
            train_dataset = ImageFolder(
                root=train_dir, 
                transform=train_transform,
            )
            val_dataset = ImageFolder(
                root=val_dir, 
                transform=eval_transform,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=8,
            prefetch_factor=6,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=8,
            prefetch_factor=6,
        )
        return train_loader, val_loader


class Metrics:
    """Provides methods to calculate evaluation metrics for model performance."""

    @staticmethod
    def calculate_balanced_accuracy(y_pred, y_true, num_classes):
        """Calculates the balanced accuracy across given predictions and true labels.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Balanced accuracy score.
        """
        correct_per_class = torch.zeros(num_classes, device=y_pred.device)
        total_per_class = torch.zeros(num_classes, device=y_pred.device)
        for c in range(num_classes):
            true_positives = ((y_pred == c) & (y_true == c)).sum()
            condition_positives = (y_true == c).sum()
            correct_per_class[c] = true_positives.float()
            total_per_class[c] = condition_positives.float()
        recall_per_class = correct_per_class / total_per_class.clamp(min=1)
        return recall_per_class.mean().item()


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

    def _initialize_scheduler(self, config):
        """Initializes the learning rate scheduler based on the configuration.

        Args:
            config (dict): Configuration dictionary specifying the scheduler type and parameters.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: Initialized scheduler, or None if not applicable.
        """
        if config["scheduler"] == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config["step_size"], gamma=config["gamma"]
            )
        elif config["scheduler"] == "CosineAnnealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["T_max"], eta_min=config["eta_min"]
            )
        elif config["scheduler"] == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=config["gamma"]
            )
        elif config["scheduler"] == "LinearLR":
            # If start_lr is None, use learning_rate from the optimizer
            start_factor = (
                1.0
                if config["start_lr"] is None
                else config["start_lr"] / config["learning_rate"]
            )
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                end_factor=config["end_lr"] / config["learning_rate"],
            )

    def step(self):
        """Executes a single step of training and validation.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.
        
        current_time = datetime.now()
        if current_time - self.last_auth_time > self.auth_interval:
            setup_ccname()  # Re-authenticate
            self.last_auth_time = current_time  # Reset the authentication time
        """

        train_loss, train_acc = self._train_one_epoch()
        if self.scheduler:
            self.scheduler.step()

        # Get the current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        val_loss, val_acc = self._validate_one_epoch()
        return {
            "loss": train_loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
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
        balanced_acc = Metrics.calculate_balanced_accuracy(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        return avg_loss, balanced_acc

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
        balanced_acc = Metrics.calculate_balanced_accuracy(
            torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
        )
        return avg_loss, balanced_acc

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


def setup_library_paths():
    """Configures additional library paths required for GPU computation.
    Note: These paths are specific to my local server.
          You might not need to define them.

    Adds specified library paths to the LD_LIBRARY_PATH environment variable
    if they are not already included, ensuring GPU dependencies are located.
    """
    library_paths = [
        "/usr/lib/xorg-nvidia-525.116.04/lib/x86_64-linux-gnu",
        "/usr/lib/xorg/lib/x86_64-linux-gnu",
        "/usr/lib/xorg-nvidia-535.113.01/lib/x86_64-linux-gnu",
    ]
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [path for path in library_paths if path not in current_ld_library_path]
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + [current_ld_library_path])
    print("Updated LD_LIBRARY_PATH:")
    print(os.environ["LD_LIBRARY_PATH"])


# Register the configurations
register_configs()


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config"
)  # Hydra Decorator
def main(cfg: DictConfig):
    """Main function to set up and execute the hyperparameter tuning."""
    print("Loaded Configuration:")
    print(cfg)
    ray.shutdown()
    ray.init(
        num_cpus=cfg.ray.num_cpus,
        num_gpus=cfg.ray.num_gpus,
        include_dashboard=cfg.ray.include_dashboard,
    )
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
        reduction_factor=2,
        stop_last_trials=False,
    )

    # Create the TuneBOHB search algorithm
    bohb_search = TuneBOHB(config_space, metric="val_acc", mode="max")
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
            #reuse_actors=True,
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

    ray.shutdown()


if __name__ == "__main__":
    main()
