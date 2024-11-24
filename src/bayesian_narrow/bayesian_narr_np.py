"""
This file performs hyperparameter optimization using Bayesian Optimization with the HyperBand scheduler from Ray Tune.
It integrates a Vision Transformer model, optimizing over a manually defined configuration space using ConfigSpace.
"""

import sys
import os
import time
from datetime import datetime, timedelta
import logging

try:
    import ray
    from ray import tune
    from ray.tune import Tuner, TuneConfig, with_resources
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray.tune.search.bohb import TuneBOHB
    import ConfigSpace as CS
except ImportError as e:
    logging.error(f"Failed to import a required module: {e}")
    sys.exit(1)

try:
    import hydra
    from omegaconf import DictConfig
except ImportError as e:
    logging.error(f"Failed to import Hydra or OmegaConf: {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torch.nn.functional as F
    from torchvision.datasets import ImageFolder
except ImportError as e:
    logging.error(f"PyTorch libraries are missing: {e}")
    sys.exit(1)

# Local Imports
sys.path.append('/home/sur06423/wacv_paper/wacv_paper')
from src.utils.utils import ImageFolderCustom, LinearClassifier
from src.utils.utils import (
    make_classification_eval_transform,
    make_classification_train_transform,
    setup_ccname,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"hyperopt_results_narrow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
    try:
        config_space = CS.ConfigurationSpace()
        # Access the chosen dataset and model keys
        chosen_dataset_key = cfg.experiment.dataset
        chosen_model_key = cfg.experiment.model
        # Retrieve the chosen dataset and model configurations
        dataset_config = cfg.datasets[chosen_dataset_key]
        model_config = cfg.models[chosen_model_key]

        dataset_name = CS.Constant("dataset_name", dataset_config.name)
        num_classes = CS.Constant("num_classes", dataset_config.num_classes)
        train_dir = CS.Constant("train_dir", dataset_config.train_dir)
        val_dir = CS.Constant("val_dir", dataset_config.val_dir)
        model_type = CS.Constant("model_type", model_config.type)
        # Define main hyperparameters
        batch_size = CS.Constant("batch_size", cfg.hyperparameters.batch_size)
        optimizer = CS.CategoricalHyperparameter("optimizer", cfg.hyperparameters.optimizer.choices)
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
        scheduler = CS.CategoricalHyperparameter("scheduler", cfg.hyperparameters.scheduler.choices)
        start_factor = CS.Constant("start_factor", cfg.hyperparameters.scheduler.start_factor)
        end_factor = CS.Constant("end_factor", cfg.hyperparameters.scheduler.end_factor)
        total_iters = CS.Constant("total_iters", cfg.hyperparameters.scheduler.total_iters)
        epochs = CS.CategoricalHyperparameter("epochs", cfg.hyperparameters.epochs.choices)
        use_gpu = CS.CategoricalHyperparameter("use_gpu", [cfg.hyperparameters.use_gpu.enabled])

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
                scheduler,
                start_factor,
                end_factor,
                total_iters,
                epochs,
                use_gpu,
            ]
        )
        return config_space

    except Exception as e:
        logging.error(f"Error in creating config space: {e}")
        raise

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
        try:
            if model_type == "vit_b_16":
                pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
                model = torchvision.models.vit_b_16(weights=pretrained_weights)
                for param in model.parameters():
                    param.requires_grad = False
                model.heads = nn.Linear(in_features=768, out_features=num_classes)
                train_transform = pretrained_weights.transforms()
                eval_transform = pretrained_weights.transforms()
                return model, train_transform, eval_transform

            elif model_type == "dinov2_vitb14":
                dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
                model = LinearClassifier(backbone=dinov2_vitb14, in_features=768, num_classes=num_classes)
                for param in model.backbone.parameters():
                    param.requires_grad = False
                train_transform = make_classification_train_transform()
                eval_transform = make_classification_eval_transform()
                return model, train_transform, eval_transform

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except ValueError as ve:
            logging.error(f"Value error in model creation: {ve}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in model creation: {e}")
            raise

class DataLoaderFactory:
    """Factory class for creating dataset loaders based on the dataset name."""

    @staticmethod
    def get_data_loaders(dataset_name, train_dir, val_dir, train_transform=None, eval_transform=None, batch_size=None):
        """Creates training and validation data loaders based on the dataset name.

        Args:
            dataset_name (str): The name of the dataset ('Statefarm', 'Pizza_dataset', 'DAA').
            train_dir (str): The path to the train data. 
            val_dir (str): The path to the validation data. 
            train_transform (callable): Transform for the training data.
            eval_transform (callable): Transform for the validation data.
            batch_size (int): Number of images in each batch of data.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Data loaders for training and validation.
        """
        try:
            if dataset_name == "DAA":
                train_dataset = ImageFolderCustom(train_dir, transform=train_transform)
                val_dataset = ImageFolderCustom(val_dir, transform=eval_transform)
            else:
                train_dataset = ImageFolder(root=train_dir, transform=train_transform)
                val_dataset = ImageFolder(root=val_dir, transform=eval_transform)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=32
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=32
            )
            return train_loader, val_loader
        except Exception as e:
            logging.error(f"Failed to create data loaders: {e}")
            raise


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
        try:
            if y_pred.size(0) != y_true.size(0):
                raise ValueError("Prediction and label tensors must have the same number of samples.")

            correct_per_class = torch.zeros(num_classes, device=y_pred.device)
            total_per_class = torch.zeros(num_classes, device=y_pred.device)

            for c in range(num_classes):
                true_positives = ((y_pred == c) & (y_true == c)).sum()
                condition_positives = (y_true == c).sum()
                correct_per_class[c] = true_positives.float()
                total_per_class[c] = condition_positives.float()

            recall_per_class = correct_per_class / total_per_class.clamp(min=1)
            balanced_accuracy = recall_per_class.mean().item()
            return balanced_accuracy

        except Exception as e:
            logging.error(f"Error calculating balanced accuracy: {e}")
            raise

class TrainViT(tune.Trainable):
    """A trainable class for Ray Tune that handles the training and validation of a Vision Transformer model."""

    def setup(self, config):
        """Prepares the model, data loaders, optimizer, and scheduler for training based on the configuration provided.
        
        Args:
            config (dict): Configuration dictionary containing hyperparameters and model settings.
        """
        try:
            # Access ConfigSpace configuration values directly
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu"
            )

            self.dataset_name = config["dataset_name"]
            self.num_classes = config["num_classes"]
            self.train_dir = config["train_dir"]
            self.val_dir = config["val_dir"]
            self.model_type = config["model_type"]
            self.batch_size = config["batch_size"]

            # Create the model and transforms using ModelFactory
            self.model, self.train_transform, self.eval_transform = ModelFactory.get_model_and_transforms(
                model_type=self.model_type, num_classes=self.num_classes
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

            # Initialize optimizer and scheduler
            self.optimizer = self._initialize_optimizer(config)
            self.scheduler = self._initialize_scheduler(config)

        except KeyError as e:
            logging.error(f"Missing configuration key in setup: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during setup: {e}")
            raise

    def _initialize_optimizer(self, config):
        """Initializes the optimizer based on the configuration.
        
        Args:
            config (dict): Configuration dictionary specifying the optimizer type and parameters.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        try:
            optimizer_type = config["optimizer"]
            learning_rate = config["learning_rate"]
            weight_decay = config["weight_decay"]

            if optimizer_type == "AdamW":
                return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        except KeyError as e:
            logging.error(f"Missing configuration key for optimizer: {e}")
            raise
        except Exception as e:
            logging.error(f"Error initializing optimizer: {e}")
            raise

    def _initialize_scheduler(self, config):
        """Initializes the learning rate scheduler based on the configuration.
        
        Args:
            config (dict): Configuration dictionary specifying the scheduler type and parameters.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: Initialized scheduler, or None if not applicable.
        """
        try:
            scheduler_type = config["scheduler"]

            if scheduler_type == "LinearLR":
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
                    total_iters=total_iters,
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        except KeyError as e:
            logging.error(f"Missing configuration key for scheduler: {e}")
            raise
        except Exception as e:
            logging.error(f"Error initializing scheduler: {e}")
            raise

    def step(self):
        """Executes a single step of training and validation.
        
        Returns:
            dict: A dictionary containing learning rate, training and validation loss and balanced accuracy.
        """
        try:
            train_loss, train_acc = self._train_one_epoch()
            if self.scheduler:
                self.scheduler.step()

            val_loss, val_acc = self._validate_one_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]

            return {
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }

        except Exception as e:
            logging.error(f"Error during training step: {e}")
            raise

    def _train_one_epoch(self):
        """Conducts a single epoch of training on the entire training dataset.
        
        Returns:
            Tuple[float, float]: Training loss and Balanced accuracy.
        """
        try:
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

        except Exception as e:
            logging.error(f"Error during training epoch: {e}")
            raise

    def _validate_one_epoch(self):
        """Conducts validation on the entire validation dataset.
        
        Returns:
            Tuple[float, float]: Validation loss and Balanced accuracy.
        """
        try:
            self.model.eval()
            running_loss = 0.0
            all_predictions, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    all_predictions.append(torch.argmax(torch.softmax(outputs, dim=1), dim=1))
                    all_labels.append(labels)

            avg_loss = running_loss / len(self.val_loader.dataset)
            balanced_acc = Metrics.calculate_balanced_accuracy(
                torch.cat(all_predictions), torch.cat(all_labels), self.num_classes
            )
            return avg_loss, balanced_acc

        except Exception as e:
            logging.error(f"Error during validation epoch: {e}")
            raise

    def save_checkpoint(self, checkpoint_dir):
        """Saves the current model and optimizer state to a checkpoint.
        
        Args:
            checkpoint_dir (str): Directory path to save the checkpoint.

        Returns:
            str: Path to the checkpoint directory.
        """
        try:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )
            return checkpoint_dir
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_dir):
        """Loads the model and optimizer state from a checkpoint.
        
        Args:
            checkpoint_dir (str): Directory path from which to load the checkpoint.
        """
        try:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise


def setup_library_paths():
    """Configures additional library paths required for GPU computation."""
    try:
        library_paths = [
            "/usr/lib/xorg-nvidia-525.116.04/lib/x86_64-linux-gnu",
            "/usr/lib/xorg/lib/x86_64-linux-gnu",
            "/usr/lib/xorg-nvidia-535.113.01/lib/x86_64-linux-gnu",
        ]
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [path for path in library_paths if path not in current_ld_library_path]
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + [current_ld_library_path])
        logging.info("LD_LIBRARY_PATH updated successfully.")
    except Exception as e:
        logging.error(f"Error while setting up library paths: {e}")

# Register the configurations
try:
    from config_narrow import register_configs  # Importing custom config registration
    register_configs()
except ImportError as e:
    logging.error(f"Configuration registration module missing: {e}")
    sys.exit(1)

@hydra.main(version_base=None, config_path="../../conf", config_name="config_narrow")
def main(cfg: DictConfig):
    """Main function to set up and execute the hyperparameter tuning."""
    try:
        logging.info("Loaded Configuration:")
        logging.info(cfg)
        print(cfg)
        setup_library_paths()

        # Create the ConfigSpace
        config_space = create_fine_tuning_config_space(cfg)
        max_epochs = max(config_space["epochs"].choices)

        # Create the HyperBandForBOHB scheduler
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

        run_config = ray.train.RunConfig(
            name=cfg.run_config.name,
            storage_path=cfg.run_config.storage_path,
            stop={"training_iteration": max_epochs},
            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_frequency=cfg.run_config.checkpoint_frequency,
                checkpoint_at_end=True,
            ),
        )

        # Run the Tuner
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
            ),
            run_config=run_config,
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="val_acc", mode="max")
        logging.info("Best trial config: {}".format(best_result.config))
        logging.info(
            "Best trial final validation accuracy: {}".format(best_result.metrics["val_acc"])
        )

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Failed to run main function: {e}")
        sys.exit(1)
