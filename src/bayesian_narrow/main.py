# main.py
import hydra
from omegaconf import DictConfig
import ConfigSpace as CS
import sys

# Add your project root to the system path
sys.path.append('/home/sur06423/wacv_paper/wacv_paper')

# Configuration registration import
try:
    from config_narrow import register_configs  # Importing custom config registration
    register_configs()
except ImportError as e:
    print(f"Configuration registration module missing: {e}")
    sys.exit(1)


def create_fine_tuning_config_space(cfg: DictConfig):
    """Creates a configuration space for fine-tuning using ConfigSpace and includes dataset and model selection.

    Args:
        cfg (DictConfig): Configuration dictionary provided by Hydra.

    Returns:
        CS.ConfigurationSpace: ConfigSpace object for BOHB.
    """
    print("Debug: Inside create_fine_tuning_config_space")
    print(f"Loaded cfg: {cfg}")

    try:
        # Access dataset and model keys
        chosen_dataset_key = cfg.experiment.dataset
        print(f"Chosen Dataset Key: {chosen_dataset_key}")
        chosen_model_key = cfg.experiment.model
        print(f"Chosen Model Key: {chosen_model_key}")

        # Retrieve dataset and model configurations
        dataset_config = cfg.datasets[chosen_dataset_key]
        print(f"Dataset Config: {dataset_config}")
        model_config = cfg.models[chosen_model_key]
        print(f"Model Config: {model_config}")

        # Initialize ConfigSpace
        config_space = CS.ConfigurationSpace()
        print("Debug: Initialized Configuration Space")

        # Add dataset and model-specific constants
        dataset_name = CS.Constant("dataset_name", str(dataset_config.name))
        num_classes = CS.Constant("num_classes", int(dataset_config.num_classes))
        train_dir = CS.Constant("train_dir", str(dataset_config.train_dir))
        val_dir = CS.Constant("val_dir", str(dataset_config.val_dir))
        model_type = CS.Constant("model_type", str(model_config.type))

        config_space.add([dataset_name, num_classes, train_dir, val_dir, model_type])
        print(f"Added dataset and model parameters: {dataset_name}, {num_classes}, {train_dir}, {val_dir}, {model_type}")

        # Add hyperparameters
        batch_size = CS.Constant("batch_size", int(cfg.hyperparameters.batch_size))
        optimizer = CS.CategoricalHyperparameter(
            "optimizer", cfg.hyperparameters.optimizer.choices
        )
        learning_rate = CS.UniformFloatHyperparameter(
            "learning_rate",
            lower=float(cfg.hyperparameters.learning_rate.lower),
            upper=float(cfg.hyperparameters.learning_rate.upper),
            log=cfg.hyperparameters.learning_rate.log,
        )
        weight_decay = CS.UniformFloatHyperparameter(
            "weight_decay",
            lower=float(cfg.hyperparameters.weight_decay.lower),
            upper=float(cfg.hyperparameters.weight_decay.upper),
            log=cfg.hyperparameters.weight_decay.log,
        )

        scheduler = CS.CategoricalHyperparameter(
            "scheduler", cfg.hyperparameters.scheduler.choices
        )
        start_factor = CS.Constant(
            "start_factor", float(cfg.hyperparameters.scheduler.start_factor)
        )
        end_factor = CS.Constant(
            "end_factor", float(cfg.hyperparameters.scheduler.end_factor)
        )
        total_iters = CS.Constant(
            "total_iters", int(cfg.hyperparameters.scheduler.total_iters)
        )
        T_zero = CS.Constant("T_0", int(cfg.hyperparameters.scheduler.T_0))
        T_mult = CS.Constant("T_mult", int(cfg.hyperparameters.scheduler.T_mult))
        eta_min = CS.Constant("eta_min", float(cfg.hyperparameters.scheduler.eta_min))
        epochs = CS.CategoricalHyperparameter(
            "epochs", cfg.hyperparameters.epochs.choices
        )
        use_gpu = CS.CategoricalHyperparameter(
            "use_gpu", [cfg.hyperparameters.use_gpu.enabled]
        )

        config_space.add(
            [
                batch_size,
                optimizer,
                learning_rate,
                weight_decay,
                scheduler,
                start_factor,
                end_factor,
                total_iters,
                T_zero,
                T_mult,
                eta_min,
                epochs,
                use_gpu,
            ]
        )
        print("Added hyperparameters to Configuration Space")

        return config_space

    except Exception as e:
        print(f"Error creating configuration space: {e}")
        raise

# Dummy class to parse and display configuration
class Dummy:
    def __init__(self, config):
        # Map the configuration from ConfigSpace
        self.dataset_name = config["dataset_name"]
        self.num_classes = config["num_classes"]
        self.train_dir = config["train_dir"]
        self.val_dir = config["val_dir"]
        self.model_type = config["model_type"]
        self.batch_size = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.scheduler = config["scheduler"]
        self.start_factor = config["start_factor"]
        self.end_factor = config["end_factor"]
        self.total_iters = config["total_iters"]
        self.T_0 = config["T_0"]
        self.T_mult = config["T_mult"]
        self.eta_min = config["eta_min"]
        self.epochs = config["epochs"]
        self.use_gpu = config["use_gpu"]

    def display(self):
        # Print each configuration on a new line
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Train Directory: {self.train_dir}")
        print(f"Validation Directory: {self.val_dir}")
        print(f"Model Type: {self.model_type}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Scheduler: {self.scheduler}")
        print(f"Start Factor (LinearLR): {self.start_factor}")
        print(f"End Factor (LinearLR): {self.end_factor}")
        print(f"Total Iterations (Warm-Up): {self.total_iters}")
        print(f"T_0 (CosineAnnealingWarmRestarts): {self.T_0}")
        print(f"T_mult (CosineAnnealingWarmRestarts): {self.T_mult}")
        print(f"Eta Min (CosineAnnealingWarmRestarts): {self.eta_min}")
        print(f"Epochs: {self.epochs}")
        print(f"Use GPU: {self.use_gpu}")


        

@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_narrow"
)  # Hydra Decorator
def main(cfg: DictConfig):
    """Main function to set up and execute the hyperparameter tuning."""
    #print("Loaded Configuration:")
    #print(cfg)

    # Create the ConfigSpace
    try:
        config_space = create_fine_tuning_config_space(cfg)
        #print("\n--- Mapped Configuration Space ---")
        #for hyperparameter in list(config_space.values()):
            #print(f"{hyperparameter.name}: {hyperparameter}")
    except Exception as e:
        print(f"Error in creating ConfigSpace: {e}")
        raise

    # Create and display Dummy instance
    dummy_instance = Dummy(config_space)
    dummy_instance.display()


if __name__ == "__main__":
    main()
