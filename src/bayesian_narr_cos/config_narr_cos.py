# config_narrow.py

from dataclasses import dataclass, field
from typing import List, Dict, Any
from hydra.core.config_store import ConfigStore

@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    train_dir: str
    val_dir: str

@dataclass
class ModelConfig:
    type: str
    in_features: int
    mode: str

@dataclass
class ExperimentConfig:
    dataset: str  # The key for the chosen dataset
    model: str  # The key for the chosen model

@dataclass
class OptimizerConfig:
    choices: List[str] = field(default_factory=lambda: ["AdamW"])

@dataclass
class CosineAnnealingLRConfig:
    choices: List[str] = field(default_factory=lambda: ["CosineAnnealingLR"])
    T_max: int = 60  # Duration in epochs before resetting the learning rate
    eta_min: float = 0.0001  # Minimum learning rate at the end of each cycle

@dataclass
class HyperparameterConfig:
    optimizer: OptimizerConfig
    learning_rate: Dict[str, Any]
    weight_decay: Dict[str, Any]
    scheduler: CosineAnnealingLRConfig
    epochs: List[int] = field(default_factory=lambda: [60])  # Total number of epochs
    use_gpu: Dict[str, bool] = field(default_factory=lambda: {"enabled": True})  # Assuming enabled by default
    batch_size: int = 1024

@dataclass
class RayConfig:
    num_cpus: int
    num_gpus: int
    include_dashboard: bool

@dataclass
class RunConfig:
    name: str
    storage_path: str
    checkpoint_frequency: int

@dataclass
class TunerConfig:
    max_concurrent: int
    num_samples: int
    resources_per_trial: Dict[str, int]

@dataclass
class BOConfig:
    datasets: Dict[str, DatasetConfig]
    models: Dict[str, ModelConfig]
    experiment: ExperimentConfig
    hyperparameters: HyperparameterConfig
    ray: RayConfig
    run_config: RunConfig
    tuner: TunerConfig

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="bo_cosLR_config", node=BOConfig)

# register_configs()
