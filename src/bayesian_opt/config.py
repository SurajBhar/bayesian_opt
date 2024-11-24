"""
This file is where you define your configuration classes and 
register them using Hydra's ConfigStore
"""

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
    choices: List[str]

@dataclass
class HyperparameterConfig:
    optimizer: OptimizerConfig
    learning_rate: Dict[str, Any]
    weight_decay: Dict[str, Any]
    momentum: Dict[str, Any]
    scheduler: Dict[str, List[str]]
    step_size: Dict[str, List[int]]
    gamma: Dict[str, float]
    T_max: Dict[str, List[int]]
    eta_min: float
    start_lr: Any
    end_lr: float
    epochs: Dict[str, List[int]]
    use_gpu: Dict[str, bool]
    batch_size: int

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
    datasets: Dict[str, DatasetConfig]  # Dictionary of all datasets
    models: Dict[str, ModelConfig]  # Dictionary of all models
    experiment: ExperimentConfig  # The chosen dataset and model
    hyperparameters: HyperparameterConfig
    ray: RayConfig
    run_config: RunConfig
    tuner: TunerConfig

def register_configs():
    cs = ConfigStore.instance()
    # Use a unique name for the schema to avoid deprecation warnings
    cs.store(name="bo_exp_config", node=BOConfig)

register_configs()
