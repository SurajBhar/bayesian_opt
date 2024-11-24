# Bayesian Optimization for Hyperparameter Tuning

This repository implements a Bayesian Optimization workflow for hyperparameter tuning using the Ray Tune framework and ConfigSpace for configuration management. The implementation supports advanced scheduling and search algorithms, including the **BOHB (Bayesian Optimization with HyperBand)** scheduler.

![Parallel Coordinate View](/Parallel_coordinate_view.PNG)
![Mean Accuracy Plot](/mean_acc_plot.PNG)

---

## Table of Contents

- [Features](#features)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [License](#license)

---

## Features

1. **Bayesian Optimization with BOHB**:
   - Efficient hyperparameter search.
   - Dynamic resource allocation for trials.

2. **Configurable Search Space**:
   - Supports a wide range of hyperparameter configurations.
   - Easily extendable using Hydra and ConfigSpace.

3. **Metrics Monitoring**:
   - Integration with **TensorBoard** and **Weights & Biases (W&B)** for real-time performance tracking.
   - AUROC, confusion matrix, precision, recall, F1-score, and balanced accuracy evaluation.

4. **Supports Imbalanced Datasets**:
   - Metrics tailored for imbalanced datasets to ensure robust evaluation.

5. **Parallelization**:
   - Distributed execution across multiple GPUs and CPUs.

---

## Configuration

The project uses Hydra and ConfigSpace for managing configurations. The main configuration files are:

1. **`config_narrow.yaml`**:
   - Defines datasets, models, hyperparameters, and runtime configurations.

2. **`config_narrow.py`**:
   - Python-based configuration schema using `dataclasses` from Hydra.

---

## Repository Structure
```bash
├── conf/
│   ├── config_narrow.yaml      # Configuration file 2 for the experiment
│   ├── config.yaml        # Configuration file 1
├── ray_job_submission_jarvis/
│   ├── cancel_ray_job.py      # Cancel an ongoing job on Ray Cluster
│   ├── job_config.yaml        # Configuration file for job scheduling
│   ├── submit_ray_job.py        # Submit a job to Ray Cluster
├── src/
│   ├── bayesian_narrow/
│   │   ├── bohb_narrowed.py    # Main script for BOHB
│   │   ├── config_narrow.py             # Hydra Configuration definitions
│   ├── utils/
│   │   ├── utils.py            # Utility functions
│   │   ├── metrics.py            # Different metrics one can utilise for tracking and validation
├── outputs/                       # Hydra Configuration outputs (Will be created upon execution)
├── README.md                   # Project documentation
└── environment.yml            # Dependencies to create conda environment
```

---

## Installation

To reproduce the experiments or run any part of this codebase, follow the steps below:

### Clone the repository:
```bash
git clone https://github.com/SurajBhar/bayesian_opt.git
cd bayesian_opt
```
### Install Dependencies:
Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
conda activate deepl
```

For DINOv2 specific task, you can create the conda environment using seperate conda yaml file from DINOv2 github repository and can install ray specific dependencies on top of it. 
Alternative: You can also use environment_2.yaml file to create a conda environment:


- **An example: For DINOv2:**
  ```bash
  conda env create -f environment_2.yaml
  conda activate dinov2_ray
  ```
---

## How to run:
Create the ray cluster and use submit_ray_job.py file to schedule a job:

```bash
ray start --head --node-ip-address=IP-Address_here --port=6379 --dashboard-host=0.0.0.0
python path/to/ray_job_submission_jarvis/submit_ray_job.py
```
---

## Dependencies

Key dependencies for this project include:

- Python >3.9
- PyTorch (for Vision Transformer models)
- torchvision
- scikit-learn
- ray
- Jupyter Notebook (for data preprocessing)
- Additional dependencies listed in `environment.yaml`.

---
## Contact

If you have any questions, suggestions, or issues, feel free to reach out:

- **Name**: Suraj Bhardwaj
- **Email**: suraj.unisiegen@gmail.com
- **LinkedIn**: [Suraj Bhardwaj](https://www.linkedin.com/in/bhardwaj-suraj)

For project-related inquiries, please use the email mentioned above.

---

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT) - feel free to modify and distribute the code with proper attribution.
