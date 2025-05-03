# ScoreOperator

Repository for the paper ["Infinite-dimensional diffusion bridge simulation via operator learning"](https://arxiv.org/abs/2405.18353).

## Overview

ScoreOperator is a framework for simulating infinite-dimensional diffusion bridges using operator learning. This approach combines continuous-time neural operators with score-based diffusion models for efficient simulation of stochastic processes.

## Installation

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bookdiver/scoreoperator.git
   cd scoreoperator
   ```

2. Create and activate a conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate scoreoperator
   ```

3. Install the package in development mode with appropriate JAX version:
   ```bash
   # For CPU only
   pip install -e ".[cpu]"
   
   # For GPU support (CUDA 12)
   pip install -e ".[gpu]"
   ```

## Running Experiments

The main script `main.py` is used to run experiments. The script accepts several command-line arguments:

```bash
python main.py --experiment <experiment_name> [--resume-step <step>] [--train-x-shape <shape>]
```

### Arguments

- `--experiment`: (Required) Name of the experiment configuration to use. Options include:
  - `toy_brownian`
  - `ellipse_cylindrical_brownian`
  - `ellipsoid_brownian`
  - `butterfly_kunita`

- `--resume-step`: (Optional) Resume training from a specific checkpoint step.
- `--train-x-shape`: (Optional) Override the shape of the training data specified in the config.

### Examples

Start training a new model with the toy Brownian experiment configuration:
```bash
python main.py --experiment toy_brownian
```

Resume training from checkpoint step 5000:
```bash
python main.py --experiment ellipse_cylindrical_brownian --resume-step 5000
```

## Project Structure

- `scoreoperator/`: Main package directory
  - `diffusion/`: Diffusion models and SDE implementations
  - `neuralop/`: Neural operator models
  - `utils/`: Utility functions, configuration, and training code
- `notebooks/`: Jupyter notebooks for experiments and visualization
- `ckpts/`: Model checkpoints

## Citation

If you use this code for your research, please cite our paper:

@inproceedings{
    yang2025infinitedimensional,
    title={Infinite-dimensional Diffusion Bridge Simulation via Operator Learning},
    author={Gefan Yang and Elizabeth Louise Baker and Michael Lind Severinsen and Christy Anna Hipsley and Stefan Sommer},
    booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
    year={2025},
    url={https://openreview.net/forum?id=RZryinonfr}
}

