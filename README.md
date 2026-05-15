# Multi-Determinant Neural Network Backflow

A JAX-based implementation of multi-determinant neural network wavefunctions with backflow transformations for quantum many-body systems.

## Overview

This project implements advanced neural network architectures for representing quantum states in many-body systems. It combines:

- **Multi-determinant Ansätze**: Flexible quantum state representations using determinantal forms
- **Backflow Transformations**: Pre-processing layers that enhance the expressiveness of neural network wavefunctions
- **JAX/Flax Framework**: High-performance automatic differentiation and GPU acceleration
- **Variational Monte Carlo**: Optimization using stochastic reconfiguration for ground state finding

## Features

- ✨ Efficient embedding modules for spin configurations
- 🔄 Static and dynamic kernel layers
- 📊 Variational Monte Carlo (VMC) with Stochastic Reconfiguration (SR)
- 🎯 Gradient normalization with clipping
- 🚀 GPU-accelerated training with JAX JIT compilation
- 📈 Training result tracking and analysis

## Project Structure

```
├── Network_lib.py           # Neural network components (embeddings, kernels, layers)
├── Optimizer_lib.py         # VMC and optimization routines
├── Run_training.ipynb       # Training script and examples
├── Training_results/        # Results from completed training runs
│   └── jupyter_hubbard_test/  # Hubbard model test case results
└── README.md
```

## Dependencies

- **JAX** - Automatic differentiation and numerical computing
- **Flax** - Neural network library built on JAX
- **NetKet** - Quantum many-body system toolkit
- **NumPy** - Numerical operations
- **Jupyter** - Interactive notebooks

## Installation

```bash
# Clone the repository
git clone https://github.com/sweetpotat5/Multi-determinant-nerual-network-backflow.git
cd Multi-determinant-nerual-network-backflow

# Install dependencies
pip install jax flax netket numpy jupyter
```

## Usage

### Running Training

Open and run `Run_training.ipynb` in Jupyter:

```bash
jupyter notebook Run_training.ipynb
```

### Core Components

#### Network Libraries

**Network_lib.py** provides:
- `EmbeddingModule`: Converts spin configurations into embedding vectors
- `StaticKernel`: Static convolutional kernels for feature extraction
- Support for patch-based processing of lattice systems

**Optimizer_lib.py** provides:
- `VMC_SR_norm_clip`: Variational Monte Carlo with gradient normalization
- Automatic differentiation utilities
- Loss computation and optimization

## Example: Training on Hubbard Model

The repository includes training results on the Hubbard model test case:

```
Training_results/jupyter_hubbard_test/
├── jupyter_hubbard_test.mpack        # Model checkpoint
├── jupyter_hubbard_test_job_summary.txt  # Training summary
└── jupyter_hubbard_test.log          # Training logs
```

## Key Algorithms

### Variational Monte Carlo with Stochastic Reconfiguration

The optimization combines:
1. **Sampling**: Monte Carlo sampling of configurations from the neural network ansatz
2. **Energy Estimation**: Computing ground state energy and gradients
3. **Parameter Updates**: Stochastic reconfiguration with adaptive gradient clipping

### Backflow Transformations

Backflow layers preprocess spin configurations to capture quantum correlations:
- Enhance expressiveness beyond simple product states
- Enable representation of complex entanglement patterns
- Improve convergence during variational optimization

## References

This implementation is based on advances in neural network quantum states and backflow-enhanced variational methods for quantum simulations.

## License

This project is available for educational and research purposes.

## Author

**sweetpotat5** - Quantum Computing & Machine Learning

## Contributing

Contributions and improvements are welcome! Please feel free to submit issues or pull requests.