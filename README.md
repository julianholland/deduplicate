<div align="center">
  <h1><code>deduplicate_lib</code></h1>
  <p><i>deduplication algorithms in python</i></p>
</div>

***
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/julianholland/deduplicate)
[![codecov](https://codecov.io/gh/julianholland/deduplicate/graph/badge.svg?token=JL3OTRCXZD)](https://codecov.io/gh/julianholland/deduplicate)
[![PyPI version](https://badge.fury.io/py/deduplicate_lib.svg)](https://badge.fury.io/py/deduplicate_lib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/julianholland/deduplicate/actions/workflows/ci.yml/badge.svg)](https://github.com/julianholland/deduplicate/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/deduplicate-lib/badge/?version=latest)](https://deduplicate-lib.readthedocs.io/en/latest/)

## Key Features


- Easy to use deduplication algorithms for any vector array
- Suite of tolerance tuning algorithms to help you find the right tolerance value for your system
- Suite of benchmarking tools to ensure rigor, accuracy, and speed (not yet implemented)
- Factory Plugin architecture, for easy extensibility and modification

***
## Implemented Algorithms

- Distance Matrix (Simple, accurate, expensive): Computes the distance matrix for all vectors and determines duplicates by finding those that fall below a given distance
- Multi Hashing (Fast): Smears and rounds the vectors using a normal distribution and computes the hashes for each which are then used to determine duplicates by proportion of hash clashes.
<!-- - Locality Sensitive Hashing (Fast, Accurate) -->

## Quick Start

```bash
pip install deduplicate_lib
```

```python
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import MultiHashing
import numpy as np

dataset = np.array([[1.0, 2.0], [1.01, 2.01], [5.0, 6.0]])
dda = MultiHashing(tolerance=0.1, dataset_array=dataset)

# return unique vectors
print(dda.deduplicate())

# check a single vector against the dataset
dda.input_vector = np.array([1.0, 2.0])
print(dda.duplicate_check())  # True
```

See the **[full documentation](https://deduplicate-lib.readthedocs.io)** for API reference, tolerance tuning, and more examples.

### Dependencies

- Python 3.9+
- `numpy`
- `numba`
- `scipy`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/julianholland/deduplicate.git
cd deduplicate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
ruff format .
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/core/
pytest tests/plugins/
pytest tests/plugins/duplicate_detection_algorithms/distance_matrix

# Run with coverage
pytest --cov
```

## 📝 Citation

If you use deduplicate_lib in your research, please cite:

```bibtex
@software{deduplicate2026,
  title={deduplicate_lib: Auto Tolerance Finding Deduplication Algorithms in Python},
  author={Julian Holland},
  year={2026},
  url={https://github.com/julianholland/deduplicate},
  version={0.0.5}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Fritz Haber Institute
- Juan Manuel Lombardi <3
- Maximillion Ach
- Chiara Panosetti


## Project Links

- [GitHub Repository](https://github.com/julianholland/deduplicate)
- [Documentation](https://deduplicate-lib.readthedocs.io)

## Project To-Do

- [x] Add example.ipynb
- [x] Create general Pre-allocation protocal
- [ ] Add benchmarks for time and robustness
- [ ] Add Locality-Sensitive Hashing as an option
- [x] Speedup slow tasks with Numba
- [x] Set up Read the Docs
- [x] Create general deduplicate function
- [x] Speed up NTPP