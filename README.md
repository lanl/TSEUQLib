# TSEUQLib: Taylor Series Expansion Uncertainty Quantification Library
Construct Taylor series expansions for surrogate modelling. Compute UQ and sensitivity analysis metrics of the Taylor series expansion.

# Features

| Function              | Description                                                     | Symbol |
|----------------------|-----------------------------------------------------------------|--------|
| `tse`                | *n*-th-order Taylor Series                                      | $T_{n}(\boldsymbol{x})$ |
| `expectation`        | Expected Value                                                  | $E\left[T_{n}(\boldsymbol{x})\right]$ |
| `central_moment`     | Central Moment                                                  | $\mu_{k}\left[T_{n}(\boldsymbol{x})\right]$ |
| `sobol_indices`      | Partial Variances                                               | $V_{ij\ldots}\left[T_{n}(\boldsymbol{x})\right]$ |
| `sobol_indices`      | Sobol’ Indices                                                  | $S_{ij\ldots}\left[T_{n}(\boldsymbol{x})\right]$ |
| `total_sobol_indices`| Total Sobol’ Indices                                            | $S_{i}^{T}\left[T_{n}(\boldsymbol{x})\right]$ |
| `shapley_values`     | Shapley effects                                                 | $\phi_{i}\left[T_{n}(\boldsymbol{x})\right]$ |
| `remainder`          | Error of $\xi\!\left[T_{n}(\boldsymbol{x})\right]$              | $\varepsilon\!\left[\xi\!\left[T_{n}(\boldsymbol{x})\right]\right]$ |
| —                    | Sensitivities, $\frac{\partial \xi[T_n(\boldsymbol{x})]}{\partial \theta_i}$ | $S^{\xi}_{\theta_i}$ |


# Prerequisites
Install OTILib from https://github.com/mauriaristi/otilib. This will create a new conda environment with OTILib installed.

Activate the otilib conda environment, for example:
```bash
conda activate pyoti
```

Add the TSEUQLib dependencies to the otilib conda environment with:
```bash
conda env update -f environment.yml
```

# Usage
Run the examples in examples/ with python. For example, :
```bash
cd TSEUQLib/examples/
python ishigami.py
```


# Copyright
LANL **O5031**

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
