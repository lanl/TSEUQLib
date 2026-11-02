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
| `remainder`          | Error of $\xi\left[T_{n}(\boldsymbol{x})\right]$              | $\varepsilon\left[\xi\left[T_{n}(\boldsymbol{x})\right]\right]$ |
| —                    | Sensitivities, $\frac{\partial \xi[T_n(\boldsymbol{x})]}{\partial \theta_i}$ | $S^{\xi}_{\theta_i}$ |

**Many inputs at low order.** `tse_moments.tse_central_moment` computes the same quantities from closed-form equations (extending those of Balcer et al., *J. Verif. Valid. Uncert. Quantif.* 8(2), 2023, doi:10.1115/1.4062459). It covers the expected value up to fifth order, the variance with partial variances and Sobol' indices up to fourth order (at fourth order, the variance only), and the third and fourth central moments up to third order. Inputs must be independent, and it evaluates many expansion points at once. It has no limit on the number of inputs, whereas pyoti caps `oti_moments` (the third and fourth central moments of a second-order expansion take at most 10 inputs), but it computes no parameter sensitivities.


# Installation
TSEUQLib runs on pyoti, from OTILib (https://github.com/mauriaristi/otilib). pyoti is published on the `mauriaristi` conda channel for Linux and macOS; there is no Windows build, so on Windows use WSL. Create a conda environment with pyoti and the other dependencies, and activate it:
```bash
conda env create -f environment.yml
conda activate tseuqlib
```
pyoti 1.2.1 requires Python 3.13. To build pyoti from source instead, follow the OTILib instructions.

Run the tests with:
```bash
python -m pytest tests
```
Add `-n auto` to spread them over all cores, as CI does.

# Usage
Run the examples in examples/ with python. For example, run the ishigami example by:
```bash
cd examples/
python ishigami_paper.py
```


# Copyright
LANL **O5031**

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
