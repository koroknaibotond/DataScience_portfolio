# Monte Carlo Integration

This project implements a three-dimensional Monte Carlo integrator in **C++** to solve numerical integration problems involving high dimensionality or complex domains where analytical solutions are difficult or impossible to obtain.

## The Core Problem

Traditional numerical integration evaluates functions on a fixed grid with fixed steps. This becomes computationally expensive as the number of dimensions or the domain complexity increases—a challenge often called the "curse of dimensionality". Monte Carlo integration avoids this by randomly sampling points within a bounding cuboid and evaluating the integrand only at points inside the target domain.

## Implementation Details

The implementation estimates the integral of the function $f(x,y,z)=e^{-x^{2}-y^{2}-z^{2}}$ over a spherical domain with a radius of $4$.

- **Algorithmic Logic:** The program generates independent random points and uses a domain lambda function to check if they fall within the integration region.
- **Statistical Foundation:** Based on the law of large numbers, the estimated results converge toward the true expected value as the number of simulations increases.
- **Convergence:** The standard error of the estimator decreases proportionally to $1/\sqrt{N}$. This means reducing the error by a factor of ten requires approximately one hundred times more simulations.
- **C++ Features:** The project utilizes the **Mersenne Twister (mt19937)** engine for unbiased random sampling and **lambda expressions** to provide a flexible framework for any integrand or domain.

## Validation and Results

The Monte Carlo estimate is validated against a deterministic **Riemann sum** that serves as a control value.

- **Sample Size:** The simulation uses $500,000$ sample points, while the control Riemann sum employs $1,000,000$ evaluation points.
- **Efficiency:** The Monte Carlo estimator achieves reliable results with approximately half the number of function evaluations required by the deterministic method.
- **Precision:** Implementation consistency is verified by ensuring the squared difference between the estimate and the control is below a tolerance of $\epsilon=1\times10^{-4}$.

