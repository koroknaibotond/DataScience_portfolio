# Monte Carlo Integration

[cite_start]This project implements a three-dimensional Monte Carlo integrator in **C++** to solve numerical integration problems involving high dimensionality or complex domains where analytical solutions are difficult or impossible to obtain[cite: 5, 8].

## The Core Problem

[cite_start]Traditional numerical integration evaluates functions on a fixed grid with fixed steps[cite: 40]. [cite_start]This becomes computationally expensive as the number of dimensions or the domain complexity increases—a challenge often called the "curse of dimensionality"[cite: 40]. [cite_start]Monte Carlo integration avoids this by randomly sampling points within a bounding cuboid and evaluating the integrand only at points inside the target domain[cite: 41].

## Implementation Details

[cite_start]The implementation estimates the integral of the function $f(x,y,z)=e^{-x^{2}-y^{2}-z^{2}}$ over a spherical domain with a radius of $4$[cite: 9, 50].

- [cite_start]**Algorithmic Logic:** The program generates independent random points and uses a domain lambda function to check if they fall within the integration region[cite: 44, 45, 68, 69].
- [cite_start]**Statistical Foundation:** Based on the law of large numbers, the estimated results converge toward the true expected value as the number of simulations increases[cite: 20].
- [cite_start]**Convergence:** The standard error of the estimator decreases proportionally to $1/\sqrt{N}$[cite: 7, 37]. [cite_start]This means reducing the error by a factor of ten requires approximately one hundred times more simulations[cite: 37].
- [cite_start]**C++ Features:** The project utilizes the **Mersenne Twister (mt19937)** engine for unbiased random sampling and **lambda expressions** to provide a flexible framework for any integrand or domain[cite: 10, 44, 46, 51, 68].

## Validation and Results

[cite_start]The Monte Carlo estimate ($\hat{I}_{MC}$) is validated against a deterministic **Riemann sum** ($I_{Riemann}$) that serves as a control value[cite: 11, 85].

- [cite_start]**Sample Size:** The simulation uses $500,000$ sample points, while the control Riemann sum employs $1,000,000$ evaluation points[cite: 92].
- [cite_start]**Efficiency:** The Monte Carlo estimator achieves reliable results with approximately half the number of function evaluations required by the deterministic method[cite: 12, 94].
- [cite_start]**Precision:** Implementation consistency is verified by ensuring the squared difference between the estimate and the control is below a tolerance of $\epsilon=1\times10^{-4}$[cite: 88, 91].

## Visualizing the Process

[cite_start]The project includes a visualization of the sampling process[cite: 13, 119]:

- [cite_start]**Green points:** Samples falling inside the spherical domain that contribute to the integral[cite: 110, 111].
- [cite_start]**Red points:** Rejected samples lying outside the target domain[cite: 110, 112].
