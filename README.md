# Data Science Portfolio

This repository documents a transition from theoretical foundations to technical implementation in Data Science. The focus is on building from the ground up, balancing high-level Python prototyping with low-level C++ logic to ensure a deep understanding of algorithmic mechanics.

---

## Project Structure

### Fundamentals

#### Supervised Learning
* **Linear Regression:** Implementation of continuous variable prediction using Ordinary Least Squares and gradient descent. Includes a C++ version to demonstrate mathematical operations without library abstraction.
* **Logistic Regression:** Binary classification focusing on sigmoid activation, log-loss optimization, and decision boundary interpretation.
* **Decision Trees:** Exploration of non-linear classification logic, entropy-based splitting, and tree-based data structures.

#### Numerical Methods
* **Monte Carlo Integration:** A C++ implementation using repeated random sampling to solve high-dimensional integrals. This project demonstrates how the Law of Large Numbers provides a scalable alternative to grid-based integration for complex domains.

* **Ising Model Simulation:** A Python-based study using the **Metropolis-Hastings algorithm** to simulate 1D and 2D spin systems. The project focuses on the numerical emergence of criticality, benchmarking results against exact **Transfer Matrix** (1D) and **Onsager** (2D) solutions. Key features include **$\mathcal{O}(1)$ local updates**, Periodic Boundary Conditions, and a critical analysis of **finite-size scaling** artifacts.

---

## Tech Stack
* **Programming Languages:** Python, C++
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
* **Focus Areas:** Numerical Integration, Statistical Simulation, Model Building, Data Preprocessing

---

## Goals
1. **Algorithmic Transparency:** Moving beyond library calls by implementing core logic in compiled C++ to understand memory management and computational efficiency.
2. **Scientific Computing:** Applying numerical methods to handle uncertainty and solve problems where analytical solutions are difficult to obtain.
3. **Foundational Mastery:** Building a robust codebase that serves as a bridge between pure mathematics and applied machine learning.

---

Created by [Botond Koroknai](https://github.com/koroknaibotond)
