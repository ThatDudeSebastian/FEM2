# FEM Code Documentation - Assignment 3 (Tasks 4-9)

This document describes the implementation details, modifications, and theoretical background of the Python FEM solver in `Afg3_solution.py`. This code extends a basic residual-based Newton-Raphson solver (Project 3 base) to handle advanced non-linear material models, dynamics, and structural sizing.

## 1. Overview of Changes

The original `Afg3_residual.py` provided a static solver for linear elastic trusses using a co-rotational formulation. The new `Afg3_solution.py` introduces:
*   **Object-Oriented Material System**: To switch easily between Hooke, Plasticity, and Viscoelasticity.
*   **Dynamic Solver**: Newmark-Beta time integration for transient analysis.
*   **Automated Sizing**: Iterative loops to dimension the structure.
*   **Enhanced Stabilization**: Regularization and Rayleigh damping to handle kinematic mechanisms (pendulum modes) and rigid body motion.

## 2. Implementation Details by Task

### Task 4: Structural Sizing (Hooke's Law)
*   **Goal**: Find cross-sectional Area $A$ such that $\sigma_{max} \approx 1000$ MPa.
*   **Implementation**: 
    *   Uses `Hooke` material class ($E=210$ GPa).
    *   Runs a static analysis with linear ramp loading.
    *   Calculates `required_area = current_area * (max_stress / target_stress)`.
    *   Re-runs verification with new area.
*   **Location**: `run_all_tasks()` -> `--- Running Task 4 ---` block.

### Task 5: Ideal Plasticity
*   **Goal**: Limit stress to yield strength $S_y = 960$ MPa.
*   **Material Model**: `PerfectPlasticity` class.
    *   **Algorithm**: 1D Return Mapping.
    *   Trial stress $\sigma_{trial} = E (\epsilon - \epsilon_p^{n})$.
    *   Yield Function $\phi = |\sigma_{trial}| - S_y$.
    *   If $\phi > 0$: Stress returned to $S_y \cdot \text{sign}(\sigma)$, plastic strain $\epsilon_p$ updated.
    *   Tangent modulus $E_t \approx 0$ (set to small value `1e-5 * E` for numerical stability).

### Task 6: Linear Hardening
*   **Goal**: Allow stress increase beyond yield with modulus $H$.
*   **Material Model**: `LinearHardening` class.
    *   Flow stress $\sigma_y(\alpha) = S_y + H \cdot \alpha$ (isotropic hardening).
    *   Consistent Tangent: $E_t = \frac{EH}{E+H}$.
    *   Allowable stress increases with accumulated plastic strain $\alpha$.

### Task 7: Viscoelasticity (Superglue)
*   **Goal**: Model crane with soft, viscous material.
*   **Material Model**: `ViscoElastic` class.
    *   Model: Kelvin-Voigt-like formulation.
    *   $\sigma = E\epsilon + \eta \dot{\epsilon} \approx E\epsilon + \eta \frac{\Delta \epsilon}{\Delta t}$.
    *   Tangent Modulus: $E_t = E + \frac{\eta}{\Delta t}$.
    *   Results in significantly larger displacements due to lower Stiffness ($E_{glue} \approx 600$ MPa vs $E_{steel} \approx 210000$ MPa).

### Task 8: Dynamics with Damping
*   **Goal**: Transient response under sudden load step, with semi-free bearings.
*   **Solver**: `FEMSolver.solve(..., dynamic=True)`
    *   **Time Integration**: Newmark-Beta method ($\beta=0.25, \gamma=0.5$).
    *   **Inertia**: Consistent Mass Matrix `M` (lumped mass approx).
    *   **Damping**: 
        1.  **Rayleigh Damping**: $C = \alpha M + \beta K$. (`self.rayleigh_alpha=0.5`, `self.rayleigh_beta=0.001`). Stabilizes the structure globally.
        2.  **Bearing Damping/Stiffness**: Nodes 0 and 1 are released from Dirichlet BCs. Discrete springs (`self.bearing_k = 1e9`) and dampers (`self.damping_c = 1e4`) are added to the diagonal of `K` and `C` matrices to model the "viscoelastic dampers".
    *   **Note on Stability**: A fine timestep ($dt=0.002s$) is used to capture the high-frequency response of the stiff bearing springs.

### Task 9: Material Comparison
*   **Goal**: Hysteresis loops under cyclic loading.
*   **Implementation**: 
    *   Runs all 3 steel models (Hooke, PerfectPlasticity, LinearHardening) with a cyclic load curve (0 -> 1 -> 0 -> -1 -> 0).
    *   Extracts stress/strain from the most loaded element.
    *   Plots results for comparison.

## 3. Key Solver Features

### Co-Rotational Formulation
The code retains the large-deformation truss formulation. Strain is calculated as Engineering Strain $\epsilon = (L - L_0) / L_0$. The global stiffness matrix $K$ is assembled including both Material Stiffness ($K_m$) and Geometric Stiffness ($K_g$ - stress stiffening).

### Regularization
To prevent "Singular Matrix" errors (esp. for the pendulum-like hanging bars 15/16 at the start of simulation), a small regularization term is added to the diagonal of the stiffness matrix: `K_solve.diagonal().add_(1.0)`. This acts as a very weak "ghost spring" (1 N/m) that stabilizes rigid body modes without affecting the physics of the steel structure.

### Visualization
*   **Real-time**: `matplotlib` is used in interactive mode (`plt.ion()`) to show the deformation and stress distribution (blue-red color map) during the calculation.
*   **Colorbar**: A dynamic colorbar shows the mapping of stress values to colors in MPa.

## 4. How to Run
Execute the script in Python environment with `torch` and `matplotlib` installed:
```bash
python Afg3_solution.py
```
*   **Interactivity**: The script pauses after plots. **Close the plot window** to proceed to the next task.
*   **Output**: Console prints max stress per step. Plots are saved as PNG files (`Task4_Stress.png`, etc.).
