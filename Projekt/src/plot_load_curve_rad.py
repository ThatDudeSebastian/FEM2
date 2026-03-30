import numpy as np
import matplotlib.pyplot as plt
import math
import os

# ============ CONFIGURATION ============
# Parameters matching Diskretisierer_Rad.py
F_total = -5.3e4     # Normalkraft [N]
n_cycles = 1
n_steps_per_cycle = 120
n_ramp_steps = 5
use_ramp_in = True

n_steps = int(n_cycles * n_steps_per_cycle)

# ============ CALCULATION ============
steps = np.arange(1, n_steps + 1)
forces = []

for step in steps:
    # Ramp-in factor
    ramp_fac = min(1.0, step / n_ramp_steps) if use_ramp_in else 1.0
    
    # Cyclic factor (Pulsating / Schwellend -> abs(sin))
    # Note: step-1 because simulation loop starts at step 1 but sin usually starts at 0
    # Diskretisierer logic: math.sin(2*math.pi*(step-1)/n_steps_per_cycle)
    cyclic_fac = abs(math.sin(2 * math.pi * (step - 1) / n_steps_per_cycle))
    
    # Total factor
    fac = ramp_fac * cyclic_fac
    
    # Force value (in kN for plotting)
    # F_total is negative (-53 kN). Prescribed force is fac * F_total.
    # User wanted "Schwellende Kraft", so we plot magnitude or values?
    # Usually F-t diagram shows actual values.
    # Dividing by 1000 for kN.
    force_kn = (fac * F_total) / 1000.0
    forces.append(force_kn)

# ============ PLOTTING ============
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "load_curve_prescribed.png")

plt.figure(figsize=(10, 5))
plt.plot(steps, forces, 'b-o', markersize=6, linewidth=2, label="Vorgegebene Last (Soll)")

plt.xlabel("Step")
plt.ylabel("Kraft [kN]")
plt.title(f"Vorgegebener Lastverlauf (Schwellend)")
plt.grid(True, alpha=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.legend()

plt.tight_layout()
try:
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.show()
