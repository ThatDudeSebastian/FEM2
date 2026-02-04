import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Parameters from HA4_main
n_cycles = 2.0
n_steps_per_cycle = 30
n_ramp_steps = 5
F_total = -150
use_ramp_in = True

def ramp(step):
    if not use_ramp_in or n_ramp_steps <= 0:
        return 1.0
    return min(1.0, step / n_ramp_steps)

def cyclic_factor(step):
    return math.sin(2.0 * math.pi * (step-1) / n_steps_per_cycle)

# Generate data
steps = np.arange(1, int(n_cycles * n_steps_per_cycle) + 1)
times = steps.astype(float) # In ANSYS we use Time
factors = [ramp(s) * cyclic_factor(s) for s in steps]
forces = [f * F_total for f in factors]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(steps, forces, 'b-o', markersize=4, label="Kraftverlauf")
plt.axhline(0, color='black', lw=1)
plt.xlabel("Step / Zeit")
plt.ylabel("Kraft [N]")
plt.title("Sinusförmige Lastkurve")
plt.grid(True)
plt.legend()

# Save plot
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "lastkurve.png"))
print(f"Plot gespeichert unter: {os.path.join(results_dir, 'lastkurve.png')}")

# Export for ANSYS (CSV)
# ANSYS Tabular Data: Discarding time as requested, one force per row
csv_path = os.path.join(results_dir, "lastkurve_ANSYS.csv")
with open(csv_path, "w") as f:
    # Start with 0.0 in scientific notation
    f.write(f"{0.0:.2e}".replace('.', ',') + "\n")
    for F in forces:
        f.write(f"{F:.2e}".replace('.', ',') + "\n")

print(f"Letzte Kraftwerte für ANSYS (nur Force) gespeichert unter: {csv_path}")

plt.show()
