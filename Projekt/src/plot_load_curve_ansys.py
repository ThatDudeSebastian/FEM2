import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Parameters from HA4_main
n_cycles = 2.0
n_steps_per_cycle = 30
n_ramp_steps = 5
F_total = -1500000.0
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
plt.plot(steps, forces, 'b-o', markersize=4, label="Kraftverlauf (F_max = -1.5 MN)")
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
# ANSYS Tabular Data format: Time, Force
csv_path = os.path.join(results_dir, "lastkurve.csv")
with open(csv_path, "w") as f:
    f.write("Time,Force\n")
    # Add point 0 to ensure start at zero
    f.write("0,0\n")
    for t, F in zip(times, forces):
        f.write(f"{t},{F}\n")

print(f"CSV für ANSYS Export gespeichert unter: {csv_path}")

# Export for ANSYS (XML)
xml_path = os.path.join(results_dir, "lastkurve.xml")
import xml.etree.ElementTree as ET
root = ET.Element("LoadData")
# Point zero
p0 = ET.SubElement(root, "Point")
ET.SubElement(p0, "Time").text = "0.0"
ET.SubElement(p0, "Force").text = "0.0"

for t, F in zip(times, forces):
    p = ET.SubElement(root, "Point")
    ET.SubElement(p, "Time").text = str(t)
    ET.SubElement(p, "Force").text = str(F)

tree = ET.ElementTree(root)
# For better readability, we can use a small hack for indenting
from xml.dom import minidom
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
with open(xml_path, "w") as f:
    f.write(xmlstr)

print(f"XML für ANSYS Export gespeichert unter: {xml_path}")
plt.show()
