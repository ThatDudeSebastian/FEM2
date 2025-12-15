import torch

# Threading and dtype
DEFAULT_DTYPE = torch.float64
TORCH_THREADS = 18
torch.set_default_dtype(DEFAULT_DTYPE)
torch.set_num_threads(TORCH_THREADS)

# Element Type Selection
USE_Q8 = False  # True for Q8 (8-node quadratic), False for Q4 (4-node linear)

# Parameters
E = 210e9  # E-Modul in Pa (210000 MPa)
NU = 0.3
RHO = 7850.0  # Dichte in kg/m^3
M = torch.rand(100)  # kg

TDM = 2
NDF = 2
NDM = 2
NEN = 8 if USE_Q8 else 4  # 8 nodes for Q8, 4 nodes for Q4
NQP = 9 if USE_Q8 else 4  # 3x3 Gauss quadrature for Q8, 2x2 for Q4

# Geometry in meters
LENGTH = 1.0  # m
HEIGHT = 0.1  # m
WIDTH = 0.1  # m

# Mesh density
NX = 20  # Elements in x-direction
NY = 2  # Elements in y-direction

# Loads / BCs
# total line load in N per meter width applied on right edge, positive downward
FORCE = -1000.0
TOTAL_FORCE = FORCE / WIDTH  # N/m on the boundary (consistent with original)

# Modal Analysis Parameters
ANZAHL_MODEN = 10  # Number of modes to display
F_THRESHOLD = 0.001  # Hz, threshold for zero frequencies
PLOT_MODE_INDEX = 4 # Which mode to plot (1-based index)

# Newmark Time Integration Parameters
RUN_TRANSIENT = True  # Set to True to run transient analysis
TIME_INTERVAL = 0.1  # Total time of simulation in seconds
TIME_STEPS = 100  # Number of time steps
NEWMARK_BETA = 0.25  # Newmark beta parameter
NEWMARK_GAMMA = 0.5  # Newmark gamma parameter

# Plotting
TOPLOT = True
DISP_SCALING = 1000  # Displacement scaling for visualization