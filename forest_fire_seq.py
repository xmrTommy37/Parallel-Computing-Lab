import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================
# 1. PARAMETERS & INITIALIZATION
# ==========================================
EMPTY, TREE, FIRE = 0, 1, 2
N = 250        # Grid dimension (N x N)
T = 500        # Number of time steps
p = 0.005      # Probability of tree growth
l = 0.0001     # Probability of lightning

# Colormap for visualization: EMPTY=Black, TREE=Green, FIRE=Red
cmap = ListedColormap(['black', 'forestgreen', 'red'])

# Initialize the grid randomly with EMPTY (60%) and TREE (40%)
grid = np.random.choice([EMPTY, TREE], size=(N, N), p=[0.6, 0.4])

# Toggle this to True if you want to see the animation 
# (Warning: Visualization slows down the execution time significantly!)
VISUALIZE = True

if VISUALIZE:
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
    ax.set_title("Sequential Forest Fire")
    plt.axis('off')

# ==========================================
# 2. SEQUENTIAL SIMULATION (Algorithm 1)
# ==========================================
print(f"Starting sequential simulation on a {N}x{N} grid for {T} steps...")
start_time = time.time()

for step in range(T):
    # Create the auxiliary grid (Double buffering to avoid race conditions)
    new_grid = grid.copy()
    
    # Identify the current state of each cell
    is_fire = (grid == FIRE)
    is_tree = (grid == TREE)
    is_empty = (grid == EMPTY)
    
    # Find neighbors on fire (Von Neumann neighborhood)
    # We pad the grid with 'False' to handle the boundaries without wrapping (no periodic boundaries)
    padded_fire = np.pad(is_fire, pad_width=1, mode='constant', constant_values=False)
    
    fire_N = padded_fire[:-2, 1:-1]
    fire_S = padded_fire[2:, 1:-1]
    fire_E = padded_fire[1:-1, 2:]
    fire_W = padded_fire[1:-1, :-2]
    
    # Boolean mask: True if AT LEAST ONE orthogonal neighbor is on fire
    neighbor_on_fire = fire_N | fire_S | fire_E | fire_W
    
    # Generate a single random matrix for all probability evaluations
    rand_matrix = np.random.rand(N, N)
    
    # --- Apply Transition Rules ---
    
    # Rule 1: Fire turns to Empty
    new_grid[is_fire] = EMPTY
    
    # Rule 2: Trees catch fire if a neighbor is burning, OR by random lightning
    catch_fire = is_tree & (neighbor_on_fire | (rand_matrix < l))
    new_grid[catch_fire] = FIRE
    
    # Rule 3: Empty cells grow into Trees based on probability p
    grow_tree = is_empty & (rand_matrix < p)
    new_grid[grow_tree] = TREE
    
    # Commit the changes for the next step
    grid = new_grid
    
    # Render the frame if visualization is enabled
    if VISUALIZE:
        img.set_data(grid)
        plt.pause(0.01)

# ==========================================
# 3. PERFORMANCE METRICS
# ==========================================
end_time = time.time()
execution_time = end_time - start_time

if VISUALIZE:
    plt.ioff()
    plt.show()

print("-" * 40)
print(f"Simulation completed.")
print(f"Total Execution Time: {execution_time:.4f} seconds")
print("-" * 40)