from mpi4py import MPI
import numpy as np
import time
import sys

# ==========================================
# 1. PARAMETERS
# ==========================================
EMPTY, TREE, FIRE = 0, 1, 2
N = 1000       # Global grid dimension (N x N)
T = 500        # Number of time steps
p = 0.005      # Probability of tree growth
l = 0.0001     # Probability of lightning

# ==========================================
# 2. MPI INITIALIZATION & TOPOLOGY
# ==========================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a 2D Cartesian Topology
# MPI.Compute_dims automatically finds the best grid dimensions (e.g., 4 processes -> 2x2 grid)
dims = MPI.Compute_dims(size, 2)
cart_comm = comm.Create_cart(dims, periods=(False, False), reorder=True)
cart_rank = cart_comm.Get_rank()

# Determine neighbors for point-to-point communication (Halo Exchange)
# Shift returns (source, destination) for a given direction
north_src, north_dst = cart_comm.Shift(0, 1)  # Shift along rows (North-South)
west_src,  west_dst  = cart_comm.Shift(1, 1)  # Shift along cols (West-East)

# ==========================================
# 3. DOMAIN DECOMPOSITION
# ==========================================
# For simplicity, we assume N is perfectly divisible by the grid dimensions.
if N % dims[0] != 0 or N % dims[1] != 0:
    if rank == 0:
        print(f"Error: Global grid {N}x{N} cannot be evenly divided by processor grid {dims[0]}x{dims[1]}")
    sys.exit(1)

local_rows = N // dims[0]
local_cols = N // dims[1]

# Allocate local grid with +2 for the Ghost Cells boundary
local_grid = np.zeros((local_rows + 2, local_cols + 2), dtype=np.int8)

# Initialize inner grid randomly (each rank needs a different seed)
np.random.seed(42 + cart_rank)
inner_initial = np.random.choice([EMPTY, TREE], size=(local_rows, local_cols), p=[0.6, 0.4])
local_grid[1:-1, 1:-1] = inner_initial

# Synchronize before starting the timer
cart_comm.Barrier()
start_time = time.time()

# ==========================================
# 4. PARALLEL SIMULATION
# ==========================================
for step in range(T):
    # --- HALO EXCHANGE (GHOST CELLS) ---
    
    # 1. Exchange with North
    send_N = local_grid[1, :].copy()    # Top inner row
    recv_S = np.empty_like(send_N)
    cart_comm.Sendrecv(sendbuf=send_N, dest=north_dst, recvbuf=recv_S, source=north_src)
    if north_src != MPI.PROC_NULL: 
        local_grid[-1, :] = recv_S      # Put received data in bottom ghost row

    # 2. Exchange with South
    send_S = local_grid[-2, :].copy()   # Bottom inner row
    recv_N = np.empty_like(send_S)
    cart_comm.Sendrecv(sendbuf=send_S, dest=north_src, recvbuf=recv_N, source=north_dst)
    if north_dst != MPI.PROC_NULL: 
        local_grid[0, :] = recv_N       # Put received data in top ghost row

    # 3. Exchange with West
    send_W = local_grid[:, 1].copy()    # Left inner col
    recv_E = np.empty_like(send_W)
    cart_comm.Sendrecv(sendbuf=send_W, dest=west_dst, recvbuf=recv_E, source=west_src)
    if west_src != MPI.PROC_NULL: 
        local_grid[:, -1] = recv_E      # Put received data in right ghost col

    # 4. Exchange with East
    send_E = local_grid[:, -2].copy()   # Right inner col
    recv_W = np.empty_like(send_E)
    cart_comm.Sendrecv(sendbuf=send_E, dest=west_src, recvbuf=recv_W, source=west_dst)
    if west_dst != MPI.PROC_NULL: 
        local_grid[:, 0] = recv_W       # Put received data in left ghost col


    # --- PARALLEL COMPUTATION  ---
    new_grid = local_grid.copy()
    
    is_fire = (local_grid == FIRE)
    is_tree = (local_grid == TREE)
    is_empty = (local_grid == EMPTY)
    
    # Evaluate neighbors on the entire grid (including ghosts)
    fire_N = is_fire[:-2, 1:-1]
    fire_S = is_fire[2:, 1:-1]
    fire_E = is_fire[1:-1, 2:]
    fire_W = is_fire[1:-1, :-2]
    
    neighbor_on_fire = fire_N | fire_S | fire_E | fire_W
    
    # Extract only the inner portion for applying rules
    inner_is_tree = is_tree[1:-1, 1:-1]
    inner_is_empty = is_empty[1:-1, 1:-1]
    inner_is_fire = is_fire[1:-1, 1:-1]
    
    rand_matrix = np.random.rand(local_rows, local_cols)
    
    # Apply rules (modifying only the inner part of new_grid)
    new_inner = new_grid[1:-1, 1:-1]
    
    new_inner[inner_is_fire] = EMPTY
    
    catch_fire = inner_is_tree & (neighbor_on_fire | (rand_matrix < l))
    new_inner[catch_fire] = FIRE
    
    grow_tree = inner_is_empty & (rand_matrix < p)
    new_inner[grow_tree] = TREE
    
    # Commit changes
    local_grid = new_grid

# ==========================================
# 5. PERFORMANCE METRICS
# ==========================================
cart_comm.Barrier()
end_time = time.time()

if rank == 0:
    print("-" * 40)
    print(f"Parallel Simulation completed on {size} MPI processes.")
    print(f"Processor Grid Topology: {dims[0]}x{dims[1]}")
    print(f"Global Grid: {N}x{N}, Steps: {T}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("-" * 40)