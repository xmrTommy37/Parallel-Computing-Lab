from mpi4py import MPI
import numpy as np
import os
import csv
import argparse

# ==========================================
# 1. PARAMETERS & COMMAND-LINE INTERFACE
# ==========================================
EMPTY, TREE, FIRE = 0, 1, 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel (MPI) Forest Fire cellular-automaton simulation."
    )
    parser.add_argument("-N", "--size", type=int, default=1000,
                        help="Global grid dimension (N x N). Default: 1000")
    parser.add_argument("-T", "--steps", type=int, default=500,
                        help="Number of time steps. Default: 500")
    parser.add_argument("-p", "--growth", type=float, default=0.005,
                        help="Probability of tree growth in an empty cell. Default: 0.005")
    parser.add_argument("-l", "--lightning", type=float, default=0.0001,
                        help="Probability of a tree igniting by lightning. Default: 0.0001")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (offset per rank). Default: 42")
    parser.add_argument("--decomp", choices=["1d", "2d"], default="2d",
                        help="Domain decomposition: '1d' (row slabs) or '2d' (blocks). Default: 2d")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a CSV file to append the timing result to (rank 0 only).")
    return parser.parse_args()


def append_csv(path, row):
    """Append one result row to a CSV file, writing the header if the file is new."""
    header = ["implementation", "N", "T", "p", "l", "procs", "decomp",
              "dims", "seed", "total_time", "compute_time", "comm_time"]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    N, T, p, l = args.size, args.steps, args.growth, args.lightning

    # ==========================================
    # 2. MPI INITIALIZATION & TOPOLOGY
    # ==========================================
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Choose the processor-grid dimensions.
    #   2d -> MPI picks a balanced factorization (e.g. 4 -> 2x2, 8 -> 4x2)
    #   1d -> a single column of "row slab" domains (size x 1)
    if args.decomp == "1d":
        dims = [size, 1]
    else:
        dims = MPI.Compute_dims(size, 2)

    cart_comm = comm.Create_cart(dims, periods=(False, False), reorder=True)
    cart_rank = cart_comm.Get_rank()

    # Determine neighbors for point-to-point communication (halo exchange).
    # Shift(dim, +1) returns (source, dest): dest = coord+1, source = coord-1.
    row_src, row_dst = cart_comm.Shift(0, 1)  # along rows  (vertical: up/down)
    col_src, col_dst = cart_comm.Shift(1, 1)  # along cols  (horizontal: left/right)

    # ==========================================
    # 3. DOMAIN DECOMPOSITION
    # ==========================================
    # For simplicity, we assume N is perfectly divisible by the grid dimensions.
    # The check is deterministic and identical on every rank, so all ranks return
    # together (clean exit, code 0) and the configuration is simply skipped.
    if N % dims[0] != 0 or N % dims[1] != 0:
        if rank == 0:
            print(f"Skipping: global grid {N}x{N} cannot be evenly divided by "
                  f"processor grid {dims[0]}x{dims[1]}")
        return

    local_rows = N // dims[0]
    local_cols = N // dims[1]

    # Allocate local grid with +2 for the ghost-cell boundary
    local_grid = np.zeros((local_rows + 2, local_cols + 2), dtype=np.int8)

    # Initialize inner grid randomly (each rank needs a different seed)
    np.random.seed(args.seed + cart_rank)
    inner_initial = np.random.choice([EMPTY, TREE], size=(local_rows, local_cols), p=[0.6, 0.4])
    local_grid[1:-1, 1:-1] = inner_initial

    # Reusable receive buffers for the halo exchange
    recv_row = np.empty(local_cols + 2, dtype=np.int8)
    recv_col = np.empty(local_rows + 2, dtype=np.int8)

    # Timers for the computation / communication breakdown
    comm_time = 0.0
    compute_time = 0.0

    # Synchronize before starting the timer
    cart_comm.Barrier()
    start_time = MPI.Wtime()

    # ==========================================
    # 4. PARALLEL SIMULATION
    # ==========================================
    for step in range(T):
        # --- HALO EXCHANGE (GHOST CELLS) ---
        # A cell's ghost layer must hold the *physically adjacent* border of the
        # neighboring domain. dest = coord+1 (down/right), source = coord-1 (up/left).
        t0 = MPI.Wtime()

        # Vertical exchange (rows)
        # 1. Send bottom inner row DOWN (to row_dst); receive TOP ghost from row_src (above)
        cart_comm.Sendrecv(sendbuf=local_grid[-2, :].copy(), dest=row_dst,
                           recvbuf=recv_row, source=row_src)
        if row_src != MPI.PROC_NULL:
            local_grid[0, :] = recv_row
        # 2. Send top inner row UP (to row_src); receive BOTTOM ghost from row_dst (below)
        cart_comm.Sendrecv(sendbuf=local_grid[1, :].copy(), dest=row_src,
                           recvbuf=recv_row, source=row_dst)
        if row_dst != MPI.PROC_NULL:
            local_grid[-1, :] = recv_row

        # Horizontal exchange (columns)
        # 3. Send right inner col to col_dst (right); receive LEFT ghost from col_src (left)
        cart_comm.Sendrecv(sendbuf=local_grid[:, -2].copy(), dest=col_dst,
                           recvbuf=recv_col, source=col_src)
        if col_src != MPI.PROC_NULL:
            local_grid[:, 0] = recv_col
        # 4. Send left inner col to col_src (left); receive RIGHT ghost from col_dst (right)
        cart_comm.Sendrecv(sendbuf=local_grid[:, 1].copy(), dest=col_src,
                           recvbuf=recv_col, source=col_dst)
        if col_dst != MPI.PROC_NULL:
            local_grid[:, -1] = recv_col

        comm_time += MPI.Wtime() - t0

        # --- PARALLEL COMPUTATION ---
        t0 = MPI.Wtime()
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
        compute_time += MPI.Wtime() - t0

    # ==========================================
    # 5. PERFORMANCE METRICS
    # ==========================================
    cart_comm.Barrier()
    end_time = MPI.Wtime()
    total_time = end_time - start_time

    # Average the per-rank compute/comm times across all processes
    mean_compute = cart_comm.reduce(compute_time, op=MPI.SUM, root=0)
    mean_comm = cart_comm.reduce(comm_time, op=MPI.SUM, root=0)

    if rank == 0:
        mean_compute /= size
        mean_comm /= size
        print("-" * 40)
        print(f"Parallel Simulation completed on {size} MPI processes.")
        print(f"Decomposition: {args.decomp}  |  Processor Grid Topology: {dims[0]}x{dims[1]}")
        print(f"Global Grid: {N}x{N}, Steps: {T}")
        print(f"Execution Time: {total_time:.4f} seconds")
        print(f"  Mean compute time: {mean_compute:.4f} s | Mean comm time: {mean_comm:.4f} s")
        print("-" * 40)

        if args.csv:
            append_csv(args.csv, {
                "implementation": "mpi", "N": N, "T": T, "p": p, "l": l,
                "procs": size, "decomp": args.decomp,
                "dims": f"{dims[0]}x{dims[1]}", "seed": args.seed,
                "total_time": f"{total_time:.6f}",
                "compute_time": f"{mean_compute:.6f}",
                "comm_time": f"{mean_comm:.6f}",
            })


if __name__ == "__main__":
    main()
