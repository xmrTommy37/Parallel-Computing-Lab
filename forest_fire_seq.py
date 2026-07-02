import numpy as np
import time
import os
import csv
import argparse

# ==========================================
# 1. PARAMETERS & COMMAND-LINE INTERFACE
# ==========================================
EMPTY, TREE, FIRE = 0, 1, 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential Forest Fire cellular-automaton simulation."
    )
    parser.add_argument("-N", "--size", type=int, default=1000,
                        help="Grid dimension (N x N). Default: 1000")
    parser.add_argument("-T", "--steps", type=int, default=500,
                        help="Number of time steps. Default: 500")
    parser.add_argument("-p", "--growth", type=float, default=0.005,
                        help="Probability of tree growth in an empty cell. Default: 0.005")
    parser.add_argument("-l", "--lightning", type=float, default=0.0001,
                        help="Probability of a tree igniting by lightning. Default: 0.0001")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--visualize", action="store_true",
                        help="Show the live animation (slows execution significantly).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a CSV file to append the timing result to.")
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

    # Fixed seed guarantees reproducibility across runs.
    np.random.seed(args.seed)

    # Colormap for visualization: EMPTY=Black, TREE=Green, FIRE=Red
    grid = np.random.choice([EMPTY, TREE], size=(N, N), p=[0.6, 0.4])

    img = None
    if args.visualize:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['black', 'forestgreen', 'red'])
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
        # Create the auxiliary grid (double buffering to avoid race conditions)
        new_grid = grid.copy()

        # Identify the current state of each cell
        is_fire = (grid == FIRE)
        is_tree = (grid == TREE)
        is_empty = (grid == EMPTY)

        # Find neighbors on fire (Von Neumann neighborhood).
        # We pad the grid with 'False' to handle the boundaries without wrapping.
        padded_fire = np.pad(is_fire, pad_width=1, mode='constant', constant_values=False)

        fire_N = padded_fire[:-2, 1:-1]
        fire_S = padded_fire[2:, 1:-1]
        fire_E = padded_fire[1:-1, 2:]
        fire_W = padded_fire[1:-1, :-2]

        # Boolean mask: True if AT LEAST ONE orthogonal neighbor is on fire
        neighbor_on_fire = fire_N | fire_S | fire_E | fire_W

        # Generate a single random matrix for all probability evaluations
        rand_matrix = np.random.rand(N, N)

        # --- Apply transition rules ---
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
        if args.visualize:
            img.set_data(grid)
            import matplotlib.pyplot as plt
            plt.pause(0.01)

    # ==========================================
    # 3. PERFORMANCE METRICS
    # ==========================================
    end_time = time.time()
    execution_time = end_time - start_time

    if args.visualize:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    print("-" * 40)
    print("Simulation completed.")
    print(f"Total Execution Time: {execution_time:.4f} seconds")
    print("-" * 40)

    if args.csv:
        append_csv(args.csv, {
            "implementation": "sequential", "N": N, "T": T, "p": p, "l": l,
            "procs": 1, "decomp": "none", "dims": f"1x1", "seed": args.seed,
            "total_time": f"{execution_time:.6f}",
            "compute_time": f"{execution_time:.6f}", "comm_time": "0.000000",
        })


if __name__ == "__main__":
    main()
