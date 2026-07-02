"""Generate the figures used in the report from results.csv.
Usage:  ff_env/bin/python make_plots.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "report/figures"
os.makedirs(FIGDIR, exist_ok=True)


def load(path="results.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("N", "procs"):
            r[k] = int(r[k])
        for k in ("total_time", "compute_time", "comm_time", "comm_frac"):
            r[k] = float(r[k])
        r["speedup"] = float(r["speedup"]) if r["speedup"] else None
        r["efficiency"] = float(r["efficiency"]) if r["efficiency"] else None
    return rows


rows = load()
NS = [500, 1000, 2000]
PROCS = [1, 2, 4, 8]


def series(decomp, N, field):
    xs, ys = [], []
    for p in PROCS:
        m = [r for r in rows if r["implementation"] == "mpi" and r["decomp"] == decomp
             and r["N"] == N and r["procs"] == p]
        if m and m[0][field] is not None:
            xs.append(p); ys.append(m[0][field])
    return xs, ys


# 1. Execution time vs p (2D)
plt.figure(figsize=(6, 4))
for N in NS:
    x, y = series("2d", N, "total_time")
    plt.plot(x, y, "o-", label=f"N={N}")
plt.xlabel("Number of processes $p$"); plt.ylabel("Total execution time [s]")
plt.title("Execution time (2D decomposition)")
plt.yscale("log"); plt.xticks(PROCS); plt.grid(True, which="both", ls=":")
plt.legend(); plt.tight_layout(); plt.savefig(f"{FIGDIR}/exec_time.png", dpi=150); plt.close()

# 2. Speedup vs p (2D)
plt.figure(figsize=(6, 4))
plt.plot(PROCS, PROCS, "k--", label="Ideal")
for N in NS:
    x, y = series("2d", N, "speedup")
    plt.plot(x, y, "o-", label=f"N={N}")
plt.xlabel("Number of processes $p$"); plt.ylabel("Speedup $S(p)$")
plt.title("Speedup (2D decomposition)")
plt.xticks(PROCS); plt.grid(True, ls=":"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/speedup.png", dpi=150); plt.close()

# 3. Efficiency vs p (2D)
plt.figure(figsize=(6, 4))
plt.axhline(1.0, color="k", ls="--", label="Ideal")
for N in NS:
    x, y = series("2d", N, "efficiency")
    plt.plot(x, y, "o-", label=f"N={N}")
plt.xlabel("Number of processes $p$"); plt.ylabel("Efficiency $E(p)$")
plt.title("Parallel efficiency (2D decomposition)")
plt.xticks(PROCS); plt.grid(True, ls=":"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/efficiency.png", dpi=150); plt.close()

# 4. Topology comparison: total time 1D vs 2D at N=2000
plt.figure(figsize=(6, 4))
for decomp, style in (("1d", "s-"), ("2d", "o-")):
    x, y = series(decomp, 2000, "total_time")
    plt.plot(x, y, style, label=f"{decomp.upper()}")
plt.xlabel("Number of processes $p$"); plt.ylabel("Total execution time [s]")
plt.title("Topology comparison at $N=2000$")
plt.xticks(PROCS); plt.grid(True, ls=":"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/topology.png", dpi=150); plt.close()

# 5. Stacked bar: compute vs comm at N=2000 (2D)
plt.figure(figsize=(6, 4))
comp = [series("2d", 2000, "compute_time")[1][PROCS.index(p)] for p in PROCS]
comm = [series("2d", 2000, "comm_time")[1][PROCS.index(p)] for p in PROCS]
xlab = [str(p) for p in PROCS]
plt.bar(xlab, comp, label="Computation", color="#4c72b0")
plt.bar(xlab, comm, bottom=comp, label="Communication", color="#c44e52")
plt.xlabel("Number of processes $p$"); plt.ylabel("Mean time per process [s]")
plt.title("Computation vs. communication ($N=2000$, 2D)")
plt.legend(); plt.grid(True, axis="y", ls=":")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/comm_compute.png", dpi=150); plt.close()

# 6. Weak scaling (2D, ~500x500 per process)
weak = {1: 500, 4: 1000, 9: 1500, 16: 2000}
xs, tot, comp = [], [], []
for p, N in weak.items():
    m = [r for r in rows if r["implementation"] == "mpi" and r["decomp"] == "2d"
         and r["N"] == N and r["procs"] == p]
    if m:
        xs.append(p); tot.append(m[0]["total_time"]); comp.append(m[0]["compute_time"])
plt.figure(figsize=(6, 4))
plt.plot(xs, tot, "o-", label="Total time")
plt.plot(xs, comp, "s-", label="Computation only")
plt.axhline(tot[0], color="k", ls="--", label="Ideal (constant)")
plt.xlabel("Number of processes $p$"); plt.ylabel("Time [s]")
plt.title("Weak scaling (~500$\\times$500 cells / process, 2D)")
plt.xticks(xs); plt.grid(True, ls=":"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/weak_scaling.png", dpi=150); plt.close()

print("Figures written to", FIGDIR)
