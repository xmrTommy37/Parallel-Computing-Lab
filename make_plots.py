"""Generate the figures used in the report from results.csv.
Usage:  python3 make_plots.py
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
NS = [1000, 2000, 4000]
PROCS = [1, 2, 4, 8, 16, 20]
BIG = 4000  # grid used for the per-process breakdown / topology plots


def get(decomp, N, p, field):
    m = [r for r in rows if r["implementation"] == "mpi" and r["decomp"] == decomp
         and r["N"] == N and r["procs"] == p]
    return m[0][field] if m and m[0][field] is not None else None


def series(decomp, N, field):
    xs, ys = [], []
    for p in PROCS:
        v = get(decomp, N, p, field)
        if v is not None:
            xs.append(p); ys.append(v)
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

# 4. Topology comparison at N=BIG: total time and communication time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for decomp, style in (("1d", "s-"), ("2d", "o-")):
    x, y = series(decomp, BIG, "total_time")
    ax1.plot(x, y, style, label=f"{decomp.upper()}")
    x, y = series(decomp, BIG, "comm_time")
    ax2.plot(x, y, style, label=f"{decomp.upper()}")
ax1.set_title(f"Total time ($N={BIG}$)"); ax2.set_title(f"Communication time ($N={BIG}$)")
for ax in (ax1, ax2):
    ax.set_xlabel("Number of processes $p$"); ax.set_ylabel("Time [s]")
    ax.set_xticks(PROCS); ax.grid(True, ls=":"); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/topology.png", dpi=150); plt.close()

# 5. Stacked bar: compute vs comm at N=BIG (2D)
plt.figure(figsize=(6, 4))
xp = [p for p in PROCS if get("2d", BIG, p, "compute_time") is not None]
comp = [get("2d", BIG, p, "compute_time") for p in xp]
comm = [get("2d", BIG, p, "comm_time") for p in xp]
xlab = [str(p) for p in xp]
plt.bar(xlab, comp, label="Computation", color="#4c72b0")
plt.bar(xlab, comm, bottom=comp, label="Communication", color="#c44e52")
plt.xlabel("Number of processes $p$"); plt.ylabel("Mean time per process [s]")
plt.title(f"Computation vs. communication ($N={BIG}$, 2D)")
plt.legend(); plt.grid(True, axis="y", ls=":")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/comm_compute.png", dpi=150); plt.close()

# 6. Weak scaling (2D, ~500x500 per process)
weak = {1: 500, 4: 1000, 9: 1500, 16: 2000}
xs, tot, comp = [], [], []
for p, N in weak.items():
    t = get("2d", N, p, "total_time"); c = get("2d", N, p, "compute_time")
    if t is not None:
        xs.append(p); tot.append(t); comp.append(c)
plt.figure(figsize=(6, 4))
plt.plot(xs, tot, "o-", label="Total time")
plt.plot(xs, comp, "s-", label="Computation only")
plt.axhline(tot[0], color="k", ls="--", label="Ideal (constant)")
plt.xlabel("Number of processes $p$"); plt.ylabel("Time [s]")
plt.title("Weak scaling (~500$\\times$500 cells / process, 2D)")
plt.xticks(xs); plt.grid(True, ls=":"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/weak_scaling.png", dpi=150); plt.close()

print("Figures written to", FIGDIR)
