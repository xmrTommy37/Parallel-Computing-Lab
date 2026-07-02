"""Aggregate results_raw.csv (multiple repetitions per configuration) into a
clean summary CSV with mean timings, speedup and efficiency, and emit LaTeX
table snippets used in the report.

Usage:  ff_env/bin/python aggregate_results.py
"""
import csv
from collections import defaultdict
from statistics import mean

RAW = "results_raw.csv"
OUT = "results.csv"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = load(RAW)

    # Group repetitions by configuration key
    groups = defaultdict(list)
    for r in rows:
        key = (r["implementation"], int(r["N"]), int(r["T"]),
               int(r["procs"]), r["decomp"], r["dims"])
        groups[key].append(r)

    # Sequential baseline time per grid size N (from the sequential runs)
    seq_time = {}
    for key, rs in groups.items():
        impl, N, T, procs, decomp, dims = key
        if impl == "sequential":
            seq_time[N] = mean(float(r["total_time"]) for r in rs)

    summary = []
    for key, rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1], k[0][4], k[0][3])):
        impl, N, T, procs, decomp, dims = key
        total = mean(float(r["total_time"]) for r in rs)
        comp = mean(float(r["compute_time"]) for r in rs)
        commt = mean(float(r["comm_time"]) for r in rs)
        base = seq_time.get(N)
        speedup = base / total if base else ""
        efficiency = (speedup / procs) if base else ""
        summary.append({
            "implementation": impl, "N": N, "T": T, "procs": procs,
            "decomp": decomp, "dims": dims, "reps": len(rs),
            "total_time": round(total, 4),
            "compute_time": round(comp, 4),
            "comm_time": round(commt, 4),
            "comm_frac": round(commt / total, 4) if total else "",
            "speedup": round(speedup, 3) if speedup != "" else "",
            "efficiency": round(efficiency, 3) if efficiency != "" else "",
        })

    fields = ["implementation", "N", "T", "procs", "decomp", "dims", "reps",
              "total_time", "compute_time", "comm_time", "comm_frac",
              "speedup", "efficiency"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote {OUT} ({len(summary)} configurations)\n")

    # Pretty print for quick inspection
    hdr = f"{'impl':<11}{'N':>6}{'p':>4}{'decomp':>8}{'dims':>7}{'total':>9}{'comp':>8}{'comm':>8}{'S':>7}{'E':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['implementation']:<11}{s['N']:>6}{s['procs']:>4}{s['decomp']:>8}"
              f"{s['dims']:>7}{s['total_time']:>9}{s['compute_time']:>8}{s['comm_time']:>8}"
              f"{str(s['speedup']):>7}{str(s['efficiency']):>7}")


if __name__ == "__main__":
    main()
