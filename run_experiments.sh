#!/usr/bin/env bash
# ==========================================================================
# Experiment driver for the Parallel Forest Fire report.
# Runs the sequential baseline and the MPI implementation (1D and 2D
# decompositions) across a range of grid sizes and process counts, plus a
# weak-scaling series, appending every run to results_raw.csv.
#
# Tuned for a machine with ~20 physical cores.
#
# Usage:
#   ./run_experiments.sh
# Optionally choose the Python interpreter (must have numpy + mpi4py):
#   PYTHON=python3 ./run_experiments.sh
#   PYTHON=/path/to/venv/bin/python ./run_experiments.sh
# ==========================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"      # interpreter with numpy + mpi4py
RAW=results_raw.csv
T=500                        # time steps (as in the report)
REPS=3                       # repetitions per configuration (averaged later)
SEED=42

# --oversubscribe only relaxes Open MPI's slot check; it is harmless when the
# process count does not exceed the number of cores.
MPIRUN="mpiexec --oversubscribe"

rm -f "$RAW"

echo "### Strong scaling + topology comparison ###"
for N in 1000 2000 4000; do
  for r in $(seq 1 $REPS); do
    echo "[rep $r] sequential  N=$N"
    $PY forest_fire_seq.py -N "$N" -T "$T" --seed "$SEED" --csv "$RAW" >/dev/null

    for np in 1 2 4 8 16 20; do
      for decomp in 2d 1d; do
        echo "[rep $r] mpi $decomp np=$np  N=$N"
        $MPIRUN -n "$np" $PY forest_fire_mpi.py -N "$N" -T "$T" \
          --seed "$SEED" --decomp "$decomp" --csv "$RAW" >/dev/null
      done
    done
  done
done

echo "### Weak scaling (2D, ~500x500 cells per process) ###"
# N = 500 * sqrt(p): p=1->500, p=4->1000, p=9->1500, p=16->2000  (all <= 20 cores)
for spec in "1 500" "4 1000" "9 1500" "16 2000"; do
  set -- $spec; np=$1; N=$2
  for r in $(seq 1 $REPS); do
    echo "[rep $r] weak-scaling mpi 2d np=$np  N=$N"
    $MPIRUN -n "$np" $PY forest_fire_mpi.py -N "$N" -T "$T" \
      --seed "$SEED" --decomp 2d --csv "$RAW" >/dev/null
  done
done

echo "Done. Raw results written to $RAW"
echo "Now run:  $PY aggregate_results.py   (produces results.csv)"
