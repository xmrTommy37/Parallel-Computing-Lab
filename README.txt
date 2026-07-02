Link al report su Overleaf (come editor)
https://overleaf.dei.unipd.it/5783919713nzbsvkzxkfrx#1e551d

Chat con Gemini:
https://share.gemini.google/6btSqS6Twg8A

============================================================
USO DEGLI SCRIPT
============================================================

Simulazione sequenziale:
  ff_env/bin/python forest_fire_seq.py [opzioni]
    -N/--size       dimensione griglia N x N (default 1000)
    -T/--steps      numero di passi temporali (default 500)
    -p/--growth     probabilita' di crescita di un albero (default 0.005)
    -l/--lightning  probabilita' di fulmine (default 0.0001)
    --seed          seme random (default 42)
    --visualize     mostra l'animazione (rallenta molto)
    --csv PATH      appende il risultato di timing a un file CSV

Simulazione parallela (MPI):
  mpiexec -n <num_processi> ff_env/bin/python forest_fire_mpi.py [opzioni]
    stesse opzioni di sopra, piu':
    --decomp 1d|2d  decomposizione a strisce (1D) o a blocchi (2D), default 2d
    --csv PATH      appende total/compute/comm time a un file CSV (solo rank 0)

Esempi:
  ff_env/bin/python forest_fire_seq.py -N 1000 -T 500 --csv results_raw.csv
  mpiexec -n 4 ff_env/bin/python forest_fire_mpi.py -N 1000 --decomp 2d --csv results_raw.csv

============================================================
RIPRODURRE GLI ESPERIMENTI DEL REPORT
============================================================
  ./run_experiments.sh              # esegue tutta la matrice -> results_raw.csv
  ff_env/bin/python aggregate_results.py   # results_raw.csv -> results.csv (medie, speedup, efficienza)
  ff_env/bin/python make_plots.py          # genera le figure in report/figures/

Compilazione report:
  cd report && latexmk -pdf main.tex
