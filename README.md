# Implementation and Integration of MLP and PSO from Scratch

The aim of this project is to implement Artificial Neural Networks (ANNs) and Particle Swarm Optimisation (PSO), and experimentally investigate how PSO can be used to optimise an ANN to carry out a specified task.

Both algorithms are written from scratch in Python with NumPy. No existing ANN or PSO library is used. The network has no back-propagation: PSO searches for the network's weights and biases directly.

## Headline result

The task is the UCI banknote authentication dataset (1,372 notes, 4 features, 2 classes). With the final configuration (one hidden layer of 8 tanh neurons, 49 parameters, 30 particles, 100 iterations, nearest-neighbour informants with k = 3), the PSO-trained network reaches a **mean test accuracy of 99.49% (± 0.43) over 10 seeds**. The per-seed range is 98.91% to 100%, and 3 of the 10 seeds classify all 274 test notes correctly.

Each run takes well under a second on the machine that ran the notebook.

## Where the results are stored

Everything the notebook produces is in the `results/` folder.

| File | What it contains |
|---|---|
| `results/final_test.csv` | The final configuration run on 10 seeds and scored on the **test set**. This is where the 98 to 100% accuracy figures come from. |
| `results/run_info.json` | The final configuration and the number of seeds used. |
| `results/best_run.json` | Snapshot of the best of the 10 final runs (seed 6), in the same style as `run_info.json`, plus a summary of all 10 seeds. |
| `results/pso_alpha.csv`, `pso_beta.csv`, `pso_delta.csv`, `pso_gamma.csv`, `pso_epsilon.csv` | PSO coefficient sweeps (validation set). |
| `results/pso_iterations.csv`, `pso_particles.csv` | Iteration-count and swarm-size sweeps (validation set). |
| `results/mlp_width.csv`, `mlp_depth.csv`, `mlp_activation.csv` | Network-size and activation sweeps (validation set). |
| `results/informants.csv` | Informant strategy (random vs nearest) against the number of informants k (validation set). |

Each sweep CSV has one row per seed and setting. Useful columns are `eval_accuracy`, `eval_loss`, `eval_f1`, `train_loss`, `iters_to_within_5pct` (how quickly the run converged), `evaluations`, `seconds`, and the `cfg_*` columns that record the configuration used.

### Best-run snapshot

`results/best_run.json` holds the snapshot of the best run: seed 6, which classified all 274 test notes correctly (test loss 0.0061, F1 1.0), took 0.38 s and 3,030 fitness evaluations, and used the final configuration.

"Best" means the highest test accuracy, with ties broken by the lowest test loss. Seeds 1, 6 and 7 all reached 100%. Because this seed was chosen by looking at test results, quote the 10-seed mean above as the headline figure and treat the seed-6 run as the best case. Seed 6 was also the quickest of the ten runs, but timings depend on the machine and are not a meaningful ranking.

### Final test results, per seed

| Seed | Test accuracy | Test loss | F1 | Seconds |
|---|---|---|---|---|
| 0 | 98.91% | 0.0219 | 0.9879 | 0.43 |
| 1 | 100.00% | 0.0085 | 1.0000 | 0.44 |
| 2 | 99.64% | 0.0061 | 0.9959 | 0.38 |
| 3 | 98.91% | 0.0218 | 0.9878 | 0.38 |
| 4 | 99.27% | 0.0211 | 0.9919 | 0.43 |
| 5 | 99.27% | 0.0271 | 0.9918 | 0.44 |
| 6 | 100.00% | 0.0061 | 1.0000 | 0.38 |
| 7 | 100.00% | 0.0114 | 1.0000 | 0.38 |
| 8 | 99.64% | 0.0140 | 0.9959 | 0.42 |
| 9 | 99.27% | 0.0210 | 0.9918 | 0.44 |

## How the experiments were run

- **Data split.** A stratified 20% test set (274 notes) is held out and used only for the final evaluation. The other 80% is split again into a fit set (60% of the data) and a validation set (20%).
- **Sweeps** train on the fit set and are compared on the validation set, so no tuning decision looks at the test set.
- **Final run** trains on the full 80% and is scored on the test set.
- **Preprocessing.** Features are standardised (z-score) using statistics from the training data only.
- **Seeds.** Every setting is repeated over 10 PSO seeds.
- **One parameter at a time.** Each sweep changes a single hyperparameter and keeps the others at the default (marked in the tables below). Interactions between hyperparameters were not explored.
- **Loss and fitness.** A particle is a full vector of network weights and biases. Its fitness is the binary cross-entropy on the training data, and PSO minimises it. PSO never sees the test data.
- **Evaluations.** Each run makes particles × (iterations + 1) fitness evaluations, which is 30 × 101 = 3,030 for the default.

## Findings

All figures in this section are validation accuracy (mean over 10 seeds) from the sweep CSVs, unless stated otherwise. Differences smaller than roughly one standard deviation (about 0.5 percentage points near the top of these tables) should not be treated as real. The explanations after each table are plausible readings of the numbers, not things the experiments tested directly.

### PSO coefficients

Inertia α: moderate values work best (0.5 and 0.7, about 99.8%). Low inertia (0.3) converges early to a worse solution, and values near 1.0 let particles overshoot.

| Inertia α | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.3` | 96.35% ± 1.54 | 0.0962 | 29.5 |
| `0.5` | 99.78% ± 0.35 | 0.0155 | 79.8 |
| `0.7` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `0.9` | 95.51% ± 2.12 | 0.1448 | 13.4 |
| `1.0` | 95.04% ± 2.54 | 0.1484 | 9.0 |

Cognitive β: 0.5 to 1.5 all perform well. β = 0 (no pull towards the particle's own best) and β = 2.5 are clearly worse.

| Cognitive β | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` | 98.10% ± 1.50 | 0.0717 | 50.4 |
| `0.5` | 99.93% ± 0.15 | 0.0065 | 90.3 |
| `1.0` | 99.96% ± 0.12 | 0.0048 | 95.1 |
| `1.5` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `2.0` | 98.39% ± 1.16 | 0.0433 | 51.9 |
| `2.5` | 94.38% ± 3.17 | 0.1481 | 13.4 |

Informant δ: this one matters most. With δ = 0 accuracy falls to 83%, so the pull towards the informants' best is what drives the search. δ of 1.0 to 1.5 is best.

| Informant δ | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` | 83.25% ± 6.17 | 0.4224 | 1.3 |
| `0.5` | 96.31% ± 0.56 | 0.1156 | 48.0 |
| `1.0` | 99.78% ± 0.35 | 0.0181 | 83.5 |
| `1.5` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `2.0` | 96.17% ± 2.31 | 0.1036 | 28.0 |
| `2.5` | 91.90% ± 4.93 | 0.2351 | 6.4 |

Global γ: the default of 0 (no global-best pull) is best. Adding it reduced accuracy in every setting tried.

| Global γ | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `0.5` | 98.18% ± 1.22 | 0.0523 | 41.5 |
| `1.0` | 98.32% ± 0.91 | 0.0473 | 51.0 |
| `1.5` | 97.01% ± 1.84 | 0.0968 | 28.9 |

Step size ε: ε = 1.0 is best, and both smaller and larger steps are worse.

| Step size ε | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.25` | 94.01% ± 2.79 | 0.1991 | 30.2 |
| `0.5` | 98.76% ± 1.09 | 0.0566 | 55.3 |
| `0.75` | 99.56% ± 0.66 | 0.0196 | 87.1 |
| `1.0` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `1.25` | 95.69% ± 3.00 | 0.1262 | 21.8 |

### Iterations and swarm size

More iterations keep helping: 400 iterations reached 100% validation accuracy on all 10 seeds, at a cost of about 4 times the evaluations of the default. More particles helped up to 50.

| Iterations | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `25` | 97.88% ± 0.77 | 0.0598 | 20.8 |
| `50` | 98.83% ± 0.99 | 0.0309 | 41.1 |
| `100` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `200` | 99.89% ± 0.25 | 0.0057 | 192.6 |
| `400` | 100.00% ± 0.00 | 0.0017 | 377.8 |

| Particles | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `10` | 98.69% ± 1.52 | 0.0292 | 86.6 |
| `20` | 99.74% ± 0.30 | 0.0139 | 85.2 |
| `30` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `50` | 100.00% ± 0.00 | 0.0055 | 91.7 |
| `100` | 99.96% ± 0.12 | 0.0041 | 94.4 |

### Network design

Smaller networks did better. A hidden layer of 2 neurons (13 parameters) scored highest and 32 neurons (193 parameters) lowest, and adding hidden layers lowered accuracy more (98.72% with two hidden layers of 8, 96.82% with three). This is consistent with PSO finding it harder to search a larger parameter space with the same swarm and budget. The dataset is also easy, so a very small network is enough.

| Hidden layer | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `[2]` | 99.93% ± 0.15 | 0.0087 | 93.2 |
| `[4]` | 99.85% ± 0.31 | 0.0085 | 91.6 |
| `[8]` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `[16]` | 99.34% ± 1.07 | 0.0260 | 88.1 |
| `[32]` | 99.01% ± 0.57 | 0.0268 | 79.2 |

| Hidden layers | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `[8]` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `[8, 8]` | 98.72% ± 1.89 | 0.0317 | 72.8 |
| `[8, 8, 8]` | 96.82% ± 1.41 | 0.1110 | 68.2 |

Tanh was slightly ahead of sigmoid, but the gap is within the seed-to-seed spread.

| Activation | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `['tanh']` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `['sigmoid']` | 99.56% ± 0.68 | 0.0189 | 87.5 |

### Informant strategy

Nearest-neighbour informants were much better than random ones when k is small: nearest with k = 1 gave 99.20% against about 96% for random with k = 1 to 5. As k approaches the swarm size (30 particles, so k = 29 means everyone), the two strategies converge to the same accuracy, which is expected because each particle then listens to the whole swarm.

| Strategy and k | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `random-k1` | 96.02% ± 2.02 | 0.1267 | 43.5 |
| `random-k2` | 96.13% ± 1.47 | 0.1251 | 32.9 |
| `random-k3` | 96.02% ± 1.64 | 0.1132 | 25.4 |
| `random-k5` | 96.50% ± 1.55 | 0.1101 | 19.8 |
| `random-k10` | 98.18% ± 1.14 | 0.0722 | 53.4 |
| `random-k20` | 99.89% ± 0.18 | 0.0046 | 86.5 |
| `random-k29` | 99.96% ± 0.12 | 0.0018 | 97.8 |
| `nearest-k1` | 99.20% ± 0.82 | 0.0331 | 87.5 |
| `nearest-k2` | 99.74% ± 0.35 | 0.0117 | 86.9 |
| `nearest-k3` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `nearest-k5` | 99.85% ± 0.26 | 0.0069 | 91.4 |
| `nearest-k10` | 99.96% ± 0.12 | 0.0029 | 95.1 |
| `nearest-k20` | 99.93% ± 0.15 | 0.0023 | 97.6 |
| `nearest-k29` | 99.96% ± 0.12 | 0.0018 | 97.8 |

### Limitations

- The final configuration is the default one, not the best setting from each sweep. Several settings scored slightly higher on validation (for example 400 iterations, 50 particles, β = 1.0, or nearest with k = 10), but they were not combined or re-tested.
- The test set holds 274 notes, so one misclassified note changes accuracy by about 0.36 percentage points.
- Runtime figures depend on the machine and on what else was running.
- Sweeps vary one parameter at a time, so combinations were not tested.

## Project structure

```
data/           banknote dataset (data_banknote_authentication.txt)
mlp/            MLP from scratch: activations, losses, layer, network, builder
pso/            PSO from scratch: fitness interface, particle, informant strategies, population
utils/          dataset loading, stratified split, standardiser, classification metrics
adapter.py      couples the MLP to PSO (the only place that knows both)
training.py     shared helper that builds the network and trains it with PSO
main.py         command-line run (one training run)
app.py          Streamlit GUI (train and inspect one run)
experiments.py  experiment helpers (data protocol, sweeps, statistics, plots)
experiments.ipynb  the experiment notebook that writes results/
tests/          automatic checks (pytest)
results/        CSV and JSON outputs of the experiments
report/         final project report
```

Design: the MLP builder follows the Builder pattern, informant selection follows the Strategy pattern, and the adapter follows the Adapter pattern. The `mlp` and `pso` packages do not import each other.

## Replicating this project

This section is for anyone who wants to run the code and reproduce the results on their own machine. Everything runs on a normal laptop with no GPU, and the dataset is included in the repository.

### 1. What you need

| Requirement | Details |
|---|---|
| Python | Python 3 (the project's virtual environment was created with Python 3.14, and it has not been tested on older versions). Check with `python3 --version`. |
| Packages | `numpy`, `pandas`, `matplotlib`, `ipykernel`, `pytest`, `streamlit` (listed in `requirements.txt`, installed in step 3). |
| Editor for the notebook | VS Code with the Python and Jupyter extensions, or any Jupyter front end. |
| Disk and network | A few hundred MB for the packages. Internet access is only needed for `pip install` and `git clone`. |

On macOS, install Python with Homebrew if you don't have it: `brew install python`.

### 2. Get the code

```bash
git clone https://github.com/shailendragadakari/Implementation-and-Integration-of-MLP-and-PSO-from-Scratch.git
cd Implementation-and-Integration-of-MLP-and-PSO-from-Scratch
```

If you received the project as a zip file instead, unzip it and `cd` into the folder. Check that `data/data_banknote_authentication.txt` is there, because every script reads it from that path.

### 3. Create a virtual environment and install the packages

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. Run every command below from the project's root folder with the environment active. The project has no non-Python dependencies.

### 4. Check that everything works

```bash
pytest tests -q
```

All tests should pass. They cover the activation functions, losses, layers, network, builder, PSO components, the data utilities, and a check that `mlp/` and `pso/` do not import each other.

### 5. Run one training run

```bash
python main.py
```

This loads the dataset, splits it, builds the MLP, trains it with PSO and prints the training loss and test metrics. With the default settings you should see test accuracy of roughly 99 to 100%, and it finishes in about a second. Options can be listed with `python main.py --help`, for example:

```bash
python main.py --hidden 8 --informants nearest --k 3 --particles 30 --iterations 100 --seed 42
python main.py --hidden 4 4 --activations tanh tanh --informants random --k 5
```

Using the same `--seed` gives the same result on the same machine.

### 6. Launch the GUI

```bash
streamlit run app.py
```

Streamlit prints a local address (normally http://localhost:8501) and opens it in your browser. The sidebar sets the network and PSO hyperparameters, and the tabs let you train a single run and inspect the convergence curve, the results, individual predictions and the data. Stop the app with Ctrl+C in the terminal.

### 7. Reproduce the experiments and the results files

1. Open `experiments.ipynb` in VS Code and choose the `.venv` environment as the kernel (top-right "Select Kernel").
2. For a quick dry run, change the line `N_SEEDS = int(os.environ.get("N_SEEDS", 10))` in the first code cell to `N_SEEDS = 3` (or set the `N_SEEDS` environment variable before starting the notebook server). Leave it as it is for the full 10 seeds used in this README.
3. Run all cells. The notebook runs the baseline, every sweep, the informant comparison and the final test evaluation, and in its last cell saves the CSV files and `run_info.json` into `results/`.

Notes:

- Every setting is run over many seeds in a process pool, so the full run takes longer than a single `main.py` run. As a rough guide, a 3-seed dry run took about 77 seconds on a 2-core Linux machine during development. The full 10-seed run was not timed there, so expect several minutes, depending on your CPU.
- Running the notebook **overwrites** the files in `results/`. Copy the folder first if you want to keep the originals for comparison.
- The notebook does not create `results/best_run.json`. That file was derived from `results/final_test.csv`: take the row with the highest `eval_accuracy`, breaking ties with the lowest `eval_loss`.
- Seeds, the data split and the PSO settings are fixed in the code, so your numbers should match `results/` very closely. Small differences are possible if your NumPy version or CPU differs from the one used here, so compare means and spreads rather than expecting identical digits.

To confirm you have reproduced the main result, check that `results/final_test.csv` has 10 rows with test accuracy between about 98.9% and 100% and a mean near 99.5%.

### 8. Make your own changes

- **Different hyperparameters in the notebook:** edit `FINAL_CONFIG` in the final-evaluation cell, or the sweep values passed to `make_configs`.
- **Different network or PSO settings from the command line:** use the `main.py` flags above.
- **A different dataset:** `mlp/builder.py` derives the input size, output size, output activation and loss from the data and a task name (`binary_classification` or `multiclass_classification`). The metrics in `utils/metrics.py` are binary-only, so multi-class data would need extra metric code.

### Troubleshooting

| Problem | Likely fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or `streamlit`, `pandas`) | The environment is not active or the install did not run. Activate `.venv` and run `pip install -r requirements.txt` again. |
| `ModuleNotFoundError: No module named 'utils'` / `mlp` / `pso` | You are not in the project's root folder. `cd` to it first. In the notebook, make sure it was opened from that folder. |
| `FileNotFoundError` for the dataset | `data/data_banknote_authentication.txt` is missing or the folder was moved. Restore it to that path. |
| `command not found: streamlit` | The environment is not active. Run `source .venv/bin/activate`, or use `python -m streamlit run app.py`. |
| The notebook cannot find the kernel | In VS Code choose "Select Kernel", then "Python Environments", then `.venv`. Install the Jupyter extension if the option is missing. |
| `pip install` is slow or fails | Update pip with `python -m pip install --upgrade pip` and retry. Check your network or proxy. |
| Port 8501 is already in use | Run `streamlit run app.py --server.port 8502`. |
| The notebook takes very long | Use `N_SEEDS = 3` for a dry run, or close other heavy applications. |
